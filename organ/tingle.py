"""Tiny Neural Graph Library (TiNGLe).

This module defines a set of abstractions and functions to program
graph neural networks (and graph convolutional networks), primarily
to be used as the approximator and the discriminator in OrGAN.
The necessity to create this library (instead of using, e.g.,
PyG) is that most existing neural graph libraries assume there is
a strictly defined set of edges (and the graph is represented using
this set). However, it is not the case in OrGAN, where graph
edges are created by the generator, and the presence of an edge is
not strictly binary (it is necessary to allow gragient flow to
the generator).

The TiNGLe uses graph representation most convenient for the
generation process, representing the graph connectivity by an
adjacency matrix. Conceptually, the library follows message passing
framework for graph neural networks and is based on the ideas,
described in https://distill.pub/2021/gnn-intro/. More precidely,
a graph is represented using the following components:

- global representation (one vector, describing graph as a whole);
- nodes representation. In TiNGLe it is assumed, that a node can have
  a type, besides, it can also have some set of features, so:

  - node types (batch x nodes x node_types);
  - node features (batch x nodes x N_F);

- edges representation. Edges can also be of multiple types,
  however, between a pair of nodes it is not possible to have
  more than one edge:

  - edge types (batch x edge_types x nodes x nodes ).
    In this representation, 0 means that there is no edge of the
    respective type, and 1 - that there is. However, other values
    are also possible - they are interpreted as a "power" of connection
    and are used during the propagation through (or from) the
    respective edge.
  - edge representation (one for all types of edges)
    (batch x nodes x nodes x V_F).

The library is based on the message passing framework, specifically,
message massing implemented in TiNGLe consists of the following steps:

- collection. At this step, the library identifies relevant components
  (depending on the message type). Any usage of the edge data is multiplied
  by a "strength" of this connection;
- aggregation. It is a mechanism to obtain one representation from several
  vectors identified during the collection step. Aggragation can be
  two-staged: aggregation of components (passed) via one type of edges and
  further aggregation across several types of edges. Two-staged aggregation
  occurs in V-V message passing. The simplest aggragation type is summation,
  some types of message passing, e.g., V-E allow concatenation, because each
  edge has exactly two incident nodes;
- merge. Merging the aggregated data with the existing component
  representation. The simplest kinds are replacement and concatenation. Some
  of the merging strategies are possible only for certain messages.

The library defines two types of tools:

#. Functions to implement collection and aggragation steps for various kinds of
   message passing.
#. Classes and "orchestration" tools to compose the architecture of a
   complete graph neural network.

As a result, one can build graph neural networks in the following way:

.. code :: python

   gn = torch.nn.ModuleList([
       VV(merge='replace', apply_to_types=True),
       GNNBlock(nodes_module=torch.nn.Linear(2, 4)),
       VV(merge='replace'),
       GNNBlock(nodes_module=torch.nn.Linear(4, 2)),
   ])

"""
import torch


# Low-level functions to perform various kinds of message passing.

def vv_collect_aggregate(nodes, edges, agg='sum', *,
                         add_loops=False,
                         smoothing_eps=1e-6,
                         edge_weights=None):
    """Collection and aggregation for vv-message passing.

    For each node and each edge type the function collects adjacent
    node representations (respecting the edge weight) and aggregates
    them according to `agg` strategy. Then, it also aggregates
    the resulting vectors along edge types (summing with optional
    weights).

    .. math ::

       v_i = EGDE\\_AGG_k(edge\\_weights_k * (AGG_j(E_{kij} * v_j)
           + add\\_loops * v_i))

    Parameters
    ----------
    nodes : torch.tensor (batch x nodes x node_repr)
        Nodes representations.
    edges : torch.tensor (batch x edge_types x nodes x nodes)
        Adjacency matrices.
    agg : str
        The strategy to aggragate nodes representations, obtained
        via one edge type. Can be either `'sum'` (summation) or
        `'avg'` (then the sum is divided by the sum of edge
        weights).
    add_loops : Bool
        If loops should be added to each of the edge types.
    smoothing_eps : float
        Important for the aggregation strategy `'avg'` to avoid
        accidental division by zero.
    edge_weights : torch.tensor (edge_types, )
        Optional edge type weights.

    Returns
    -------
    torch.tensor (batch x nodes)
        Nodes representation.
    """
    if agg not in ('sum', 'avg'):
        raise ValueError('Only ''sum'' and ''avg'' '
                         'aggregation types are supported.')

    n_edge_types = edges.shape[1]
    nodes_ = torch.unsqueeze(nodes, 1).\
        expand(-1, n_edge_types, -1, -1)

    # Суммирование результатов по всем вершинам, смежным с j
    # В результате получается
    #   batch x n_edge_types x n_nodes x node_repr
    hidden = torch.einsum('bijk,bikl->bijl', (edges, nodes_))
    # Если требуется, добавляем петли. То есть в сумму всех
    # соседних вершин добавится еще и представление самой вершины.
    if add_loops:
        hidden += nodes_

    if agg == 'avg':
        degrees = torch.sum(edges, -1, keepdims=True) + smoothing_eps
        hidden = hidden / degrees

    # Агрегация по всем типам дуг
    #
    # В результате получается batch x n_nodes x out_features_list[0]
    # Вот тут "настоящий" R-GCN предполагает взвешенную сумму (с
    # рассчитываемыми или обучаемыми параметрами)
    if edge_weights is not None:
        # Преобразуем форму тензора с коэффициентами, чтобы
        # воспользоваться broadcasting
        hidden = torch.mul(hidden,
                           torch.reshape(edge_weights, (1, -1, 1, 1)))
    hidden = torch.sum(hidden, 1)
    return hidden


def _cartesian(x):
    """Получение описаний для каждой пары индексов.

    Вычисляет две n x n матрицы с описаниями вершин - одна
    соответствует вершине-строке, другая вершине - столбцу.
    На основе этой пары можно реализовывать различные способы
    объединения вершин, инцидентных заданной дуге - конкатенировать,
    вычитать и пр.

    Parameters
    ----------
    x : torch.tensor
        Батч описаний вершин (..., vertexes, k).
    Returns
    -------
    pytorch.tensor, pytorch.tensor
        Результирующие тензоры размерности (..., vertexes, vertexes, k).
    """
    start_dim = len(x.shape) - 2
    n = x.shape[start_dim]
    x1 = torch.unsqueeze(x, start_dim + 1).\
        expand(*([-1] * (start_dim + 1) + [n, -1]))
    x2 = torch.transpose(x1, start_dim, start_dim + 1)
    return x1, x2


def ve_collect_aggregate(nodes, agg='sum'):
    """Collection and aggregation for ve-message passing.

    For each edge the function collects the representations of the
    incident nodes and aggregates them using the specified strategy.
    As a result, there is a new representation for each edge (and
    each edge type). According to the general principles of the
    library, edge weigts are applied only for "outbound" information,
    so it is not the case here.

    .. math ::

       e_{ij} = AGG(v_i, v_j)

    Parameters
    ----------
    nodes : torch.tensor (batch x nodes x node_repr)
        Nodes representation.
    agg : str
        Node representation aggregation strategy. Can be
        `'sum'` (summation), `'avg'` (arithmetic average),
        `'subtract'` (subtraction), or '`cat'` (concatenation).

    Returns
    -------
    torch.tensor (batch x nodes x nodes x k)
        Edge representation. This tensor describes a full graph
        (there is a representation for each pair of nodes).
    """
    t1, t2 = _cartesian(nodes)

    if agg == 'sum':
        return t1 + t2
    elif agg == 'subtract':
        return t1 - t2
    elif agg == 'avg':
        return t1 + t2 / 2
    elif agg == 'cat':
        return torch.cat([t1, t2], dim=-1)
    else:
        raise ValueError('Only (''sum'', ''subtract'', ''avg'','
                         ' ''cat'') aggregation types are supported.')


def ev_collect_aggregate(edge_types, edges,
                         agg='sum', *,
                         outbound=True,
                         smoothing_eps=1e-6,
                         edge_weights=None):
    """Collection and aggregation for ev-message passing.

    Collects all edges incident to a node and aggregates them (adjacency value
    determines weight of an edge).

    if outbound == True:

    .. math ::

       v_i = EGDE\\_AGG_k(edge\\_weights_k * AGG_j(e_{ij} * E_{kij}))

    if outbound == False:

    .. math ::

       v_i = EGDE\\_AGG_k(edge\\_weights_k * AGG_j(e_{ji} * E_{kji}))

    Parameters
    ----------
    edge_types : torch.tensor (batch x edge_types x nodes x nodes)
        Adjacency matrix.
    edges : torch.tensor (batch x nodes x nodes x edge_repr)
        Edge representation.
    agg : str
        Node representations aggregation strategy.
        Can be `'sum'` (summation) or `'avg'` (the sum is divided by the
        total weight of the contributing edges).
    outbound : Bool
        Aggregation should be for outbound edges. ``False`` means that
        the aggregation is across the inbound edges of a node.
    smoothing_eps : float
        Important for the aggregation strategy `'avg'` to avoid
        accidental division by zero.
    edge_weights : torch.tensor (edge_types, )
        Edge type weights for the "second stage" aggregation.

    Returns
    -------
    torch.tensor (batch x nodes)
        Nodes representation.
    """
    if agg not in ('sum', 'avg'):
        raise ValueError('Only ''sum'' and ''avg'' '
                         'aggregation types are supported.')

    if outbound:
        tmp = torch.einsum('bkij,bijl->bkil',
                           (edge_types, edges))
    else:
        tmp = torch.einsum('bkij,bijl->bkjl',
                           (edge_types, edges))

    if agg == 'avg':
        if outbound:
            degrees = torch.sum(edge_types,
                                -1,
                                keepdims=True)
        else:
            degrees = torch.transpose(torch.sum(edge_types,
                                                -2,
                                                keepdims=True),
                                      -1, -2)
        tmp = tmp / (degrees + smoothing_eps)

    # Агрегация по всем типам дуг
    #
    # В результате получается batch x n_nodes x out_features_list[0]
    if edge_weights is not None:
        # Преобразуем форму тензора с коэффициентами, чтобы
        # воспользоваться broadcasting
        tmp = torch.mul(tmp,
                        torch.reshape(edge_weights, (1, -1, 1, 1)))
    tmp = torch.sum(tmp, 1)

    return tmp


# Эти "высокоуровневые" классы уже работают с графом и отвечают за поддержание
# всех пяти компонент, описывающих граф, и за отработку "объединения"
# результатов распространения с текущим содержимым графа.

class GraphSequential(torch.nn.Module):

    def __init__(self, *args):
        super(GraphSequential, self).__init__()
        self.module_list = torch.nn.ModuleList(args)

    def forward(self, gl, nt, n, et, e):
        for module in self.module_list:
            gl, nt, n, et, e = module(gl, nt, n, et, e)
        return gl, nt, n, et, e


class VV(torch.nn.Module):
    """V-V pooling.

    For each node, collects information from all adjacent nodes (using
    each edge type), aggregates and merges to the new node representation.
    """

    def __init__(self, merge='cat', apply_to_types=False, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        merge: str
            A strategy for merging the message passing results to the
            existing nodes representation. Can be `'cat'` (concatenation),
            `'replace'` (replacement), or `'add'` (summation).
        apply_to_types : Bool
            Use node types as input (instead of node representations).
            Can be useful if there are no node representations (yet).
        kwargs : dict
            Parameters for `vv_collect_aggregate`.
        """
        super(VV, self).__init__()
        self.merge = merge
        self.apply_to_types = apply_to_types
        self.args = kwargs

    def forward(self, global_repr, node_types, nodes, edge_types, edges):
        """Forward pass."""
        if self.apply_to_types:
            new_nodes = vv_collect_aggregate(node_types, edge_types,
                                             **self.args)
        else:
            new_nodes = vv_collect_aggregate(nodes, edge_types,
                                             **self.args)
        if self.merge == 'cat':
            nodes = torch.cat([nodes, new_nodes], -1)
        elif self.merge == 'replace':
            nodes = new_nodes
        elif self.merge == 'add':
            nodes = nodes + new_nodes
        else:
            raise ValueError('merge must be in '
                             '(''cat'', ''replace'', ''add'')')
        return global_repr, node_types, nodes, edge_types, edges


class VE(torch.nn.Module):
    """V-E pooling.

    For each edge, collects and aggregates information from the incident
    nodes, then merges with the existing edge representation.
    """

    def __init__(self, merge='cat', apply_to_types=False, **kwargs):
        """Constructor.

        Parameters
        ----------
        merge: str
            Merge strategy. Can be `'cat'` (concatenation) or
            `'replace'` (replacement).
        apply_to_types : Bool
            Use node types as input (instead of node representations).
            Can be useful if there are no node representations (yet).
        kwargs : dict
            Параметры для `ve_collect_aggregate`.
        """
        super(VE, self).__init__()
        self.merge = merge
        self.apply_to_types = apply_to_types
        self.args = kwargs

    def forward(self, global_repr, node_types, nodes, edge_types, edges):
        """Forward pass."""

        if self.apply_to_types:
            new_edges = ve_collect_aggregate(node_types,
                                             **self.args)
        else:
            new_edges = ve_collect_aggregate(nodes,
                                             **self.args)

        if self.merge == 'cat':
            edges = torch.cat([edges, new_edges], -1)
        elif self.merge == 'replace':
            edges = new_edges
        else:
            raise ValueError('merge must be in '
                             '(''cat'', ''replace'')')

        return global_repr, node_types, nodes, edge_types, edges


class EV(torch.nn.Module):
    """E-V pooling.

    For each node, collects and aggregates information from the incident
    edges, then merges with the existing node representation.
    """

    def __init__(self, merge='cat', **kwargs):
        """Constructor.

        Parameters
        ----------
        merge: str
            Merge strategy. Can be `'cat'` (concatenation) or
            `'replace'` (replacement).
        kwargs : dict
            Parameters for `ev_collect_aggregate`.
        """
        super(EV, self).__init__()
        self.merge = merge
        self.args = kwargs

    def forward(self, global_repr, node_types, nodes, edge_types, edges):
        """Forward pass."""
        new_nodes = ev_collect_aggregate(edge_types, edges,
                                         **self.args)

        if self.merge == 'cat':
            nodes = torch.cat([nodes, new_nodes], -1)
        elif self.merge == 'replace':
            nodes = new_nodes
        else:
            raise ValueError('merge must be in '
                             '(''cat'', ''replace'')')

        return global_repr, node_types, nodes, edge_types, edges


class GNNBlock(torch.nn.Module):
    """Plain GNN block, performing independent transformations on
    the selected graph components."""

    def __init__(self, global_module=None,
                 nodes_module=None,
                 edges_module=None):
        """Constructor.

        Parameters
        ----------
        global_module: torch.nn.Module
            Pytorch `Module` to transform the global state.
        nodes_module: torch.nn.Module
            Pytorch `Module` to transform node representations.
        edges_module: torch.nn.Module
            Pytorch `Module` to transform edge represenations.
        """
        super(GNNBlock, self).__init__()
        self.global_module = global_module
        self.nodes_module = nodes_module
        self.edges_module = edges_module

    def forward(self, global_repr, node_types, nodes, edge_types, edges):
        """Forward pass."""
        if self.global_module is not None:
            global_repr = self.global_module(global_repr)
        if self.nodes_module is not None:
            nodes = self.nodes_module(nodes)
        if self.edges_module is not None:
            edges = self.edges_module(edges)
        return global_repr, node_types, nodes, edge_types, edges
