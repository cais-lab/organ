"""
Definition of neural network layers, used in the generative adversarial
network OrGAN.

The module includes the definition of graph convolution layer,
graph aggregation (to aggregate several node representations into
one vector), and edge convolution.

.. warning::
    This module is deprecated and in future releases it will be
    replaced by the Tiny Neural Graph Library (organ.tingle).

"""

import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """Graph convolution layer.

    In the original MolGAN paper (https://arxiv.org/pdf/1805.11973.pdf)
    it is proposed to use Relational GCN,
    (https://arxiv.org/pdf/1703.06103.pdf), however, this class implements
    a usual GCN. Difference is following:

    - parameters of the graph convoltion are the same for all edge types
      (in R-GCN they may be different);
    - in R-GCN components, corresponding to different kinds of edges
      are normalized (the paper discusses several types of such normalization),
      it is not done here.

    In practice, this class implements a block, consisting of two
    convolutions (referred to as hidden and output).
    """

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        """Constructor.

        Parameters
        ----------
        in_features : int
            The number of features, describing each graph node
            (or the number of graph node types, if they are one-hot
            encoded).
        out_feature_list : list
            A list, consisting of two elements - the number of features
            per node in the hidden and output layers respectively.
        b_dim : int
            Not used.
        dropout : float
            Dropout after each convolution [0; 1.0].
        """
        super(GraphConvolution, self).__init__()

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        """Forward pass.
        """
        # input : batch x n_nodes x n_node_types
        # adj : batch x n_edge_types x n_nodes x n_nodes

        # Вычисление
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))],
                             1)
        # Суммирование результатов по всем вершинам, смежным с j
        # В результате получается
        #   batch x n_edge_types x n_nodes x out_features_list[0]
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        # Суммирование по всем типам дуг (плюс результаты преобразования
        # самой вершины).
        # В результате получается batch x n_nodes x out_features_list[0]
        # Вот тут "настоящий" R-GCN предполагает взвешенную сумму (с
        # рассчтиваемыми или обучаемыми параметрами)
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        # Применение активации
        hidden = activation(hidden) if activation is not None else hidden
        # ...и дропаута
        hidden = self.dropout(hidden)

        # Следующий блок аналогичен (именно поэтому желателен рефакторинг)
        output = torch.stack([self.linear2(hidden)
                              for _ in range(adj.size(1))],
                             1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):
    """Aggregation of node descriptions.

    The layer aggregates nodes descriptions into a global graph representation
    vector. The implemented aggregation is done in the following way.
    There are two representations of the nodes (n_nodes x in_features and
    n_nodes x m_dim). They are concetenated (outside this class),
    and after that several non-linear transformations are applied to the
    result of this concatenation, so that in is mapped into new
    feature space (out_features), the results are multiplied and then
    summed for all the nodes.

    It is used in the following way: during graph convolution new node
    representations are obtained for each node. This representation is
    concatenated with the original one (outside this class) and then
    is transformed into one vector using this class.

    .. warning::
        TODO (hatter): I think, this class is not very logical - the
        constructor receives the dimensions of two (aggregated)
        representations, but `forward()` receives only one (concatenated)
        tensor. One should either do concatenation, or construct a layer
        with the concatenated dimension size.
    """

    def __init__(self, in_features, out_features, m_dim, dropout):
        """Constructor.

        Parameters
        ----------
        in_features : int
            Number of features in the first node representation.
        out_features : int
            Number of features in the output (global, aggregated)
            representation.
        m_dim : int
            Number of features in the second input representation.
        dropout : float
            Droupout [0; 1].
        """
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + m_dim, out_features),
            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + m_dim, out_features),
            nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        """Forward pass.

        Parameters
        ----------
        input : torch.tensor
            Concatenated nodes representation
            batch x n_nodes x (in_features + m_dim).
        activation : Callable
            Activation function for the aggregated representation.

        Returns
        -------
        torch.tensor
            Aggregated (global) graph representation batch x out_features
        """
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), 1)
        output = activation(output) if activation is not None \
            else output
        output = self.dropout(output)

        return output


def cartesian(x):
    """Obtain descriptions for each pair of indices.

    Calculates two n x n tensors with node descriptions -
    the first one corresponds to the row node, the second one
    to the column node.
    Based on this pair, one can implement various ways of
    aggregating the representations of nodes, incident to one edge -
    concatenate, subtract, etc.

    Parameters
    ----------
    x : torch.tensor
        Batch of vertex descriptions (..., vertexes, k).

    Returns
    -------
    pytorch.tensor, pytorch.tensor
        Resulting tensors of the shape (..., vertexes, vertexes, k).
    """
    start_dim = len(x.shape) - 2
    n = x.shape[start_dim]
    x1 = torch.unsqueeze(x, start_dim + 1).\
        expand(*([-1] * (start_dim + 1) + [n, -1]))
    x2 = torch.transpose(x1, start_dim, start_dim + 1)
    return x1, x2


class EdgeConvolution(torch.nn.Module):
    """Edge convolution layer."""

    def __init__(self, node_dim, out_dim, edge_types):
        """Constructor.

        Parameters
        ----------
        node_dim : int
            Number of input node features.
        out_dim : int
            Number of output node features.
        edge_types : int
            Number of edge types.
        """
        super(EdgeConvolution, self).__init__()
        self.linears = [nn.Linear(2 * node_dim, out_dim)
                        for _ in range(edge_types)]
        self.linears = nn.ModuleList(self.linears)

    def forward(self, nodes, adj):
        """Forward pass.

        Parameters
        ----------
        nodes : torch.tensor
            Batch of nodes representations
            (batch x nodes x node_dim).
        adj : torch.tensor
            Adjacency matrix
            (batch x edge_types x nodes x nodes).

        Returns
        -------
        torch.tensor
            New node representations
            (batch x nodes x out_dim).
        """
        # Получение двух n x n матриц с описаниями
        # вершин - одна соответствует вершине-строке,
        # другая вершине - столбцу. На основе этой
        # пары можно порождать различные способы объединения
        # вершин, инцидентных заданной дуге - конкатенировать,
        # вычитать и пр.
        t1, t2 = cartesian(nodes)
        # В данном случае, представления смежных вершин
        # просто конкатенируются
        tmp = torch.cat([t1, t2], dim=-1)
        # К описанию различных типов дуг применяются
        # различные преобразования. Результаты суммируются.
        out = []
        for i, l in enumerate(self.linears):
            # Матрица смежности соответствующего типа
            type_adj = torch.unsqueeze(adj[:, i, :, :], -1)
            # Применение преобразования к описаниям пар вершин
            x = l(tmp)
            # Выбираем только те результаты, которые соответствуют
            # дугам рассматриваемого на данной итерации типа
            x = torch.mul(x, type_adj)
            out.append(x)
        # Суммирование результатов (без весов)
        out = torch.stack(out)
        out = torch.sum(out, 0)
        return out


def edge_aggregation(edges):
    return torch.mean(edges, dim=(1, 2))
