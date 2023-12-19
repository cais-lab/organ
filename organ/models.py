"""Neural models OrGAN is built of.

This module defines several flavours of basic generator and
discriminator neural networks.

You can as well define your own generator and discriminator
architectures.

Both generator and discriminator must be PyTorch modules
(derive from `torch.nn.Module`).

Generator's `forward()` method has to accept two positional
parameters:

- `condition` (`torch.tensor` of shape (batch, cond_dim) or
  ``None``) with input condition (requirements to the
  sample to be generated). If a generator model doesn't
  support conditional generation it may ignore this parameter;
- `x` (`torch.tensor` of shape (batch, z_dim)) with input
  noise.

and return a 3-tuple:

- edges specification (batch, nodes, nodes, edge_types);
- nodes specification (batch, nodes, node_types);
- optional node parameters (batch, nodes, node_features).

Discriminator's `forward()` method has to accept following
parameters:

- `edges` (`torch.tensor` of shape
  (batch, nodes, nodes, edge_types)) - adjacency matrices;
- `nodes` (`torch.tensor` of shape
  (batch, nodes, node_types)) - types of nodes;
- `node_params` (`torch.tensor` of shape
  (batch, nodes, node_features) or ``None``) - parameters of
  each node. If the discriminator doesn't support parameters
  it may ignore this parameter;
- `condition` (`torch.tensor` of shape
  (batch, condition_features) or ``None``) - condition,
  under which the graph was generated. If the discriminator
  doesn't support conditional generation it may ignore this
  parameter;
- `activation` - an activation function to apply to the
  results.

"""
import torch
import torch.nn as nn

import organ.tingle

from organ.layers import GraphConvolution, GraphAggregation, \
    EdgeConvolution, edge_aggregation


class SimpleGenerator(nn.Module):
    """Generator network for OrGAN.

    Generator is a non-linear neural transformation from an input
    vector (consisting of `z_dim` features) to a graph, describing an
    organization structure.

    The generator is built of several fully connected layers, making
    a series of transformations, followed by "forking" the representation
    into nodes description and adjacency matrix::

                               Input (batch x z_dim)
                                          |
                       Fully connected (FC) layers (tanh, dropout)
                                 |                 |
                           FC layer for       FC layer for
                               edges             nodes
                          (no activation)    (no activation)
    """

    def __init__(self, conv_dims, z_dim, vertexes, edges, dropout):
        """Constructor.

        Parameters
        ----------
        conv_dims : list
            List, describing the FC layers in the beginning of the
            generator.
        z_dim : int
            Input dimensions.
        vertexes : int
            Number of vertexes in the graph (which is the same as
            the number of node types).
        edges : int
            Number of connections (edges).
        dropout : float
            Droupout [0; 1] (applied to each layer, including output).
        """
        super(SimpleGenerator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = vertexes

        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1],
                                     edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1],
                                     vertexes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, _, x):
        """Forward pass.

        .. note::
           Output values are not bounded, activation is not applied.

        Parameters
        ----------
        x : torch.tensor
            Input tensor of batch x z_dim.
        Returns
        -------
        tuple
            A tuple, consisting of edges specification
            (batch x vertexes x vertexes x edges) and nodes specification
            (batch x vertexes). It is assumed, that a vertex of certain type
            can be placed only in certain position (overall, vertex type
            is equivalent to its position), therefore, it is enough to
            form only presence of a node in certain position, its type
            is known automatically.
        """
        # Применение начальной группы полносвязных слоев
        output = self.layers(x)

        # Получение спецификации связей графа
        # (здесь view() осуществляет преобразование размерности из плоского
        # вектора
        edges_logits = self.edges_layer(output) \
            .view(-1, self.edges, self.vertexes, self.vertexes)
        # Получение симметричной (!) матрицы смежности
        # edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        # TODO: (hatter) Мне странно применение дропаута к выходному слою
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        # Получение спецификации вершин графа
        nodes_logits = self.nodes_layer(output)
        # TODO: (hatter) Мне странно применение дропаута к выходному слою
        nodes_logits = self.dropout(
            nodes_logits.view(-1, self.vertexes))

        return edges_logits, nodes_logits, None


class EdgeAwareGenerator(nn.Module):
    """Generator that creates edges based on types of nodes.

    .. note::
       This generator *does NOT* support conditional generation
       and parametric organizations. For such full-fledged
       generator see `CPGenerator`.

    """

    def __init__(self, conv_dims, edge_conv_dims, z_dim,
                 vertexes, edges, dropout):
        """Constructor.

        Parameters
        ----------
        conv_dims : list
            List, describing the FC layers in the beginning of the
            generator.
        edge_conv_dims : list
            List, describint the edge layers.
        z_dim : int
            Input dimensions.
        vertexes : int
            Number of vertexes in the graph (which is the same as
            the number of node types).
        edges : int
            Number of connections (edges).
        dropout : float
            Droupout [0; 1] (applied to each layer, including output).
        """
        super(EdgeAwareGenerator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = vertexes

        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_ctx_layer = nn.Linear(conv_dims[-1],
                                         32)
        edge_layers = []
        for c0, c1 in zip([self.nodes + 32] + edge_conv_dims[:-1],
                          edge_conv_dims):
            edge_layers.append(nn.Linear(c0, c1))
            edge_layers.append(nn.Tanh())
        self.edge_layers = nn.Sequential(*edge_layers)

        self.edges_layer = nn.Linear(edge_conv_dims[-1],
                                     edges)
        self.nodes_layer = nn.Linear(conv_dims[-1],
                                     vertexes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, _, x):
        """Forward pass."""

        # Применение начальной группы полносвязных слоев
        output = self.layers(x)

        # Получение спецификации вершин
        nodes_logits = self.nodes_layer(output)

        # Описание вершин в развернутую форму
        nodes_sigm = torch.sigmoid(nodes_logits)
        nodes_hat = torch.diag_embed(nodes_sigm)
        nodes_hat[:, :, 0] += (1 - nodes_sigm)

        # Получение спецификации связей графа
        # Контекст генерации графа
        ctx = self.edges_ctx_layer(output)
        # Описания (типы) вершин, инцидентных
        # ребру
        cc = organ.tingle._cartesian(nodes_hat)
        # Контекст + данные об инцидентных вершинах
        edges_data = torch.cat([cc[0] - cc[1],
                                ctx.view(-1, 1, 1, 32).
                                expand(-1, self.nodes, self.nodes, 32)],
                               axis=-1)
        edges = self.edge_layers(edges_data)
        edges_logits = self.edges_layer(edges)

        return edges_logits, nodes_logits, None


class Discriminator(nn.Module):
    """Discriminator for OrGAN.

    Discriminator receives a graph (described by edges and nodes),
    applies a series of graph convolutions and fully connected layers to
    obtain a single number (characterizing graph as a whole, e.g., its
    consistency or verisimilitude).

    .. note::
       This discriminator *does NOT* support conditional generation
       and parametric organizations. For such full-fledged
       disciminator see `CPDiscriminator`.

    """

    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        """Constructor.

        Parameters
        ----------
        conv_dim : tuple, list
            Transformation complexity specification. Consists of
            three components:
            - graph_conv_dim (a list, describing graph convolution
              parameters),
            - aux_dim (a number of features in global graph
              representation), and
            - linear_dim (a list, describing the numbers of neurons in
              fully connected layers).
        m_dim : int
            The number of node types (including 0, absense of a node).
        b_dim : int
            The number of edge types (including 0, absense of an edge).
        dropout : float
            Dropout [0; 1]. Applied at each stage (for all
            graph transformations and after each fully connected layer).
        """
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim

        # Серия графовых преобразований
        self.gcn_layer = GraphConvolution(m_dim,
                                          graph_conv_dim,
                                          b_dim,
                                          dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1],
                                          aux_dim,
                                          m_dim,
                                          dropout)
        self.edge_layer = EdgeConvolution(m_dim,
                                          16,  # TODO
                                          b_dim - 1)

        # Группа полносвязных слоев
        layers = []
        for c0, c1 in zip([aux_dim + 16] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, nodes, ignored, ignored_, activation=None):
        """Forward pass.

        Parameters
        ----------
        adj : torch.tensor
            Adjacency matrices, batch x vertexes x vertexes x edges.
        nodes : torch.tensor
            Nodes specification, batch x vertexes x nodes.
        ignored
            Ignored.
        activation : Callable
            Activation function for the last layer.
        """
        # Предполагается, что между вершинами может присутствовать
        # только один тип связей (либо не присутствовать вовсе). И тип связи
        # 0 означает как раз отсутствие связи между вершинами.
        # Для графовых сверток отсутствующие связи не нужны, поэтому
        # матрица смежности для типа связи 0 уничтожается, после чего
        # матрицы преобразуются в вид "тип сначала"
        # (batch x edges x vertexes x vertexes).
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        annotations = nodes
        h = self.gcn_layer(annotations, adj)
        # Представления ребер
        h1 = self.edge_layer(annotations, adj)
        h1 = edge_aggregation(h1)
        h1 = torch.tanh(h1)
        # Свертка графа в один вектор
        annotations = torch.cat((h, nodes), -1)
        h = self.agg_layer(annotations, torch.tanh)
        # Объединим интегральное описание ребер и вершин
        h = torch.cat([h, h1], -1)
        # Применение серии полносвязных слоев
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output


class CPDiscriminator(nn.Module):
    """Conditional parametric discriminator for OrGAN.

    Discriminator receives a graph (described by edges, nodes, node
    features and condition), applies a series of graph convolutions and
    fully connected layers to obtain a single number (characterizing
    the graph as a whole, e.g., its consistency or verisimilitude).
    """

    def __init__(self, conv_dim, fc_dim, cond_encoder_dim,
                 n_node_types, n_edge_types, n_cond_features,
                 n_node_features, dropout):
        """Constructor.

        Parameters
        ----------
        conv_dim : tuple, list
            Transformation complexity specification. Consists of
            three components:
            - graph_conv_dim (a list, describing graph convolution
              parameters),
            - aux_dim (a number of features in global graph
              representation), and
            - linear_dim (a list, describing the numbers of neurons in
              fully connected layers).
        fc_dim : list
            Specification of a fully-connected block processing graph
            representation and parameter values.
        cond_encoder_dim : list
            Specification of the condition encoder.
        n_node_types : int
            The number of node types (including 0, absense of a node).
        n_edge_types : int
            The number of edge types (including 0, absense of an edge).
        n_cond_features : in
            The number of condition parameters (0 or None to disable).
        n_node_features : in
            The number of node features (0 or None to disable).
        dropout : float
            Dropout [0; 1]. Applied at each stage (for all
            graph transformations and after each fully connected layer).
        """
        super(CPDiscriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim

        self.n_nodes = n_node_types
        if n_node_features is not None:
            self.n_node_features = n_node_features
        else:
            self.n_node_features = 0

        # Серия графовых преобразований
        self.gcn_layer = GraphConvolution(n_node_types +
                                          self.n_node_features,
                                          graph_conv_dim,
                                          n_edge_types,
                                          dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] +
                                          self.n_node_features,
                                          aux_dim,
                                          n_node_types,
                                          dropout)
        self.edge_layer = EdgeConvolution(n_node_types +
                                          self.n_node_features,
                                          16,  # TODO
                                          n_edge_types - 1)

        # If there is some condition, we must encode it
        if n_cond_features is not None and n_cond_features > 0:
            self.encode_condition = FCBlock(n_cond_features,
                                            cond_encoder_dim[:-1],
                                            cond_encoder_dim[-1],
                                            nn.Tanh, 0)
            encoded_cond = cond_encoder_dim[-1]
        else:
            self.encode_condition = None
            encoded_cond = 0

        self.fc_group = FCBlock(aux_dim + 16 +  # graph
                                encoded_cond +  # condition
                                n_node_types +  # node presence
                                # node features
                                n_node_features * n_node_types,
                                fc_dim,
                                aux_dim,
                                nn.Tanh, 0)

        # Группа полносвязных слоев
        layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, nodes, node_params, cond, activation=None):
        """Forward pass.

        Parameters
        ----------
        adj : torch.tensor
            Adjacency matrices, batch x vertexes x vertexes x edges.
        nodes : torch.tensor
            Nodes specification, batch x vertexes x nodes.
        node_params : torch.tensor
            Node parameter values, batch x vertexes x node_features.
        cond : torch.tensor
            Condition, batch x cond_features.
        activation : Callable
            Activation function for the last layer.
        """
        # Предполагается, что между вершинами может присутствовать
        # только один тип связей (либо не присутствовать вовсе). И тип связи
        # 0 означает как раз отсутствие связи между вершинами.
        # Для графовых сверток отсутствующие связи не нужны, поэтому
        # матрица смежности для типа связи 0 уничтожается, после чего
        # матрицы преобразуются в вид "тип сначала"
        # (batch x edges x vertexes x vertexes).
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        annotations = nodes if node_params is None else \
            torch.cat((nodes, node_params), -1)
        h = self.gcn_layer(annotations, adj)
        # Представления ребер
        h1 = self.edge_layer(annotations, adj)
        h1 = edge_aggregation(h1)
        h1 = torch.tanh(h1)
        # Свертка графа в один вектор
        annotations = torch.cat((h, annotations), -1)
        h = self.agg_layer(annotations, torch.tanh)
        # Объединим интегральное описание ребер и вершин
        # h = torch.cat([h, h1], -1)

        # Закодируем контекст
        if self.encode_condition is not None:
            cond = self.encode_condition(cond)
        else:
            cond = None

        # Collect a group from nodes, edges and parameters
        comps = [h,                                             # graph nodes
                 h1,                                            # graph edges
                 (1 - nodes[:, :, 0]).view(-1, self.n_nodes),   # node presence
                 ]
        # Condition, if present
        if cond is not None:
            comps.append(cond)
        # Params, if present
        if node_params is not None:
            comps.append(node_params.view(-1,
                                          self.n_nodes * self.n_node_features))

        h = torch.cat(comps, -1)
        h = self.fc_group(h)

        # Применение серии полносвязных слоев
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output


class FCBlock(nn.Module):
    """A fully-connected block."""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation, dropout=0.0):
        """Constructor.

        Parameters
        ----------
        input_dim : int
            Input features.
        hidden_dims : list[int]
            Dimensions of the hidden layers.
        output_dim : int
            Number of output features.
        activation : Callable
            A constructor for an activation layer (e.g., nn.Tanh).
        droupout : float
            Dropout probability after each layer.
        """
        super(FCBlock, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for c0, c1 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(c0, c1))
            if activation is not None:
                layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class CPGenerator(nn.Module):
    """Conditional parametric generator."""

    def __init__(self, conv_dims, edge_conv_dims, param_dims,
                 z_dim, cond_dim, nodes, edge_types, node_features,
                 dropout):
        """Constructor.

        Parameters
        ----------
        conv_dims : list
            List, describing the FC layers in the beginning of the
            generator.
        edge_conv_dims : list
            List, describing the FC layers for edge generation.
        param_dims : list
            Param generation specification.
        z_dim : int
            Input dimensions.
        cond_dim : lit
            Condition dimensions.
        nodes : int
            Number of types of nodes.
        edges : int
            Number of connections (edges).
        node_features : int
            Number of features per node.
        dropout : float
            Droupout [0; 1] (applied to each layer, including output).
        """
        super(CPGenerator, self).__init__()

        self.nodes = nodes
        self.edges = edge_types
        if node_features is not None:
            self.features = node_features
        else:
            self.features = 0
        if cond_dim is not None:
            self.cond_dim = cond_dim
        else:
            self.cond_dim = 0

        # Encode context
        if self.cond_dim > 0:
            self.cond_encoder = FCBlock(cond_dim, [64], 12, nn.Tanh, 0)
            cond_encoded_dim = 12
        else:
            self.cond_encoder = None
            cond_encoded_dim = 0

        # Encode noise + context
        self.layers = FCBlock(z_dim + cond_encoded_dim,
                              conv_dims[:-1],
                              conv_dims[-1],
                              nn.Tanh, dropout)

        # General edge specification
        self.edges_spec_layer = nn.Linear(conv_dims[-1], 32)
        self.edge_layers = FCBlock(nodes + 32,
                                   edge_conv_dims[:-1],
                                   edge_conv_dims[-1],
                                   nn.Tanh, dropout)

        # Output layers
        self.nodes_layer = nn.Linear(conv_dims[-1],
                                     nodes)
        self.edges_layer = nn.Linear(edge_conv_dims[-1],
                                     edge_types)

        if self.features > 0:
            # TODO: Probably, it may be useful to
            # employ graph convolutions here
            self.params_layer = FCBlock(conv_dims[-1] +
                                        z_dim +
                                        self.nodes +
                                        cond_encoded_dim,
                                        param_dims[:-1],
                                        param_dims[-1],
                                        nn.ELU, 0)
            self.params_out = nn.Linear(param_dims[-1],
                                        nodes * self.features)

    def forward(self, cond, z):
        """Forward pass."""

        if cond is not None:
            # Encode condition
            cond = self.cond_encoder(cond)
            # Concat inputs
            x = torch.cat((cond, z), dim=-1)
        else:
            cond = torch.empty((z.shape[0], 0))
            x = z

        # Apply the fully connected layers to obtain
        # global graph representation
        x = self.layers(x)

        # Nodes specification
        nodes_logits = self.nodes_layer(x)

        # Nodes to extended form
        nodes_sigm = torch.sigmoid(nodes_logits)
        nodes_hat = torch.diag_embed(nodes_sigm)
        nodes_hat[:, :, 0] += (1 - nodes_sigm)

        # Edges specification
        # Account for the global graph representation
        ctx = self.edges_spec_layer(x)
        # Collect node types incident to each edge
        cc = organ.tingle._cartesian(nodes_hat)
        # Global + incident nodes
        edges_data = torch.cat([cc[0] - cc[1],
                                ctx.view(-1, 1, 1, 32).
                                expand(-1, self.nodes, self.nodes, 32)],
                               axis=-1)
        edges = self.edge_layers(edges_data)
        edges_logits = self.edges_layer(edges)

        # Node parameters
        # node_params = self.params_layer(x)
        # node_params = node_params.view(-1, self.nodes, 1)

        if self.features > 0:
            # Node parameters
            node_params = torch.cat((x,
                                     z,
                                     nodes_sigm.view(-1, self.nodes).detach(),
                                     cond), -1)
            node_params = self.params_layer(node_params)
            node_params = self.params_out(node_params)
            node_params = node_params.view(-1, self.nodes, self.features)
        else:
            node_params = None

        return edges_logits, nodes_logits, node_params


class __Deprecated(nn.Module):
    """Discriminator for OrGAN.

    Discriminator receives a graph (described by edges and nodes),
    applies a series of graph convolutions and fully connected layers to
    obtain a single number (characterizing graph as a whole, e.g., its
    consistency or verisimilitude).
    """

    def __init__(self, cond_dim, node_types, node_params, dropout):
        """Constructor.

        Parameters
        ----------
        conv_dim : tuple, list
            Transformation complexity specification. Consists of
            three components:
            - graph_conv_dim (a list, describing graph convolution
              parameters),
            - aux_dim (a number of features in global graph
              representation), and
            - linear_dim (a list, describing the numbers of neurons in
              fully connected layers).
        m_dim : int
            The number of node types (including 0, absense of a node).
        b_dim : int
            The number of edge types (including 0, absense of an edge).
        dropout : float
            Dropout [0; 1]. Applied at each stage (for all
            graph transformations and after each fully connected layer).
        """
        super(__Deprecated, self).__init__()

        self.node_encoder = FCBlock(node_params,
                                    [64],
                                    32,
                                    None, 0.0)

        # Graph convolution layers:
        self.layers = organ.tingle.GraphSequential(
            organ.tingle.GNNBlock(
                nodes_module=FCBlock(node_types + 32,
                                     [],
                                     32,
                                     None, 0)),
            organ.tingle.VV(merge='replace', agg='avg'),
            organ.tingle.GNNBlock(
                nodes_module=FCBlock(32,
                                     [],
                                     64,
                                     None, 0)),
            organ.tingle.VV(merge='replace', agg='avg'),
            organ.tingle.VE(merge='replace', agg='subtract'),
            organ.tingle.EV(merge='cat', agg='avg'),
            organ.tingle.GNNBlock(
                nodes_module=FCBlock(64 + 64,
                                     [256],
                                     128,
                                     None, 0)),
        )

        self.cond_encoder = FCBlock(cond_dim,
                                    [32],
                                    16, None, 0)

        self.fc_layers = FCBlock(16 + 128,
                                 [128, 64, 32],
                                 1, None, 0)

    def forward(self, edges, hidden, nodes, node_params, cond,
                activation=None):
        """Forward pass.

        Parameters
        ----------
        adj : torch.tensor
            Adjacency matrices, batch x vertexes x vertexes x edges.
        hidden : torch.tensor
            MUST be None! Initially, it should be some optional
            information about nodes batch x vertexes x nodes that could
            be accesible by graph convlutions and graph aggregations.
            However, currently, the discriminator is created in such
            a way, that graph convolutions are constructed without
            such possibility, and if something is passed here,
            there will be a dimensions mismatch problems.
        node : torch.tensor
            Nodes specification, batch x vertexes x nodes.
        activation : Callable
            Activation function for the last layer.
        """
        # Предполагается, что между вершинами может присутствовать
        # только один тип связей (либо не присутствовать вовсе). И тип связи
        # 0 означает как раз отсутствие связи между вершинами.
        # Для графовых сверток отсутствующие связи не нужны, поэтому
        # матрица смежности для типа связи 0 уничтожается, после чего
        # матрицы преобразуются в вид "тип сначала"
        # (batch x edges x vertexes x vertexes).
        edges = edges[:, :, :, 1:].permute(0, 3, 1, 2)

        n = self.node_encoder(node_params)
        n = torch.cat((nodes, n), axis=-1)

        # Apply graph transformations
        gl, nt, n, et, e = self.layers(None, nodes, n, edges, None)

        gl = torch.mean(n, axis=-2)

        # Process condition
        cond = self.cond_encoder(cond)
        # Concat condition and graph and process
        gl = torch.cat((gl, cond), axis=-1)
        gl = self.fc_layers(gl)
        output = activation(gl) if activation is not None else gl

        return output

class CompletionGenerator(nn.Module):
    """Structure completion generator."""

    def __init__(self, conv_dims, edge_conv_dims,
                 z_dim, node_types, edge_types,
                 dropout):
        """Constructor.

        Parameters
        ----------
        conv_dims : list
            List, describing the FC layers in the beginning of the
            generator.
        edge_conv_dims : list
            List, describing the FC layers for edge generation.
        z_dim : int
            Input dimensions.
        node_types : int
            Number of types of nodes.
        edge_types : int
            Number of connections (edges).
        dropout : float
            Droupout [0; 1] (applied to each layer, including output).
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types

        self.node_repr_size = 3
        self.node_hidden_size = 10
        
        # Encode the input noise
        self.noise_encoder = FCBlock(z_dim,
                                     conv_dims[:-1],
                                     conv_dims[-1],
                                     nn.Tanh,
                                     dropout)

        # Initial data for each node
        self.global_to_nodes = nn.Linear(conv_dims[-1],
                                         self.node_types * self.node_repr_size)
        # Hidden representation of the initial nodes
        # data and the specification
        self.nodes_to_hidden = nn.Sequential(
            nn.Linear(self.node_repr_size + 2,
                      self.node_hidden_size),
            nn.ReLU()
        )
        # Weights
        self.context = nn.Parameter(torch.rand(node_types,
                                               node_types,
                                               dtype=torch.float32))
        # Nodes encoder
        self.nodes_encoder = nn.Sequential(
            nn.Linear(2 * self.node_hidden_size, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # General edge specification
        self.edges_spec_layer = nn.Linear(conv_dims[-1], 32)
        self.edge_layers = FCBlock(self.node_types + 32,
                                   edge_conv_dims[:-1],
                                   edge_conv_dims[-1],
                                   nn.Tanh, dropout)

        # Edge output layer
        self.edges_layer = nn.Linear(edge_conv_dims[-1],
                                     edge_types)


    def forward(self, nodes, edges, nodes_mask, edges_mask, ignored, z):
        """Forward pass.
        
        Parameters
        ----------
        nodes : torch.tensor
            The specification of existing nodes,
            (batch, node_types, node_types)
        edges : torch.tensor
            The specification of existing edges,
            (batch, node_types, node_types, edge_types)
        nodes_mask : torch.tensor
            The mask defining the presence of which nodes
            should be conserved in the output,
            (batch, node_types)
        edges_mask : torch.tensor
            The mask defining the presence of which nodes
            should be conserved in the output,
            (batch, node_types, node_types)
        z : torch.tensor
            Noise vector.
        """

        batch_size = z.shape[0]
        
        # Apply the fully connected layers to obtain
        # global graph representation
        z = self.noise_encoder(z)

        # Nodes cue from the global graph
        # representation
        x = self.global_to_nodes(z).view(-1,
                                         self.node_types,
                                         self.node_repr_size)

        # Join nodes cue from the global representation with the 
        # external specification (required nodes)
        x = torch.cat([x, 
                       (1. - nodes[:, :, 0]).unsqueeze(-1),
                       nodes_mask.unsqueeze(-1)], dim=2)
        # To a hidden representation of nodes
        x = self.nodes_to_hidden(x)
        # Account for other nodes
        x_ = organ.tingle.vv_collect_aggregate(x,
                                         self.context.expand(batch_size,
                                                             1,
                                                             -1,
                                                             -1))
        x = torch.cat([x, x_], dim=2)
        
        # Final nodes encoding
        nodes_logits = self.nodes_encoder(x).squeeze(-1)
        nodes_sigm = torch.sigmoid(nodes_logits)
       
        # Nodes to extended form
        nodes_hat = torch.diag_embed(nodes_sigm)
        nodes_hat[:, :, 0] += (1 - nodes_sigm)

        # Edges specification
        # Account for the global graph representation
        ctx = self.edges_spec_layer(z)
        # Collect node types incident to each edge
        cc = organ.tingle._cartesian(nodes_hat)

        # Global + incident nodes
        edges_data = torch.cat([cc[0] - cc[1],
                                ctx.view(-1, 1, 1, 32).
                                    expand(-1,
                                           self.node_types, 
                                           self.node_types,
                                           -1)],
                               axis=-1)
        edges = self.edge_layers(edges_data)
        edges_logits = self.edges_layer(edges)

        return edges_logits, nodes_logits, None
