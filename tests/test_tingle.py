import torch

import organ.tingle

import pytest


def make_sample_graph():

    nodes = torch.tensor([[1., 0.],
                          [0., 1.],
                          [1., 0.]],
                         dtype=torch.float32)
    # edge x node x node
    # 0-type:
    #   0 - 1 - 2
    # 1-type:
    #   0 ----- 2
    edges = torch.tensor([[[0., 1., 0.],
                           [1., 0., 1.],
                           [0., 1., 0.]],
                          [[0., 0., 1.],
                           [0., 0., 0.],
                           [1., 0., 0.]],
                          ],
                         dtype=torch.float32)

    batch_nodes = torch.stack([nodes, nodes])
    batch_edges = torch.stack([edges, edges])

    return batch_nodes, batch_edges


def test_vv_collect_aggregate():
    nodes, edges = make_sample_graph()
    # Such setting just sums all adjacent representations
    new_nodes = organ.tingle.vv_collect_aggregate(nodes,
                                                  edges,
                                                  agg='sum',
                                                  add_loops=False,
                                                  edge_weights=None)
    assert torch.allclose(new_nodes[0],
                          torch.tensor([[1.0, 1.0],
                                        [2.0, 0.0],
                                        [1.0, 1.0]],
                                       dtype=torch.float32))

    # 'edge_weights' allow to select only one type
    # of connections (also, mind adding loops)
    new_nodes = organ.tingle.vv_collect_aggregate(nodes,
                                                  edges,
                                                  agg='sum',
                                                  add_loops=True,
                                                  edge_weights=torch.tensor([1, 0]))  # noqa: E501
    assert torch.allclose(new_nodes[0],
                          torch.tensor([[1.0, 1.0],
                                        [2.0, 1.0],
                                        [1.0, 1.0]],
                                       dtype=torch.float32))

    # averaging
    new_nodes = organ.tingle.vv_collect_aggregate(nodes,
                                                  edges,
                                                  agg='avg')
    assert torch.allclose(new_nodes[0],
                          torch.tensor([[1.0, 1.0],
                                        [1.0, 0.0],
                                        [1.0, 1.0]],
                                       dtype=torch.float32))

    # unknown aggregation type
    with pytest.raises(ValueError):
        organ.tingle.vv_collect_aggregate(nodes, edges, agg='asdfasdf')


def test_ve_collect_aggregate():
    nodes, edges = make_sample_graph()

    # sum incident nodes
    new_edges = organ.tingle.ve_collect_aggregate(nodes, agg='sum')
    for i in range(nodes.shape[1]):
        for j in range(nodes.shape[1]):
            assert torch.allclose(new_edges[0, i, j, :],
                                  nodes[0, i, :] + nodes[0, j, :])

    # unknown aggregation
    with pytest.raises(ValueError):
        organ.tingle.ve_collect_aggregate(nodes, agg='1234')


def test_ev_collect_aggregate():
    nodes, edges = make_sample_graph()

    # let each edge is described by one number (and its 'one')
    batch_size = nodes.shape[0]
    n_nodes = nodes.shape[1]
    edge_repr = torch.ones((batch_size, n_nodes, n_nodes, 1),
                           dtype=torch.float32)

    new_nodes = organ.tingle.ev_collect_aggregate(edges, edge_repr,
                                                  agg='sum', outbound=True)
    assert new_nodes.shape == (2, n_nodes, 1)
    # Each node has two outbound edges
    assert torch.allclose(new_nodes[0], torch.ones_like(new_nodes[0]) * 2.0)

    new_nodes = organ.tingle.ev_collect_aggregate(edges, edge_repr,
                                                  agg='sum', outbound=True,
                                                  edge_weights=torch.tensor([1, 0]))  # noqa: E501

    # however, if we consider only the zeroth edge type, then
    # the situation is different (nodes 0 and 2 have only one, while
    # node 1 still has two)
    assert torch.allclose(new_nodes[0],
                          torch.tensor([[1], [2], [1]], dtype=torch.float32))

    # unknown aggregation
    with pytest.raises(ValueError):
        organ.tingle.ev_collect_aggregate(edges, edge_repr, agg='1234')


def test_VV():
    nodes, edges = make_sample_graph()

    vv = organ.tingle.VV(apply_to_types=True, merge='replace')
    g, nt, n, et, e = vv(None, nodes, None, edges, None)
    # Doesn't touch global state, edge representations and types
    assert g is None
    assert e is None
    assert id(nt) == id(nodes)
    assert id(et) == id(edges)
    # New node representation
    assert torch.allclose(n,
                          organ.tingle.vv_collect_aggregate(nodes,
                                                            edges))


def test_VE():
    nodes, edges = make_sample_graph()

    ve = organ.tingle.VE(apply_to_types=True, merge='replace', agg='sum')
    g, nt, n, et, e = ve(None, nodes, None, None, None)
    # Doesn't touch global state, edge and node types
    assert g is None
    assert n is None
    assert et is None
    assert id(nt) == id(nodes)
    # New edge representations
    assert torch.allclose(e, organ.tingle.ve_collect_aggregate(nodes))


def test_GNN_global():
    prev_g = torch.ones((4, 3), dtype=torch.float32)

    block = organ.tingle.GNNBlock(global_module=torch.nn.Linear(3, 5))
    g, nt, n, et, e = block(prev_g, None, None, None, None)
    assert g.shape == (4, 5)
    assert nt is None
    assert n is None
    assert et is None
    assert e is None


def test_GNN_nodes():
    node_repr = torch.ones((4, 5, 7), dtype=torch.float32)

    block = organ.tingle.GNNBlock(nodes_module=torch.nn.Linear(7, 2))
    g, nt, n, et, e = block(None, None, node_repr, None, None)
    assert g is None
    assert nt is None
    assert n.shape == (4, 5, 2)
    assert et is None
    assert e is None


def test_GNN_edges():
    edge_repr = torch.ones((4, 5, 5, 7), dtype=torch.float32)

    block = organ.tingle.GNNBlock(edges_module=torch.nn.Linear(7, 2))
    g, nt, n, et, e = block(None, None, None, None, edge_repr)
    assert g is None
    assert nt is None
    assert n is None
    assert et is None
    assert e.shape == (4, 5, 5, 2)


def test_complete():
    nodes, edges = make_sample_graph()

    gn = torch.nn.ModuleList([
        organ.tingle.VV(merge='replace', apply_to_types=True),
        organ.tingle.GNNBlock(nodes_module=torch.nn.Linear(2, 4)),
        organ.tingle.VV(merge='replace'),
        organ.tingle.GNNBlock(nodes_module=torch.nn.Linear(4, 2)),
    ])

    g = (None, nodes, None, edges, edges.permute(0, 2, 3, 1))
    for m in gn:
        g = m(*g)

    g, nt, n, et, e = g
    assert g is None
    assert id(nt) == id(nodes)
    assert n.shape == (2, 3, 2)
    assert id(et) == id(edges)
    assert e.shape == (2, 3, 3, 2)
