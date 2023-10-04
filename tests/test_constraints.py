import torch

import pytest

import organ.structure.constraints as C

import tests.util


def test_edge_consistent():

    # Conforming
    # There are 4 possible nodes, three of them exist (0, 1, 2),
    # two are connected by edges (0-1).
    nodes = torch.tensor([[[0.0, 1.0, 0.0],    # type 1
                           [0.0, 0.0, 1.0],    # type 2
                           [0.0, 1.0, 0.0],    # type 1
                           [1.0, 0.0, 0.0]]])  # no node
    edges1 = torch.zeros(4, 4)
    edges1[0, 1] = 1.0
    edges1 = edges1 + edges1.t()
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    edges = torch.unsqueeze(edges, 0)
    assert C.edge_consistent(nodes, edges).item() == pytest.approx(0.0)

    # Non-conforming
    # The same as above, but add an edge 1-3
    edges1 = torch.zeros(4, 4)
    edges1[0, 1] = 1.0
    edges1[1, 3] = 1.0
    edges1 = edges1 + edges1.t()
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    edges = torch.unsqueeze(edges, 0)

    # Violation of two edges
    assert C.edge_consistent(nodes, edges).item() == pytest.approx(2.0)

    nodes.requires_grad = True
    edges.requires_grad = True
    assert tests.util.can_drive_learning([nodes, edges],
                                         lambda x: C.edge_consistent(*x),
                                         n_iters=10)


def test_edge_symmetric():

    # Conforming
    edges1 = torch.zeros(3, 3)
    edges1[0, 1] = 1.0
    edges1 = edges1 + edges1.t()
    # Add zero-type edge
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    # Make batch
    edges = torch.unsqueeze(edges, 0)
    assert C.edge_symmetric(edges).item() == pytest.approx(0.0)

    # Non-conforming
    edges1 = torch.zeros(3, 3)
    edges1[0, 1] = 1.0
    # Add zero-type edge
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    # Make batch
    edges = torch.unsqueeze(edges, 0)
    assert C.edge_symmetric(edges).item() == pytest.approx(1.0)

    edges.requires_grad = True
    assert tests.util.can_drive_learning([edges],
                                         lambda x: C.edge_symmetric(*x),
                                         n_iters=10)
