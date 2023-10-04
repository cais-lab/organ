"""Some generic differentiable constraints on structures."""

import torch
import torch.nn.functional as F


def edge_consistent(nodes, edges):
    """Penalizes edges incident to non-existing nodes.

    The constraint that the function is enforcing
    is :math:`y_{ij} <= x_i * x_j`, where :math:`x_i, x_j`
    is presence of a node in respective locations, and
    :math:`y_{ij}` is presence of an edge.

    As a penalty, this is transformed to:

    .. math::

       ReLU(y_{ij} - x_i * x_j)

    Parameters
    ----------
    nodes : torch.tensor
        Batch of node descriptions (batch, nodes, f).
        Assumes that sum across the last dimension is 1 and
        node type 0 is the absence of a node.
    edges : torch.tensor
        Batch of edge descriptions
        (batch, nodes, nodes, edge_types).
        Assumes that the sum across the last dimension is 1
        and edge type 0 is the absence of an edge.
    Returns
    -------
    float
        Penalty for edge inconsistence.
    """
    # The probability of node presence (excluding
    # zero node type)
    x = torch.sum(nodes[:, :, 1:], -1)
    x = torch.einsum('bi,bj->bij', x, x)
    # Exclude absent edges (zero-type)
    return F.relu(edges[:, :, :, 1:] - torch.unsqueeze(x, -1)).sum()


def edge_symmetric(edges):
    """Penalizes non-symmetric edges.

    Parameters
    ----------
    edges : torch.tensor
        Batch of edge descriptions
        (batch, nodes, nodes, edge_types).

    Returns
    -------
    float
        Penalty for non-symmetric adjecency matrix.
    """
    return torch.norm((edges - edges.permute(0, 2, 1, 3)) / 2.0)
