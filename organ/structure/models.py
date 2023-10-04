"""Describes some of the organization structure models."""
from abc import ABC, abstractmethod

import numpy as np

import organ.structure.constraints as C


class Organization:
    """Holder for an organization description."""

    def __init__(self, nodes: np.ndarray,
                 edges: np.ndarray, *,
                 node_features: np.ndarray = None,
                 condition: np.ndarray = None):
        self.nodes = nodes
        self.edges = edges
        self.node_features = node_features
        self.condition = condition


class OrganizationModel(ABC):
    """Base class for organization structure models.

    This is an abstract class, defining an interface that must
    be implemented by custom organization structure model
    classes. See, e.g., `Generic`.
    """

    @abstractmethod
    def validness(self, org: Organization) -> bool:
        pass

    @abstractmethod
    def metrics(self, org: Organization) -> dict:
        pass

    def soft_constraints(self, nodes, edges, features, cond):
        return 0.0


class Generic(OrganizationModel):
    """Generic organization structure model.

    This model considers every non-empty and properly connected organization
    structure as valid.
    """

    def __init__(self):
        pass

    def check_nodes(self, nodes: np.array) -> bool:
        """Checks the validness of the set of nodes.

        All non-empty sets of nodes are considered valid.

        Parameters
        ----------
        nodes : np.array
            One-dimensional integer numpy array with types
            of nodes (on certain positions). Zero means
            the position is empty.

        Returns
        -------
        bool
            Validness of structure nodes.
        """
        return not np.all(nodes == 0)

    def check_relations(self, nodes: np.array, edges: np.ndarray) -> bool:
        """Checks the validness of edges.

        Only edges between existing nodes are considered
        to be valid.

        Parameters
        ----------
        nodes : np.array
            One-dimensional integer numpy array with types of nodes (on
            certain positions). Zero means the position is empty.
        edges : np.ndarray
            An adjacency matrix. The value of a cell corresponds
            to the type of the respective edge, 0 is no edge.

        Returns
        -------
        bool
            Validness of structure edges.
        """
        node_active = (nodes != 0).reshape(-1, 1)
        edge_active = (edges != 0)
        return np.all(edge_active <= node_active @ node_active.T), None

    def validness(self, org) -> bool:
        """Checks structure validness."""
        return org is not None and \
            self.check_nodes(org.nodes) and \
            self.check_relations(org.nodes, org.edges)[0]

    def metrics(self, org) -> dict:
        """Returns a dict with relevant metric values."""
        return {
            'node score': self.check_nodes(org.nodes),
            'edge score': self.check_relations(org.nodes, org.edges)[0],
        }

    def soft_constraints(self, nodes, edges, *ignored):
        """Differentiable constraints on the graph structure."""
        return C.edge_consistent(nodes, edges) + C.edge_symmetric(edges)
