"""
Tests for multilevel configuration.

The tests are based on a mock generator system, where the
context determines the number (and the set) of nodes
that have to be included into the generated structure, i.e.

   - generate(2) => nodes 1 and 2
   - generate(3) => nodes 1, 2, and 3
   
"""
import numpy as np

import pytest

import organ.configurator
from organ.configurator import make_rule
from organ.structure.models import Organization


class MockGenerator:
    """Mock generator class.
    
    Generates a graph with the required number of nodes
    connected into a 'chain'.
    """
    
    def __init__(self, node_types: int):
        """
        Parameters:
        node_types: int
            A number of node types, including no-node.
        """
        self.node_types = node_types
        # self.edge_types = edge_types

    def generate(self, batch_size: int = 1, ctx=None):

        assert ctx is None or ctx.shape == (1, )

        if ctx is None:  ctx = 1
        else:            ctx = int(np.round(ctx[0]))
        
        ctx = min(ctx + 1, self.node_types)
        orgs = []
        for i in range(batch_size):
            # Fill nodes
            n = np.zeros(self.node_types, dtype=np.int8)
            for i in range(1, ctx):
                n[i] = i
            # Fill edges
            e = np.zeros((self.node_types, self.node_types), dtype=np.int8)
            for i in range(1, ctx - 1):
                if i + 1 <= ctx:
                    e[i, i+1] = 1
                    e[i+1, i] = 1

            node_features = np.arange(self.node_types) * (n != 0)
           
            orgs.append(Organization(n,
                                     e,
                                     node_features=node_features.reshape((-1, 1)),
                                     condition=np.array([ctx])))
                                     
        return orgs
    
    def generate_valid(self, n: int, ctx=None, max_generate: int=1000):
        return self.generate(n, ctx)
    
    def complete(self, batch_size: int = 1,
                 nodes=None,
                 nodes_mask=None,
                 edges=None,
                 edges_mask=None,
                 params=None,
                 params_mask=None,
                 ctx=None):
        return self.generate(batch_size, ctx)
    
    def comlpete_valid(self, n: int, *, 
                 nodes=None,
                 nodes_mask=None,
                 edges=None,
                 edges_mask=None,
                 params=None,
                 params_mask=None,
                 ctx=None, 
                 max_generate: int = 1000):
        return self.complete(n, ctx)


def mock_link_foo(upper_level: Organization,
                  initial_context: np.ndarray) -> np.ndarray:
    """Builds input specification for the lower-level structure
    based on upper-level structure and initial input specification.
    
    The function is called to find out what input features should
    be for the lower-level structure, taking into account the
    upper level one. It should return the specification,
    or raise one of the exceptions:
       - `ConfigurationConflict` if it turns out to 
       be impossible to merge the requirements of the upper level 
       `upper_level` with the requirements `initial_context`,
       - `SubmodelNotNeeded` if there should
       be no lower level system for the upper level one.

    This particular function just tries to replicate the upper-level
    structure."""

    assert (initial_context is None 
            or initial_context.shape == (1,))
    # None means there are no requirements
    if initial_context is None:
        context = np.empty((1, ))
        context[0] = np.nan
    else:
        context = np.array(initial_context)   # copy

    # Find out the number of active nodes in the upper-level
    # organization
    n_nodes = (upper_level.nodes != 0).sum()

    # Try to merge it with the requirement
    if np.isnan(context[0]):
        context[0] = n_nodes
    elif np.isclose(context[0], n_nodes):
        pass
    else:
        raise organ.configurator.ConfigurationConflict()

    return context


def test_mock_generator():
    N_NODES = 5      # Including the NO-NODE
    BATCH_SIZE = 4
    g = MockGenerator(N_NODES)

    # Default is one active node (and no edges)
    orgs = g.generate(batch_size=BATCH_SIZE)
    assert len(orgs) == BATCH_SIZE
    assert orgs[0].nodes.shape == (N_NODES,)
    assert (orgs[0].nodes != 0).sum() == 1
    assert orgs[0].edges.sum() == 0

    # Three nodes
    orgs = g.generate(batch_size=BATCH_SIZE, ctx=np.array([3]))
    assert len(orgs) == BATCH_SIZE
    assert orgs[0].nodes.shape == (N_NODES,)
    assert (orgs[0].nodes != 0).sum() == 3
    assert orgs[0].edges.sum() == 4   # 2 edges in two directions
    

def test_create_configurator():
    configurator = organ.configurator.Configurator(
        generators={'top': MockGenerator(3),
                    'one': MockGenerator(5),
                    'three': MockGenerator(5)},
        dependencies={('top', 'one'): make_rule(1, mock_link_foo),
                      ('top', 'three'): make_rule(3, mock_link_foo),
                     },
        sequence=['top', 'one', 'three']
    )

def test_generate_multilevel():
    configurator = organ.configurator.Configurator(
        generators={'top': MockGenerator(4),
                    'one': MockGenerator(5),
                    'three': MockGenerator(5)},
        dependencies={('top', 'one'): make_rule(1, mock_link_foo),
                      ('top', 'three'): make_rule(3, mock_link_foo),
                     },
        sequence=['top', 'one', 'three']
    )
    
    # A possible case:
    # Three nodes on the top level (1, 2, 3) and 
    # submodels for the one and three
    multi_org = configurator.generate({
        'top': np.array([3])
    })
    assert multi_org is not None
    assert len(multi_org) == 3
    assert 'top' in multi_org
    assert 'one' in multi_org
    assert 'three' in multi_org
    assert (multi_org['one'].nodes != 0).sum() == 3
    # mock_link_foo just propagates the number of nodes
    assert (multi_org['three'].nodes != 0).sum() == 3

    # An impossible case:
    # There is a requirement for 'one' (a submodel of 
    # 'top' that is inconsistent with the propagated
    # value (3)
    multi_org = configurator.generate({
        'top': np.array([3]),
        'one': np.array([2]),
    })
    assert multi_org is None
   
    # When there is some sub-model generation scheme,
    # however, there is no respective node in the upper
    # level
    # E.g., in this case in the upper level there
    # are only 1 and 2, therefore, there is no need
    # to create the submodel 'three'
    multi_org = configurator.generate({
        'top': np.array([2]),
    })
    assert multi_org is not None
    assert len(multi_org) == 2    # top and one
    assert 'top' in multi_org
    assert 'one' in multi_org
    assert (multi_org['top'].nodes != 0).sum() == 2
    assert (multi_org['one'].nodes != 0).sum() == 2
