import torch
import numpy as np

import pytest

import organ.structure.models
import organ.demo


def test_generic_conforming():

    org_model = organ.structure.models.Generic()

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

    # The above is a batch description, it is OK for checking
    # soft constraints. However, to check the validness and
    # calculate metrics we have to transform it to a compact
    # representation
    nodes_hard = torch.max(nodes[0], -1)[1].numpy()
    edges_hard = torch.max(edges[0], -1)[1].numpy()

    org = organ.structure.models.Organization(nodes_hard, edges_hard)

    assert org_model.validness(org)
    metrics = org_model.metrics(org)
    assert len(metrics) == 2
    assert 'node score' in metrics
    assert 'edge score' in metrics
    assert org_model.soft_constraints(nodes, edges).item() == \
        pytest.approx(0.0)


def test_generic_nonconforming():

    org_model = organ.structure.models.Generic()

    # There are 4 possible nodes, three of them exist (0, 1, 2),
    # two are connected by edges (0-1), but there is also an
    # edge 1-3
    nodes = torch.tensor([[[0.0, 1.0, 0.0],    # type 1
                           [0.0, 0.0, 1.0],    # type 2
                           [0.0, 1.0, 0.0],    # type 1
                           [1.0, 0.0, 0.0]]])  # no node
    edges1 = torch.zeros(4, 4)
    edges1[0, 1] = 1.0
    edges1[1, 3] = 1.0
    edges1 = edges1 + edges1.t()
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    edges = torch.unsqueeze(edges, 0)

    # The above is a batch description, it is OK for checking
    # soft constraints. However, to check the validness and
    # calculate metrics we have to transform it to a compact
    # representation
    nodes_hard = torch.max(nodes[0], -1)[1].numpy()
    edges_hard = torch.max(edges[0], -1)[1].numpy()

    org = organ.structure.models.Organization(nodes_hard, edges_hard)

    assert not org_model.validness(org)
    metrics = org_model.metrics(org)
    assert len(metrics) == 2
    assert 'node score' in metrics
    assert 'edge score' in metrics
    assert org_model.soft_constraints(nodes, edges).item() > 0.0


def test_demo_management_soft_constraints():
    nodes = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 0
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 1
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 2
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 3
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],    # node 4 Marketing
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # node 5
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 6
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],    # node 7
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],   # node 8
         [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 0
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 1
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 2
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 3
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 4 Marketing
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # node 5
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # node 6
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],    # node 7
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],    # node 8
          ]], requires_grad=True)

    edges1 = torch.zeros(4, 4)
    edges1[0, 1] = 1.0
    edges1 = edges1 + edges1.t()
    edges = torch.stack([1 - edges1, edges1], dim=-1)
    edges = torch.unsqueeze(edges, 0)

    params = torch.tensor([[[0.0, 0.0],    # node 0
                            [0.0, 0.0],    # node 1
                            [0.0, 0.0],    # node 2
                            [0.0, 0.0],    # node 3
                            [0.0, 0.0],    # node 4 Marketing
                            [0.0, 0.0],    # node 5
                            [0.0, 0.0],    # node 6
                            [0.0, 30.0],    # node 7
                            [1000.0, 0.0]],   # node 8
                           [[0.0, 0.0],    # node 0
                            [0.0, 0.0],    # node 1
                            [0.0, 0.0],    # node 2
                            [0.0, 0.0],    # node 3
                            [0.0, 0.0],    # node 4 Marketing
                            [0.0, 0.0],    # node 5
                            [0.0, 0.0],    # node 6
                            [0.0, 12.0],    # node 7
                            [1000.0, 0.0],    # node 8
                            ]], requires_grad=True)

    ctx = torch.tensor([[1000.0, 1.0],
                        [500.0, 1.0]], requires_grad=True)

    org = organ.demo.ManagementModel()
    res = org.soft_constraints(nodes, edges, params, ctx)
    assert res > 0
    assert nodes.grad is None
    res.backward()
    assert nodes.grad is not None


def test_demo_management_validness():
    org_model = organ.demo.ManagementModel()
    # valid configuration
    org = organ.structure.models.Organization(
                       nodes=np.array([0, 0, 2, 0, 0, 5, 0, 7, 8]),
                       edges=np.array([  # 0  1  2  3  4  5  6  7  8
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                         [0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 5
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 8
                                         ]),
                       node_features=np.array([[0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 25],
                                               [1000, 0]]),
                       condition=np.array([1000, 0]))

    assert org_model.validness(org)

    # invalid nodes
    org = organ.structure.models.Organization(
                       nodes=np.array([0, 0, 0, 0, 0, 5, 0, 7, 8]),
                       edges=np.array([  # 0  1  2  3  4  5  6  7  8
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                         [0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 5
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 8
                                         ]),
                       node_features=np.array([[0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 25],
                                               [1000, 0]]),
                       condition=np.array([1000, 0]))

    assert not org_model.validness(org)

    # invalid relations
    org = organ.structure.models.Organization(
                       nodes=np.array([0, 1, 2, 0, 0, 5, 0, 7, 8]),
                       edges=np.array([  # 0  1  2  3  4  5  6  7  8
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                         [0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 5
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 8
                                         ]),
                       node_features=np.array([[0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 25],
                                               [1000, 0]]),
                       condition=np.array([10000, 0]))

    assert not org_model.validness(org)

    # invalid parameters
    org = organ.structure.models.Organization(
                       nodes=np.array([0, 0, 2, 0, 0, 5, 0, 7, 8]),
                       edges=np.array([  # 0  1  2  3  4  5  6  7  8
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                         [0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 5
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 8
                                         ]),
                       node_features=np.array([[0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 20],
                                               [1000, 0]]),
                       condition=np.array([1000, 0]))

    assert not org_model.validness(org)


def test_demo_management_generate_key_values():
    org_model = organ.demo.ManagementModel()
    nodes = np.array([0, 0, 2, 0, 1, 0, 0, 0, 0])

    v = org_model.generate_key_values(nodes)
    assert (v[0] > 999.99 and
            v[0] < 1201.01 and
            v[1] == pytest.approx(1.0))


def test_demo_management_generate_augmentation():
    org_model = organ.demo.ManagementModel()
    # valid configuration
    org = organ.structure.models.Organization(
                       nodes=np.array([0, 0, 2, 0, 0, 5, 0, 7, 8]),
                       edges=np.array([  # 0  1  2  3  4  5  6  7  8
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                         [0, 0, 0, 0, 0, 0, 0, 2, 0],  # 2
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 5
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                         [0, 0, 0, 0, 0, 0, 0, 3, 0],  # 8
                                         ]),
                       node_features=np.array([[0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 0],
                                               [0, 25],
                                               [1000, 0]]),
                       condition=np.array([1000, 0]))
    aug_configuration = org_model.generate_augmentation(
                              org.nodes,
                              org.edges,
                              org.node_features,
                              logging=False,
                              max_iterations=1000)
    aug_org = organ.structure.models.Organization(
                           nodes=aug_configuration[0],
                           edges=aug_configuration[1],
                           node_features=aug_configuration[2],
                           condition=aug_configuration[3])
    assert org_model.validness(aug_org)


def test_sapsam_dataset():
    org_model = organ.demo.SapSamEMStructureModel()
    nodes, relations = org_model.generate_parametrized_model()
    assert nodes.shape == (17, 22)
    assert relations.shape == (17, 22, 22)
    
    for i in range(17):
        assert org_model.check_nodes(nodes[i])
        assert org_model.check_relations(nodes[i], relations[i])[0]
    
    assert not org_model.check_nodes(np.ones((22, )))
    assert not org_model.check_nodes(np.zeros((22, )))
    assert not org_model.check_relations(np.ones((22, )), np.ones((22, 22)))[0]
    
    aug_nodes, aug_edges, _, _ = org_model.generate_augmentation(nodes, relations, [], False, 100) 
    assert org_model.check_nodes(aug_nodes)
    assert org_model.check_relations(aug_nodes, aug_edges)[0]      

    org = organ.structure.models.Organization(
        nodes=aug_nodes,
        edges=aug_edges
    )
    assert org_model.validness(org)

    assert len(org_model.metrics(org)) == 2
    assert 'node score' in org_model.metrics(org)
    assert 'edge score' in org_model.metrics(org)
