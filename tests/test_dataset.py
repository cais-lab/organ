import sys

import numpy as np

sys.path.append('.')

from organ.data.organization_structure_dataset import OrganizationStructureDataset       # noqa: E402 E501


def test_load():
    dataset = OrganizationStructureDataset()
    dataset.load('tests/data')

    # Общий размер
    assert len(dataset) == 100
    # Размер обучающего подмножества
    assert len(dataset.train_idx) == 80
    assert dataset.train_count == 80
    # Размер валидационного подмножества
    assert len(dataset.validation_idx) == 10
    assert dataset.validation_count == 10
    # Размер тестового подмножества
    assert len(dataset.test_idx) == 10
    assert dataset.test_count == 10

    # Подмножества не должны попарно пересекаться
    train_idx_set = frozenset(dataset.train_idx)
    validation_idx_set = frozenset(dataset.validation_idx)
    test_idx_set = frozenset(dataset.test_idx)
    assert train_idx_set.isdisjoint(validation_idx_set)
    assert train_idx_set.isdisjoint(test_idx_set)
    assert validation_idx_set.isdisjoint(test_idx_set)

    # Размерности
    assert dataset.nodes.shape == (100, 12)
    assert dataset.nodes.shape == (100,
                                   dataset.vertexes)
    assert dataset.edges.shape == (100, 12, 12)
    assert dataset.edges.shape == (100,
                                   dataset.vertexes,
                                   dataset.vertexes)

    assert dataset.node_num_types == 12
    assert dataset.edge_num_types == 3
    assert dataset.vertexes == 12
    assert np.all(dataset.nodes) <= dataset.node_num_types


def test_batch_iteration():

    batch_iteration(False, False)
    batch_iteration(True, False)
    batch_iteration(False, True)
    batch_iteration(True, True)


def test_batches_different():

    dataset = OrganizationStructureDataset(load_params=True)
    dataset.load('tests/data')

    batch_size = 7

    _, __, params1, __ = dataset.next_train_batch(batch_size)
    _, __, params2, __ = dataset.next_train_batch(batch_size)

    assert not np.allclose(params1, params2)


def batch_iteration(load_cond: bool, load_params: bool):

    dataset = OrganizationStructureDataset(load_cond=load_cond,
                                           load_params=load_params)
    dataset.load('tests/data')

    batch_size = 7

    subsets = [('next_train_batch', 'train_count'),
               ('next_validation_batch', 'validation_count'),
               ('next_test_batch', 'test_count')]

    for subset_batch_iter_foo_name, subset_size_attr_name in subsets:

        subset_batch_iter_foo = getattr(dataset, subset_batch_iter_foo_name)
        subset_size = getattr(dataset, subset_size_attr_name)

        # Получение батча
        nodes, edges, params, cond = subset_batch_iter_foo(batch_size)
        assert nodes.shape == (batch_size, dataset.vertexes)
        assert edges.shape == (batch_size, dataset.vertexes, dataset.vertexes)
        if load_params:
            assert params.shape == (batch_size, dataset.vertexes, 1)
        else:
            assert params is None
        if load_cond:
            assert cond.shape == (batch_size, 2)
        else:
            assert cond is None

        # Получение всего набора
        nodes, edges, params, cond = subset_batch_iter_foo()
        assert nodes.shape == (subset_size, dataset.vertexes)
        assert edges.shape == (subset_size,
                               dataset.vertexes,
                               dataset.vertexes)
