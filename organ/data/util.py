import os
import pickle

import numpy as np

from organ.structure.models import Organization


def generate_dataset(org_model, dataset_size):
    nodes_list = []
    edges_list = []
    staff_list = []
    cond_list = []
    for i in range(dataset_size):
        nodes, relations, staff, cond = org_model.generate_parametrized_model()
        nodes_list.append(nodes)
        edges_list.append(relations)
        staff_list.append(staff)
        cond_list.append(cond)
    return np.stack(nodes_list, axis=0), \
        np.stack(edges_list, axis=0), \
        np.stack(staff_list, axis=0), \
        np.stack(cond_list, axis=0)


def save_dataset(nodes, edges, staff, cond,
                 dataset_dir, org_model,
                 validation=0.1,
                 test=0.1):

    dataset_size = len(nodes)

    if type(validation) == float:
        if validation <= 0.0 or validation >= 1.0:
            raise ValueError('validation must be float (0.0; 1.0) '
                             'or integer')
        validation = int(validation * dataset_size)

    if type(test) == float:
        if test <= 0.0 or test >= 1.0:
            raise ValueError('test must be float (0.0; 1.0) '
                             'or integer')
        test = int(test * dataset_size)

    train = dataset_size - validation - test

    all_idx = np.random.permutation(dataset_size)
    train_idx = all_idx[0:train]
    validation_idx = all_idx[train:train + validation]
    test_idx = all_idx[train + validation:]

    # Make sure that even one param has its dimension
    if len(staff.shape) < 3:
        staff = np.expand_dims(staff, axis=-1)
    if len(cond.shape) < 2:
        cond = np.expand_dims(cond, axis=-1)

    np.save(os.path.join(dataset_dir, 'data_nodes.npy'), nodes)
    np.save(os.path.join(dataset_dir, 'data_edges.npy'), edges)
    np.save(os.path.join(dataset_dir, 'data_staff.npy'), staff)
    np.save(os.path.join(dataset_dir, 'data_cond.npy'), cond)

    with open(os.path.join(dataset_dir, 'data_meta.pkl'), 'wb') as f:
        pickle.dump({'train_idx': train_idx,
                     'train_count': train,
                     'train_counter': 0,
                     'validation_idx': validation_idx,
                     'validation_count': validation,
                     'validation_counter': 0,
                     'test_idx': test_idx,
                     'test_count': test,
                     'test_counter': 0,

                     'node_num_types': org_model.NODE_N_TYPES,
                     'edge_num_types': org_model.EDGE_N_TYPES,
                     'vertexes': org_model.MAX_NODES_PER_GRAPH,

                     'features_per_node': staff.shape[2],
                     'condition_dim': cond.shape[1],
                     }, f)


def create_dataset(org_model,
                   dataset_size,
                   dataset_dir,
                   validation=0.1,
                   test=0.1):

    nodes, edges, staff, cond = generate_dataset(org_model, dataset_size)
    save_dataset(nodes, edges, staff, cond,
                 dataset_dir, org_model, validation, test)


def augment_dataset(dataset, new_instances: int, org_model):
    """Creates an augmented dataset.

    The augmented dataset will include all the samples
    from the original dataset and `new_instances` of
    new samples.

    Parameters
    ----------
    dataset
        Original dataset.
    new_instances : int
        Number of new instances to create.
    org_model
        A model implementing augmentation logics.

    Returns
    -------
    nodes : np.ndarray
        Nodes. NumPy array, (..., n_nodes)
    edges : np.ndarray
        Edges. NumPy array, (..., n_nodes, n_nodes)
    features : np.ndarray
        Node features. NumPy array, (..., n_nodes, n_features)
    condition : np.ndarray
        Condition. NumPy array, (..., n_condition_params)
    """
    def get_org(dataset, idx: int) -> Organization:
        return Organization(dataset.nodes[idx],
                            dataset.edges[idx],
                            node_features=dataset.node_params[idx]
                            if dataset.load_params else None,
                            condition=dataset.cond[idx]
                            if dataset.load_cond else None)

    original_size = len(dataset)
    nodes_list = []
    edges_list = []
    features_list = []
    cond_list = []
    for i in range(new_instances):
        # Pick an instance
        idx = np.random.choice(original_size)
        org = get_org(dataset, idx)
        # Try to create an augmentation
        nodes, edges, features, cond = org_model.generate_augmentation(
            org.nodes,
            org.edges,
            org.node_features,
            logging=False,
            max_iterations=100)
        if isinstance(nodes, np.ndarray) and \
            org_model.validness(Organization(nodes, edges,
                                             node_features=features,
                                             condition=cond)):
            nodes_list.append(nodes)
            edges_list.append(edges)
            if features.ndim < 2:
                features = np.expand_dims(features, axis=-1)
            features_list.append(features)
            cond_list.append(cond)
        else:
            pass
    nodes = np.stack(nodes_list)
    edges = np.stack(edges_list)
    features = np.stack(features_list)
    cond = np.stack(cond_list)
    # Add the original data
    nodes = np.concatenate([nodes, dataset.nodes], 0)
    edges = np.concatenate([edges, dataset.edges], 0)
    features = np.concatenate([features, dataset.node_params], 0)
    cond = np.concatenate([cond, dataset.cond], 0)
    return nodes, edges, features, cond
