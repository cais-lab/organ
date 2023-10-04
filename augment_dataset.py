"""A script to create an augmented dataset."""
import os
import sys
import random
import argparse

import numpy as np

import organ.data.util

from organ.config import _structure_rules
from organ.data.organization_structure_dataset import OrganizationStructureDataset  # noqa E501


if __name__ == '__main__':

    # Make it reproducible
    random.seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser()

    # Organization structure rules configuration
    parser.add_argument('org_rules', type=_structure_rules,
                        help='organization structure rules description. '
                        'Can be either "demo_logistics", '
                        '"demo_management", "generic", or a '
                        'fully-qualified class name')

    # Dataset size
    parser.add_argument('n', type=int,
                        help='New instances to create.')
    # Source dataset
    parser.add_argument('source', type=str,
                        help='Source dataset directory.')

    # Destination directory
    parser.add_argument('destination', type=str, nargs='?',
                        default='data',
                        help='Destination directory.')
    # Potentially overwrite a dataset
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force storing he augmented dataset in an '
                             'existing directory.')
    # Test size
    parser.add_argument('--test', type=int,
                        help='Test subset size.')
    # Validation set size
    parser.add_argument('--validation', type=int,
                        help='Validation subset size.')

    config = parser.parse_args()

    test_size = config.test if config.test is not None else 0.1
    val_size = config.validation if config.validation is not None else 0.1

    if os.path.isdir(config.destination):
        if not config.force:
            print('The destination directory exists and may contain a '
                  'dataset. Use --force flag to overwrite it.')
            sys.exit(1)
    else:
        os.makedirs(config.destination)

    # Load the dataset
    dataset = OrganizationStructureDataset(load_cond=True, load_params=True)
    dataset.load(config.source)

    # Make the components of the augmented dataset
    org_model = config.org_rules
    nodes, edges, features, cond = organ.data.util.augment_dataset(
        dataset,
        config.n,
        org_model
    )

    # Store the augmented dataset
    organ.data.util.save_dataset(
        nodes,
        edges,
        features,
        cond,
        config.destination,
        org_model,
        val_size, test_size)
