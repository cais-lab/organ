import os

from torch.backends import cudnn

from organ.solver import Solver
from organ.config import parse_args


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if config.samples_dir is not None and \
            not os.path.exists(config.samples_dir):
        os.makedirs(config.samples_dir)

    # Solver for training and testing OrGAN.
    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':

    config = parse_args()
    print(config)
    main(config)
