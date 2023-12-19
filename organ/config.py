"""Configuration tools."""
import argparse
import importlib

import organ.demo
import organ.structure.models


def _load_rules(python_class_str: str):
    """Creates an instance of a class by fully qualified name.

    Parameters
    ----------
    python_class_str : str
        Fully qualified class name. MUST contain at least one
        dot symbol, separating module name and class name.
    Returns
    -------
    An instance of the specified class.
    """
    last_dot_pos = python_class_str.rfind('.')
    if last_dot_pos <= 0 or last_dot_pos >= len(python_class_str) - 1:
        raise ValueError('Must be a fully-qualified name')

    module_name = python_class_str[:last_dot_pos]
    class_name = python_class_str[last_dot_pos + 1:]

    rules_module = importlib.import_module(module_name)
    return getattr(rules_module, class_name)()


def _structure_rules(val: str):
    """Process 'rules' command-line argument."""
    if val == 'demo_logistics':
        return organ.demo.LogisticsDepartmentModel()
    elif val == 'demo_management':
        return organ.demo.ManagementModel()
    elif val == 'sapsam':
        return organ.demo.SapSamEMStructureModel()
    elif val == 'generic':
        return organ.structure.models.Generic()
    else:
        return _load_rules(val)


class _Configurator:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Organization structure rules configuration
        parser.add_argument('--rules', type=_structure_rules,
                            default='generic',
                            help='organization structure rules description. '
                            'Can be either "demo_logistics", '
                            '"demo_management", "generic", or a '
                            'fully-qualified class name')

        # Model configuration.
        parser.add_argument('--z_dim', type=int, default=8,
                            help='input dimension of G')
        # Размерности группы полносвязных слоев в начале генератора
        parser.add_argument('--g_conv_dim', type=int, nargs='+',
                            default=[128, 256, 512],
                            help='neurons in the dense layers '
                                 'in the encoder of G')
        parser.add_argument('--g_edge_conv_dim', type=int, nargs='+',
                            default=[128, 64, 32],
                            help='specification edge convolutions in the G')
        parser.add_argument('--g_params_fc_dim', type=int,
                            default=[128, 64, 64],
                            help='specification of G"s fully-connected block '
                                 'to create parameter values')
        # Спецификация сложности преобразований, которые должны
        # быть реализованы дискриминатором (и аппроксиматором).
        # Состоит из трех компонент:
        #   - список, описывающий параметры графовых сверток, в частности,
        #     размерности представлений вершин,
        #   - количество признаков в глобальном представлении графа,
        #   - список, задающий количества нейронов в серии полносвязных слоев.
        parser.add_argument('--d_conv_dim', type=int,
                            default=[[128, 64], 128, [128, 64]],
                            help='specification of D')
        parser.add_argument('--d_fc_dim', type=int, nargs='+',
                            default=[256, 128, 64],
                            help='specification of fc group in the D')
        parser.add_argument('--d_cond_enc_dim', type=int, nargs='+',
                            default=[32, 16],
                            help='specification of the condition encoder '
                                 'in the D')

        # Вес для штрафа на величину градиента в функции оптимизации
        parser.add_argument('--lambda_gp', type=float, default=10,
                            help='weight for gradient penalty')
        # Метод постобработки сгенерированных графов
        parser.add_argument('--post_method', type=str, default='softmax',
                            choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

        # Training configuration.
        # Размер батча
        parser.add_argument('--batch_size', type=int, default=16,
                            help='mini-batch size')
        # Количество итераций (батчей) в процессе обучения
        parser.add_argument('--num_iters', type=int, default=200000,
                            help='number of total iterations for training D')
        # Количество итераций (перед последней, `num_iters`) в течение
        # которых будет осуществляться снижение константы обучения
        parser.add_argument('--num_iters_decay', type=int, default=100000,
                            help='number of iterations for decaying lr')
        # Константа обучения для генератора
        parser.add_argument('--g_lr', type=float, default=0.0001,
                            help='learning rate for G')
        # Константа обучения для дискриминатора
        parser.add_argument('--d_lr', type=float, default=0.0001,
                            help='learning rate for D')
        # Дропаут (одно и то же значение используется везде, между
        # каждой парой слоев)
        parser.add_argument('--dropout', type=float, default=0.,
                            help='dropout rate')
        # Периодичность тренировки генератора
        # (каждые `n_critic` батчей)
        parser.add_argument('--n_critic', type=int, default=5,
                            help='number of D updates per each G update')
        # beta1 для Adam (при обучении всех моделей)
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for Adam optimizer')
        # beta2 для Adam (при обучении всех моделей)
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='beta2 for Adam optimizer')
        # Итерация, с которой нужно продолжить процесс обучения.
        # Если значение не 0, то все модели будут загружены из
        # точек сохранения и процесс продолжен.
        parser.add_argument('--resume_iters', type=int, default=None,
                            help='resume training from this step')

        # Test configuration.
        # Указание на то, какую именно модель следует тестировать
        # (модель, созданную после test_iters итераций обучения).
        parser.add_argument('--test_iters', type=int, default=200000,
                            help='test model from this step')

        # Miscellaneous.
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--mode', type=str, default='train',
                            choices=['train', 'test'])
        parser.add_argument('--use_tensorboard', action='store_true',
                            default=False)
        parser.add_argument('--augment', action='store_true', default=False,
                            help='apply augmentations during training')
        parser.add_argument('--no_pretrain', action='store_false',
                            default=True, dest='pretrain',
                            help='disable pretraining')
        parser.add_argument('--non_conditional', action='store_false',
                            default=True, dest='conditional',
                            help='disable conditional generation')
        parser.add_argument('--non_parametric', action='store_false',
                            default=True, dest='parametric',
                            help='disable parametric generation')
        parser.add_argument('--train_completion', action='store_true',
                            default=False, dest='completion',
                            help='train for structure completion')

        # Directories.
        # Директория с данными
        parser.add_argument('--data_dir', type=str, default='data')
        # Директория записи журнала (используется только с
        # Tensorboard)
        parser.add_argument('--log_dir', type=str, default='output/logs')
        # Директория для сохранения моделей
        # (из этой же директории они будут подгружаться при необходимости
        # продолжить обучение)
        parser.add_argument('--model_save_dir', type=str,
                            default='output/models')
        # Directory to put samples
        parser.add_argument('--samples_dir', type=str, default=None,
                            help='directory to put samples')

        # Настройка периодичности вывода информации
        #
        # Периодичность записи данных в журнал (для Tensorboard)
        parser.add_argument('--log_step', type=int, default=10)
        # Периодичность сохранения моделей
        parser.add_argument('--model_save_step', type=int, default=10000)
        # Периодичность изменения констант обучения.
        # Этим параметром регулируется то, как часто будет оцениваться
        # необходимость ревизии констант. См. также `num_iters_decay`.
        parser.add_argument('--lr_update_step', type=int, default=1000)

        self.parser = parser


_config_maker = _Configurator()


def parse_args(args=None):
    """Parse arguments.

    Parameters
    ----------
    args : list
        List of strings with arguments or ``None``.
        In the latter case, the arguments will be taken
        from the `sys.argv`. The contents of the list
        is governed by `argparse.ArgumentParser.parse_args`.

    Returns
    -------
    Namespace
        A namespace object with configuration parameter values.
    """
    return _config_maker.parser.parse_args(args)


def make_config(**kwargs):
    """Build a configuration object with the specified values.

    Sets values of the given parameter keys. Other (not mentioned)
    parameters will receive their default values.

    Parameters
    ----------
    kwargs
        Key-value pairs with parameter values.

    Returns
    -------
    Namespace
        A namespace object with configuration parameter values.
    """
    defaults_dict = vars(_config_maker.parser.parse_args([]))
    defaults_dict.update(kwargs)
    # Fix some keys with non-trivial logics
    if isinstance(defaults_dict['rules'], str):
        defaults_dict['rules'] = _structure_rules(defaults_dict['rules'])
    return argparse.Namespace(**defaults_dict)
