"""Main module for training and testing of the OrGAN.
"""

import numpy as np
import os
import time
import datetime
from itertools import repeat

import torch
import torch.nn.functional as F

from organ.models import CPGenerator, CPDiscriminator, \
    EdgeAwareGenerator, Discriminator, CompletionGenerator
from organ.structure.models import Organization
from organ.data.organization_structure_dataset \
    import OrganizationStructureDataset
from organ.utils import MetricsAggregator, all_scores


class Normalizer:

    def __init__(self, device, per_feature=False):
        self.per_feature = per_feature
        self.device = device

    def fit(self, x: np.ndarray):
        if self.per_feature:
            self.m = np.max(x, axis=0)
        else:
            self.m = np.max(x)
        self.mt = torch.tensor(self.m).to(self.device)

    def transform(self, x):
        if isinstance(x, np.ndarray):
            return x / self.m
        return x / self.mt

    def reverse_transform(self, x):
        if isinstance(x, np.ndarray):
            return x * self.m
        return x * self.mt


def compliance_loss(expected_nodes, generated_nodes):
    """Compliance loss.
    
    Checks if generated nodes match the expected ones.
    
    Parameters
    ----------
    expected_nodes : tuple
        A pair of node specification and the mask.
    generated_nodes : torch.tensor
        Generated nodes specification.
    Returns
    -------
        Loss value.
    """
    nodes_spec, nodes_mask = expected_nodes
    x = generated_nodes.squeeze(-1) * nodes_mask
    gt_x = (nodes_spec >= 0.5).float() * nodes_mask
    return F.binary_cross_entropy(x, gt_x)


def make_random_completion_mask(nodes, edges, node_params, _, *,
                                node_p=0.4,
                                edge_p=0.4):
    """Generates a random mask to use in structure completion training.

    Selects some subset of nodes and edges as 'fixed' - these nodes
    and edges should be close (or have to be the same) as in the 
    original structure.

    For example. Let the original node description be [0, 1, 0, 3],
    with the mask [False, True, True, False] the output should
    also contain node 1, and should NOT contain node 2, the presence
    of node 3 is not restricted."""
    if len(nodes.shape) == 1:
        is_batch = False
    elif len(nodes.shape) == 2:
        is_batch = True
    else:
        raise ValueError('"nodes" must have either 1 (non-batched) or 2 (batched) dims')

    nodes_mask = torch.rand(nodes.shape) < node_p
    if is_batch:
        nodes_mask[:, 0] = 0
    else:
        nodes_mask[0] = 0
    edges_mask = torch.rand(edges.shape) < edge_p
    if is_batch:
        edges_mask[:, :, 0] = 0
        edges_mask[:, 0, :] = 0
    else:
        edges_mask[:, 0] = 0
        edges_mask[0, :] = 0
    return nodes_mask, edges_mask


class Solver(object):
    """Class for training and testing the OrGAN model."""

    def __init__(self, config):
        """Constructor.

        Parameters
        ----------
        config : namespace, argparse.Namespace
            An object with configuration parameter values.
        """

        # Training problem specification
        #
        # Conditional generation (cGAN)
        self.conditional = config.conditional
        # Parametric generation
        self.parametric = config.parametric
        # Train completion
        self.completion = config.completion

        # Organization structure model (describing how
        # the organization should be evaluated)
        self.org_model = config.rules

        # Quality metrics
        self.org_metrics = MetricsAggregator(self.org_model)

        # Dataset
        self.data = OrganizationStructureDataset(load_params=self.parametric,
                                                 load_cond=self.conditional)
        self.data.load(config.data_dir)

        # Models configuration (generator, discriminator,
        # approximator)
        # Dimensions of the generator input
        self.z_dim = config.z_dim
        # The number of node types
        self.m_dim = self.data.node_num_types
        # The number of edge types
        self.b_dim = self.data.edge_num_types
        # Dimensions of the fully-connected layers group
        # at the beginning of the generator
        self.g_conv_dim = config.g_conv_dim
        # G's edge convolution specification
        self.g_edge_conv_dim = config.g_edge_conv_dim
        # Specification of G's fully connected layers for parameter values
        self.g_params_fc_dim = config.g_params_fc_dim
        # Спецификация преобразований, которые должны
        # быть реализованы дискриминатором и аппроксиматором.
        # Состоит из трех компонент:
        # graph_conv_dim (список, описывающий параметры графовых сверток,
        # в частности, размерности представлений вершин), aux_dim
        # (количество признаков в глобальном представлении графа) и
        # linear_dim (список, задающий количества нейронов в серии
        # полносвязных слоев)
        self.d_conv_dim = config.d_conv_dim
        # Specification of a fully-connected block at the end of the
        # discriminator
        self.d_fc_dim = config.d_fc_dim
        # Condition encoder parameters of the D
        self.d_cond_enc_dim = config.d_cond_enc_dim
        # Вес для штрафа на величину градиента в функции оптимизации
        self.lambda_gp = config.lambda_gp
        # Метод постобработки сгенерированных графов
        self.post_method = config.post_method

        # Список метрик организационной структуры, которые будут
        # использоваться при обучении (all - все)
        self.metric = 'all'

        # Конфигурация процесса обучения
        #
        # Размер батча
        self.batch_size = config.batch_size
        # Количество итераций (батчей) в процессе обучения
        self.num_iters = config.num_iters
        # Количество итераций (перед последней, num_iters) в течение
        # которых будет осуществляться снижение константы обучения
        self.num_iters_decay = config.num_iters_decay
        # Константа обучения для генератора
        self.g_lr = config.g_lr
        # Константа обучения для дискриминатора
        self.d_lr = config.d_lr
        # Дропаут (одно и то же значение используется везде, между
        # каждой парой слоев)
        self.dropout = config.dropout
        # Периодичность тренировки генератора
        # (каждые n_critic батчей)
        self.n_critic = config.n_critic
        # beta1 для Adam (при обучении всех моделей)
        self.beta1 = config.beta1
        # beta2 для Adam (при обучении всех моделей)
        self.beta2 = config.beta2
        # Итерация, с которой нужно продолжить процесс обучения.
        # Если значение не 0, то все модели будут загружены из
        # точек сохранения и процесс продолжен.
        self.resume_iters = config.resume_iters

        # Конфигурация процесса тестирования
        #
        # Указание на то, какую именно модель следует тестировать
        # (модель, созданную после test_iters итераций обучения).
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

        # Директории
        #
        # Директория записи журнала (используется только с
        # Tensorboard)
        self.log_dir = config.log_dir
        # Директория для сохранения моделей
        # (из этой же директории они будут подгружаться при необходимости
        # продолжить обучение)
        self.model_save_dir = config.model_save_dir
        # Directory to write samples during training
        self.samples_dir = config.samples_dir

        # Настройка периодичности вывода информации
        #
        # Периодичность записи данных в журнал (для Tensorboard)
        self.log_step = config.log_step
        # Периодичность сохранения моделей
        self.model_save_step = config.model_save_step
        # Периодичность изменения констант обучения.
        # Этим параметром регулируется то, как часто будет оцениваться
        # необходимость ревизии констант. См. также `num_iters_decay`.
        self.lr_update_step = config.lr_update_step

        # Should we pretrain?
        self.pretrain = config.pretrain

        # For the log to be informative, it should contain quality
        # characteristics of only generated structures
        assert self.log_step % self.n_critic == 0

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Build normalizers for float features of the dataset
        if self.parametric:
            self.node_features_normalizer = Normalizer(self.device)
            self.node_features_normalizer.fit(self.data.node_params)

        if self.conditional:
            self.cond_normalizer = Normalizer(self.device, per_feature=True)
            self.cond_normalizer.fit(self.data.cond)

    def build_model(self):
        """Create neural models (generator, discriminator, and approximator).
        """

        print('Max nodes:', self.data.vertexes)
        print('Node types:', self.data.node_num_types, self.m_dim)
        print('Edge types:', self.data.edge_num_types, self.b_dim)

        if self.completion:
            self.G = CompletionGenerator(self.g_conv_dim,
                                         self.g_edge_conv_dim,
                                         self.z_dim,
                                         self.data.vertexes,
                                         self.data.edge_num_types,
                                         self.dropout)
        elif not self.parametric and not self.conditional:
            self.G = EdgeAwareGenerator(self.g_conv_dim,
                                        self.g_edge_conv_dim,
                                        self.z_dim,
                                        self.data.vertexes,
                                        self.data.edge_num_types,
                                        self.dropout)
        else:
            self.G = CPGenerator(self.g_conv_dim,        # Graph encoding
                                 self.g_edge_conv_dim,   # Edge convs
                                 self.g_params_fc_dim,   # Parameters FC
                                 self.z_dim,
                                 self.data.condition_dim,
                                 self.data.vertexes,
                                 self.data.edge_num_types,
                                 self.data.features_per_node,
                                 self.dropout)

        # NOTE: Архитектуры дискриминатора и аппроксиматора полностью
        # идентичны.
        if not self.parametric and not self.conditional:
            self.D = Discriminator(self.d_conv_dim,
                                   self.m_dim,
                                   self.b_dim,
                                   self.dropout)
            self.V = Discriminator(self.d_conv_dim,
                                   self.m_dim,
                                   self.b_dim,
                                   self.dropout)
        else:
            self.D = CPDiscriminator(self.d_conv_dim,      #
                                     self.d_fc_dim,        # FC at the end
                                     self.d_cond_enc_dim,  # Condition encoding
                                     self.m_dim,
                                     self.b_dim,
                                     self.data.condition_dim,
                                     self.data.features_per_node,
                                     self.dropout)
            self.V = CPDiscriminator(self.d_conv_dim,      #
                                     self.d_fc_dim,        # FC at the end
                                     self.d_cond_enc_dim,  # Condition encoding
                                     self.m_dim,
                                     self.b_dim,
                                     self.data.condition_dim,
                                     self.data.features_per_node,
                                     self.dropout)

        # Совместный оптимизатор для генератора
        self.g_optimizer = torch.optim.Adam(self.G.parameters(),
                                            self.g_lr,
                                            [self.beta1, self.beta2])
        # Оптимизатор для дискриминатора
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                            self.d_lr,
                                            [self.beta1, self.beta2])
        # Оптимизатор для аппроксиматора
        self.v_optimizer = torch.optim.Adam(self.V.parameters(),
                                            self.g_lr,  # SIC!
                                            [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.V, 'V')

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    def load_pretrained(self):
        """Load pretrained models."""

        for model_code, model in [('G', self.G),
                                  ('D', self.D),
                                  ('V', self.V)]:
            # if there are pre-trained models and they are compatible
            path = os.path.join(self.model_save_dir, f'pre-{model_code}.ckpt')
            try:
                model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))   # noqa: E501
                print(f'Pretrained {model_code} has been loaded.')
            except Exception:
                print(f'Can"t load pre-trained {model_code} model, starting from scratch.')  # noqa: E501

    def print_network(self, model, name):
        """Print model description.

        Parameters
        ----------
        model : torch.Module
            Model to print.
        name : str
            Model name (only for readability purposes).
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Load models from a savepoint.

        Load the state of all models (generator, discriminator, and
        approximator) from a savepoint, located at `model_save_dir`.

        Parameters
        ----------
        resume_iters : int
            Iteration number, to specify a model savepoint.
        """

        print('Loading the trained models from step {}...'.format(resume_iters))               # noqa: E501
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))           # noqa: E501
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))           # noqa: E501
        V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))           # noqa: E501
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))  # noqa: E501
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))  # noqa: E501
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))  # noqa: E501

    def build_tensorboard(self):
        """Tensorboard logging initialization."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Sets learning rate constants (for all the models).

        Parameters
        ----------
        g_lr : float
            Learning rate for the generator (and approximator).
        d_lr : float
            Learning rate for the discriminator.
        """
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.v_optimizer.param_groups:
            param_group['lr'] = g_lr  # SIC!

    def reset_grad(self):
        """Reset gradients of all optimizers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Gradient penalty.

        (L2_norm(dy/dx) - 1)**2

        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1)**2)

    def label2onehot(self, labels, dim):
        """Transform labels into one-hot encoded vectors.

        Given tensor with integer values `labels` is extended
        by one dimensions, in which these labels are converted into
        one-hot codes.

        Parameters
        ----------
        labels : torch.tensor (int64)
            Tensor with non-negative integer labels.
        dim : int
            Number of categories in `labels` tensor. This number becomes
            the size of the new dimension of the output tensor.
            The specified number must be greater than the max
            value of `labels`.
        Returns
        -------
        torch.tensor (float)
            Real-valued tensor, consisting of zeros and ones.
        """
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        """Form samples from the input distribution of the generator."""
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):
        """Postprocessing by one of the differentiable discretization methods.

        The method is used to transform matrices, describing
        edges of a graph (without activations) to a representation, where
        an edge can have only one type (or be marked as absent).
        In other words, the representation is transformed into one,
        consisting of ones and zeroes (almost).

        Parameters
        ----------
        inputs : torch.tensor, tuple [torch.tensor], list [torch.tensor]
            Input tensors to transform.
        method : str
            Transformation type: `soft_gumbel`, `hard_gumbel`,
            `softmax`.
        temperature : float
            Transformation parameter.

        Returns
        -------
        list [torch.tensor]
            The list of output tensors, with transformation applied to the
            last dimension. If `inputs` was one tensor, the result is
            still a list, though one-element.
        """

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().
                                        view(-1, e_logits.size(-1)) / temperature,  # noqa: E501
                                        hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().
                                        view(-1, e_logits.size(-1)) / temperature,  # noqa: E501
                                        hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def postprocess_nodes(self, nodes_logits):
        """Transforms a list of node logits into richer form.

        Most code assumes, that the set of graph nodes is described
        by tensor vertexes x node_types. However, in the case of
        organization structures it turns out that a node and a node
        type are mostly synonyms (there can be at most one node of
        a given type). Therefore, generator returns only logits of
        presence of certain types of nodes, and this method transforms
        these logits into batch of vertex x node_types tensors,
        placing the values on diagonal and complementing the probability
        of node absence.

        Parameters
        ----------
        nodes_logits : pytorch.tensor
            Batch of logits for node presence,
            batch x vertexes.
        Returns
        -------
        torch.tensor
            Batch of specifications batch x vertexes x nodes.
        """
        nodes_sigm = torch.sigmoid(nodes_logits)
        nodes_hat = torch.diag_embed(nodes_sigm)
        nodes_hat[:, :, 0] += (1 - nodes_sigm)
        return nodes_hat

    def reward(self, orgs):
        """Structural reward.

        The method calculates a vector of structural reward values
        for the given batch of organization descriptions. The
        definition of structural reward can be project-specific (the
        list of metrics is defined in `self.metric`) and relies on
        various metrics defined in `org_model` passed to the
        constructor.

        Parameters
        ----------
        orgs : list
            A list of organization specifications.

        Returns
        -------
        numpy.ndarray, shape (batch_size, 1)
            Batch of reward values.
        """
        return self.org_metrics.valid_scores(orgs).reshape(-1, 1)

    def train(self):
        """Training cycle."""

        def compute_gp_loss(a_tensor, x_tensor, edges_hat, nodes_hat):
            eps = torch.rand(a_tensor.size(0), 1, 1, 1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)  # noqa: E501
            x_int1 = (eps.squeeze(-1) * x_tensor +
                      (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, x_int1, None)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + \
                self.gradient_penalty(grad1, x_int1)
            return d_loss_gp

        def process_batch(a_tensor, x_tensor, params, cond,
                          orgs, z, masks,
                          critic=True, is_training=True):
            """Обработка батча."""

            # Set the networks into the required mode
            self.G.train(is_training)
            self.D.train(is_training)
            self.V.train(is_training)
            torch.set_grad_enabled(is_training)

            # =============================================================== #
            #                1. Train the discriminator                       #
            # =============================================================== #

            # Compute loss with real structures.
            logits_real = self.D(a_tensor,
                                 x_tensor,
                                 params,    # node features
                                 cond)      # condition
            # minimize: -log(D(real)) - log(1-D(G(z)))
            d_loss_real = -torch.mean(torch.log(torch.sigmoid(logits_real)))

            # Compute loss with fake structures.
            edges_hat, nodes_hat, params_hat = self._invoke_G(z, cond,
                                                              partial=(x_tensor, a_tensor),
                                                              partial_masks=masks)
            logits_fake = self.D(edges_hat,
                                 nodes_hat,
                                 params_hat,  # node features
                                 cond)        # condition
            d_loss_fake = -torch.mean(torch.log(1 - torch.sigmoid(logits_fake)))  # noqa: E501

            # Compute loss for gradient penalty.
            # NOTE: It doesn't account for parametric gradient
            if True:
                d_loss_gp = 0.0
            else:
                d_loss_gp = compute_gp_loss(a_tensor, x_tensor,
                                            edges_hat, nodes_hat)

            if is_training:
                # Backward and optimize.
                d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            # loss['D/loss_gp'] = d_loss_gp.item()

            # =============================================================== #
            #            2. Train the generator and approximator              #
            # =============================================================== #

            if critic:

                # =========================================================== #
                #                   2.1 Train the approximator                #
                # =========================================================== #

                # Получить батч из генератора
                edges_hat, nodes_hat, params_hat = self._invoke_G(z, cond,
                                                                  partial=(x_tensor, a_tensor),
                                                                  partial_masks=masks)
                # Получить оценку настоящих образцов с помощью "черного ящика"
                # Real Reward
                rewardR = torch.from_numpy(self.reward(orgs)).to(self.device)
                # Получить оценку сгенерированного батча с помощью
                # "черного ящика"
                # Fake Reward
                orgs = self._orgs_from(edges_hat, nodes_hat, params_hat, cond)
                rewardF = torch.from_numpy(self.reward(orgs)).to(self.device)
                # Скорректировать веса аппроксиматора с учетом ошибки
                # предсказаний "валидности" образцов
                # Value loss
                value_proba_real = self.V(a_tensor,
                                          x_tensor,
                                          params,
                                          cond,
                                          torch.sigmoid)
                value_proba_fake = self.V(edges_hat,
                                          nodes_hat,
                                          params_hat,
                                          cond,
                                          torch.sigmoid)
                v_loss = torch.mean((value_proba_real - rewardR) ** 2 +
                                    (value_proba_fake - rewardF) ** 2)
                if is_training:
                    self.reset_grad()
                    v_loss.backward()
                    self.v_optimizer.step()

                loss['V/loss'] = v_loss.item()

                # =========================================================== #
                #               2.2 Train the generator                       #
                # =========================================================== #

                # Получить батч из генератора
                edges_hat, nodes_hat, params_hat = self._invoke_G(z, cond,
                                                                  partial=(x_tensor, a_tensor),
                                                                  partial_masks=masks)

                # Оценить правдоподобие с точки зрения дискриминатора
                logits_fake = self.D(edges_hat,
                                     nodes_hat,
                                     params_hat,    # node features
                                     cond)          # condition
                # minimize: - log(D(G(z)))  (mimic real)
                # g_loss_fake = -torch.mean(logits_fake)
                g_loss_fake = -torch.mean(torch.log(torch.sigmoid(logits_fake)))  # noqa: E501

                # Оценить выполнение требований, описываемых аппроксиматором
                value_proba_fake = self.V(edges_hat,
                                          nodes_hat,
                                          params_hat,     # node features
                                          cond,           # condition
                                          torch.sigmoid)
                # Мы хотим, чтобы сгенерированные образцы удовлетворяли
                # критериям, аппроксимируемым V, то есть V выдавал
                # для них 1.0
                g_loss_value = -torch.mean(torch.log(value_proba_fake))
                
                if self.completion:
                    g_loss_comp = compliance_loss((1 - x_tensor[:, :, 0], masks[0]),
                                                  1 - nodes_hat[:, :, 0])
                else:
                    g_loss_comp = torch.tensor(0.0, device=self.device)

                # Тут также может быть расчет других, дифференцируемых,
                # характеристик сгенерированной структуры
                if hasattr(self.org_model, 'soft_constraints'):
                    # User-level function has to deal with non-normalized
                    # values
                    params_hat_ = self.node_features_normalizer.\
                        reverse_transform(params_hat) \
                        if params_hat is not None else None
                    cond_ = self.cond_normalizer.reverse_transform(
                        cond) if cond is not None else None
                    g_loss_soft_constraints = self.org_model.soft_constraints(
                        nodes_hat, edges_hat, params_hat_, cond_)
                else:
                    g_loss_soft_constraints = torch.tensor(0.0,
                                                           device=self.device)

                # В итоге функция потерь для генератора складывается из
                # потерь неправдоподобности (g_loss_fake) потерь, связанных с
                # нарушением ограничений V (g_loss_value) и прочих потерь
                # Backward and optimize.
                g_loss = g_loss_fake + \
                         g_loss_value + \
                         g_loss_soft_constraints + \
                         g_loss_comp
                         
                if is_training:
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_value'] = g_loss_value.item()
                loss['G/loss_soft'] = g_loss_soft_constraints.item()
                if self.completion:
                    loss['G/loss_compliance'] = g_loss_comp.item()

            return orgs, loss

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        elif self.pretrain:
            print('Start pre-training...')
            self._pretrain()
            # self.load_pretrained()

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # Получение очередного батча, его подготовка и загрузка на
            # устройство
            x_tensor, a_tensor, params, cond, orgs, z, masks = self._next_batch('train')    # noqa: E501

            # Обработка обучающего батча, пересчет весов
            orgs, loss = process_batch(a_tensor, x_tensor, params, cond,
                                       orgs, z, masks,
                                       critic=((i+1) % self.n_critic == 0),
                                       is_training=True)

            # Валидация и вывод информации о текущем качестве моделей
            if (i+1) % self.log_step == 0:

                # Получение валидационного батча
                x_tensor, a_tensor, params, cond, orgs, z, masks = self._next_batch('validation')  # noqa: E501

                # Обработка обучающего батча, пересчет весов
                orgs, loss = process_batch(a_tensor, x_tensor, params, cond,
                                           orgs, z, masks,
                                           critic=((i+1) % self.n_critic == 0),
                                           is_training=False)

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)  # noqa: E501

                # Log update
                m0, m1 = all_scores(self.org_metrics, orgs, self.data, norm=True)     # 'orgs' is output of Fake Reward  # noqa: E501
                m0 = {k: np.array(v)[np.nonzero(v)].mean()
                      for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # =============================================================== #
            #                 4. Miscellaneous                                #
            # =============================================================== #

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))  # noqa: E501
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))  # noqa: E501
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))  # noqa: E501
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))  # noqa: E501

            if (i+1) % 10000 == 0:
                if self.samples_dir is not None:
                    self._write_samples(os.path.join(self.samples_dir,
                                                     f'samples-{i+1}.txt'),
                                        orgs[:self.batch_size])

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and \
               (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))  # noqa: E501

    def test(self):
        """Model testing."""

        # Load the trained generator.
        self.restore_model(self.test_iters)

        self.G.eval()
        self.D.eval()
        self.V.eval()

        with torch.no_grad():
            # В сущности, для тестирования реальные образцы нам
            # не нужны, потому что мы просто хотим оценить
            # правдоподобность генерируемых изображений
            #
            # Note, that testing code loads all models at once,
            # potentially it may result in memory problems.
            n, _, __, cond = self.data.next_test_batch()

            if self.conditional:
                cond = self.cond_normalizer.transform(cond)
                cond = torch.from_numpy(cond).to(self.device).float()
            else:
                cond = None

            z = self.sample_z(n.shape[0])
            z = torch.from_numpy(z).to(self.device).float()

            # Z-to-target
            edges_hat, nodes_hat, params_hat = self._invoke_G(z, cond)
            orgs = self._orgs_from(edges_hat, nodes_hat, params_hat, cond)

            # Log update
            m0, m1 = all_scores(self.org_metrics, orgs, self.data, norm=True)     # 'orgs' is output of Fake Reward  # noqa: E501
            m0 = {k: np.array(v)[np.nonzero(v)].mean()
                  for k, v in m0.items()}
            m0.update(m1)

            log = 'Testing on {} structures: '.format(n.shape[0])
            if m0:
                log += ', '.join(["{}: {:.4f}".format(tag, value)
                                  for tag, value in m0.items()])
            print(log)

    def generate(self, batch_size: int = 1, ctx=None):
        """Generate a batch of samples.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        ctx
            Context for the samples to be generated. May be optional.
        """

        if ctx is not None:
            if not isinstance(ctx, np.ndarray):
                ctx = np.array(ctx)
            if ctx.ndim == 1:
                ctx = np.stack([ctx] * batch_size, axis=0)
            elif ctx.ndim == 2:
                if ctx.shape[0] == 1:
                    ctx = np.concatenate([ctx] * batch_size, axis=0)
                elif ctx.shape[0] != batch_size:
                    raise ValueError('For two-dimensional ctx, the first '
                                     'dimension must be 1 or match the '
                                     'batch size')
                else:
                    pass   # ctx is fine as it is

            ctx = self.cond_normalizer.transform(ctx)
            ctx = torch.from_numpy(ctx).to(self.device).float()

        # Load the trained generator.
        self.restore_model(self.test_iters)

        self.G.eval()
        self.D.eval()
        self.V.eval()

        with torch.no_grad():
            # Sample noise
            z = self.sample_z(batch_size)
            # Make tensor and pass to the device
            z = torch.from_numpy(z).to(self.device).float()
            # Z-to-target
            edges_hat, nodes_hat, params_hat = self._invoke_G(z, ctx)
            # Convert to organizations
            return self._orgs_from(edges_hat, nodes_hat, params_hat, ctx)

    def generate_valid(self, n: int, ctx=None, max_generate: int = 1000):
        """Generate valid organizations.

        Parameters
        ----------
        n : int
            The number of valid organizations to generate.
        ctx : np.ndarray
            Condition (context) features, (n_features, ).
        max_generate : int
            Maximal number of instances to generate. If the underlying
            model accuracy is low, it may take too much time to generate
            the required number of valid organizations. This parameter
            helps to control the process and stop generation even if
            the required count isn't achieved.

        Returns
        -------
        list
            The list of organizations, containing not more than `n`
            instances of `Organization` class.
        """
        if ctx is not None:
            if not isinstance(ctx, np.ndarray):
                ctx = np.array(ctx)
            if ctx.ndim == 1 or (ctx.ndim == 2 and ctx.shape[0] == 1):
                pass
            else:
                raise ValueError('ctx must be (n_features, ) or (1, n_features)')  # noqa: E501

        valid_orgs = []
        batch_size = 32
        n_generated = 0
        while len(valid_orgs) < n and n_generated < max_generate:
            candidates = self.generate(batch_size, ctx=ctx)
            valid_orgs.extend([org for org in candidates
                               if self.org_model.validness(org)])
            n_generated += batch_size
        return valid_orgs[:n]

    def complete(self, batch_size: int = 1,
                 nodes=None,
                 nodes_mask=None,
                 edges=None,
                 edges_mask=None,
                 params=None,
                 params_mask=None,
                 ctx=None):
        """Complete a batch of samples.

        Parameters
        ----------
        batch_size : int
            Number of samples to complete.
        nodes : torch.tensor
            Partial specification of nodes.
        nodes_mask : torch.tensor
            Mask for the nodes. If an element is 1 (or True)
            then the `Solver` will try to save the value of this
            element in the generate structure.
        edges : torch.tensor
            Partial specification of edges.
        edges_mask : torch.tensor
            Mask for edges.
        params : torch.tensor 
            Partial specification of features.
        params_mask : torch.tensor
            Mask for parameter values (`params`).        
        ctx
            Context for the samples to be generated. May be optional.
        """

        if not self.completion:
            raise Exception('The model must be trained for completion to use this method.')

        if ctx is not None:
            if not isinstance(ctx, np.ndarray):
                ctx = np.array(ctx)
            if ctx.ndim == 1:
                ctx = np.stack([ctx] * batch_size, axis=0)
            elif ctx.ndim == 2:
                if ctx.shape[0] == 1:
                    ctx = np.concatenate([ctx] * batch_size, axis=0)
                elif ctx.shape[0] != batch_size:
                    raise ValueError('For two-dimensional ctx, the first '
                                     'dimension must be 1 or match the '
                                     'batch size')
                else:
                    pass   # ctx is fine as it is

            ctx = self.cond_normalizer.transform(ctx)
            ctx = torch.from_numpy(ctx).to(self.device).float()

        if (nodes is not None) != (nodes_mask is not None):
            raise ValueError('nodes and nodes_mask must be either both specified or not')
        
        if nodes is not None and nodes_mask is not None:
            if nodes.shape != nodes_mask.shape:
                raise ValueError('nodes and nodes_mask must have the same shape')
            if nodes.ndim == 1:
                nodes = np.stack([nodes] * batch_size, axis=0)
                nodes_mask = np.stack([nodes_mask] * batch_size, axis=0)
            elif nodes.ndim == 2:
                if nodes.shape[0] == 1:
                    nodes = np.concatenate([nodes] * batch_size, axis=0)
                    nodes_mask = np.concatenate([nodes_mask] * batch_size, axis=0)
                elif nodes.shape[0] != batch_size:
                    raise ValueError('For two-dimensional nodes specification, the first '
                                     'dimension must be 1 or match the '
                                     'batch size')
                else:
                    pass

        # Load the trained generator.
        self.restore_model(self.test_iters)

        self.G.eval()
        self.D.eval()
        self.V.eval()

        with torch.no_grad():
            # Sample noise
            z = self.sample_z(batch_size)
            # Make tensor and pass to the device
            z = torch.from_numpy(z).to(self.device).float()
            
            # Transform partial specification into the G's format
            x = torch.from_numpy(nodes).to(self.device).long()         # Nodes.
            x_tensor = self.label2onehot(x, self.m_dim)
            nodes_mask = torch.from_numpy(nodes_mask).to(self.device)
           
            # Z-to-target
            edges_hat, nodes_hat, params_hat = self._invoke_G(z, ctx,
                                                              partial=(x_tensor, None),
                                                              partial_masks=(nodes_mask, None))

            # Convert to organizations
            return self._orgs_from(edges_hat, nodes_hat, params_hat, ctx)

    def complete_valid(self, n: int, *, 
                 nodes=None,
                 nodes_mask=None,
                 edges=None,
                 edges_mask=None,
                 params=None,
                 params_mask=None,
                 ctx=None, 
                 max_generate: int = 1000):
        """Make samples completing the specified one.

        Parameters
        ----------
        n : int
            Number of samples completing the given one.
        nodes : torch.tensor
            Partial specification of nodes.
        nodes_mask : torch.tensor
            Mask for the nodes. If an element is 1 (or True)
            then the `Solver` will try to save the value of this
            element in the generate structure.
        edges : torch.tensor
            Partial specification of edges.
        edges_mask : torch.tensor
            Mask for edges.
        params : torch.tensor 
            Partial specification of features.
        params_mask : torch.tensor
            Mask for parameter values (`params`).        
        ctx
            Context for the samples to be generated. May be optional.
        max_generate : int
            Maximal number of instances to generate. If the underlying
            model accuracy is low, it may take too much time to generate
            the required number of valid organizations. This parameter
            helps to control the process and stop generation even if
            the required count isn't achieved.            
        """
        
        if not self.completion:
            raise Exception('The model must be trained for completion to use this method.')
        
        valid_orgs = []
        batch_size = 32
        n_generated = 0
        while len(valid_orgs) < n and n_generated < max_generate:
            candidates = self.complete(batch_size,
                                       nodes=nodes,
                                       nodes_mask=nodes_mask,
                                       edges=edges,
                                       edges_mask=edges_mask,
                                       params=params,
                                       params_mask=params_mask,
                                       ctx=ctx)
            valid_orgs.extend([org for org in candidates
                               if self.org_model.validness(org)])
            n_generated += batch_size
        return valid_orgs[:n]        

    def _pretrain(self):
        """Pretrain models."""

        BATCH_SIZE = min(50, self.data.train_count)

        def pretrain_validator(model,
                               target_nodes,
                               target_edges,
                               target_params,
                               cond, masks,
                               max_iters=1000):
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for i in range(max_iters):
                # Батч "мусора" от генератора
                optimizer.zero_grad()
                input_z = self.sample_z(BATCH_SIZE)
                z = torch.from_numpy(input_z).to(self.device).float()
                edges_hat, nodes_hat, params_hat = self._invoke_G(z, cond,
                                                                  partial=(target_nodes,
                                                                           target_edges,
                                                                           target_params),
                                                                  partial_masks=masks)
                value_bad = model(edges_hat,
                                  nodes_hat,
                                  params_hat,
                                  cond,
                                  torch.sigmoid)
                loss_bad = loss_fn(value_bad, torch.zeros((BATCH_SIZE, 1),
                                                          dtype=torch.float32,
                                                          device=self.device))
                loss_bad.backward()
                optimizer.step()
                loss_bad = loss_bad.detach().cpu().item()
                # Хорошие модели
                optimizer.zero_grad()
                value_good = model(target_edges,
                                   target_nodes,
                                   target_params,
                                   cond,
                                   torch.sigmoid)
                loss_good = loss_fn(value_good, torch.ones((BATCH_SIZE, 1),
                                                           dtype=torch.float32,
                                                           device=self.device))
                loss_good.backward()
                optimizer.step()

        def pretrain_generator(z, target_nodes, target_edges,
                               target_params, cond, masks,
                               max_iters=1000, loss_eps=0.01):
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(self.G.parameters(), lr=0.001)
            for i in range(max_iters):
                optimizer.zero_grad()
                edges_hat, nodes_hat, params_hat = self._invoke_G(z,
                                                                  cond,
                                                                  partial=(target_nodes,
                                                                           target_edges,
                                                                           target_params),
                                                                  partial_masks=masks)
                loss = loss_fn(nodes_hat, target_nodes) + \
                    loss_fn(edges_hat, target_edges)
                # If train for completion then additional loss for
                # non-compliance
                if self.completion:
                    loss = loss + compliance_loss((1 - target_nodes[:, :, 0], masks[0]),
                                                  1 - nodes_hat[:, :, 0])
                loss.backward()
                optimizer.step()
                loss_value = loss.detach().cpu().item()
                if loss_value < loss_eps:
                    break
            print(f'Generator loss@pretrain: {loss_value:.5f}')

        # Get a batch of real examples
        x_tensor, a_tensor, params, cond, _, z, masks = self._next_batch('train', BATCH_SIZE)     # noqa: E501

        # Use these examples to give the validator and discriminator
        # ideas of what is good and evil
        pretrain_validator(self.V, x_tensor, a_tensor, params, cond,
                           masks, max_iters=10)
        pretrain_validator(self.D, x_tensor, a_tensor, params, cond,
                           masks, max_iters=10)

        # Use the noise to pretrain the generator.
        # It tries to map each point to the respective
        # sample
        pretrain_generator(z, x_tensor, a_tensor, params, cond,
                           masks, max_iters=1000, loss_eps=0.01)

    def _invoke_G(self, z, cond, partial=None, partial_masks=None):
        """Generate a batch of graphs."""
        if self.completion:
            batch_size = z.shape[0]
            if partial is None:
                partial_nodes = torch.zeros((batch_size,
                                             self.m_dim,
                                             self.m_dim))
            else:
                partial_nodes = partial[0]
            if partial_masks is None:
                partial_nodes_mask = torch.zeros((batch_size,
                                                  self.m_dim))
            else:
                partial_nodes_mask = partial_masks[0]
            edges_logits, nodes_logits, node_params = self.G(partial_nodes,  # nodes
                                                             None,           # edges
                                                             partial_nodes_mask,   # nodes mask
                                                             None,                 # edges mask                                                             
                                                             cond, z)
        else:
            edges_logits, nodes_logits, node_params = self.G(cond, z)
        # Postprocess with Gumbel softmax
        edges_hat = self.postprocess((edges_logits, ),
                                     self.post_method)[0]
        nodes_hat = self.postprocess_nodes(nodes_logits)
        return edges_hat, nodes_hat, node_params

    def _next_batch(self, mode: str, batch_size=None):
        """Retrieve next batch and load it to the device.

        Parameters
        ----------
        mode: str
            Specification of what set to use: 'train' or 'validation'.

        Returns
        -------
            tuple
                A tensor of nodes (batch, nodes, nodes), a tensor
                of edges (batch, nodes, nodes, edges), a list
                of structures corresponding to the batch, and z-noise
                to use as an input for the generator.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if mode == 'train':
            x, a, p, c = self.data.next_train_batch(batch_size)
        elif mode == 'validation':
            x, a, p, c = self.data.next_validation_batch()
        else:
            raise ValueError(f'Unknown mode: \'{mode}\'. '
                             'Only ''train'' and ''validation'' supported')

        z = self.sample_z(x.shape[0])  # Батчи одинакового размера
        # If train for completion then generate a random mask
        if self.completion:
            nodes_mask, edges_mask = make_random_completion_mask(x, a, None, None)
        else:
            nodes_mask, edges_mask = None, None
        orgs = [Organization(x_, a_, node_features=p_, condition=c_)
                for x_, a_, p_, c_ in zip(x,
                                          a,
                                          p if p is not None else repeat(None),
                                          c if c is not None else repeat(None)
                                          )]

        #   a is a (self.batch_size, 12, 12) numpy array - adjacency matrices (a_ij is the number of connections)  # noqa: E501
        #   x is a (self.batch_size, 12) numpy array - node type (categorical, 0 for no-node)  # noqa: E501

        # Загрузим данные на вычислительное устройство и приведем в вид,
        # ожидаемый нейронными сетями

        a = torch.from_numpy(a).to(self.device).long()         # Adjacency.
        x = torch.from_numpy(x).to(self.device).long()         # Nodes.
        a_tensor = self.label2onehot(a, self.b_dim)
        x_tensor = self.label2onehot(x, self.m_dim)
        z = torch.from_numpy(z).to(self.device).float()

        # If it is parametric generation, we have to normalize
        # the parameters. Otherwise, it will be None
        if self.parametric:
            p = self.node_features_normalizer.transform(p)
            p = torch.from_numpy(p).to(self.device).float()
        # If it is conditional generation, then the condition
        # must also be normalized
        if self.conditional:
            c = self.cond_normalizer.transform(c)
            c = torch.from_numpy(c).to(self.device).float()

        return x_tensor, a_tensor, p, c, orgs, z, (nodes_mask, edges_mask)

    def _orgs_from(self, edges_hat, nodes_hat, params_hat, cond):
        """Получение организационных структур из выходных
        данных генератора.
        """
        # Раньше было вот так
        # edges_hard = self.postprocess((edges_logits, ),
        #                               self.post_method)[0]
        # nodes_hard = self.postprocess_nodes(nodes_logits)
        edges_hard = torch.max(edges_hat, -1)[1]  # раньше было hard
        nodes_hard = torch.max(nodes_hat, -1)[1]  # раньше было hard
        orgs = [self.data.matrices2graph(n_.detach().cpu().numpy(),
                                         e_.detach().cpu().numpy(),
                                         strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]

        if params_hat is not None:
            # Back to the original domain
            params = params_hat.detach().cpu().numpy()
            params = self.node_features_normalizer.reverse_transform(params)
            # Post-process parameters
            # TODO move this code somewhere else
            # Get rid of the negative, remove for non-existing nodes
            params = np.maximum(params, np.zeros_like(params))
            params = params * (np.expand_dims(nodes_hard.detach()
                                              .cpu().numpy(), -1) > 0)
        else:
            params = repeat(None)

        if cond is not None:
            cond = cond.detach().cpu().numpy()
            cond = self.cond_normalizer.reverse_transform(cond)
        else:
            cond = repeat(None)

        return [Organization(x_, a_, node_features=p_, condition=c_)
                for (x_, a_), p_, c_ in zip(orgs, params, cond)]

    def _write_samples(self, filename: str, orgs, log: str = None) -> None:
        """Writes samples to a given file."""
        with open(filename, 'w') as f:
            for i, org in enumerate(orgs):
                if hasattr(self.org_model, 'check_paramater_feasibility'):
                    check = self.org_model.check_paramater_feasibility(org.nodes,                  # noqa: E501
                                                                       org.node_features,          # noqa: E501
                                                                       # logging=True,             # noqa: E501
                                                                       ctx=org.condition)
                else:
                    check = self.org_model.validness(org)
                print(f'Sample #{i}:',
                      '\nContext\n', org.condition,
                      '\nNodes:\n', org.nodes,
                      '\nStaff:\n', org.node_features,
                      '\nEdges:\n', org.edges,
                      '\nCheck results:\n', check,
                      file=f)
                print('=======', file=f)
            if log is not None:
                print(log, file=f)
