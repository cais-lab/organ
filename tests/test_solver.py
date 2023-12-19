import os

import torch
import numpy as np

import pytest

import organ.solver
import organ.structure.models
import organ.config
from organ.demo import LogisticsDepartmentModel


def make_config(pretrain: bool = False):
    args = organ.config.make_config(
        rules=LogisticsDepartmentModel(),
        z_dim=8,
        g_conv_dim=[128, 256, 512],
        d_conv_dim=[[128, 64], 128, [128, 64]],
        num_iters=20,
        resume_iters=None,
        test_iters=20,
        mode='train',
        # Directories.
        data_dir='tests/data',
        log_dir='.tmp/organ/logs',
        model_save_dir='.tmp/organ/models',
        model_save_step=10,
        pretrain=False,
        parametric=True,
        conditional=True)
    return args


def test_compliance_loss():
    nodes = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    nodes_mask = torch.tensor([[False, True, True]], dtype=torch.int8)

    # Compliant
    gen_nodes = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    assert organ.solver.compliance_loss((nodes, nodes_mask), gen_nodes) < 1e-3

    # Non-compliant
    gen_nodes = torch.tensor([[0, 1, 1]], dtype=torch.float32)
    assert organ.solver.compliance_loss((nodes, nodes_mask), gen_nodes) > 1e-3


def test_create_solver():
    args = make_config()
    organ.solver.Solver(args)


def test_label2onehot():
    args = make_config()
    s = organ.solver.Solver(args)

    labels = torch.tensor([[0, 1, 2, 3]],
                          dtype=torch.int64).to(s.device)
    r = s.label2onehot(labels, 4).cpu()
    # Добавляется еще одна размерность
    assert r.shape == (1, 4, 4)
    # Результат будет диагональной матрицей
    assert torch.allclose(r[0, :, :],
                          torch.diag(torch.ones((4,),
                                                dtype=torch.float32)))
    # Те же исходные данные, но количество категорий больше
    # (повлияет на размерность результата)
    r = s.label2onehot(labels, 5).cpu()
    assert r.shape == (1, 4, 5)
    assert torch.sum(r).item() == pytest.approx(4.)

    # Общий тест для некоторого "условно произвольного" вектора
    values = [3, 5, 1, 0, 7]
    labels = torch.tensor(values,
                          dtype=torch.int64).to(s.device)
    r = s.label2onehot(labels, 8).cpu()
    assert torch.sum(r).item() == pytest.approx(len(values))
    for i, v in enumerate(values):
        assert r[i, v] == pytest.approx(1.)


def test_postprocess():
    args = make_config()
    s = organ.solver.Solver(args)

    # Обработка одного тензора
    x = torch.tensor([[2., 3., 4.],
                      [4., 3., 2.]],
                     dtype=torch.float32)
    r = s.postprocess(x, 'softmax')
    # Возвращает в любом случае список
    assert type(r) == list
    # В данном случае, он будет состоять из одного элемента
    assert len(r) == 1
    assert r[0].shape == x.shape
    assert torch.allclose(torch.sum(r[0], -1),
                          torch.ones((2,), dtype=torch.float32))

    # Обработка списка (или кортежа) тензоров
    xs = [x, x]
    rs = s.postprocess(xs, 'soft_gumbel')
    assert type(rs) == list
    assert len(rs) == len(xs)
    for x, r in zip(xs, rs):
        assert r.shape == x.shape
        assert torch.allclose(torch.sum(r, -1),
                              torch.ones(x.shape[:-1],
                                         dtype=torch.float32))


def test_fake():
    config = make_config()
    config.rules = organ.structure.models.Generic()

    # Необходимо, чтобы эти папки были созданы заранее
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    s = organ.solver.Solver(config)
    s.train()


@pytest.mark.integration
def test_training_and_testing():
    config = make_config(pretrain=True)

    # Необходимо, чтобы эти папки были созданы заранее
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    s = organ.solver.Solver(config)
    s.train()

    # Проверим, действительно ли в ходе обучения создавались
    # контрольные точки (периодичность их создания задается
    # параметром конфигурации `model_save_step`:
    for savepoint_i in range(config.model_save_step,
                             config.num_iters + 1,
                             config.model_save_step):
        assert os.path.exists(os.path.join(config.model_save_dir,
                                           f'{savepoint_i}-G.ckpt'))
        assert os.path.exists(os.path.join(config.model_save_dir,
                                           f'{savepoint_i}-D.ckpt'))
        assert os.path.exists(os.path.join(config.model_save_dir,
                                           f'{savepoint_i}-V.ckpt'))

    # После обучения должна быть возможность запустить
    # и тестирование модели
    s = organ.solver.Solver(config)
    s.test()

    # It should also be possible to run inference (generation)
    batch_size = 10
    ctx = np.random.random((batch_size, 2))
    s = organ.solver.Solver(config)
    orgs = s.generate(batch_size=batch_size, ctx=ctx)
    assert len(orgs) == batch_size
    assert type(orgs[0]) == organ.structure.models.Organization
    n_nodes = LogisticsDepartmentModel.MAX_NODES_PER_GRAPH
    assert orgs[0].nodes.shape == (n_nodes, )
    assert orgs[0].edges.shape == (n_nodes, n_nodes)

    # orgs = s.generate(batch_size=10)
    # assert len(orgs) == 10
    # assert type(orgs[0]) == organ.structure.models.Organization
