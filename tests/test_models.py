import sys

import torch

sys.path.append('.')

import organ.models       # noqa: E402
import tests.util         # noqa: E402 F401


def check_generator_protocol(g,
                             z_dim,
                             cond_dim,
                             n_nodes,
                             n_edge_types,
                             n_features):
    batch_size = 7
    z = torch.randn(batch_size, z_dim)
    # If conditional, make condition values
    if cond_dim is not None:
        cond = torch.randn(batch_size, cond_dim)
    else:
        cond = None
    edges, nodes, params = g(cond, z)
    # Проверка размерности генерируемых массивов, описывающих граф
    assert edges.shape == (batch_size, n_nodes, n_nodes, n_edge_types)
    assert nodes.shape == (batch_size, n_nodes)
    if n_features is not None:
        assert params.shape == (batch_size, n_nodes, n_features)
    else:
        assert params is None


def test_simple_generator():
    z_dim = 3
    n_nodes = 6
    # Данная версия дискриминатора рассчитывает на то, что есть
    # не более одной вершины каждого типа. Соответственно,
    # слот под вершину определенного типа и есть определенная позиция.
    n_edge_types = 2
    g = organ.models.SimpleGenerator([4, 5],
                                     z_dim,         # Размерность входного вектора  # noqa: E501
                                     n_nodes,       # Количество вершин в графе
                                     n_edge_types,  # Количество типов дуг
                                     0.0)
    check_generator_protocol(g,
                             z_dim,
                             None,         # Unconditional
                             n_nodes,
                             n_edge_types,
                             None)         # Non-parametric

    # Проверка чувствительности ко всем компонентам входного вектора
    # TODO: проверка чувствительности не поддерживает несколько входов
    # assert tests.util.is_input_sensitive(g, (1, 3))
    # Проверка обучаемости по каждому из возвращаемых компонент
    # Дуги (матрицы смежности)
    # assert tests.util.is_learnable(g, (1, 3), output_getter=lambda x: x[0])
    # Типы вершин
    # assert tests.util.is_learnable(g, (1, 3), output_getter=lambda x: x[1])


def test_edge_aware_generator():
    z_dim = 3
    n_nodes = 6
    # Данная версия дискриминатора рассчитывает на то, что есть
    # не более одной вершины каждого типа. Соответственно,
    # слот под вершину определенного типа и есть определенная позиция.
    n_edge_types = 2
    g = organ.models.EdgeAwareGenerator([4, 5],
                                        [7, 13],
                                        z_dim,         # Размерность входного вектора  # noqa: E501
                                        n_nodes,       # Количество вершин в графе     # noqa: E501
                                        n_edge_types,  # Количество типов дуг
                                        0.0)
    check_generator_protocol(g,
                             z_dim,
                             None,         # Unconditional
                             n_nodes,
                             n_edge_types,
                             None)         # Non-parametric


def test_cpgenerator():
    z_dim = 3
    cond_dim = 5
    n_nodes = 6
    # Данная версия дискриминатора рассчитывает на то, что есть
    # не более одной вершины каждого типа. Соответственно,
    # слот под вершину определенного типа и есть определенная позиция.
    n_edge_types = 2
    n_node_features = 1
    g = organ.models.CPGenerator([4, 5],     # conv_dims
                                 [7, 13],    # edge_conv_dims
                                 [3, 22],    # param_dims
                                 z_dim,      # Размерность входного вектора  # noqa: E501
                                 cond_dim,   # condition dim
                                 n_nodes,       # Количество вершин в графе
                                 n_edge_types,  # Количество типов дуг
                                 n_node_features,
                                 0.0)
    check_generator_protocol(g,
                             z_dim,
                             cond_dim,
                             n_nodes,
                             n_edge_types,
                             n_node_features)


def test_discriminator():
    n_nodes = 6
    n_node_types = 3
    n_edge_types = 2
    node_repr = 2  # Размерность вершин после серии графовых сверток
    global_node_repr = 4  # Размерность глобального представления

    d = organ.models.Discriminator(([5, node_repr],
                                    global_node_repr,
                                    [7, 3]),
                                   n_node_types,
                                   n_edge_types,
                                   0.0)

    # В качестве слоя дополнительной информации, доступной слою
    # агрегации, ничего передавать нельзя, потому что
    # дискриминатор реализован таким образом, что эта информация
    # дополняет определение вершин, в том числе, и в слоях свертки,
    # но внутри дискриминатора они создаются без расчета на это
    # (входная размерность этого не учитывает).
    d(torch.randn(1, n_nodes, n_nodes, n_edge_types),
      torch.randn(1, n_nodes, n_node_types),
      None,
      None)

    # TODO: Протестировать чувствительность и обучаемость


def test_discriminator_degenerate():
    n_nodes = 6
    n_node_types = 3
    n_edge_types = 2
    node_repr = 2  # Размерность вершин после серии графовых сверток
    global_node_repr = 4  # Размерность глобального представления

    d = organ.models.Discriminator(([5, node_repr],
                                    global_node_repr,
                                    [7, 3]),
                                   n_node_types,
                                   n_edge_types,
                                   0.0)

    edges = torch.zeros((1, n_nodes, n_nodes, n_edge_types),
                        dtype=torch.float32)
    edges[:, :, :, 0] = 1.
    nodes = torch.zeros((1, n_nodes, n_node_types),
                        dtype=torch.float32)
    nodes[:, :, 0] = 1.
    # В качестве слоя дополнительной информации, доступной слою
    # агрегации, ничего передавать нельзя, потому что
    # дискриминатор реализован таким образом, что эта информация
    # дополняет определение вершин, в том числе, и в слоях свертки,
    # но внутри дискриминатора они создаются без расчета на это
    # (входная размерность этого не учитывает).
    output = d(edges,
               nodes,
               None,
               None)
    assert torch.isfinite(torch.log(torch.sigmoid(output)))
    assert torch.isfinite(torch.log(1 - torch.sigmoid(output)))

    output = d(edges,
               nodes,
               None,
               None,
               torch.sigmoid)
    assert torch.isfinite(torch.log(output))
    assert torch.isfinite(torch.log(1 - output))
