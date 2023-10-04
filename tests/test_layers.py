import sys
sys.path.append('.')

import numpy as np  # noqa: E402
import torch        # noqa: E402

import organ.layers       # noqa: E402


def test_graph_convolution():

    # Тестовый граф имеет три вершины двух типов и
    # два вида дуг:
    #   дуги первого вида ведут из вершины типа 0 в вершины типа 1,
    #   дуги второго вида - наоборот

    nodes = np.array([[1, 0],  # v0 (тип 0)
                      [0, 1],  # v1 (тип 1)
                      [0, 1],  # v2 (тип 1)
                      ], dtype=np.float32)
    edges = np.array([
                      # дуги первого вида
                      [[0, 1, 1],
                       [0, 0, 0],
                       [0, 0, 0]],
                      # дуги второго вида
                      [[0, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]]
                      ], dtype=np.float32)
    # Делаем батч (из одного примера)
    batch_nodes = torch.stack([torch.from_numpy(nodes)])
    batch_edges = torch.stack([torch.from_numpy(edges)])

    layer = organ.layers.GraphConvolution(2, [3, 4], None, 0.0)
    # Установим "неслучайные" параметры транформации, чтобы получить ожидаемый
    # результат свертки
    # Эти "неслучайные" преобразования по сути просто увеличение размерности
    # представления вершин. В результате вся свертка сводится к подсчету
    # смежных вершин различных типов и ее результат может быть проверен
    # достаточно легко.
    with torch.no_grad():
        layer.linear1.weight = torch.nn.Parameter(
                                    torch.from_numpy(
                                        np.array([[0, 0],
                                                  [1, 0],
                                                  [0, 1]],
                                                 dtype=np.float32)))
        layer.linear1.bias = torch.nn.Parameter(
                                 torch.zeros_like(layer.linear1.bias))
        layer.linear2.weight = torch.nn.Parameter(
                                    torch.from_numpy(
                                        np.array([[1, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1],
                                                  [0, 0, 0]],
                                                 dtype=np.float32)))
        layer.linear2.bias = torch.nn.Parameter(
                                 torch.zeros_like(layer.linear2.bias))

    output = layer.forward(batch_nodes, batch_edges)
    # Батч из одного графа, 3 вершины, 4 признака
    assert output.shape == torch.Size([1, 3, 4])
    assert np.allclose(output.detach().cpu().numpy(),
                       np.array([[[0.0, 3.0, 4.0, 0.0],
                                  [0.0, 2.0, 3.0, 0.0],
                                  [0.0, 2.0, 3.0, 0.0]]]))

    # Тест дропаута
    layer = organ.layers.GraphConvolution(2, [3, 4], None, 1.0)
    output = layer.forward(batch_nodes, batch_edges)
    assert output.shape == torch.Size([1, 3, 4])
    # При вероятности дропаута 1.0 все выходы должны стать нулями
    assert np.all(output.detach().cpu().numpy() == np.zeros((3, 4)))


def test_graph_aggregation():

    # Первое представление вершин
    v1 = np.array([[1, 0],
                   [0, 1]], dtype=np.float32)
    # Второе представление вершин
    v2 = np.array([[0, 1, 2],
                   [1, 0, 2]], dtype=np.float32)
    # Делаем батч (из одного примера)
    batch_v1 = torch.stack([torch.from_numpy(v1)])
    batch_v2 = torch.stack([torch.from_numpy(v2)])

    layer = organ.layers.GraphAggregation(2, 1, 3, dropout=0.0)
    # Установим "неслучайные" параметры транформации, чтобы получить ожидаемый
    # результат.
    # - Коэффициенты для сигмоидного блока подобраны так, чтобы под сигмодидой
    #    получился 0.0 а значит, 0.5 после применения сигмоиды
    # - Коэффициенты для блока гиперболического тангенса подобраны так, чтобы
    #   после применения тангенса получилось 0.5
    # Их перемножение даст 0.25 (для каждой вершины), итого по графу будет 0.5
    with torch.no_grad():
        layer.sigmoid_linear[0].weight = torch.nn.Parameter(
                                            torch.from_numpy(
                                                np.array([[1, 1, 1, 1, -1]],
                                                         dtype=np.float32)))
        layer.sigmoid_linear[0].bias = torch.nn.Parameter(
                                            torch.zeros_like(
                                                layer.sigmoid_linear[0].bias))
        layer.tanh_linear[0].weight = torch.nn.Parameter(
                                            torch.from_numpy(0.5493 * \
                                                np.array([[-0.5, -0.5, -0.5, -0.5, 1]], # noqa
                                                         dtype=np.float32)))
        layer.tanh_linear[0].bias = torch.nn.Parameter(
                                        torch.zeros_like(
                                            layer.tanh_linear[0].bias))

    inp = torch.cat((batch_v1, batch_v2), -1)
    h = layer(inp, activation=None)

    # Батч из одного графа, представление графа размерности 1
    assert h.shape == torch.Size([1, 1])
    assert np.allclose(h.detach().cpu().numpy(),
                       np.array([[0.5]], dtype=np.float32))

    h = layer(inp, activation=torch.tanh)
    assert h.shape == torch.Size([1, 1])
    assert np.allclose(h.detach().cpu().numpy(),
                       np.tanh(np.array([[0.5]], dtype=np.float32)))

    # Тест дропаута
    layer = organ.layers.GraphAggregation(2, 1, 3, dropout=1.0)
    h = layer.forward(inp, activation=None)
    # При вероятности дропаута 1.0 все выходы должны стать нулями
    assert np.allclose(h.detach().cpu().numpy(), np.zeros((1, 1)))


def test_cartesian():
    v = torch.tensor([[[1., 11.],
                       [2., 22.],
                       [3., 33.]]])

    t1, t2 = organ.layers.cartesian(v)
    assert t1.shape == t2.shape
    assert t1.shape == (1, 3, 3, 2)
