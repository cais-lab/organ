"""Вспомогательные функции для проверки нейронных сетей и их фрагментов.
"""
import numpy as np
import torch
import torch.nn
import torch.optim as optim


def is_input_sensitive(module, input_shape, eps=1.0):
    """Проверка влияния каждого входного элемента результат."""

    def simple_comparator(a, b):
        return not np.allclose(a, b)

    def tuple_comparator(tuple_a, tuple_b):
        return any(map(lambda x: simple_comparator(*x), zip(tuple_a, tuple_b)))

    def detach(x):
        if isinstance(x, tuple):
            return tuple(xx.detach().cpu().numpy() for xx in x)
        else:
            return x.detach().cpu().numpy()

    base_input = torch.randn(input_shape)
    base_output = detach(module(base_input))
    comparator = tuple_comparator if isinstance(base_output, tuple) \
        else simple_comparator
    # Дадим возмущение каждому элементу входного тензора
    # и проверим, что результат применения исследуемого модуля
    # изменяется.
    flat_input = torch.flatten(base_input)
    for i in range(len(flat_input)):
        eps_vector = torch.zeros_like(flat_input)
        eps_vector[i] = eps
        new_input = torch.reshape(flat_input + eps_vector, input_shape)
        output = detach(module(new_input))
        # Если результат такой же, как и без возмущения,
        # то модуль оказывается нечувствителен к этому
        # входному элементу
        if not comparator(base_output, output):
            return False
    return True


def is_learnable(module, input_shape, *,
                 output_getter=lambda x: x,
                 n_iters=10,
                 loss_fn=None,
                 optimizer_lr=None):
    """Проверка того, что модуль является обучаемым.

    Суть проверки заключается в том, что мы пытаемся подстраивать
    веса (с помощью нормального процесса распространения ошибки),
    чтобы при фиксированном образце на входе у сети был
    заданный образец на выходе). Это не всегда возможно,
    но проверка идет по тому, приближается ли образец на
    выходе к желаемому и вообще, идет ли нормальное распространение
    градиента.
    """
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()

    if optimizer_lr is None:
        optimizer_lr = 0.01

    optimizer = optim.SGD(module.parameters(), lr=optimizer_lr)

    base_input = torch.randn(input_shape)
    base_output = output_getter(module(base_input))
    # Желаемый выход
    desired_output = torch.randn(base_output.shape)

    prev_loss = None
    for epoch in range(n_iters):

        # Обнулить градиенты оптимизатора
        optimizer.zero_grad()
        # Прямое распространение
        output = output_getter(module(base_input))
        loss = loss_fn(output, desired_output)
        # Обратное распространение
        loss.backward()
        # Корректировка параметров
        optimizer.step()

        if prev_loss is not None:
            if loss.item() >= prev_loss:
                return False

        prev_loss = loss.item()

    return True


def can_drive_learning(t, loss_fn, n_iters, optimizer_lr=None):
    """Checks if the specified function can be reduced by gradient descent.

    The idea of the check is that the loss function is used during
    several iterations to decrease its own value for the given
    set of tensors. The check is passed if its value actually
    decreases.
    """

    if optimizer_lr is None:
        optimizer_lr = 0.01

    optimizer = optim.SGD(t, lr=optimizer_lr)

    prev_loss = None
    for epoch in range(n_iters):

        # Обнулить градиенты оптимизатора
        optimizer.zero_grad()
        # Оценить значение функции
        loss = loss_fn(t)
        # Обратное распространение
        loss.backward()
        # Корректировка параметров
        optimizer.step()

        if prev_loss is not None:
            if loss.item() >= prev_loss:
                return False

        prev_loss = loss.item()

    return True


if __name__ == '__main__':

    layer = torch.nn.Linear(2, 2)

    print('Linear layer', 'is sensitive to all inputs'
          if is_input_sensitive(layer, (1, 2), eps=0.1)
          else 'isn''t sensitive to all inputs')

    print('Linear layer', 'is learnable'
          if is_learnable(layer, (1, 2))
          else 'isn''t learnable')
