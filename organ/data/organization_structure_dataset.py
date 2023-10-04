"""Модуль для чтения набора данных организационных структур."""

import pickle
import os
import numpy as np


class OrganizationStructureDataset:
    """Класс для работы с набором организационных структур.

    Обеспечивает получение базовой информации по набору данных
    (размер, количество типов вершин и пр.), а также итерацию
    по батчам.

    Предполагается, что набор данных представлен набором
    файлов, размещенных в некоторой (одной) директории:
        - data_nodes.npy - numpy-файл с информацией по вершинам
        - data_edges.npy - numpy-файл с информацией по связям
        - data_staff.npy - numpy-файл со атрибутами вершин
        - data_meta.pkl - numpy-файл с общей информацией о
               наборе (разбиение на обучающую и валидационную
               выборки и пр.)
    """
    def __init__(self, load_cond=False, load_params=False):
        self.load_cond = load_cond
        self.load_params = load_params

    def load(self, path, subset=1):
        """Загрузка набора данных.

        Считывает набор данных, размещенный в указанной директории.
        Данную операцию необходимо совершить перед всеми прочими
        действиями с набором данных.

        Parameters
        ----------
            path : str
                Директория из которой должен быть считан набор.
            subset : float
                Доля загружаемого набора (загрузка части набора
                может быть полезна для отладки или для экспериментов
                по чувствительности моделей к размеру обучающей
                выборки).
        """

        # Train part
        self.nodes = np.load(os.path.join(path, 'data_nodes.npy'))
        self.edges = np.load(os.path.join(path, 'data_edges.npy'))
        if self.load_params:
            self.node_params = np.load(os.path.join(path, 'data_staff.npy'))
        if self.load_cond:
            self.cond = np.load(os.path.join(path, 'data_cond.npy'))

        # В частности, из этого файла будут загружены такие поля
        # как
        #   - train_idx - список индексов обучающего множества
        #   - validation_idx - список индексов валидационного множества
        #   - test_idx - список индексов тестового множества
        with open(os.path.join(path, 'data_meta.pkl'), 'rb') as f:
            self.__dict__.update(pickle.load(f))

        # Одновременное сокращение набора (если это было необходимо
        # и его перемешивание)
        self.train_idx = np.random.choice(self.train_idx,
                                          int(len(self.train_idx) * subset),
                                          replace=False)
        self.validation_idx = np.random.choice(self.validation_idx,
                                               int(len(self.validation_idx) * subset),  # noqa: E501
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx,
                                         int(len(self.test_idx) * subset),
                                         replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

        # Components that will be returned on iteration
        components = [self.nodes, self.edges]
        components.append(self.node_params if self.load_params else None)
        components.append(self.cond if self.load_cond else None)

        self.components = tuple(components)

    def matrices2graph(self, node_labels, edge_labels, strict=False):
        """
        Transforms matrix definition of a labeled graph into a graph instance.

        Currently, this function just glues inputs. In general, it can be used
        to transform it to some optimized representation.

        Parameters
        ----------
        node_labels : (nodes, )
            Numpy array with node types.
        edge_labels : (nodes, nodes, )
            2D numpy array with edge types.

        Returns
        -------
        tuple ((nodes,), (nodes, nodes))
            tuple representing a graph.
        """

        return node_labels, edge_labels

    def _next_batch(self, counter, count, idx, batch_size):

        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = tuple(obj[idx[counter:counter + batch_size]]
                           if obj is not None else None
                           for obj in self.components)

            counter += batch_size
        else:
            output = tuple(obj[idx]
                           if obj is not None else None
                           for obj in self.components)

        return counter, output

    def next_train_batch(self, batch_size=None):
        """Получение батча из обучающей выборки.

        Parameters
        ----------
        batch_size : int
            Размер батча. Если не задан, возвращается все множество.

        Returns
        -------
        list [numpy.ndarray]
            Список, состоящий из батча описаний вершин и батча описаний
            связей.
        """
        self.train_counter, out = self._next_batch(
            counter=self.train_counter,
            count=self.train_count,
            idx=self.train_idx,
            batch_size=batch_size
        )

        return out

    def next_validation_batch(self, batch_size=None):
        """Получение батча из валидационной выборки.

        Parameters
        ----------
        batch_size : int
            Размер батча. Если не задан, возвращается все множество.

        Returns
        -------
        list [numpy.ndarray]
            Список, состоящий из батча описаний вершин и батча описаний
            связей.
        """
        self.validation_counter, out = self._next_batch(
            counter=self.validation_counter,
            count=self.validation_count,
            idx=self.validation_idx,
            batch_size=batch_size
        )

        return out

    def next_test_batch(self, batch_size=None):
        """Получение батча из тестовой выборки.

        Parameters
        ----------
        batch_size : int
            Размер батча. Если не задан, возвращается все множество.

        Returns
        -------
        list [numpy.ndarray]
            Список, состоящий из батча описаний вершин и батча описаний
            связей.
        """
        self.test_counter, out = self._next_batch(
            counter=self.test_counter,
            count=self.test_count,
            idx=self.test_idx,
            batch_size=batch_size
        )

        return out

    def __len__(self):
        return self.__len


if __name__ == '__main__':

    pass
