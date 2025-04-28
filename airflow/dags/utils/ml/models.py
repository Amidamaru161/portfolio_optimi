from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import Sequential
from torch_geometric.utils import to_dense_batch


class EIIE(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_size=3,
        conv_mid_features=2,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
        """Инициализатор EIIE (ансамбль идентичных независимых оценщиков).

        Аргументы:
            initial_features: Количество входных признаков.
            k_size: Размер первого сверточного ядра.
            conv_mid_features: Размер промежуточных сверточных каналов.
            conv_final_features: Размер финальных сверточных каналов.
            time_window: Размер временного окна, используемого в качестве состояния агента.
            device: Устройство, на котором будет выполняться нейронная сеть.

        Примечание:
            Ссылка на статью: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        n_size = time_window - k_size + 1

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_size),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1, out_channels=1, kernel_size=(1, 1)
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Определяет наиболее благоприятное действие данной политики при входе x.

        Аргументы:
          observation: наблюдение среды.
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Наиболее благоприятное действие.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        output = self.sequential(observation)  # форма [N, 20, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [last_stocks, output], dim=1
        )  # форма [N, 21, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # форма [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # форма [N, 1, PORTFOLIO_SIZE + 1, 1]

        # форма выхода должна быть [N, features] = [1, PORTFOLIO_SIZE + 1], где N — размер батча (1)
        # и size — количество признаков (вектор весов).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # форма [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Прямое распространение политики сети.

        Аргументы:
          observation: наблюдение среды (словарь).
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Действие, которое нужно выполнить (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Обрабатывает последнее действие для получения cash bias и последних акций.

        Аргументы:
          last_action: последнее выполненное действие.

        Возвращает:
            Последние акции и cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


class EI3(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_short=3,
        k_medium=21,
        conv_mid_features=3,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
        """Инициализатор EI3 (ансамбль идентичных независимых inception).

        Аргументы:
            initial_features: Количество входных признаков.
            k_short: Размер короткого сверточного ядра.
            k_medium: Размер среднего сверточного ядра.
            conv_mid_features: Размер промежуточных сверточных каналов.
            conv_final_features: Размер финальных сверточных каналов.
            time_window: Размер временного окна, используемого в качестве состояния агента.
            device: Устройство, на котором будет выполняться нейронная сеть.

        Примечание:
            Ссылка на статью: https://doi.org/10.1145/3357384.3357961.
        """
        super().__init__()
        self.device = device

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_medium),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        self.final_convolution = nn.Conv2d(
            in_channels=2 * conv_final_features + initial_features + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Определяет наиболее благоприятное действие данной политики при входе x.

        Аргументы:
          observation: наблюдение среды.
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Наиболее благоприятное действие.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        features = torch.cat(
            [last_stocks, short_features, medium_features, long_features], dim=1
        )
        output = self.final_convolution(features)
        output = torch.cat([cash_bias, output], dim=2)

        # форма выхода должна быть [N, features] = [1, PORTFOLIO_SIZE + 1], где N — размер батча (1)
        # и size — количество признаков (вектор весов).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # форма [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Прямое распространение политики сети.

        Аргументы:
          observation: наблюдение среды (словарь).
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Действие, которое нужно выполнить (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Обрабатывает последнее действие для получения cash bias и последних акций.

        Аргументы:
          last_action: последнее выполненное действие.

        Возвращает:
            Последние акции и cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


class GPM(nn.Module):
    def __init__(
        self,
        edge_index,
        edge_type,
        nodes_to_select,
        initial_features=3,
        k_short=3,
        k_medium=21,
        conv_mid_features=3,
        conv_final_features=20,
        graph_layers=1,
        time_window=50,
        softmax_temperature=1,
        device="cpu",
    ):
        """Инициализатор GPM (графовое управление портфелем).

        Аргументы:
            edge_index: Связность графа в формате COO.
            edge_type: Тип каждого ребра в edge_index.
            nodes_to_select: ID узлов, которые будут выбраны в портфель.
            initial_features: Количество входных признаков.
            k_short: Размер короткого сверточного ядра.
            k_medium: Размер среднего сверточного ядра.
            conv_mid_features: Размер промежуточных сверточных каналов.
            conv_final_features: Размер финальных сверточных каналов.
            graph_layers: Количество слоев графовой нейронной сети.
            time_window: Размер временного окна, используемого в качестве состояния агента.
            softmax_temperature: Параметр температуры для softmax-функции.
            device: Устройство, на котором будет выполняться нейронная сеть.

        Примечание:
            Ссылка на статью: https://doi.org/10.1016/j.neucom.2022.04.105.
        """
        super().__init__()
        self.device = device
        self.softmax_temperature = softmax_temperature

        num_relations = np.unique(edge_type).shape[0]

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        self.edge_index = edge_index.to(self.device).long()

        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)
        self.edge_type = edge_type.to(self.device).long()

        if isinstance(nodes_to_select, np.ndarray):
            nodes_to_select = torch.from_numpy(nodes_to_select)
        elif isinstance(nodes_to_select, list):
            nodes_to_select = torch.tensor(nodes_to_select)
        self.nodes_to_select = nodes_to_select.to(self.device)

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_medium),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        feature_size = 2 * conv_final_features + initial_features

        graph_layers_list = []
        for i in range(graph_layers):
            graph_layers_list += [
                (
                    RGCNConv(feature_size, feature_size, num_relations),
                    "x, edge_index, edge_type -> x",
                ),
                nn.LeakyReLU(),
            ]

        self.gcn = Sequential("x, edge_index, edge_type", graph_layers_list)

        self.final_convolution = nn.Conv2d(
            in_channels=2 * feature_size + 1,
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Определяет наиболее благоприятное действие данной политики при входе x.

        Аргументы:
          observation: наблюдение среды.
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Наиболее благоприятное действие.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        temporal_features = torch.cat(
            [short_features, medium_features, long_features], dim=1
        )  # форма [N, feature_size, num_stocks, 1]

        # добавляем признаки в граф
        graph_batch = self._create_graph_batch(temporal_features, self.edge_index)

        # устанавливаем edge index для батча
        edge_type = self._create_edge_type_for_batch(graph_batch, self.edge_type)

        # выполняем графовую свертку
        graph_features = self.gcn(
            graph_batch.x, graph_batch.edge_index, edge_type
        )  # форма [N * num_stocks, feature_size]
        graph_features, _ = to_dense_batch(
            graph_features, graph_batch.batch
        )  # форма [N, num_stocks, feature_size]
        graph_features = torch.transpose(
            graph_features, 1, 2
        )  # форма [N, feature_size, num_stocks]
        graph_features = torch.unsqueeze(
            graph_features, 3
        )  # форма [N, feature_size, num_stocks, 1]
        graph_features = graph_features.to(self.device)

        # объединяем графовые признаки и временные признаки
        features = torch.cat(
            [temporal_features, graph_features], dim=1
        )  # форма [N, 2 * feature_size, num_stocks, 1]

        # выполняем выборку и добавляем последние акции
        features = torch.index_select(
            features, dim=2, index=self.nodes_to_select
        )  # форма [N, 2 * feature_size, portfolio_size, 1]
        features = torch.cat([last_stocks, features], dim=1)

        # финальная свертка
        output = self.final_convolution(features)  # форма [N, 1, portfolio_size, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # форма [N, 1, portfolio_size + 1, 1]

        # форма выхода должна быть [N, portfolio_size + 1] = [1, portfolio_size + 1], где N — размер батча
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # форма [N, portfolio_size + 1]

        output = self.softmax(output / self.softmax_temperature)

        return output

    def forward(self, observation, last_action):
        """Прямое распространение политики сети.

        Аргументы:
          observation: наблюдение среды (словарь).
          last_action: последнее действие, выполненное агентом.

        Возвращает:
          Действие, которое нужно выполнить (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().сжать()
        return action

    def _process_last_action(self, last_action):
        """Обрабатывает последнее действие для получения cash bias и последних акций.

        Аргументы:
          last_action: последнее выполненное действие.

        Возвращает:
          Последние акции и cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

    def _create_graph_batch(self, features, edge_index):
        """Создает батч графов с признаками.

        Аргументы:
          features: Тензор формы [batch_size, feature_size, num_stocks, 1].
          edge_index: Связность графа в формате COO.

        Возвращает:
          Батч графов с временными признаками, ассоциированными с каждым узлом.
        """
        batch_size = features.shape[0]
        graphs = []
        for i in range(batch_size):
            x = features[i, :, :, 0]  # форма [feature_size, num_stocks]
            x = torch.transpose(x, 0, 1)  # форма [num_stocks, feature_size]
            new_graph = Data(x=x, edge_index=edge_index).to(self.device)
            graphs.append(new_graph)
        return Batch.from_data_list(graphs)

    def _create_edge_type_for_batch(self, batch, edge_type):
        """Создает тензор типов ребер для батча графов.

        Аргументы:
          batch: Батч графовых данных.
          edge_type: Оригинальный тензор типов ребер.

        Возвращает:
          Тензор типов ребер, адаптированный для батча.
        """
        batch_edge_type = torch.clone(edge_type).detach()
        for i in range(1, batch.batch_size):
            batch_edge_type = torch.cat(
                [batch_edge_type, torch.clone(edge_type).detach()]
            )
        return batch_edge_type