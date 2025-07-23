import torch
from torch import nn
import numpy as np

class FidelityLoss(nn.Module):
    def __init__(self, reduction: str = "mean", compare_all: bool = False):
        """
        Args:
            reduction (str): 'none' | 'mean' | 'sum'
            compare_all (bool): if True, compare output[i] with all targets[j]
                                and return [batch_size, num_classes] matrix of fidelity losses
        """
        super().__init__()
        assert reduction in {"none", "mean", "sum"}
        self.reduction = reduction
        self.compare_all = compare_all

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Args:
            output: [batch_size, dim] — complex
            target:
                - if compare_all=False → [batch_size, dim]
                - if compare_all=True  → [num_classes, dim]

        Returns:
            loss:
                - if compare_all=False → [batch_size] or scalar
                - if compare_all=True  → [batch_size, num_classes]
        """
        if self.compare_all:
            # output: [B, D], target: [C, D]
            B, D = output.shape
            C = target.shape[0]
            # → [B, C, D]
            output_exp = output[:, None, :].expand(B, C, D)
            target_exp = target[None, :, :].expand(B, C, D)

            inner = torch.sum(output_exp.conj() * target_exp, dim=-1)
            fidelity = torch.abs(inner) ** 2  # [B, C]
            return 1 - fidelity  # [B, C]

        else:
            # Pairwise: output[i] vs target[i]
            inner = torch.sum(output.conj() * target, dim=-1)
            fidelity = torch.abs(inner) ** 2
            loss = 1 - fidelity
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

class QuantumClassifierEnsemble:
    def __init__(self, classifiers, class_states, device="cpu"):
        """
        :param classifiers: список обученных квантовых моделей (по 1 на класс)
        :param class_states: эталонные состояния (тензор [C, D])
        """
        self.device = device
        self.classifiers = [model.to(device).eval() for model in classifiers]
        self.class_states = class_states.to(device)

    def predict(self, x):
        """
        :param x: torch.Tensor [B, D] — входные данные
        :return: torch.LongTensor [B] — предсказанные метки классов
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)
        B = x.shape[0]
        C = len(self.classifiers)

        with torch.no_grad():
            outputs = []
            for clf in self.classifiers:
                state = clf(x)                         # [B, D]
                state = state / state.norm(dim=-1, keepdim=True)  # нормируем
                outputs.append(state)

            outputs = torch.stack(outputs, dim=1)      # [B, C, D]
            cs = self.class_states.unsqueeze(0)        # [1, C, D]
            inner = torch.sum(outputs.conj() * cs, dim=-1)  # [B, C]
            fidelity = torch.abs(inner) ** 2            # [B, C]
            return torch.argmax(fidelity, dim=-1)       # [B]

    def save(self, path):
        torch.save({
            "state_dicts": [clf.state_dict() for clf in self.classifiers],
            "class_states": self.class_states,
        }, path)

    @classmethod
    def load(cls, path, model_cls, device="cpu", in_features=None, depth=None, **kwargs):
        checkpoint = torch.load(path, map_location=device)
        state_dicts = checkpoint["state_dicts"]
        class_states = checkpoint["class_states"]

        classifiers = []
        for state_dict in state_dicts:
            model = model_cls(in_features=in_features, depth=depth, measurement_mode="state", **kwargs)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            classifiers.append(model)

        return cls(classifiers, class_states, device=device)

import torch

def predict_class(X, classifiers, class_states):
    """
    Предсказывает метки классов по наибольшей fidelity с эталонными состояниями.

    :param X: torch.Tensor или np.ndarray, shape [B, D]
    :param classifiers: список моделей, каждая возвращает statevector
    :param class_states: torch.Tensor, shape [num_classes, D], нормализованные эталоны
    :return: predicted_labels: torch.Tensor, shape [B]
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)

    X = X.to(class_states.device)
    B = X.shape[0]
    num_classes = len(classifiers)
    state_dim = class_states.shape[1]

    # Вычисляем выходы всех классификаторов
    with torch.no_grad():
        outputs = []
        for clf in classifiers:
            out = clf(X)  # [B, D]
            out = out / out.norm(dim=-1, keepdim=True)  # нормируем
            outputs.append(out)
        # outputs: list of [B, D] → [B, num_classes, D]
        outputs = torch.stack(outputs, dim=1)

        # class_states: [C, D] → [1, C, D]
        class_states = class_states.unsqueeze(0).expand(B, -1, -1)

        # Fidelity: [B, C]
        inner = torch.sum(outputs.conj() * class_states, dim=-1)
        fidelity = torch.abs(inner) ** 2

        # Предсказание — индекс класса с максимальной fidelity
        predicted = torch.argmax(fidelity, dim=-1)

    return predicted
