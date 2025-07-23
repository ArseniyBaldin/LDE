import torch
from torch.utils.data import TensorDataset, DataLoader
from utilities.train_utils import FidelityLoss, predict_class, QuantumClassifierEnsemble
from utilities.visual_utils import plot_clusters
from utilities.data_utils import generate_clusters, pad_to_statevector_dim
from utilities.speedy import VQAmplitudeEmbedding
from encoder.LDE import LDE
import numpy as np

def train_quantum_classifiers(X, y, model_cls, epochs=50, batch_size=64, lr=1e-2, log_every=10, device="cpu"):
    """
    Обучает по одному классификатору на каждый класс. Каждый обучается
    на свои примеры и приближается к опорному состоянию (class state).

    :param X: np.ndarray, shape [N, dim]
    :param y: np.ndarray, shape [N]
    :param model_cls: функция-класс для создания квантовой модели (например, VQAmplitudeEmbedding)
    :return: список моделей и соответствующих эталонных состояний
    """
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    num_classes = int(y.max().item()) + 1
    dim = X.shape[1]

    classifiers = []
    class_states = []

    for c in range(num_classes):
        # Модель на класс c
        model = model_cls(in_features=dim, depth=3, measurement_mode="state").to(device)

        # Целевое состояние = усреднённый нормализованный state
        X_class = X[y == c]
        phi = X_class / X_class.norm(dim=-1, keepdim=True)
        phi_mean = phi.mean(dim=0)
        phi_target = phi_mean / phi_mean.norm()
        class_states.append(phi_target.detach().clone())

        # Данные и оптимизация
        loader = DataLoader(TensorDataset(X_class), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = FidelityLoss()

        for epoch in range(epochs):
            losses = []
            for (batch,) in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, phi_target.expand_as(out))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch % log_every == 0:
                avg_loss = np.mean(losses)
                print(f"[Class {c}] Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

        classifiers.append(model.eval())

    return classifiers, torch.stack(class_states)

if __name__ == '__main__':
    configs = [
        {"type": "gaussian", "mean": (0, 0), "sigma": (1, 1), "angle_deg": 0, "n_points": 100},
        {"type": "gaussian", "mean": (5, 5), "sigma": (1, 1), "angle_deg": 0, "n_points": 100},
        {"type": "ball", "mean": (2, -3), "radius": 1.0, "n_points": 100}
    ]
    X, y = generate_clusters(configs)

    plot_clusters(X, y)

    X_pad = pad_to_statevector_dim(X)
    encoder = LDE(1, X_pad.shape[-1], center=np.mean(X_pad, axis=0))

    q_data = encoder.forward(X_pad)

    classifiers, class_states = train_quantum_classifiers(
        q_data, y, model_cls=VQAmplitudeEmbedding, epochs=100, device="cpu"
    )

    ensemble = QuantumClassifierEnsemble(classifiers, class_states)
    ensemble.save("saved_model.pth")
    ensemble = QuantumClassifierEnsemble.load(
        "saved_model.pth",
        model_cls=VQAmplitudeEmbedding,
        in_features=4,  # размер входа
        depth=3,
        entangling="strong"
    )
    y_pred = ensemble.predict(q_data)  # X: [B, D]

    plot_clusters(X, y_pred)

