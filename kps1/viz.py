from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt

from .models import DenseLayerSpec


def plot_training_curves(history: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    if "loss" in history:
        ax.plot(history["loss"], label="обучение")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="валидация")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("функция ошибки")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_network_graph(
    input_dim: int,
    layers: list[DenseLayerSpec],
    *,
    model_type: str,
    output_dim: int = 1,
):
    model_type_u = (model_type or "").upper()

    titles: list[str] = ["X"]
    bodies: list[str] = [f"вход\n{input_dim}"]
    for i, sp in enumerate(layers):
        titles.append(f"H{i+1}")
        bodies.append(f"{sp.units}\n{sp.activation}\ndrop={sp.dropout}")
    titles.append("Y")
    bodies.append(f"выход\n{output_dim}")

    n = len(titles)
    fig_w = max(8.0, 1.4 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    ax.set_axis_off()

    xs = list(range(n))
    y = 0.0
    r = 0.28

    def draw_node(i: int):
        ax.add_patch(plt.Circle((xs[i], y), r, fill=False, linewidth=1.6))
        ax.text(xs[i], y + 0.07, titles[i], ha="center", va="center", fontsize=12)
        ax.text(xs[i], y - 0.23, bodies[i], ha="center", va="center", fontsize=8)

    def arrow(i: int, j: int, *, alpha: float = 1.0):
        ax.annotate(
            "",
            xy=(xs[j] - r * 1.05, y),
            xytext=(xs[i] + r * 1.05, y),
            arrowprops=dict(arrowstyle="->", linewidth=1.1, alpha=alpha),
        )

    for i in range(n):
        draw_node(i)

    if model_type_u == "CFNN":
        for i in range(n - 1):
            arrow(i, i + 1, alpha=0.9)
        for i in range(0, n - 2):
            for j in range(i + 2, n):
                arrow(i, j, alpha=0.25)
    else:
        for i in range(n - 1):
            arrow(i, i + 1, alpha=0.9)

    ax.set_title(f"Схема сети ({model_type_u})")
    fig.tight_layout()
    return fig


def plot_backprop_gradients(backprop_steps: Iterable[dict[str, float]]):
    keys = None
    steps = []
    mats = []

    for i, d in enumerate(backprop_steps):
        if keys is None:
            keys = list(d.keys())
        steps.append(i)
        mats.append([d.get(k, 0.0) for k in keys])

    if keys is None:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "нет данных по градиентам", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 4))
    for j, k in enumerate(keys[:12]):
        ax.plot(steps, [row[j] for row in mats], label=k)

    ax.set_xlabel("батч (демо)")
    ax.set_ylabel("норма градиента ||grad||")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    return fig
