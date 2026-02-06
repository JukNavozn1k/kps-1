from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt

from .models import DenseLayerSpec


def plot_training_curves(history: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    if "loss" in history:
        ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
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
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_axis_off()

    nodes: list[tuple[str, str]] = [("Input", f"dim={input_dim}")]
    for i, spec in enumerate(layers):
        nodes.append((f"Dense {i+1}", f"units={spec.units}\nact={spec.activation}\ndrop={spec.dropout}"))
    nodes.append(("Output", f"dim={output_dim}\nlinear"))

    n = len(nodes)
    xs = list(range(n))

    def draw_box(x: float, y: float, title: str, body: str):
        ax.add_patch(
            plt.Rectangle(
                (x - 0.42, y - 0.32),
                0.84,
                0.64,
                fill=False,
                linewidth=1.4,
            )
        )
        ax.text(x, y + 0.16, title, ha="center", va="center", fontsize=10)
        ax.text(x, y - 0.08, body, ha="center", va="center", fontsize=8)

    y0 = 0.0
    for i, (title, body) in enumerate(nodes):
        draw_box(xs[i], y0, title, body)

    def arrow(x1: float, y1: float, x2: float, y2: float, alpha: float = 1.0):
        ax.annotate(
            "",
            xy=(x2 - 0.44, y2),
            xytext=(x1 + 0.44, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.0, alpha=alpha),
        )

    model_type_u = (model_type or "").upper()
    if model_type_u == "CFNN":
        hidden_count = max(0, len(layers))
        for i in range(1, 1 + hidden_count):
            arrow(xs[0], y0, xs[i], y0, alpha=0.9)

        for i in range(1, 1 + hidden_count):
            for j in range(i + 1, 1 + hidden_count):
                arrow(xs[i], y0, xs[j], y0, alpha=0.35)

        for i in range(0, 1 + hidden_count):
            arrow(xs[i], y0, xs[-1], y0, alpha=0.7 if i in (0, hidden_count) else 0.35)
    else:
        for i in range(n - 1):
            arrow(xs[i], y0, xs[i + 1], y0, alpha=0.95)

    ax.set_title(model_type_u)
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
        ax.text(0.5, 0.5, "no gradient data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 4))
    for j, k in enumerate(keys[:12]):
        ax.plot(steps, [row[j] for row in mats], label=k)

    ax.set_xlabel("demo batch")
    ax.set_ylabel("||grad||")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    return fig
