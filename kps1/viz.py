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
    model_type_u = (model_type or "").upper()

    max_neurons = 8
    max_input = 6
    max_output = 3

    layer_sizes: list[int] = [input_dim] + [sp.units for sp in layers] + [output_dim]

    def display_count(size: int, *, kind: str) -> int:
        if kind == "input":
            return min(max_input, max(1, size))
        if kind == "output":
            return min(max_output, max(1, size))
        return min(max_neurons, max(1, size))

    shown_counts: list[int] = []
    for i, sz in enumerate(layer_sizes):
        if i == 0:
            shown_counts.append(display_count(sz, kind="input"))
        elif i == len(layer_sizes) - 1:
            shown_counts.append(display_count(sz, kind="output"))
        else:
            shown_counts.append(display_count(sz, kind="hidden"))

    n_layers = len(layer_sizes)
    fig_w = max(9.0, 1.8 * n_layers)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    ax.set_axis_off()

    xs = [i * 1.6 for i in range(n_layers)]
    r = 0.08

    def y_positions(count: int) -> list[float]:
        if count == 1:
            return [0.0]
        span = 1.2
        top = span / 2
        step = span / (count - 1)
        return [top - k * step for k in range(count)]

    node_pos: list[list[tuple[float, float]]] = []
    for li in range(n_layers):
        cnt = shown_counts[li]
        ys = y_positions(cnt)
        node_pos.append([(xs[li], y) for y in ys])

    def draw_neuron(x: float, y: float, *, alpha: float = 1.0):
        ax.add_patch(plt.Circle((x, y), r, fill=False, linewidth=1.0, alpha=alpha))

    def draw_edge(p1: tuple[float, float], p2: tuple[float, float], *, alpha: float = 0.35):
        (x1, y1), (x2, y2) = p1, p2
        ax.annotate(
            "",
            xy=(x2 - r * 1.2, y2),
            xytext=(x1 + r * 1.2, y1),
            arrowprops=dict(arrowstyle="-", linewidth=0.6, alpha=alpha),
        )

    def connect_layers(src: int, dst: int, *, alpha: float):
        for p1 in node_pos[src]:
            for p2 in node_pos[dst]:
                draw_edge(p1, p2, alpha=alpha)

    if model_type_u == "CFNN":
        for i in range(0, n_layers - 1):
            connect_layers(i, i + 1, alpha=0.25)
        for i in range(0, n_layers - 2):
            for j in range(i + 2, n_layers):
                connect_layers(i, j, alpha=0.10)
    else:
        for i in range(n_layers - 1):
            connect_layers(i, i + 1, alpha=0.25)

    for li in range(n_layers):
        for (x, y) in node_pos[li]:
            draw_neuron(x, y)

        real_sz = layer_sizes[li]
        shown = shown_counts[li]
        if real_sz > shown:
            ax.text(xs[li], -0.85, "…", ha="center", va="center", fontsize=14)

        if li == 0:
            ax.text(xs[li], 0.95, f"Вход\n{input_dim}", ha="center", va="center", fontsize=9)
        elif li == n_layers - 1:
            ax.text(xs[li], 0.95, f"Выход\n{output_dim}", ha="center", va="center", fontsize=9)
        else:
            sp = layers[li - 1]
            ax.text(
                xs[li],
                0.95,
                f"Слой {li}\nDense {sp.units}\n{sp.activation}, drop={sp.dropout}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Граф нейросети: " + model_type_u)
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
