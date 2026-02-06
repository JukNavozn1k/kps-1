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
    # Build per-layer info
    layer_units = [input_dim] + [sp.units for sp in layers] + [output_dim]
    layer_infos = ["вход"] + [f"H{i+1}" for i in range(len(layers))] + ["выход"]

    n_layers = len(layer_units)

    # Display limits for very wide/tall networks
    max_draw_neurons = 12
    max_neurons = min(max(layer_units), max_draw_neurons)

    fig_w = max(8.0, 1.6 * n_layers)
    fig_h = max(3.6, 0.7 * max_neurons)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # horizontal positions
    x_gap = 2.0
    xs = [i * x_gap for i in range(n_layers)]

    # helper to compute vertical positions for neurons in a layer
    def neuron_ys(count: int, vspacing: float = 1.0):
        if count == 1:
            return [0.0]
        mid = (count - 1) / 2.0
        return [(i - mid) * vspacing for i in range(count)]

    # determine vspacing so large layers fit
    vspacing = 1.0

    # store neuron coordinates per layer
    layer_positions: list[list[tuple[float, float]]] = []

    # draw neurons
    for i, units in enumerate(layer_units):
        draw_count = units if units <= max_draw_neurons else max_draw_neurons
        ys = neuron_ys(draw_count, vspacing)
        pts: list[tuple[float, float]] = []
        for y in ys:
            circ = plt.Circle((xs[i], y), 0.18, facecolor="white", edgecolor="black", linewidth=0.9)
            ax.add_patch(circ)
            pts.append((xs[i], y))
        # if compressed, annotate actual count
        if units > max_draw_neurons:
            ax.text(xs[i], ys[-1] - 0.6, f"... (+{units - max_draw_neurons})", ha="center", va="center", fontsize=8)
        # layer label and small body info
        title = layer_infos[i]
        body = (
            f"{units}"
            + (f"\n{layers[i-1].activation}\n(drop={layers[i-1].dropout})" if 0 < i < n_layers - 1 else "")
        )
        ax.text(xs[i], max(ys) + 0.6, title, ha="center", va="center", fontsize=11)
        ax.text(xs[i], min(ys) - 0.8, body, ha="center", va="center", fontsize=8)
        layer_positions.append(pts)

    # draw full connections between adjacent layers
    for i in range(n_layers - 1):
        pos_a = layer_positions[i]
        pos_b = layer_positions[i + 1]
        for (x1, y1) in pos_a:
            for (x2, y2) in pos_b:
                ax.plot([x1 + 0.18, x2 - 0.18], [y1, y2], color="black", linewidth=0.6, alpha=0.35)

    # for CFNN, draw additional skip connections (lighter)
    if model_type_u == "CFNN":
        for i in range(n_layers - 2):
            for j in range(i + 2, n_layers):
                for (x1, y1) in layer_positions[i]:
                    for (x2, y2) in layer_positions[j]:
                        ax.plot([x1 + 0.18, x2 - 0.18], [y1, y2], color="black", linewidth=0.5, alpha=0.10)

    # adjust limits
    x_min, x_max = min(xs) - 1.0, max(xs) + 1.0
    # vertical extents from positions
    all_ys = [y for pts in layer_positions for (_x, y) in pts]
    if all_ys:
        y_min, y_max = min(all_ys) - 1.0, max(all_ys) + 1.0
    else:
        y_min, y_max = -1.0, 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

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
