from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tensorflow as tf


ActivationName = Literal[
    "relu",
    "tanh",
    "sigmoid",
    "elu",
    "selu",
    "gelu",
    "linear",
]


@dataclass
class DenseLayerSpec:
    units: int
    activation: ActivationName
    dropout: float = 0.0


def _activation(act: ActivationName):
    if act == "gelu":
        return tf.keras.activations.gelu
    return act


def build_fnn(
    input_dim: int,
    layers: list[DenseLayerSpec],
    *,
    output_dim: int = 1,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="x")
    x = inputs

    for i, spec in enumerate(layers):
        x = tf.keras.layers.Dense(
            spec.units,
            activation=_activation(spec.activation),
            name=f"dense_{i+1}",
        )(x)
        if spec.dropout and spec.dropout > 0:
            x = tf.keras.layers.Dropout(spec.dropout, name=f"dropout_{i+1}")(x)

    outputs = tf.keras.layers.Dense(output_dim, activation="linear", name="y")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="FNN")


def build_cfnn(
    input_dim: int,
    layers: list[DenseLayerSpec],
    *,
    output_dim: int = 1,
    include_input_to_hidden: bool = True,
    include_prev_hidden_to_hidden: bool = True,
    include_input_to_output: bool = True,
    include_hidden_to_output: bool = True,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="x")

    hidden_outputs: list[tf.Tensor] = []
    for i, spec in enumerate(layers):
        to_concat: list[tf.Tensor] = []
        if include_input_to_hidden:
            to_concat.append(inputs)
        if include_prev_hidden_to_hidden and hidden_outputs:
            to_concat.extend(hidden_outputs)
        if not to_concat:
            to_concat.append(inputs)

        if len(to_concat) == 1:
            concat = to_concat[0]
        else:
            concat = tf.keras.layers.Concatenate(name=f"concat_{i+1}")(to_concat)

        h = tf.keras.layers.Dense(
            spec.units,
            activation=_activation(spec.activation),
            name=f"dense_{i+1}",
        )(concat)
        if spec.dropout and spec.dropout > 0:
            h = tf.keras.layers.Dropout(spec.dropout, name=f"dropout_{i+1}")(h)
        hidden_outputs.append(h)

    out_concat: list[tf.Tensor] = []
    if include_input_to_output:
        out_concat.append(inputs)
    if include_hidden_to_output and hidden_outputs:
        out_concat.extend(hidden_outputs)

    if not out_concat:
        out_inp = inputs
    elif len(out_concat) == 1:
        out_inp = out_concat[0]
    else:
        out_inp = tf.keras.layers.Concatenate(name="concat_out")(out_concat)

    outputs = tf.keras.layers.Dense(output_dim, activation="linear", name="y")(out_inp)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CFNN")
