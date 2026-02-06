from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class BackpropStepInfo:
    batch: int
    loss: float
    grad_norms: dict[str, float]


@dataclass
class TrainResult:
    history: dict[str, list[float]]
    backprop_steps: list[BackpropStepInfo]


def make_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    name_l = name.lower()
    if name_l == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name_l == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
    if name_l == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name_l == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def make_loss(name: str):
    name_l = name.lower()
    if name_l in {"mse", "mean_squared_error"}:
        return tf.keras.losses.MeanSquaredError()
    if name_l in {"mae", "mean_absolute_error"}:
        return tf.keras.losses.MeanAbsoluteError()
    if name_l == "huber":
        return tf.keras.losses.Huber()
    raise ValueError(f"Unknown loss: {name}")


def train_with_backprop_demo(
    model: tf.keras.Model,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
    epochs: int,
    batch_size: int,
    demo_batches: int = 3,
    seed: int = 42,
) -> TrainResult:
    rng = np.random.default_rng(seed)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    backprop_steps: list[BackpropStepInfo] = []

    n = len(X_train)
    steps_per_epoch = int(np.ceil(n / batch_size))

    for epoch in range(epochs):
        order = np.arange(n)
        rng.shuffle(order)
        X_train = X_train[order]
        y_train = y_train[order]

        epoch_losses: list[float] = []

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(n, (step + 1) * batch_size)
            xb = tf.convert_to_tensor(X_train[start:end])
            yb = tf.convert_to_tensor(y_train[start:end])

            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss = loss_fn(yb, pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            loss_val = float(loss.numpy())
            epoch_losses.append(loss_val)

            if epoch == 0 and step < demo_batches:
                grad_norms: dict[str, float] = {}
                for var, grad in zip(model.trainable_variables, grads):
                    if grad is None:
                        continue
                    grad_norms[var.name] = float(tf.linalg.norm(grad).numpy())
                backprop_steps.append(
                    BackpropStepInfo(
                        batch=step,
                        loss=loss_val,
                        grad_norms=grad_norms,
                    )
                )

        train_loss = float(np.mean(epoch_losses))

        val_pred = model(tf.convert_to_tensor(X_val), training=False)
        val_loss = float(loss_fn(tf.convert_to_tensor(y_val), val_pred).numpy())

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return TrainResult(history=history, backprop_steps=backprop_steps)
