
import pandas as pd
import streamlit as st

from kps1.data import load_ftball_dataset
from kps1.experiments import ExperimentRecord, record_to_dict
from kps1.models import DenseLayerSpec, build_cfnn, build_fnn
from kps1.training import make_loss, make_optimizer, train_with_backprop_demo
from kps1.viz import plot_backprop_gradients, plot_network_graph, plot_training_curves


def _init_state():
    if "layers" not in st.session_state:
        st.session_state.layers = [
            {"units": 32, "activation": "relu", "dropout": 0.0},
            {"units": 16, "activation": "relu", "dropout": 0.0},
        ]
    if "experiments" not in st.session_state:
        st.session_state.experiments = []


def _layers_to_specs(layers_ui: list[dict]) -> list[DenseLayerSpec]:
    specs: list[DenseLayerSpec] = []
    for l in layers_ui:
        specs.append(
            DenseLayerSpec(
                units=int(l["units"]),
                activation=str(l["activation"]),
                dropout=float(l["dropout"]),
            )
        )
    return specs


st.set_page_config(page_title="FNN/CFNN Football Regression", layout="wide")
_init_state()

st.title("Прогнозирование результата футбольных матчей (регрессия)")


with st.sidebar:
    st.header("Данные")
    target = st.selectbox(
        "Цель (регрессия)",
        ["home_goals", "away_goals", "total_goals", "goal_diff"],
        index=2,
    )
    val_fraction = st.slider("Доля валидации", 0.05, 0.5, 0.2, 0.05)
    standardize_X = st.checkbox("Стандартизовать X", value=True)
    standardize_y = st.checkbox("Стандартизовать y", value=False)

    st.divider()
    st.header("Модель")
    model_type = st.selectbox("Тип сети", ["FNN", "CFNN"], index=0)

    st.subheader("Слои")
    if st.button("+ Добавить слой"):
        st.session_state.layers.append({"units": 16, "activation": "relu", "dropout": 0.0})
    if st.button("- Удалить последний слой"):
        if len(st.session_state.layers) > 0:
            st.session_state.layers.pop()

    for i, layer in enumerate(st.session_state.layers):
        with st.expander(f"Layer {i+1}", expanded=True):
            layer["units"] = st.number_input(
                "units",
                min_value=1,
                max_value=2048,
                value=int(layer["units"]),
                key=f"units_{i}",
            )
            layer["activation"] = st.selectbox(
                "activation",
                ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"],
                index=["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"].index(str(layer["activation"])),
                key=f"act_{i}",
            )
            layer["dropout"] = st.slider(
                "dropout",
                0.0,
                0.8,
                float(layer["dropout"]),
                0.05,
                key=f"drop_{i}",
            )

    st.divider()
    st.header("Обучение")
    optimizer_name = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "Adagrad"], index=0)
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
    loss_name = st.selectbox("Loss", ["mse", "mae", "huber"], index=0)
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30)
    batch_size = st.number_input("Batch size", min_value=4, max_value=4096, value=64, step=4)
    demo_batches = st.number_input("Backprop demo batches (epoch 1)", min_value=1, max_value=20, value=3)

    train_clicked = st.button("Запустить обучение", type="primary")


ds = load_ftball_dataset(
    "ftball.csv",
    target=target,
    val_fraction=float(val_fraction),
    seed=42,
    standardize_X=bool(standardize_X),
    standardize_y=bool(standardize_y),
)

if ds.X_train.size == 0 or ds.y_train.size == 0:
    st.error("После фильтрации данных не осталось обучающих примеров. Проверь ftball.csv")
    st.stop()

col_a, col_b = st.columns([1.1, 0.9])

with col_a:
    st.subheader("Датасет")
    st.write(
        {
            "X_train": ds.X_train.shape,
            "y_train": ds.y_train.shape,
            "X_val": ds.X_val.shape,
            "y_val": ds.y_val.shape,
            "n_features": len(ds.feature_names),
        }
    )
    st.caption("Первые 10 признаков")
    st.write(ds.feature_names[:10])

with col_b:
    st.subheader("Граф сети")
    specs_preview = _layers_to_specs(st.session_state.layers)
    fig_graph = plot_network_graph(
        input_dim=ds.X_train.shape[1],
        layers=specs_preview,
        model_type=model_type,
        output_dim=1,
    )
    st.pyplot(fig_graph, clear_figure=True)


if train_clicked:
    specs = _layers_to_specs(st.session_state.layers)
    if model_type == "FNN":
        model = build_fnn(ds.X_train.shape[1], specs)
    else:
        model = build_cfnn(ds.X_train.shape[1], specs)

    opt = make_optimizer(optimizer_name, float(learning_rate))
    loss_fn = make_loss(loss_name)

    with st.spinner("Обучение..."):
        result = train_with_backprop_demo(
            model,
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_val=ds.X_val,
            y_val=ds.y_val,
            optimizer=opt,
            loss_fn=loss_fn,
            epochs=int(epochs),
            batch_size=int(batch_size),
            demo_batches=int(demo_batches),
            seed=42,
        )

    st.success("Готово")

    st.subheader("Кривые обучения")
    st.pyplot(plot_training_curves(result.history), clear_figure=True)

    st.subheader("Backprop demo: нормы градиентов по параметрам (первые батчи первого epoch)")
    steps_grad_dicts = [s.grad_norms for s in result.backprop_steps]
    st.pyplot(plot_backprop_gradients(steps_grad_dicts), clear_figure=True)

    if result.backprop_steps:
        st.subheader("Backprop demo: подробные значения")
        rows = []
        for s in result.backprop_steps:
            row = {"batch": s.batch, "loss": s.loss}
            for k, v in list(s.grad_norms.items())[:20]:
                row[k] = v
            rows.append(row)
        st.dataframe(pd.DataFrame(rows))

    final_train_loss = float(result.history["loss"][-1])
    final_val_loss = float(result.history["val_loss"][-1])

    layers_str = "; ".join([f"{sp.units}/{sp.activation}/drop={sp.dropout}" for sp in specs])
    st.session_state.experiments.append(
        record_to_dict(
            ExperimentRecord(
                model_type=model_type,
                target=target,
                layers=layers_str,
                optimizer=optimizer_name,
                learning_rate=float(learning_rate),
                loss=loss_name,
                batch_size=int(batch_size),
                epochs=int(epochs),
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
            )
        )
    )


st.subheader("Таблица экспериментов")
if st.session_state.experiments:
    st.dataframe(pd.DataFrame(st.session_state.experiments), width="stretch")
else:
    st.info("Пока нет экспериментов. Запусти обучение хотя бы один раз.")
