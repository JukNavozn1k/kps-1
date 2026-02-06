
import pandas as pd
import streamlit as st
import numpy as np

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
    if "n_layers" not in st.session_state:
        st.session_state.n_layers = len(st.session_state.layers)
    if "experiments" not in st.session_state:
        st.session_state.experiments = []
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "trained_feature_names" not in st.session_state:
        st.session_state.trained_feature_names = None
    if "trained_target" not in st.session_state:
        st.session_state.trained_target = None
    if "raw_df" not in st.session_state:
        try:
            st.session_state.raw_df = pd.read_csv("ftball.csv")
        except Exception:
            st.session_state.raw_df = None


def _sync_layers_count(n_layers: int):
    n_layers = int(n_layers)
    if n_layers < 1:
        n_layers = 1

    layers = list(st.session_state.layers)
    if len(layers) < n_layers:
        while len(layers) < n_layers:
            layers.append({"units": 16, "activation": "relu", "dropout": 0.0})
    elif len(layers) > n_layers:
        layers = layers[:n_layers]

    st.session_state.layers = layers
    st.session_state.n_layers = n_layers


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
    model_type = st.selectbox("Тип сети", ["FNN", "CFNN"], index=0, format_func=lambda x: "FNN (прямого распространения)" if x == "FNN" else "CFNN (каскадная FNN)")

    st.subheader("Слои")
    n_layers = st.number_input(
        "Количество слоёв",
        min_value=1,
        max_value=50,
        value=int(st.session_state.n_layers),
        step=1,
    )
    _sync_layers_count(int(n_layers))

    for i, layer in enumerate(st.session_state.layers):
        with st.expander(f"Слой {i+1}", expanded=True):
            layer["units"] = st.number_input(
                "Нейронов (units)",
                min_value=1,
                max_value=2048,
                value=int(layer["units"]),
                key=f"units_{i}",
            )
            layer["activation"] = st.selectbox(
                "Функция активации",
                ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"],
                index=["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"].index(str(layer["activation"])),
                key=f"act_{i}",
            )
            layer["dropout"] = st.slider(
                "Dropout",
                0.0,
                0.8,
                float(layer["dropout"]),
                0.05,
                key=f"drop_{i}",
            )

    cfnn_include_input_to_hidden = True
    cfnn_include_prev_hidden_to_hidden = True
    cfnn_include_input_to_output = True
    cfnn_include_hidden_to_output = True
    if model_type == "CFNN":
        st.subheader("Связность CFNN")
        cfnn_include_input_to_hidden = st.checkbox("Вход → каждый скрытый слой", value=True)
        cfnn_include_prev_hidden_to_hidden = st.checkbox("Предыдущие скрытые → следующий", value=True)
        cfnn_include_input_to_output = st.checkbox("Вход → выход", value=True)
        cfnn_include_hidden_to_output = st.checkbox("Скрытые → выход", value=True)

    st.divider()
    st.header("Обучение")
    optimizer_name = st.selectbox("Оптимизатор", ["Adam", "SGD", "RMSprop", "Adagrad"], index=0)
    learning_rate = st.number_input("Скорость обучения (learning rate)", min_value=1e-6, max_value=1.0, value=5e-4, format="%.6f")
    loss_name = st.selectbox("Функция ошибки", ["mse", "mae", "huber"], index=0, format_func=lambda x: {"mse": "MSE", "mae": "MAE", "huber": "Huber"}[x])
    epochs = st.number_input("Эпохи", min_value=1, max_value=500, value=100)
    batch_size = st.number_input("Размер батча", min_value=4, max_value=4096, value=32, step=4)
    demo_batches = st.number_input("Демо backprop (батчей в 1-й эпохе)", min_value=1, max_value=200, value=15)

    st.subheader("Регуляризация")
    l2_strength = st.slider("L2 регуляризация", 0.0, 0.1, 0.01, 0.001, help="Штраф за большие веса")
    l1_strength = st.slider("L1 регуляризация", 0.0, 0.1, 0.0, 0.001, help="Обнуление незначительных весов")
    early_stopping_patience = st.number_input("Early stopping (эпохи)", min_value=5, max_value=100, value=15, help="Останов если валидация не улучшается")

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

tab_train, tab_data, tab_pred = st.tabs(["Обучение", "Датасет", "Предикт"])


with tab_train:
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.subheader("Датасет")
        m1, m2, m3 = st.columns(3)
        m1.metric("Обучение (строк)", int(ds.X_train.shape[0]))
        m2.metric("Валидация (строк)", int(ds.X_val.shape[0]))
        m3.metric("Признаков", int(ds.X_train.shape[1]))

        st.caption("Целевая переменная")
        st.write(target)

        st.caption("Первые признаки (после предобработки)")
        st.dataframe(pd.DataFrame({"Признак": ds.feature_names[:25]}), width="stretch", height=320)

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
            model = build_fnn(ds.X_train.shape[1], specs, l1_strength=float(l1_strength), l2_strength=float(l2_strength))
        else:
            model = build_cfnn(
                ds.X_train.shape[1],
                specs,
                include_input_to_hidden=bool(cfnn_include_input_to_hidden),
                include_prev_hidden_to_hidden=bool(cfnn_include_prev_hidden_to_hidden),
                include_input_to_output=bool(cfnn_include_input_to_output),
                include_hidden_to_output=bool(cfnn_include_hidden_to_output),
                l1_strength=float(l1_strength),
                l2_strength=float(l2_strength),
            )

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
                early_stopping_patience=int(early_stopping_patience),
                seed=42,
            )

        st.success("Готово")

        st.session_state.trained_model = model
        st.session_state.trained_feature_names = list(ds.feature_names)
        st.session_state.trained_target = target

        y_train_pred = model.predict(ds.X_train, verbose=0).reshape(-1)
        y_val_pred = model.predict(ds.X_val, verbose=0).reshape(-1)
        y_train_true = ds.y_train.reshape(-1)
        y_val_true = ds.y_val.reshape(-1)

        def _rmse(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

        def _mae(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(np.mean(np.abs(y_t - y_p)))

        def _r2(y_t: np.ndarray, y_p: np.ndarray) -> float:
            ss_res = float(np.sum((y_t - y_p) ** 2))
            ss_tot = float(np.sum((y_t - float(np.mean(y_t))) ** 2))
            if ss_tot == 0:
                return 0.0
            return 1.0 - ss_res / ss_tot

        st.subheader("Метрики качества")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² (валидация)", f"{_r2(y_val_true, y_val_pred):.4f}")
        m2.metric("MAE (валидация)", f"{_mae(y_val_true, y_val_pred):.4f}")
        m3.metric("RMSE (валидация)", f"{_rmse(y_val_true, y_val_pred):.4f}")

        st.subheader("Кривые обучения (функция ошибки)")
        st.pyplot(plot_training_curves(result.history), clear_figure=True)

        st.subheader("Демо backprop: нормы градиентов по параметрам (несколько первых батчей 1-й эпохи)")
        steps_grad_dicts = [s.grad_norms for s in result.backprop_steps]
        st.pyplot(plot_backprop_gradients(steps_grad_dicts), clear_figure=True)

        if result.backprop_steps:
            st.subheader("Демо backprop: подробные значения")
            rows = []
            for s in result.backprop_steps:
                row = {"batch": s.batch, "loss": s.loss}
                for k, v in list(s.grad_norms.items())[:20]:
                    row[k] = v
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch")

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


with tab_data:
    st.subheader("Описание датасета")
    st.markdown(
        """
Датасет `ftball.csv` содержит информацию о футбольных матчах и букмекерские коэффициенты.

Что прогнозируем (регрессия):
- `home_goals` — голы хозяев
- `away_goals` — голы гостей
- `total_goals` — сумма голов
- `goal_diff` — разница голов (home - away)

Признаки строятся на основе:
- коэффициентов `odds` (1, X, 2, 1X, X2, 12)
- даты начала матча
- категориальных полей (маркет, лига/страна и т.д.) через one-hot
"""
    )

    raw_df = st.session_state.raw_df
    if raw_df is None:
        st.error("Не удалось прочитать ftball.csv")
    else:
        st.caption("Первые строки")
        st.dataframe(raw_df.head(200), width="stretch")

    st.divider()
    st.subheader("Предобработанный датасет (X и y)")
    split = st.selectbox("Выбор части", ["train", "val"], index=0)
    n_rows = st.number_input("Строк для просмотра", min_value=5, max_value=500, value=100, step=5)

    if split == "train":
        X_view = ds.X_train
        y_view = ds.y_train
    else:
        X_view = ds.X_val
        y_view = ds.y_val

    with st.expander("Основные характеристики (mean/std/min/max)", expanded=True):
        y_vec = y_view.reshape(-1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("y mean", f"{float(np.mean(y_vec)):.4f}")
        m2.metric("y std", f"{float(np.std(y_vec)):.4f}")
        m3.metric("y min", f"{float(np.min(y_vec)):.4f}")
        m4.metric("y max", f"{float(np.max(y_vec)):.4f}")

        n_feat = st.number_input(
            "Признаков в таблице статистик",
            min_value=5,
            max_value=int(len(ds.feature_names)),
            value=min(30, int(len(ds.feature_names))),
            step=5,
        )

        X_stats = pd.DataFrame(X_view, columns=ds.feature_names).describe().T[["mean", "std", "min", "max"]]
        st.dataframe(X_stats.head(int(n_feat)), width="stretch")

    X_df = pd.DataFrame(X_view[: int(n_rows)], columns=ds.feature_names)
    y_df = pd.DataFrame({"y": y_view[: int(n_rows)].reshape(-1)})
    st.caption("X (после предобработки; если включена стандартизация X — здесь уже стандартизовано)")
    st.dataframe(X_df, width="stretch")
    st.caption("y")
    st.dataframe(y_df, width="stretch")


with tab_pred:
    st.subheader("Предикт")
    if st.session_state.trained_model is None or st.session_state.trained_feature_names is None:
        st.info("Сначала обучи модель во вкладке 'Обучение'.")
    else:
        model = st.session_state.trained_model
        feature_names = list(st.session_state.trained_feature_names)

        raw_df = st.session_state.raw_df
        def _choices(col: str) -> list[str]:
            if raw_df is None or col not in raw_df.columns:
                return ["<NA>"]
            vals = raw_df[col].astype("string")
            uniq = sorted(set(vals.fillna("<NA>").tolist()))
            if "<NA>" not in uniq:
                uniq = ["<NA>"] + uniq
            return uniq[:500]

        with st.form("predict_form"):
            st.caption("Введи параметры матча (упрощённо).")
            c1, c2, c3 = st.columns(3)
            with c1:
                odds_1 = st.number_input("odds 1", min_value=0.0, value=2.0)
                odds_x = st.number_input("odds X", min_value=0.0, value=3.2)
                odds_2 = st.number_input("odds 2", min_value=0.0, value=3.0)
            with c2:
                odds_1x = st.number_input("odds 1X", min_value=0.0, value=1.3)
                odds_x2 = st.number_input("odds X2", min_value=0.0, value=1.6)
                odds_12 = st.number_input("odds 12", min_value=0.0, value=1.3)
            with c3:
                start_date = st.date_input("Дата матча")
                is_expired = st.checkbox("is_expired", value=True)

            prediction = st.selectbox("prediction", _choices("prediction"))
            market = st.selectbox("market", _choices("market"))
            competition_name = st.selectbox("competition_name", _choices("competition_name"))
            competition_cluster = st.selectbox("competition_cluster", _choices("competition_cluster"))
            federation = st.selectbox("federation", _choices("federation"))

            submitted = st.form_submit_button("Посчитать предикт")

        if submitted:
            row = {name: 0.0 for name in feature_names}

            num_map = {
                "odds_1": float(odds_1),
                "odds_X": float(odds_x),
                "odds_2": float(odds_2),
                "odds_1X": float(odds_1x),
                "odds_X2": float(odds_x2),
                "odds_12": float(odds_12),
                "start_year": float(start_date.year),
                "start_month": float(start_date.month),
                "start_day": float(start_date.day),
                "is_expired": float(1 if is_expired else 0),
            }
            for k, v in num_map.items():
                if k in row:
                    row[k] = v

            cat_map = {
                "prediction": prediction,
                "market": market,
                "competition_name": competition_name,
                "competition_cluster": competition_cluster,
                "federation": federation,
            }
            for col, val in cat_map.items():
                if val is None:
                    val = "<NA>"
                key = f"{col}_{val}"
                if key in row:
                    row[key] = 1.0
                else:
                    key_na = f"{col}_<NA>"
                    if key_na in row:
                        row[key_na] = 1.0

            X = pd.DataFrame([row], columns=feature_names).to_numpy(dtype="float32")
            y_pred = model.predict(X, verbose=0)
            val = float(y_pred.reshape(-1)[0])
            st.success(f"Предсказание для '{st.session_state.trained_target}': {val:.4f}")
