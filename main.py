
import pandas as pd
import streamlit as st
import numpy as np

from kps1.data import load_ftball_dataset, filter_dataset_features, apply_dataset_custom_features
from kps1.feature_engineering import CustomFeature, get_operation_info, create_feature_from_expression, suggest_interesting_features
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
    if "trained_label_encoders" not in st.session_state:
        st.session_state.trained_label_encoders = None
    if "raw_df" not in st.session_state:
        try:
            st.session_state.raw_df = pd.read_csv("ftball.csv")
        except Exception:
            st.session_state.raw_df = None
    if "selected_features_all" not in st.session_state:
        st.session_state.selected_features_all = []
    if "available_features" not in st.session_state:
        st.session_state.available_features = []
    if "custom_features" not in st.session_state:
        st.session_state.custom_features = []


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

# Обновляем список доступных признаков
if st.session_state.available_features != ds.feature_names:
    st.session_state.available_features = list(ds.feature_names)
    st.session_state.selected_features_all = list(ds.feature_names)

# Интерфейс выбора признаков
st.sidebar.divider()
st.sidebar.header("Признаки")
selected_features = st.sidebar.multiselect(
    "Выбери признаки для обучения",
    options=st.session_state.available_features,
    default=st.session_state.selected_features_all,
    help="Оставь пустым = используются все признаки",
)

# Обновляем session state
if selected_features:
    st.session_state.selected_features_all = selected_features
else:
    st.session_state.selected_features_all = list(st.session_state.available_features)

# Применяем фильтр к датасету
if selected_features:
    ds = filter_dataset_features(ds, selected_features)
else:
    ds = filter_dataset_features(ds, st.session_state.available_features)

# ===== Интерфейс кастомных признаков =====
st.sidebar.divider()
st.sidebar.header("🔧 Кастомные признаки")

with st.sidebar.expander("Создать новый признак", expanded=False):
    st.caption("Выбери операцию")
    
    col1, col2 = st.columns(2)
    with col1:
        operation = st.selectbox(
            "Операция",
            options=[
                "product", "ratio", "sum", "diff",
                "sin", "cos", "tan", "exp", "log", "sqrt",
                "square", "cube", "abs"
            ],
            format_func=lambda op: get_operation_info(op).get("name", op),
            key="custom_op_select",
        )
    
    with col2:
        custom_feature_name = st.text_input("Имя признака", value="", key="custom_name_input")
    
    # Получаем информацию об операции
    op_info = get_operation_info(operation)
    is_binary = op_info.get("binary", False)
    
    st.caption(f"📝 {op_info.get('description', '')}")
    
    # Выбор признаков
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox(
            "Признак 1",
            options=st.session_state.available_features,
            key="custom_feat1_select",
        )
    
    if is_binary:
        with col2:
            feature2 = st.selectbox(
                "Признак 2",
                options=st.session_state.available_features,
                key="custom_feat2_select",
            )
    else:
        feature2 = None
    
    # Кнопка добавления
    if st.button("✅ Добавить признак", key="add_custom_feature_btn"):
        if not custom_feature_name:
            st.error("Введи имя признака")
        else:
            cf = create_feature_from_expression(
                name=custom_feature_name,
                feature1_name=feature1,
                feature2_name=feature2,
                operation=operation,
                feature_names=st.session_state.available_features,
            )
            if cf:
                st.session_state.custom_features.append(cf)
                st.success(f"Добавлен признак: {custom_feature_name}")
                st.rerun()
            else:
                st.error("Ошибка при создании признака")

# Список созданных кастомных признаков
if st.session_state.custom_features:
    st.sidebar.subheader(f"Активные ({len(st.session_state.custom_features)})")
    for i, cf in enumerate(st.session_state.custom_features):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.caption(f"• {cf.name}")
        with col2:
            if st.button("❌", key=f"del_custom_{i}", help="Удалить"):
                st.session_state.custom_features.pop(i)
                st.rerun()

# Предложения интересных признаков
st.sidebar.subheader("💡 Идеи")
if st.sidebar.button("Предложить интересную комбинацию"):
    ideas = suggest_interesting_features(st.session_state.available_features, max_count=3)
    for idea in ideas:
        if idea not in st.session_state.custom_features:
            st.session_state.custom_features.append(idea)
    st.sidebar.success(f"Добавлено {len(ideas)} признаков!")
    st.rerun()

# Применяем кастомные признаки к датасету
if st.session_state.custom_features:
    ds = apply_dataset_custom_features(ds, st.session_state.custom_features)

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
            include_input_to_hidden=bool(cfnn_include_input_to_hidden),
            include_prev_hidden_to_hidden=bool(cfnn_include_prev_hidden_to_hidden),
            include_input_to_output=bool(cfnn_include_input_to_output),
            include_hidden_to_output=bool(cfnn_include_hidden_to_output),
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
        st.session_state.trained_label_encoders = ds.label_encoders if ds.label_encoders else {}

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
- категориальных полей (маркет, лига/страна и т.д.) через label encoding (целые числа)
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
            min_value=0,
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
        label_encoders = st.session_state.trained_label_encoders if st.session_state.trained_label_encoders else {}

        with st.form("predict_form"):
            st.caption("Введи параметры матча.")
            
            row = {}
            for feat_name in feature_names:
                # Численный признак
                if feat_name in ["odds_1", "odds_X", "odds_2", "odds_1X", "odds_X2", "odds_12", 
                                 "start_year", "start_month", "start_day", "is_expired"]:
                    labels_map = {
                        "odds_1": "odds 1 (коэффициент на победу хозяев)",
                        "odds_X": "odds X (коэффициент на ничью)",
                        "odds_2": "odds 2 (коэффициент на победу гостей)",
                        "odds_1X": "odds 1X",
                        "odds_X2": "odds X2",
                        "odds_12": "odds 12",
                        "start_year": "Год матча",
                        "start_month": "Месяц матча",
                        "start_day": "День матча",
                        "is_expired": "Матч завершён",
                    }
                    defaults_map = {
                        "odds_1": 1.699, "odds_X": 3.989, "odds_2": 4.377,
                        "odds_1X": 1.193, "odds_X2": 2.099, "odds_12": 1.216,
                        "start_year": 2024.0, "start_month": 1.0, "start_day": 1.0,
                        "is_expired": 0.0,
                    }
                    label = labels_map.get(feat_name, feat_name)
                    default = defaults_map.get(feat_name, 0.0)
                    row[feat_name] = st.number_input(label, min_value=0.0, value=default, key=f"input_{feat_name}")
                
                # Категориальный признак (label encoded)
                elif feat_name in label_encoders:
                    le = label_encoders[feat_name]
                    # Получаем список доступных категорий
                    available_cats = list(le.classes_)
                    selected_cat = st.selectbox(
                        f"{feat_name} (категория)",
                        available_cats,
                        key=f"select_{feat_name}"
                    )
                    # Кодируем выбор пользователя
                    encoded_val = int(le.transform([selected_cat])[0])
                    row[feat_name] = encoded_val

            submitted = st.form_submit_button("Посчитать предикт")

        if submitted:
            # Создаём полный vector (заполняем нулями отсутствующие признаки)
            full_row = {name: 0.0 for name in feature_names}
            full_row.update(row)
            
            X = pd.DataFrame([full_row], columns=feature_names).to_numpy(dtype="float32")
            y_pred = model.predict(X, verbose=0)
            val = float(y_pred.reshape(-1)[0])
            st.success(f"Предсказание для '{st.session_state.trained_target}': {val:.4f}")
