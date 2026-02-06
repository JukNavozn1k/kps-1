"""
Примеры использования модуля feature_engineering для создания кастомных признаков.

Эти примеры показывают, как использовать функциональность программно, 
без интерфейса Streamlit.
"""

import numpy as np
import pandas as pd
from kps1.feature_engineering import (
    CustomFeature,
    get_operation_info, 
    create_feature_from_expression,
    apply_custom_features,
    suggest_interesting_features,
)


def example_1_basic_product():
    """Пример 1: Произведение двух признаков"""
    print("\n" + "="*60)
    print("ПРИМЕР 1: Произведение двух признаков")
    print("="*60)
    
    # Создаём синтетические данные
    X = np.array([
        [2.0, 3.0, 1.0],  # Sample 1
        [4.0, 5.0, 2.0],  # Sample 2
        [6.0, 7.0, 3.0],  # Sample 3
    ])
    
    feature_names = ["odds_1", "odds_X", "odds_2"]
    
    print(f"\nИсходные признаки:")
    print(f"Имена: {feature_names}")
    print(f"Данные:\n{X}")
    
    # Создаём кастомный признак: произведение odds_1 и odds_X
    cf = CustomFeature(
        name="odds_1_times_odds_X",
        operation="product",
        feature1_idx=0,  # odds_1
        feature2_idx=1,  # odds_X
    )
    
    result = cf.apply(X)
    print(f"\nНовый признак '{cf.name}' = odds_1 × odds_X:")
    print(result)
    print(f"Объяснение: [2*3, 4*5, 6*7] = [6, 20, 42]")


def example_2_nonlinear_transformation():
    """Пример 2: Нелинейные преобразования"""
    print("\n" + "="*60)
    print("ПРИМЕР 2: Нелинейные преобразования")
    print("="*60)
    
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    
    feature_names = ["some_feature"]
    
    print(f"\nИсходный признак: {X.flatten()}")
    
    operations = ["square", "sqrt", "log"]
    
    for op in operations:
        cf = CustomFeature(
            name=f"feature_{op}",
            operation=op,
            feature1_idx=0,
        )
        result = cf.apply(X)
        op_info = get_operation_info(op)
        print(f"\n{op_info['name']}:")
        print(f"  {result.flatten()}")


def example_3_batch_features():
    """Пример 3: Применение нескольких кастомных признаков сразу"""
    print("\n" + "="*60)
    print("ПРИМЕР 3: Применение нескольких признаков сразу")
    print("="*60)
    
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    
    feature_names = ["A", "B", "C"]
    
    print(f"\nИсходные признаки (матрица 3x3):")
    df = pd.DataFrame(X, columns=feature_names)
    print(df)
    
    # Создаём несколько кастомных признаков
    custom_features = [
        CustomFeature("A_squared", "square", 0),
        CustomFeature("A_times_B", "product", 0, 1),
        CustomFeature("B_div_C", "ratio", 1, 2),
    ]
    
    # Применяем их к матрице X
    X_extended, names_extended = apply_custom_features(X, feature_names, custom_features)
    
    print(f"\nПосле добавления 3 кастомных признаков:")
    print(f"Количество признаков: {X_extended.shape[1]}")
    print(f"Имена признаков: {names_extended}")
    
    df_extended = pd.DataFrame(X_extended, columns=names_extended)
    print(df_extended)


def example_4_create_from_names():
    """Пример 4: Создание признака по названиям"""
    print("\n" + "="*60)
    print("ПРИМЕР 4: Создание признака по названиям признаков")
    print("="*60)
    
    feature_names = ["odds_1", "odds_X", "odds_2", "year", "month"]
    
    print(f"\nДоступные признаки: {feature_names}")
    
    # Создаём признак через функцию по названиям
    cf = create_feature_from_expression(
        name="ratio_1_to_X",
        feature1_name="odds_1",
        feature2_name="odds_X",
        operation="ratio",
        feature_names=feature_names,
    )
    
    if cf:
        print(f"\n✅ Успешно создан признак:")
        print(f"  Имя: {cf.name}")
        print(f"  Операция: {cf.operation}")
        print(f"  Признак 1 (индекс {cf.feature1_idx}): {feature_names[cf.feature1_idx]}")
        print(f"  Признак 2 (индекс {cf.feature2_idx}): {feature_names[cf.feature2_idx]}")


def example_5_operation_info():
    """Пример 5: Информация об операциях"""
    print("\n" + "="*60)
    print("ПРИМЕР 5: Справка по всем операциям")
    print("="*60)
    
    operations = [
        "product", "ratio", "sum", "diff",
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "square", "cube", "abs"
    ]
    
    print(f"\n{'Операция':<15} {'Тип':<10} {'Описание':<45}")
    print("-" * 70)
    
    for op in operations:
        info = get_operation_info(op)
        op_type = "Бинарная" if info.get("binary") else "Унарная"
        desc = info.get("description", "")
        print(f"{op:<15} {op_type:<10} {desc:<45}")


def example_6_suggestions():
    """Пример 6: Автоматическое предложение интересных признаков"""
    print("\n" + "="*60)
    print("ПРИМЕР 6: Автоматические рекомендации")
    print("="*60)
    
    feature_names = [
        "odds_1", "odds_X", "odds_2", 
        "odds_1X", "odds_X2", "odds_12",
        "start_year", "start_month", "start_day"
    ]
    
    print(f"\nДоступные признаки ({len(feature_names)}):")
    for fname in feature_names:
        print(f"  • {fname}")
    
    suggestions = suggest_interesting_features(feature_names, max_count=5)
    
    print(f"\n💡 Предложены {len(suggestions)} интересных признаков:")
    for i, cf in enumerate(suggestions, 1):
        if cf.feature2_idx is not None:
            f1_name = feature_names[cf.feature1_idx]
            f2_name = feature_names[cf.feature2_idx]
            print(f"  {i}. {cf.name} ({cf.operation}): {f1_name} и {f2_name}")
        else:
            f1_name = feature_names[cf.feature1_idx]
            print(f"  {i}. {cf.name} ({cf.operation}): {f1_name}")


def example_7_safety_features():
    """Пример 7: Защита от специальных случаев"""
    print("\n" + "="*60)
    print("ПРИМЕР 7: Безопасность обработки данных")
    print("="*60)
    
    # Данные с проблемами
    X = np.array([
        [5.0, 0.0],      # Деление на ноль
        [-1.0, 2.0],     # Отрицательное значение
        [1000.0, 50.0],  # Очень большое значение для exp
    ])
    
    print(f"\nПроблемные данные:")
    print(X)
    
    # Отношение (защита от деления на ноль)
    cf_ratio = CustomFeature("safe_ratio", "ratio", 0, 1)
    result_ratio = cf_ratio.apply(X)
    print(f"\nХараритет (х1/х2) с защитой от нуля:")
    print(result_ratio)
    
    # Логарифм отрицательного (защита через |x|)
    cf_log = CustomFeature("safe_log", "log", 0)
    result_log = cf_log.apply(X)
    print(f"\nЛогарифм (log(|x|+1)):")
    print(result_log)
    
    # Корень квадратный отрицательного (защита через |x|)
    cf_sqrt = CustomFeature("safe_sqrt", "sqrt", 0)
    result_sqrt = cf_sqrt.apply(X)
    print(f"\nКорень квадратный (sqrt(|x|)):")
    print(result_sqrt)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ МОДУЛЯ FEATURE ENGINEERING")
    print("="*60)
    
    example_1_basic_product()
    example_2_nonlinear_transformation()
    example_3_batch_features()
    example_4_create_from_names()
    example_5_operation_info()
    example_6_suggestions()
    example_7_safety_features()
    
    print("\n" + "="*60)
    print("✅ Все примеры завершены!")
    print("="*60 + "\n")
