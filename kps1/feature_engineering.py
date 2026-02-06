"""
Модуль для создания кастомных признаков через математические операции.
Поддерживает произведения, триг. функции, степени и их комбинации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd


OperationType = Literal[
    "product",      # x1 * x2
    "ratio",        # x1 / x2 (с защитой от нуля)
    "sum",          # x1 + x2
    "diff",         # x1 - x2
    "sin",          # sin(x)
    "cos",          # cos(x)
    "tan",          # tan(x)
    "exp",          # e^x
    "log",          # log(|x| + 1)
    "sqrt",         # sqrt(|x|)
    "square",       # x^2
    "cube",         # x^3
    "abs",          # |x|
]


@dataclass
class CustomFeature:
    """Определение кастомного признака."""
    name: str
    operation: OperationType
    feature1_idx: int | None = None  # Индекс первого признака
    feature2_idx: int | None = None  # Индекс второго признака (для бинарных операций)
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Применяет операцию к признакам.
        
        Args:
            X: матрица признаков shape (n_samples, n_features)
            
        Returns:
            Векторstyle признаков shape (n_samples,)
        """
        if self.operation in ["sin", "cos", "tan", "exp", "log", "sqrt", "square", "cube", "abs"]:
            return self._apply_unary(X)
        else:
            return self._apply_binary(X)
    
    def _apply_unary(self, X: np.ndarray) -> np.ndarray:
        """Унарные операции."""
        x = X[:, self.feature1_idx]
        
        if self.operation == "sin":
            return np.sin(x)
        elif self.operation == "cos":
            return np.cos(x)
        elif self.operation == "tan":
            return np.tan(x)
        elif self.operation == "exp":
            # Защита от overflow
            x_safe = np.clip(x, -100, 100)
            return np.exp(x_safe)
        elif self.operation == "log":
            # log(|x| + 1) для любых x
            return np.log(np.abs(x) + 1.0)
        elif self.operation == "sqrt":
            # sqrt(|x|)
            return np.sqrt(np.abs(x))
        elif self.operation == "square":
            return x ** 2
        elif self.operation == "cube":
            return x ** 3
        elif self.operation == "abs":
            return np.abs(x)
        else:
            raise ValueError(f"Неизвестная унарная операция: {self.operation}")
    
    def _apply_binary(self, X: np.ndarray) -> np.ndarray:
        """Бинарные операции."""
        x1 = X[:, self.feature1_idx]
        x2 = X[:, self.feature2_idx]
        
        if self.operation == "product":
            return x1 * x2
        elif self.operation == "ratio":
            # x1 / x2 с защитой от нуля
            denominator = np.where(np.abs(x2) < 1e-8, 1e-8, x2)
            return x1 / denominator
        elif self.operation == "sum":
            return x1 + x2
        elif self.operation == "diff":
            return x1 - x2
        else:
            raise ValueError(f"Неизвестная бинарная операция: {self.operation}")


def get_operation_info(op: OperationType) -> dict:
    """Возвращает информацию об операции."""
    info_map = {
        "product": {"name": "Произведение (x1 * x2)", "binary": True, "description": "Умножение двух признаков"},
        "ratio": {"name": "Отношение (x1 / x2)", "binary": True, "description": "Деление первого на второй (с защитой от нуля)"},
        "sum": {"name": "Сумма (x1 + x2)", "binary": True, "description": "Сложение двух признаков"},
        "diff": {"name": "Разность (x1 - x2)", "binary": True, "description": "Вычитание второго из первого"},
        "sin": {"name": "sin(x)", "binary": False, "description": "Синус признака"},
        "cos": {"name": "cos(x)", "binary": False, "description": "Косинус признака"},
        "tan": {"name": "tan(x)", "binary": False, "description": "Тангенс признака"},
        "exp": {"name": "exp(x)", "binary": False, "description": "Экспонента признака (с ограничением)"},
        "log": {"name": "log(|x|+1)", "binary": False, "description": "Логарифм модуля признака"},
        "sqrt": {"name": "sqrt(|x|)", "binary": False, "description": "Корень квадратный модуля признака"},
        "square": {"name": "x²", "binary": False, "description": "Квадрат признака"},
        "cube": {"name": "x³", "binary": False, "description": "Куб признака"},
        "abs": {"name": "|x|", "binary": False, "description": "Модуль признака"},
    }
    return info_map.get(op, {})


def apply_custom_features(
    X: np.ndarray,
    feature_names: list[str],
    custom_features: list[CustomFeature],
) -> tuple[np.ndarray, list[str]]:
    """
    Добавляет кастомные признаки к матрице X.
    
    Args:
        X: матрица признаков shape (n_samples, n_features)
        feature_names: список имён оригинальных признаков
        custom_features: список кастомных признаков для добавления
        
    Returns:
        Большая матрица X и обновлённый список имён признаков
    """
    if not custom_features:
        return X, feature_names
    
    new_features = []
    new_feature_names = []
    
    for cf in custom_features:
        try:
            feature_vector = cf.apply(X)
            # Защита от NaN и Inf
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            new_features.append(feature_vector)
            new_feature_names.append(cf.name)
        except Exception as e:
            print(f"Ошибка при применении признака '{cf.name}': {e}")
            continue
    
    if not new_features:
        return X, feature_names
    
    # Конкатенируем оригинальные с новыми признаками
    new_features_array = np.column_stack(new_features)
    X_extended = np.column_stack([X, new_features_array])
    
    extended_feature_names = list(feature_names) + new_feature_names
    
    return X_extended, extended_feature_names


def create_feature_from_expression(
    name: str,
    feature1_name: str,
    feature2_name: str | None,
    operation: OperationType,
    feature_names: list[str],
) -> CustomFeature | None:
    """
    Создаёт CustomFeature из названий признаков.
    
    Args:
        name: имя нового признака
        feature1_name: имя первого признака
        feature2_name: имя второго признака (опционально)
        operation: тип операции
        feature_names: полный список доступных признаков
        
    Returns:
        объект CustomFeature или None если ошибка
    """
    try:
        if feature1_name not in feature_names:
            print(f"Признак '{feature1_name}' не найден")
            return None
        
        idx1 = feature_names.index(feature1_name)
        idx2 = None
        
        # Проверяем информацию об операции
        op_info = get_operation_info(operation)
        if op_info.get("binary"):
            if feature2_name is None or feature2_name not in feature_names:
                print(f"Для операции '{operation}' требуется второй признак")
                return None
            idx2 = feature_names.index(feature2_name)
        
        return CustomFeature(
            name=name,
            operation=operation,
            feature1_idx=idx1,
            feature2_idx=idx2,
        )
    except Exception as e:
        print(f"Ошибка при создании признака: {e}")
        return None


def suggest_interesting_features(feature_names: list[str], max_count: int = 5) -> list[CustomFeature]:
    """
    Предлагает несколько интересных комбинаций признаков для экспериментов.
    
    Args:
        feature_names: список доступных признаков
        max_count: максимальное количество предложений
        
    Returns:
        Список CustomFeature
    """
    suggestions = []
    
    # Предлагаем квадраты для числовых признаков
    numeric_features = [f for f in feature_names if "odds" in f or f.startswith("start_")]
    for i, feat in enumerate(numeric_features[:2]):
        if i >= max_count:
            break
        suggestions.append(
            CustomFeature(
                name=f"{feat}_squared",
                operation="square",
                feature1_idx=feature_names.index(feat),
            )
        )
    
    # Предлагаем произведения для коэффициентов
    odds_features = [f for f in feature_names if "odds" in f]
    if len(odds_features) >= 2 and len(suggestions) < max_count:
        idx1 = feature_names.index(odds_features[0])
        idx2 = feature_names.index(odds_features[1])
        suggestions.append(
            CustomFeature(
                name=f"{odds_features[0]}_x_{odds_features[1]}",
                operation="product",
                feature1_idx=idx1,
                feature2_idx=idx2,
            )
        )
    
    # Предлагаем отношения для коэффициентов
    if len(odds_features) >= 2 and len(suggestions) < max_count:
        idx1 = feature_names.index(odds_features[0])
        idx2 = feature_names.index(odds_features[-1])
        if idx1 != idx2:
            suggestions.append(
                CustomFeature(
                    name=f"{odds_features[0]}_div_{odds_features[-1]}",
                    operation="ratio",
                    feature1_idx=idx1,
                    feature2_idx=idx2,
                )
            )
    
    return suggestions[:max_count]
