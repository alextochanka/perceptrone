"""
Модуль нейронных сетей для классификации химических реакций
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, Any, List, Optional, Union
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """Класс нейронной сети с расширенной функциональностью"""

    def __init__(self, model_type: str = 'perceptron'):
        """
        Инициализация модели
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'timestamps': []
        }
        self.is_trained = False
        self.training_time = None
        self.n_features_ = 20  # Фиксированное количество признаков
        self.n_classes_ = 2    # Только 2 класса для 2 типов реакций

        logger.info(f"Инициализирована модель типа: {model_type}")

    def create_model(self, hidden_layer_sizes: tuple = None,
                    activation: str = 'relu',
                    solver: str = 'adam',
                    max_iter: int = 3000,
                    learning_rate_init: float = 0.001,
                    alpha: float = 0.0001,
                    batch_size: str = 'auto',
                    early_stopping: bool = True,
                    validation_fraction: float = 0.1,
                    n_iter_no_change: int = 10,
                    verbose: bool = False) -> None:
        """
        Создание модели без случайного семени
        """
        if hidden_layer_sizes is None:
            if self.model_type == 'perceptron':
                hidden_layer_sizes = ()
            else:
                hidden_layer_sizes = (128, 64)

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            batch_size=batch_size,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=verbose
        )

        logger.info(f"Создана модель типа: {self.model_type}")

    def generate_training_data(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация реалистичных данных для обучения без фиксированного seed"""
        n_features = 20
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Случайно выбираем тип реакции (без seed)
            reaction_type = np.random.randint(0, 2)  # 0 или 1

            if reaction_type == 0:  # Последовательная реакция A→B→C→D
                # Признаки для последовательной реакции
                X[i, 0] = np.random.uniform(0.8, 1.2)
                X[i, 1] = np.random.uniform(0.0, 0.2)
                X[i, 2] = np.random.uniform(0.0, 0.1)
                X[i, 3] = 0.0

                # Скорости уменьшения
                X[i, 4] = np.random.uniform(0.1, 0.3)
                X[i, 5] = np.random.uniform(0.05, 0.15)
                X[i, 6] = np.random.uniform(0.02, 0.08)
                X[i, 7] = np.random.uniform(0.01, 0.05)

                # Отношения концентраций
                X[i, 8] = X[i, 5] / (X[i, 4] + 1e-10)
                X[i, 9] = X[i, 6] / (X[i, 5] + 1e-10)
                X[i, 10] = X[i, 7] / (X[i, 6] + 1e-10)

                # Максимальные концентрации
                X[i, 11] = np.random.uniform(0.3, 0.5)
                X[i, 12] = np.random.uniform(0.2, 0.4)
                X[i, 13] = np.random.uniform(0.4, 0.6)

            else:  # Разветвленная реакция A→B→D и A→C→D
                # Признаки для разветвленной реакции
                X[i, 0] = np.random.uniform(0.8, 1.2)
                X[i, 1] = np.random.uniform(0.0, 0.1)
                X[i, 2] = np.random.uniform(0.0, 0.1)
                X[i, 3] = 0.0

                # Скорости уменьшения
                X[i, 4] = np.random.uniform(0.2, 0.4)
                X[i, 5] = np.random.uniform(0.08, 0.15)
                X[i, 6] = np.random.uniform(0.08, 0.15)
                X[i, 7] = np.random.uniform(0.1, 0.2)

                # Отношения концентраций
                X[i, 8] = X[i, 5] / (X[i, 4] + 1e-10)
                X[i, 9] = X[i, 6] / (X[i, 4] + 1e-10)
                X[i, 10] = (X[i, 5] + X[i, 6]) / (X[i, 7] + 1e-10)

                # Максимальные концентрации
                X[i, 11] = np.random.uniform(0.2, 0.3)
                X[i, 12] = np.random.uniform(0.2, 0.3)
                X[i, 13] = np.random.uniform(0.5, 0.8)

            # Добавляем случайный шум
            noise = np.random.normal(0, 0.05, 6)
            X[i, 14:20] = noise

            y[i] = reaction_type

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Обучение модели с отслеживанием прогресса
        """
        import time
        start_time = time.time()

        try:
            # Если не переданы данные, генерируем их
            if X_train is None or len(X_train) == 0:
                X_train, y_train = self.generate_training_data(1000)
                if X_val is None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=0.2
                    )

            # Сохраняем количество признаков
            if X_train is not None and len(X_train.shape) > 1:
                self.n_features_ = X_train.shape[1]

            # Нормализация данных
            logger.info(f"Нормализация {len(X_train)} обучающих образцов...")
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Обучение модели
            logger.info(f"Начало обучения модели {self.model_type}...")
            self.model.fit(X_train_scaled, y_train)

            # Сохранение истории обучения
            if hasattr(self.model, 'loss_curve_'):
                self.training_history['loss'] = self.model.loss_curve_

            # Расчет метрик на обучающей выборке
            y_train_pred = self.model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            self.training_history['accuracy'].append(train_accuracy)

            # Расчет на валидационной выборке
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_pred = self.model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                self.training_history['val_accuracy'].append(val_accuracy)

                logger.info(f"Точность на валидации: {val_accuracy:.4f}")

            # Генерация имен признаков
            self.feature_names = [f'feature_{i}' for i in range(self.n_features_)]

            # Расчет времени обучения
            self.training_time = time.time() - start_time
            self.is_trained = True

            logger.info(f"Модель успешно обучена за {self.training_time:.2f} сек")
            logger.info(f"Точность на обучении: {train_accuracy:.4f}")
            logger.info(f"Точность на валидации: {val_accuracy:.4f}")
            logger.info(f"Количество признаков: {self.n_features_}")

            return {
                'status': 'success',
                'training_time': self.training_time,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'n_samples': len(X_train),
                'n_features': self.n_features_,
                'n_iterations': self.model.n_iter_
            }

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        # Проверяем количество признаков
        if X.shape[1] != self.n_features_:
            logger.warning(f"Ожидалось {self.n_features_} признаков, получено {X.shape[1]}.")

            if X.shape[1] > self.n_features_:
                X = X[:, :self.n_features_]
            else:
                padding = np.zeros((X.shape[0], self.n_features_ - X.shape[1]))
                X = np.hstack([X, padding])

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей классов"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        # Проверяем количество признаков
        if X.shape[1] != self.n_features_:
            logger.warning(f"Ожидалось {self.n_features_} признаков, получено {X.shape[1]}.")

            if X.shape[1] > self.n_features_:
                X = X[:, :self.n_features_]
            else:
                padding = np.zeros((X.shape[0], self.n_features_ - X.shape[1]))
                X = np.hstack([X, padding])

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Оценка модели"""
        if not self.is_trained:
            return {
                'status': 'error',
                'message': 'Модель не обучена'
            }

        try:
            # Проверяем количество признаков
            if X_test.shape[1] != self.n_features_:
                logger.warning(f"Ожидалось {self.n_features_} признаков, получено {X_test.shape[1]}.")

                if X_test.shape[1] > self.n_features_:
                    X_test = X_test[:, :self.n_features_]
                else:
                    padding = np.zeros((X_test.shape[0], self.n_features_ - X_test.shape[1]))
                    X_test = np.hstack([X_test, padding])

            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)

            # Основные метрики
            accuracy = accuracy_score(y_test, y_pred)

            # Детальный отчет
            report = classification_report(y_test, y_pred, output_dict=True)

            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)

            # Дополнительные метрики
            try:
                if len(np.unique(y_test)) > 1:
                    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else:
                    auc_score = 0.0
            except:
                auc_score = 0.0

            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Точность модели: {accuracy:.4f}")
            logger.info(f"AUC-ROC: {auc_score:.4f}, F1-score: {f1:.4f}")

            return {
                'status': 'success',
                'accuracy': accuracy,
                'auc_roc': auc_score,
                'f1_score': f1,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'training_history': self.training_history,
                'n_test_samples': len(y_test),
                'n_features': self.n_features_
            }

        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if self.model is None:
            return {}

        try:
            info = {
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'training_time': self.training_time,
                'n_features': self.n_features_,
                'n_classes': self.n_classes_,
                'n_layers': len(self.model.coefs_) if self.model.coefs_ else 0,
                'activation': self.model.activation,
                'solver': self.model.solver,
                'max_iter': self.model.max_iter,
                'n_iter': self.model.n_iter_,
                'loss': getattr(self.model, 'loss_', None),
                'best_loss': getattr(self.model, 'best_loss_', None),
                'training_history': self.training_history
            }

            if self.model_type == 'mlp':
                info['hidden_layer_sizes'] = self.model.hidden_layer_sizes

            return info

        except Exception as e:
            logger.error(f"Ошибка получения информации о модели: {e}")
            return {}

    def save_model(self, filepath: Union[str, Path]) -> bool:
        """Сохранение модели"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'is_trained': self.is_trained,
                'training_time': self.training_time,
                'n_features_': self.n_features_,
                'save_timestamp': datetime.now().isoformat()
            }

            joblib.dump(save_data, filepath)
            logger.info(f"Модель сохранена в {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            return False

    def load_model(self, filepath: Union[str, Path]) -> bool:
        """Загрузка модели"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"Файл модели не существует: {filepath}")
                return False

            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.feature_names = data.get('feature_names')
            self.training_history = data.get('training_history', {})
            self.is_trained = data.get('is_trained', False)
            self.training_time = data.get('training_time')
            self.n_features_ = data.get('n_features_', 20)

            logger.info(f"Модель загружена из {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False