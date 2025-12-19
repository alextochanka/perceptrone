"""
Модуль нейронных сетей для классификации химических реакций с учетом материального баланса
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
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """Класс нейронной сети с учетом материального баланса химических реакций"""

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

    def solve_sequential_reaction(self, A0: float, B0: float, C0: float, D0: float,
                                 k1: float, k2: float, k3: float,
                                 time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Численное решение системы ОДУ для последовательной реакции A→B→C→D
        с сохранением материального баланса
        """
        def ode_system(t, y):
            A, B, C, D = y
            dA_dt = -k1 * A
            dB_dt = k1 * A - k2 * B
            dC_dt = k2 * B - k3 * C
            dD_dt = k3 * C
            return [dA_dt, dB_dt, dC_dt, dD_dt]

        # Начальные условия
        y0 = [A0, B0, C0, D0]

        # Решаем систему ОДУ
        try:
            solution = solve_ivp(ode_system, [time_points[0], time_points[-1]], y0,
                               t_eval=time_points, method='RK45', rtol=1e-6, atol=1e-8)

            # Извлекаем результаты
            A = np.maximum(0, solution.y[0])  # Гарантируем неотрицательность
            B = np.maximum(0, solution.y[1])
            C = np.maximum(0, solution.y[2])
            D = np.maximum(0, solution.y[3])

            # Нормализуем для точного сохранения массы (A+B+C+D = 1)
            for i in range(len(time_points)):
                total = A[i] + B[i] + C[i] + D[i]
                if total > 0:
                    A[i] /= total
                    B[i] /= total
                    C[i] /= total
                    D[i] /= total

            return {'A': A, 'B': B, 'C': C, 'D': D}

        except Exception as e:
            logger.error(f"Ошибка решения ОДУ для последовательной реакции: {e}")
            # Возвращаем данные по умолчанию в случае ошибки
            n_points = len(time_points)
            A = np.maximum(0, A0 * np.exp(-k1 * time_points))
            B = np.maximum(0, B0 + (A0 - A) * 0.5)
            C = np.maximum(0, C0 + (A0 - A) * 0.3)
            D = np.maximum(0, D0 + (A0 - A) * 0.2)

            # Нормализуем
            for i in range(n_points):
                total = A[i] + B[i] + C[i] + D[i]
                if total > 0:
                    A[i], B[i], C[i], D[i] = A[i]/total, B[i]/total, C[i]/total, D[i]/total

            return {'A': A, 'B': B, 'C': C, 'D': D}

    def solve_parallel_reaction(self, A0: float, B0: float, C0: float, D0: float,
                               k1: float, k2: float, k3: float,
                               time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Численное решение системы ОДУ для параллельной реакции A→B и A→C→D
        с сохранением материального баланса
        """
        def ode_system(t, y):
            A, B, C, D = y
            dA_dt = -k1 * A - k2 * A
            dB_dt = k1 * A
            dC_dt = k2 * A - k3 * C
            dD_dt = k3 * C
            return [dA_dt, dB_dt, dC_dt, dD_dt]

        # Начальные условия
        y0 = [A0, B0, C0, D0]

        # Решаем систему ОДУ
        try:
            solution = solve_ivp(ode_system, [time_points[0], time_points[-1]], y0,
                               t_eval=time_points, method='RK45', rtol=1e-6, atol=1e-8)

            # Извлекаем результаты
            A = np.maximum(0, solution.y[0])
            B = np.maximum(0, solution.y[1])
            C = np.maximum(0, solution.y[2])
            D = np.maximum(0, solution.y[3])

            # Нормализуем для точного сохранения массы
            for i in range(len(time_points)):
                total = A[i] + B[i] + C[i] + D[i]
                if total > 0:
                    A[i] /= total
                    B[i] /= total
                    C[i] /= total
                    D[i] /= total

            return {'A': A, 'B': B, 'C': C, 'D': D}

        except Exception as e:
            logger.error(f"Ошибка решения ОДУ для параллельной реакции: {e}")
            # Возвращаем данные по умолчанию в случае ошибки
            n_points = len(time_points)
            A = np.maximum(0, A0 * np.exp(-(k1 + k2) * time_points))
            B = np.maximum(0, B0 + (A0 - A) * (k1/(k1 + k2)))
            C = np.maximum(0, C0 + (A0 - A) * (k2/(k1 + k2)) * 0.7)
            D = np.maximum(0, D0 + (A0 - A) * (k2/(k1 + k2)) * 0.3)

            # Нормализуем
            for i in range(n_points):
                total = A[i] + B[i] + C[i] + D[i]
                if total > 0:
                    A[i], B[i], C[i], D[i] = A[i]/total, B[i]/total, C[i]/total, D[i]/total

            return {'A': A, 'B': B, 'C': C, 'D': D}

    def extract_features_with_mass_balance(self, concentrations: Dict[str, np.ndarray],
                                          reaction_type: int) -> np.ndarray:
        """
        Извлечение признаков из концентраций с учетом материального баланса
        """
        A = concentrations['A']
        B = concentrations['B']
        C = concentrations['C']
        D = concentrations['D']
        n_points = len(A)

        features = []

        # 1. ПРОВЕРКА МАТЕРИАЛЬНОГО БАЛАНСА (самые важные признаки)
        # Среднее отклонение от идеального баланса (A+B+C+D=1)
        balance_errors = []
        for t in range(n_points):
            total = A[t] + B[t] + C[t] + D[t]
            balance_errors.append(abs(total - 1.0))

        features.append(np.mean(balance_errors))      # Средняя ошибка баланса
        features.append(np.max(balance_errors))       # Максимальная ошибка баланса
        features.append(np.std(balance_errors))       # Стандартное отклонение ошибки

        # 2. ИНТЕГРАЛЬНЫЕ ХАРАКТЕРИСТИКИ (сохранение массы во времени)
        # Интегралы по времени для каждого вещества
        time_points = np.linspace(0, 10, n_points)
        integral_A = np.trapz(A, time_points)
        integral_B = np.trapz(B, time_points)
        integral_C = np.trapz(C, time_points)
        integral_D = np.trapz(D, time_points)

        total_integral = integral_A + integral_B + integral_C + integral_D
        if total_integral > 0:
            features.append(integral_A / total_integral)  # Доля A в общем интеграле
            features.append(integral_B / total_integral)  # Доля B в общем интеграле
            features.append(integral_C / total_integral)  # Доля C в общем интеграле
            features.append(integral_D / total_integral)  # Доля D в общем интеграле
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 3. БАЛАНС ПРЕВРАЩЕНИЙ (эффективность реакций)
        # Для A→B
        delta_A = A[0] - A[-1]
        delta_B = B[-1] - B[0]
        if delta_A > 0.01:
            features.append(delta_B / delta_A)  # Эффективность превращения A в B
        else:
            features.append(0.0)

        # Для B→C (только для последовательной реакции)
        if reaction_type == 0:
            delta_B_for_C = B[0] - B[-1] if B[0] > B[-1] else 0
            delta_C = C[-1] - C[0]
            if delta_B_for_C > 0.01:
                features.append(delta_C / delta_B_for_C)  # Эффективность B→C
            else:
                features.append(0.0)
        else:
            features.append(0.0)  # Для параллельной реакции это не применимо

        # 4. ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ С УЧЕТОМ БАЛАНСА
        # Время достижения максимумов
        time_points_norm = time_points / time_points[-1]  # Нормализованное время

        max_B_idx = np.argmax(B)
        max_C_idx = np.argmax(C)

        features.append(time_points_norm[max_B_idx])  # Нормализованное время максимума B
        features.append(time_points_norm[max_C_idx])  # Нормализованное время максимума C

        # Разница во времени максимумов (ключевой признак)
        if reaction_type == 0:
            # Для последовательной: B раньше C
            features.append(time_points_norm[max_B_idx] - time_points_norm[max_C_idx])
        else:
            # Для параллельной: B и C одновременно
            features.append(abs(time_points_norm[max_B_idx] - time_points_norm[max_C_idx]))

        # 5. СТАТИСТИЧЕСКИЕ ПРИЗНАКИ С УЧЕТОМ БАЛАНСА
        features.append(np.mean(A))
        features.append(np.std(A))
        features.append(np.mean(B))
        features.append(np.std(B))
        features.append(np.mean(C))
        features.append(np.std(C))
        features.append(A[-1] / (A[0] + 1e-10))  # Отношение конечной к начальной для A

        # 6. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ДЛЯ БАЛАНСА
        # Общее изменение массовой доли
        features.append((A[-1] + B[-1] + C[-1] + D[-1]) - (A[0] + B[0] + C[0] + D[0]))

        # Отношение максимальных концентраций B и C
        max_B = np.max(B) if np.max(B) > 0 else 0.001
        max_C = np.max(C) if np.max(C) > 0 else 0.001
        features.append(max_B / max_C)

        # Если признаков меньше 20, дополняем нулями
        if len(features) < self.n_features_:
            features.extend([0.0] * (self.n_features_ - len(features)))
        elif len(features) > self.n_features_:
            features = features[:self.n_features_]

        return np.array(features)

    def generate_training_data(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация реалистичных данных для обучения с учетом материального баланса
        и физических законов химической кинетики
        """
        n_features = self.n_features_
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)

        logger.info(f"Генерация {n_samples} обучающих образцов с учетом материального баланса...")

        for i in range(n_samples):
            # Случайный выбор типа реакции
            reaction_type = np.random.randint(0, 2)  # 0 - последовательная, 1 - параллельная

            # НАЧАЛЬНЫЕ УСЛОВИЯ С СОХРАНЕНИЕМ МАССЫ
            # A0 - основной реагент, остальные - продукты или промежуточные вещества
            A0 = np.random.uniform(0.7, 1.0)          # Основной реагент
            B0 = np.random.uniform(0.0, 0.15)         # Обычно малая начальная концентрация
            C0 = np.random.uniform(0.0, 0.1)          # Обычно малая начальная концентрация
            D0 = np.random.uniform(0.0, 0.05)         # Обычно малая начальная концентрация

            # Нормализуем для сохранения материального баланса (A+B+C+D=1)
            total = A0 + B0 + C0 + D0
            A0, B0, C0, D0 = A0/total, B0/total, C0/total, D0/total

            # Временные точки (нормализованные)
            n_time_points = 50
            time_points = np.linspace(0, 10, n_time_points)

            if reaction_type == 0:  # ПОСЛЕДОВАТЕЛЬНАЯ РЕАКЦИЯ A→B→C→D
                # Константы скоростей (физически осмысленные значения)
                k1 = np.random.uniform(0.1, 0.5)  # A→B
                k2 = np.random.uniform(0.05, 0.3) # B→C
                k3 = np.random.uniform(0.02, 0.2) # C→D

                # Решаем систему ОДУ
                concentrations = self.solve_sequential_reaction(A0, B0, C0, D0,
                                                               k1, k2, k3, time_points)

            else:  # Параллельная РЕАКЦИЯ A→B и A→C→D
                # Константы скоростей для параллельных путей
                k1 = np.random.uniform(0.1, 0.4)  # A→B
                k2 = np.random.uniform(0.1, 0.4)  # A→C
                k3 = np.random.uniform(0.05, 0.3) # C→D

                # Решаем систему ОДУ
                concentrations = self.solve_parallel_reaction(A0, B0, C0, D0,
                                                            k1, k2, k3, time_points)

            # ФИНАЛЬНАЯ ПРОВЕРКА МАТЕРИАЛЬНОГО БАЛАНСА
            # Проверяем, что сумма концентраций равна 1 с допустимой погрешностью
            for t in range(n_time_points):
                total_t = (concentrations['A'][t] + concentrations['B'][t] +
                         concentrations['C'][t] + concentrations['D'][t])
                if abs(total_t - 1.0) > 0.001:  # Допустимая погрешность 0.1%
                    # Корректируем
                    scale = 1.0 / total_t
                    concentrations['A'][t] *= scale
                    concentrations['B'][t] *= scale
                    concentrations['C'][t] *= scale
                    concentrations['D'][t] *= scale

            # Извлекаем признаки с учетом материального баланса
            features = self.extract_features_with_mass_balance(concentrations, reaction_type)

            X[i] = features
            y[i] = reaction_type

            # Логируем прогресс
            if (i + 1) % 500 == 0:
                logger.info(f"Сгенерировано {i + 1}/{n_samples} образцов")

        # Проверяем качество сгенерированных данных
        logger.info(f"Генерация завершена. Проверка материального баланса...")

        # Проверяем первый образец для демонстрации
        if n_samples > 0:
            sample_features = X[0]
            # Признаки 0-2: показатели материального баланса
            balance_error = sample_features[0]
            if balance_error > 0.01:
                logger.warning(f"Высокая ошибка баланса в образцах: {balance_error:.6f}")
            else:
                logger.info(f"Материальный баланс сохранен (ошибка: {balance_error:.6f})")

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Обучение модели с отслеживанием прогресса и проверкой материального баланса
        """
        import time
        start_time = time.time()

        try:
            # Если не переданы данные, генерируем их с учетом материального баланса
            if X_train is None or len(X_train) == 0:
                X_train, y_train = self.generate_training_data(1000)
                if X_val is None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=0.2
                    )
                logger.info("Данные сгенерированы с учетом материального баланса")

            # Сохраняем количество признаков
            if X_train is not None and len(X_train.shape) > 1:
                self.n_features_ = X_train.shape[1]

            # Проверяем признаки материального баланса в данных
            balance_indicators = []
            if X_train is not None and len(X_train) > 0:
                # Первые 3 признака - показатели материального баланса
                balance_errors = X_train[:, 0]  # Средняя ошибка баланса
                avg_balance_error = np.mean(balance_errors)
                max_balance_error = np.max(balance_errors)

                logger.info(f"Средняя ошибка материального баланса в данных: {avg_balance_error:.6f}")
                logger.info(f"Максимальная ошибка материального баланса: {max_balance_error:.6f}")

                if avg_balance_error > 0.01:
                    logger.warning("Высокая средняя ошибка материального баланса в обучающих данных")
                else:
                    logger.info("Материальный баланс в обучающих данных хорошо сохранен")

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
            val_accuracy = 0
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_pred = self.model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                self.training_history['val_accuracy'].append(val_accuracy)

                logger.info(f"Точность на валидации: {val_accuracy:.4f}")

            # Генерация имен признаков
            self.feature_names = [
                'balance_error_mean', 'balance_error_max', 'balance_error_std',
                'integral_A_share', 'integral_B_share', 'integral_C_share', 'integral_D_share',
                'A_to_B_efficiency', 'B_to_C_efficiency',
                'time_max_B_norm', 'time_max_C_norm', 'time_diff_max',
                'A_mean', 'A_std', 'B_mean', 'B_std', 'C_mean', 'C_std',
                'A_final_initial_ratio', 'total_mass_change', 'max_B_to_C_ratio'
            ]

            # Обрезаем или дополняем до нужного количества
            if len(self.feature_names) > self.n_features_:
                self.feature_names = self.feature_names[:self.n_features_]
            elif len(self.feature_names) < self.n_features_:
                self.feature_names.extend([f'feature_{i}' for i in range(len(self.feature_names), self.n_features_)])

            # Расчет времени обучения
            self.training_time = time.time() - start_time
            self.is_trained = True

            logger.info(f"Модель успешно обучена за {self.training_time:.2f} сек")
            logger.info(f"Точность на обучении: {train_accuracy:.4f}")
            if X_val is not None:
                logger.info(f"Точность на валидации: {val_accuracy:.4f}")
            logger.info(f"Количество признаков: {self.n_features_}")
            logger.info(f"Признаки материального баланса в модели: {self.feature_names[:3]}")

            return {
                'status': 'success',
                'training_time': self.training_time,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'n_samples': len(X_train),
                'n_features': self.n_features_,
                'n_iterations': self.model.n_iter_,
                'balance_error_mean': avg_balance_error if 'avg_balance_error' in locals() else 0,
                'balance_error_max': max_balance_error if 'max_balance_error' in locals() else 0
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
        """Оценка модели с анализом материального баланса"""
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

            # Анализ материального баланса в тестовых данных
            balance_analysis = {}
            if X_test.shape[0] > 0 and X_test.shape[1] >= 3:
                balance_errors = X_test[:, 0]  # Первый признак - средняя ошибка баланса
                balance_analysis = {
                    'balance_error_mean': float(np.mean(balance_errors)),
                    'balance_error_std': float(np.std(balance_errors)),
                    'balance_error_max': float(np.max(balance_errors)),
                    'well_balanced_samples': int(np.sum(balance_errors < 0.01)),
                    'poorly_balanced_samples': int(np.sum(balance_errors > 0.05))
                }

            logger.info(f"Точность модели: {accuracy:.4f}")
            logger.info(f"AUC-ROC: {auc_score:.4f}, F1-score: {f1:.4f}")
            logger.info(f"Анализ материального баланса в тестовых данных: {balance_analysis}")

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
                'balance_analysis': balance_analysis,
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
        """Получение информации о модели с данными о материальном балансе"""
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
                'training_history': self.training_history,
                'feature_names': self.feature_names[:5] if self.feature_names else [],
                'mass_balance_features': self.feature_names[:3] if self.feature_names and len(self.feature_names) >= 3 else []
            }

            if self.model_type == 'mlp':
                info['hidden_layer_sizes'] = self.model.hidden_layer_sizes

            return info

        except Exception as e:
            logger.error(f"Ошибка получения информации о модели: {e}")
            return {}

    def save_model(self, filepath: Union[str, Path]) -> bool:
        """Сохранение модели с информацией о материальном балансе"""
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
                'save_timestamp': datetime.now().isoformat(),
                'mass_balance_info': {
                    'has_mass_balance_features': True,
                    'balance_feature_names': self.feature_names[:3] if self.feature_names else []
                }
            }

            joblib.dump(save_data, filepath)
            logger.info(f"Модель сохранена в {filepath}")
            logger.info(f"Информация о материальном балансе сохранена: {save_data['mass_balance_info']}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            return False

    def load_model(self, filepath: Union[str, Path]) -> bool:
        """Загрузка модели с информацией о материальном балансе"""
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

            # Проверяем информацию о материальном балансе
            mass_balance_info = data.get('mass_balance_info', {})
            if mass_balance_info.get('has_mass_balance_features', False):
                logger.info("Модель загружена с поддержкой материального баланса")
                if self.feature_names and len(self.feature_names) >= 3:
                    logger.info(f"Признаки материального баланса: {self.feature_names[:3]}")
            else:
                logger.info("Модель загружена без информации о материальном балансе")

            logger.info(f"Модель загружена из {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False