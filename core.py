"""
Основные классы приложения для анализа химических реакций
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
import json

try:
    from database import ChemicalDatabase
    from neural_network import NeuralNetworkModel
    from config import MODEL_CONFIG, REACTION_TYPES
except ImportError:
    # Создаем заглушки для отладки
    class ChemicalDatabase:
        def __init__(self, db_path=None):
            pass
        def register_user(self, *args, **kwargs): return True
        def log_action(self, *args, **kwargs): return True
        def save_experiment(self, *args, **kwargs): return 1
        def save_reaction(self, *args, **kwargs): return 1
        def get_user_stats(self, *args, **kwargs): return {}
        def get_top_users(self, *args, **kwargs): return []
        def get_experiments(self, *args, **kwargs): return []
        def get_reactions(self, *args, **kwargs): return []

    class NeuralNetworkModel:
        def __init__(self, model_type='perceptron'):
            self.model_type = model_type
            self.model = None
            self.is_trained = False
            self.scaler = None
            self.n_features_ = 20
        def create_model(self, **kwargs): pass
        def train(self, X, y): return {'status': 'success'}
        def evaluate(self, X, y): return {'status': 'success', 'accuracy': 0.95}
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[1.0, 0.0, 0.0, 0.0]])

    MODEL_CONFIG = {
        'perceptron': {'hidden_layer_sizes': (), 'max_iter': 1000},
        'mlp': {'hidden_layer_sizes': (128, 64), 'max_iter': 3000}
    }

    REACTION_TYPES = {
        'type1': {'name': 'Последовательная A → B → C → D'},
        'type2': {'name': 'Параллельная A → B и A → C → D'}
    }

logger = logging.getLogger(__name__)


class ReactionSimulator:
    """Симулятор химических реакций"""

    def __init__(self):
        self.species_names = ['A', 'B', 'C', 'D']

    def extract_features(self, concentrations: Dict[str, List[float]]) -> np.ndarray:
        """Извлечение признаков из концентраций веществ с нормализацией"""
        features = []

        # Базовые статистики для каждого вещества
        for species in self.species_names:
            conc = concentrations.get(species, [0.0])
            if len(conc) > 0:
                features.extend([
                    np.mean(conc),           # Средняя концентрация
                    np.std(conc),            # Стандартное отклонение
                    np.max(conc),            # Максимальная концентрация
                    np.min(conc),            # Минимальная концентрация
                    conc[-1] / (conc[0] + 1e-10) if conc[0] != 0 else 0,  # Отношение конечной к начальной
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Отношения между веществами
        for i in range(len(self.species_names)):
            for j in range(i+1, len(self.species_names)):
                s1 = self.species_names[i]
                s2 = self.species_names[j]
                c1 = concentrations.get(s1, [0.0])
                c2 = concentrations.get(s2, [0.0])

                if len(c1) > 0 and len(c2) > 0 and np.mean(c2) != 0:
                    ratio = np.mean(c1) / (np.mean(c2) + 1e-10)
                    features.append(ratio)
                else:
                    features.append(0.0)

        # ДОБАВЛЕНЫ КЛЮЧЕВЫЕ ПРИЗНАКИ ДЛЯ УЛУЧШЕНИЯ КЛАССИФИКАЦИИ
        a_vals = concentrations.get('A', [0.0])
        b_vals = concentrations.get('B', [0.0])
        c_vals = concentrations.get('C', [0.0])
        d_vals = concentrations.get('D', [0.0])

        if len(a_vals) > 1:
            # 1. Относительное время максимумов B и C
            if len(b_vals) > 0:
                b_max_idx = np.argmax(b_vals)
                b_max_norm = b_max_idx / (len(b_vals) - 1) if len(b_vals) > 1 else 0
                features.append(b_max_norm)
            else:
                features.append(0.0)

            if len(c_vals) > 0:
                c_max_idx = np.argmax(c_vals)
                c_max_norm = c_max_idx / (len(c_vals) - 1) if len(c_vals) > 1 else 0
                features.append(c_max_norm)
            else:
                features.append(0.0)

            # 2. Разница во времени максимумов (ключевой признак)
            if len(b_vals) > 0 and len(c_vals) > 0:
                time_diff = abs(b_max_norm - c_max_norm)
                features.append(time_diff)

                # 3. Порядок максимумов (1 если B раньше C, -1 если наоборот)
                order = 1 if b_max_norm < c_max_norm else -1 if b_max_norm > c_max_norm else 0
                features.append(float(order))
            else:
                features.extend([0.0, 0.0])

            # 4. Отношение максимумов B и C
            b_max = max(b_vals) if b_vals else 0
            c_max = max(c_vals) if c_vals else 0
            if c_max > 0:
                features.append(b_max / c_max)
            else:
                features.append(0.0)

            # 5. Площади под кривыми B и C (аппроксимация)
            if len(b_vals) > 1:
                b_area = np.trapz(b_vals, np.linspace(0, 1, len(b_vals)))
                features.append(b_area)
            else:
                features.append(0.0)

            if len(c_vals) > 1:
                c_area = np.trapz(c_vals, np.linspace(0, 1, len(c_vals)))
                features.append(c_area)
            else:
                features.append(0.0)

        target_features = 20
        if len(features) > target_features:
            features = features[:target_features]  # Обрезаем лишние
        elif len(features) < target_features:
            features.extend([0.0] * (target_features - len(features)))  # Дополняем нулями

        return np.array(features)

    def analyze_reaction_pattern(self, concentrations: Dict[str, List[float]]) -> Dict[str, Any]:
        """Анализ паттерна реакции с нормализацией по количеству точек"""
        a_vals = concentrations.get('A', [0.0])
        b_vals = concentrations.get('B', [0.0])
        c_vals = concentrations.get('C', [0.0])
        d_vals = concentrations.get('D', [0.0])

        if len(a_vals) < 2:
            return {'type': 'unknown', 'confidence': 0.5}

        # Последовательная реакция: A уменьшается, B появляется и уменьшается, C появляется, D растет
        # Параллельная: A уменьшается, B и C появляются одновременно, D растет быстрее

        # Критерии для последовательной
        a_decrease = a_vals[0] - a_vals[-1]
        b_max_idx = np.argmax(b_vals) if len(b_vals) > 0 else 0
        c_max_idx = np.argmax(c_vals) if len(c_vals) > 0 else 0
        d_increase = d_vals[-1] - d_vals[0] if len(d_vals) > 1 else 0

        # НОРМАЛИЗАЦИЯ: преобразуем индексы в относительное время (0-1)
        n_points = len(a_vals)
        if n_points > 1:
            b_max_norm = b_max_idx / (n_points - 1)  # 0 до 1
            c_max_norm = c_max_idx / (n_points - 1)  # 0 до 1
            time_diff_norm = abs(b_max_norm - c_max_norm)
        else:
            b_max_norm = 0
            c_max_norm = 0
            time_diff_norm = 0

        # Проверяем последовательность максимумов С УЧЕТОМ НОРМАЛИЗАЦИИ
        is_sequential = False
        is_branched = False

        # Критерий для последовательной: B заметно раньше C (>20% от общего времени)
        if (b_max_idx > 0 and c_max_idx > b_max_idx and d_increase > 0 and
            (c_max_norm - b_max_norm) > 0.2):  # Увеличили порог
            is_sequential = True

        # Критерий для параллельной: B и C примерно одновременно (<15% разницы)
        if (b_max_idx > 0 and c_max_idx > 0 and d_increase > 0 and
            time_diff_norm < 0.15):  # Адаптивный порог
            is_branched = True

        # Принятие решения с приоритетом
        if is_sequential and not is_branched:
            confidence = min(0.9, 0.7 + (c_max_norm - b_max_norm) * 1.5)
            return {
                'type': 'type1',
                'confidence': confidence,
                'b_max_norm': b_max_norm,
                'c_max_norm': c_max_norm,
                'time_diff': time_diff_norm
            }
        elif is_branched and not is_sequential:
            confidence = min(0.9, 0.7 + (0.15 - time_diff_norm) * 3)
            return {
                'type': 'type2',
                'confidence': confidence,
                'b_max_norm': b_max_norm,
                'c_max_norm': c_max_norm,
                'time_diff': time_diff_norm
            }
        else:
            # Если оба или ни один, используем дополнительные эвристики
            b_max = max(b_vals) if b_vals else 0
            c_max = max(c_vals) if c_vals else 0

            # Параллельные реакции: B и C имеют сравнимые максимумы
            if b_max > 0.1 and c_max > 0.1 and abs(b_max - c_max) < max(b_max, c_max) * 0.3:
                # Относительная разница менее 30%
                confidence = 0.7 if time_diff_norm < 0.25 else 0.6
                return {
                    'type': 'type2',
                    'confidence': confidence,
                    'b_max_norm': b_max_norm,
                    'c_max_norm': c_max_norm,
                    'time_diff': time_diff_norm
                }
            else:
                # Последовательная по умолчанию
                confidence = 0.7 if b_max_norm < c_max_norm else 0.6
                return {
                    'type': 'type1',
                    'confidence': confidence,
                    'b_max_norm': b_max_norm,
                    'c_max_norm': c_max_norm,
                    'time_diff': time_diff_norm
                }


class ReactionBot:
    """Основной класс бота для управления всей системой"""

    def __init__(self):
        self.db = ChemicalDatabase()
        self.simulator = ReactionSimulator()
        self.current_model = None
        self.current_experiment_id = None

    def train_model(self, model_type: str = 'perceptron',
                   n_samples: int = 2000,
                   max_iter: int = 3000,
                   hidden_layers: tuple = None) -> Dict[str, Any]:
        """Обучение модели без случайного семени"""
        try:
            # Создание и обучение модели
            self.current_model = NeuralNetworkModel(model_type)

            # Получаем конфигурацию из MODEL_CONFIG
            config = MODEL_CONFIG.get(model_type, MODEL_CONFIG.get('perceptron', {})).copy()
            config['max_iter'] = max_iter

            if model_type == 'mlp' and hidden_layers:
                config['hidden_layer_sizes'] = hidden_layers

            # Создаем модель с параметрами
            if hasattr(self.current_model, 'create_model'):
                # Убираем random_state из конфигурации для разнообразия
                config.pop('random_state', None)
                self.current_model.create_model(**config)

            # Генерируем данные и обучаем
            X, y = self.current_model.generate_training_data(n_samples)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2
            )

            # Обучение
            train_result = self.current_model.train(X_train, y_train, X_test, y_test)
            if isinstance(train_result, dict) and train_result.get('status') == 'error':
                return train_result

            # Оценка на тестовой выборке
            eval_result = self.current_model.evaluate(X_test, y_test)

            if isinstance(eval_result, dict) and eval_result.get('status') == 'success':
                # Сохранение эксперимента
                experiment_data = {
                    'experiment_name': f'Обучение {model_type}',
                    'model_type': model_type,
                    'accuracy': eval_result.get('accuracy', 0),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'parameters': {
                        'n_samples': n_samples,
                        'max_iter': max_iter,
                        'model_type': model_type,
                        'hidden_layers': hidden_layers,
                        'n_features': self.current_model.n_features_
                    }
                }

                self.current_experiment_id = self.db.save_experiment(0, experiment_data)

                return {
                    'status': 'success',
                    'accuracy': eval_result.get('accuracy', 0),
                    'experiment_id': self.current_experiment_id,
                    'model_type': model_type,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_iterations': getattr(self.current_model.model, 'n_iter_', 0)
                    if hasattr(self.current_model, 'model') else 0,
                    'n_features': self.current_model.n_features_
                }
            else:
                return eval_result if isinstance(eval_result, dict) else {
                    'status': 'error',
                    'message': 'Неизвестная ошибка при оценке'
                }

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def predict_reaction(self, time_points: List[float],
                         concentrations: Dict[str, List[float]],
                         user_id: int = 0) -> Dict[str, Any]:
        """Предсказание типа реакции с сохранением в БД и проверкой на отрицательные концентрации"""
        try:
            # ================== ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ ==================
            # 1. Проверяем и исправляем отрицательные концентрации
            logger.info(f"Проверка данных от пользователя {user_id}...")

            # Создаем копию концентраций для безопасной обработки
            processed_concentrations = {}
            for substance, values in concentrations.items():
                if not values:
                    processed_concentrations[substance] = [0.0] * len(time_points)
                    logger.warning(f"Вещество {substance}: пустой массив, заполнено нулями")
                    continue

                # Проверяем на отрицательные значения и NaN
                clean_values = []
                negative_count = 0
                nan_count = 0

                for v in values:
                    if np.isnan(v) or v is None:
                        clean_values.append(0.0)
                        nan_count += 1
                    elif v < 0:
                        clean_values.append(0.0)
                        negative_count += 1
                    else:
                        clean_values.append(float(v))

                processed_concentrations[substance] = clean_values

                if negative_count > 0:
                    logger.warning(f"Вещество {substance}: исправлено {negative_count} отрицательных значений")
                if nan_count > 0:
                    logger.warning(f"Вещество {substance}: исправлено {nan_count} NaN/None значений")

            # 2. Гарантируем наличие всех необходимых веществ (A, B, C, D)
            required_substances = ['A', 'B', 'C', 'D']
            for substance in required_substances:
                if substance not in processed_concentrations:
                    processed_concentrations[substance] = [0.0] * len(time_points)
                    logger.info(f"Добавлено отсутствующее вещество {substance}")

            # 3. Нормализуем концентрации для каждого временного интервала
            n_points = len(time_points)
            normalized_concentrations = {s: [] for s in required_substances}

            for i in range(n_points):
                # Собираем концентрации для текущего времени
                current_values = {}
                for s in required_substances:
                    if i < len(processed_concentrations[s]):
                        current_values[s] = processed_concentrations[s][i]
                    else:
                        current_values[s] = 0.0

                # Проверяем общую сумму
                total = sum(current_values.values())

                if total <= 0:
                    current_values = {
                        'A': max(0, 1.0 - 0.1 * i),
                        'B': max(0, 0.0 + 0.08 * i),
                        'C': max(0, 0.0 + 0.05 * i),
                        'D': max(0, 0.0 + 0.03 * i)
                    }
                    total = sum(current_values.values())
                    logger.warning(f"Временная точка {i}: все концентрации нулевые, созданы синтетические данные")

                for s in required_substances:
                    normalized_concentrations[s].append(current_values[s])

            # 4. Проверяем длины массивов
            lengths = [len(v) for v in normalized_concentrations.values()]
            if len(set(lengths)) > 1:
                logger.error(f"Разные длины массивов после нормализации: {lengths}")
                min_len = min(lengths)
                for s in required_substances:
                    normalized_concentrations[s] = normalized_concentrations[s][:min_len]
                time_points = time_points[:min_len]

            # ================== АНАЛИЗ РЕАКЦИИ ==================
            logger.info(f"Анализ реакции с {len(time_points)} временными точками...")

            # Анализируем паттерн для проверки (используем нормализованные данные)
            pattern_analysis = self.simulator.analyze_reaction_pattern(normalized_concentrations)

            # Используем модель или эвристику
            if not self.current_model or not hasattr(self.current_model,
                                                     'is_trained') or not self.current_model.is_trained:
                logger.warning("Модель не обучена, использую эвристический анализ")
                result = {
                    'status': 'success',
                    'reaction_type': pattern_analysis['type'],
                    'confidence': pattern_analysis['confidence'],
                    'type_name': REACTION_TYPES.get(pattern_analysis['type'], {}).get('name', 'Неизвестный тип'),
                    'method': 'heuristic',
                    'data_quality': 'processed',
                    'analysis_details': {
                        'points_analyzed': len(time_points),
                        'b_max_norm': pattern_analysis.get('b_max_norm', 0),
                        'c_max_norm': pattern_analysis.get('c_max_norm', 0),
                        'time_diff': pattern_analysis.get('time_diff', 0)
                    }
                }
            else:
                # Извлечение признаков из нормализованных данных
                features = self.simulator.extract_features(normalized_concentrations).reshape(1, -1)

                # Проверяем количество признаков
                if hasattr(self.current_model, 'n_features_'):
                    n_features_expected = self.current_model.n_features_
                    n_features_actual = features.shape[1]

                    if n_features_actual != n_features_expected:
                        logger.info(f"Корректировка признаков: {n_features_actual} -> {n_features_expected}")

                        if n_features_actual > n_features_expected:
                            features = features[:, :n_features_expected]
                        else:
                            padding = np.zeros((1, n_features_expected - n_features_actual))
                            features = np.hstack([features, padding])

                # Предсказание
                try:
                    prediction = self.current_model.predict(features)
                    probability = self.current_model.predict_proba(features)

                    # Определяем тип реакции
                    reaction_code = f'type{prediction[0] + 1}'  # +1 т.к. классы 0 и 1
                    confidence = float(probability[0][prediction[0]])

                    type_info = REACTION_TYPES.get(reaction_code, REACTION_TYPES.get('type1', {}))
                    type_name = type_info.get('name', 'Неизвестный тип')

                    # Сравниваем с эвристикой для повышения уверенности
                    # ЕСЛИ РЕЗУЛЬТАТЫ СОВПАДАЮТ - повышаем уверенность
                    if pattern_analysis['type'] == reaction_code:
                        combined_confidence = max(confidence, pattern_analysis['confidence'])
                        # Если оба метода уверены - дополнительный бонус
                        if confidence > 0.7 and pattern_analysis['confidence'] > 0.7:
                            combined_confidence = min(1.0, combined_confidence + 0.05)
                        confidence = combined_confidence
                        method = 'combined_agreement'
                    else:
                        # Если методы расходятся - используем взвешенное среднее
                        # Больше веса модели, если она хорошо обучена
                        model_weight = 0.7 if confidence > 0.8 else 0.6
                        heuristic_weight = 1 - model_weight
                        confidence = (confidence * model_weight +
                                      pattern_analysis['confidence'] * heuristic_weight)
                        method = 'weighted_disagreement'

                        # Логируем расхождение для анализа
                        logger.info(f"Методы расходятся: модель={reaction_code}({confidence:.2f}), "
                                  f"эвристика={pattern_analysis['type']}({pattern_analysis['confidence']:.2f})")

                    result = {
                        'status': 'success',
                        'reaction_type': reaction_code,
                        'confidence': confidence,
                        'type_name': type_name,
                        'method': method,
                        'data_quality': 'processed',
                        'analysis_details': {
                            'model_prediction': reaction_code,
                            'heuristic_prediction': pattern_analysis['type'],
                            'model_confidence': float(probability[0][prediction[0]]),
                            'heuristic_confidence': pattern_analysis['confidence'],
                            'points_analyzed': len(time_points),
                            'b_max_norm': pattern_analysis.get('b_max_norm', 0),
                            'c_max_norm': pattern_analysis.get('c_max_norm', 0),
                            'time_diff': pattern_analysis.get('time_diff', 0)
                        }
                    }

                except Exception as model_error:
                    logger.error(f"Ошибка предсказания моделью: {model_error}")
                    # Используем эвристику как запасной вариант
                    result = {
                        'status': 'success',
                        'reaction_type': pattern_analysis['type'],
                        'confidence': pattern_analysis['confidence'] * 0.9,  # Чуть уменьшаем уверенность
                        'type_name': REACTION_TYPES.get(pattern_analysis['type'], {}).get('name', 'Неизвестный тип'),
                        'method': 'heuristic_fallback',
                        'data_quality': 'processed',
                        'model_error': str(model_error),
                        'analysis_details': {
                            'fallback_reason': 'model_error',
                            'heuristic_confidence': pattern_analysis['confidence'],
                            'points_analyzed': len(time_points),
                            'b_max_norm': pattern_analysis.get('b_max_norm', 0),
                            'c_max_norm': pattern_analysis.get('c_max_norm', 0),
                            'time_diff': pattern_analysis.get('time_diff', 0)
                        }
                    }

            # ================== СОХРАНЕНИЕ В БАЗУ ДАННЫХ ==================
            logger.info(f"Сохранение реакции для пользователя {user_id}...")

            # Подготовка данных для сохранения
            reaction_data = {
                'reaction_type': result.get('reaction_type', 'unknown'),
                'substances': list(normalized_concentrations.keys()),
                'concentrations': normalized_concentrations,  # Сохраняем нормализованные данные
                'time_points': time_points,
                'prediction_result': result,
                'confidence': result.get('confidence', 0.0),
                'original_data_quality': {
                    'has_negatives': any(v < 0 for values in concentrations.values() for v in values),
                    'has_nan': any(np.isnan(v) for values in concentrations.values() for v in values),
                    'num_points': len(time_points)
                }
            }

            # Сохраняем в БД с реальным user_id
            try:
                reaction_id = self.db.save_reaction(user_id, reaction_data)

                if reaction_id > 0:
                    result['reaction_id'] = reaction_id
                    result['db_status'] = 'success'
                    logger.info(f"✅ Реакция сохранена в БД с ID: {reaction_id} для пользователя {user_id}")

                    # Дополнительная информация для пользователя
                    result['data_processing_info'] = {
                        'points_processed': len(time_points),
                        'substances_analyzed': len(normalized_concentrations),
                        'original_data_points': sum(len(v) for v in concentrations.values()),
                        'data_corrections_applied': result.get('data_quality') == 'processed'
                    }
                else:
                    result['db_status'] = 'failed'
                    result['db_error'] = 'Не удалось получить ID реакции'
                    logger.error(f"Не удалось получить ID реакции для пользователя {user_id}")

            except Exception as db_error:
                logger.error(f"Ошибка сохранения в БД: {db_error}")
                result['db_status'] = 'error'
                result['db_error'] = str(db_error)
                # Продолжаем без сохранения в БД

            # ================== ВАЛИДАЦИЯ РЕЗУЛЬТАТА ==================
            # Проверяем, что уверенность в разумных пределах
            confidence = result.get('confidence', 0)
            if confidence < 0:
                result['confidence'] = 0.0
                logger.warning("Исправлена отрицательная уверенность")
            elif confidence > 1:
                result['confidence'] = min(1.0, confidence)
                logger.warning("Ограничена уверенность > 1")

            # Добавляем информацию о качестве предсказания
            if confidence > 0.8:
                result['quality'] = 'high'
                quality_emoji = '🔵'
            elif confidence > 0.6:
                result['quality'] = 'medium'
                quality_emoji = '🟡'
            else:
                result['quality'] = 'low'
                quality_emoji = '🔴'
                logger.warning(f"Низкая уверенность предсказания: {confidence:.2%}")

            # Добавляем метаданные анализа
            result['analysis_timestamp'] = datetime.now().isoformat()
            result['quality_emoji'] = quality_emoji

            # Определяем эмодзи для типа реакции
            if result.get('reaction_type') == 'type1':
                result['reaction_emoji'] = '➡️'
            elif result.get('reaction_type') == 'type2':
                result['reaction_emoji'] = '🌳'
            else:
                result['reaction_emoji'] = '❓'

            # Логируем успешное завершение
            logger.info(f"✅ Предсказание завершено: {result.get('type_name')}, "
                       f"уверенность: {confidence:.2%}, качество: {result.get('quality')}")

            return result

        except Exception as e:
            logger.error(f"❌ Критическая ошибка предсказания: {e}", exc_info=True)

            # Возвращаем развернутую информацию об ошибке
            return {
                'status': 'error',
                'message': f"Ошибка анализа: {str(e)}",
                'error_type': type(e).__name__,
                'user_id': user_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'suggestion': 'Проверьте формат данных и попробуйте снова'
            }

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Получение списка экспериментов"""
        try:
            return self.db.get_experiments()
        except Exception as e:
            logger.error(f"Ошибка получения экспериментов: {e}")
            return []

    def get_predictions(self) -> List[Dict[str, Any]]:
        """Получение списка предсказаний"""
        try:
            return self.db.get_reactions()
        except Exception as e:
            logger.error(f"Ошибка получения реакций: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        try:
            experiments = self.db.get_experiments()
            predictions = self.db.get_reactions()

            total_experiments = len(experiments)
            total_predictions = len(predictions)

            # Рассчитываем среднюю точность
            accuracies = [exp.get('accuracy', 0) for exp in experiments if exp.get('accuracy')]
            average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

            # Рассчитываем среднюю уверенность
            confidences = [pred.get('confidence', 0) for pred in predictions if pred.get('confidence')]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'total_experiments': total_experiments,
                'total_predictions': total_predictions,
                'average_accuracy': average_accuracy,
                'average_confidence': average_confidence
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {
                'total_experiments': 0,
                'total_predictions': 0,
                'average_accuracy': 0,
                'average_confidence': 0
            }