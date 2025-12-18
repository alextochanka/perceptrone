"""
Модуль для работы с базой данных химических реакций
"""
import sqlite3
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ChemicalDatabase:
    """Класс для работы с базой данных химических реакций"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "chemical_reactions.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def _get_current_time(self):
        """Получение текущего времени"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _safe_json_loads(self, json_str):
        """Безопасный парсинг JSON строки"""
        if json_str is None:
            return {}
        if isinstance(json_str, (dict, list)):
            return json_str
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Не удалось распарсить JSON: {str(e)}. Строка: {str(json_str)[:100]}...")
            try:
                # Пробуем заменить одинарные кавычки на двойные
                fixed_str = str(json_str).replace("'", '"')
                return json.loads(fixed_str)
            except:
                return {"raw_data": str(json_str), "parse_error": True}

    def init_database(self):
        """Инициализация базы данных (без удаления существующих данных)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Таблица пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TEXT  -- Храним явно как текст
                )
            ''')

            # Таблица экспериментов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    experiment_name TEXT,
                    model_type TEXT,
                    accuracy REAL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    parameters TEXT,
                    created_at TEXT,  -- Храним явно как текст
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Таблица реакций
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    reaction_type TEXT,
                    substances TEXT,
                    concentrations TEXT,
                    time_points TEXT,
                    prediction_result TEXT,
                    confidence REAL,
                    created_at TEXT,  -- Храним явно как текст
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Таблица действий
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT,
                    details TEXT,
                    created_at TEXT,  -- Храним явно как текст
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()
            conn.close()

            # Проверяем и обновляем структуру, если нужно
            self._check_and_update_schema()

            logger.info("База данных инициализирована успешно")

        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            self._alternative_init()

    def _check_and_update_schema(self):
        """Проверяем и обновляем структуру базы данных при необходимости"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Проверяем существующие таблицы
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [table[0] for table in cursor.fetchall()]
            logger.info(f"Существующие таблицы: {existing_tables}")

            # Проверяем структуру каждой таблицы
            for table_name in ['users', 'experiments', 'reactions', 'actions']:
                if table_name in existing_tables:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    logger.info(f"Структура таблицы {table_name}: {columns}")

                    # Если нужно добавить отсутствующие колонки
                    if table_name == 'reactions' and 'created_at' not in columns:
                        logger.info("Добавляю колонку created_at в таблицу reactions")
                        cursor.execute("ALTER TABLE reactions ADD COLUMN created_at TEXT")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Ошибка при проверке схемы: {e}")


    def register_user(self, telegram_id: int, username: str = None,
                      first_name: str = None, last_name: str = None):
        """Регистрация пользователя"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            current_time = self._get_current_time()
            logger.info(f"Регистрация пользователя {telegram_id} в {current_time}")

            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (telegram_id, username, first_name, last_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (telegram_id, username, first_name, last_name, current_time))

            conn.commit()
            conn.close()

            # Проверяем, что записалось в базу данных
            self._check_last_record('users')

            return True

        except Exception as e:
            logger.error(f"Ошибка регистрации пользователя: {e}", exc_info=True)
            return False

    def _check_last_record(self, table_name: str):
        """Проверка последней записи в таблице"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f'SELECT COUNT(*) as count FROM {table_name}')
            total = cursor.fetchone()[0]

            cursor.execute(f'SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            conn.close()

            if row:
                logger.info(f"Записей в {table_name}: {total}, последняя: {row}")
            else:
                logger.info(f"Таблица {table_name} пуста (0 записей)")
        except Exception as e:
            logger.error(f"Ошибка проверки таблицы {table_name}: {e}")

    def log_action(self, user_id: int, action: str, details: str = None):
        """Логирование действий пользователя"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            current_time = self._get_current_time()
            logger.info(f"Логирование действия {action} в {current_time}")

            cursor.execute('''
                INSERT INTO actions (user_id, action, details, created_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, action, details, current_time))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Ошибка логирования действия: {e}")
            return False

    def save_experiment(self, user_id: int, experiment_data: Dict[str, Any]) -> int:
        """Сохранение эксперимента"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            current_time = self._get_current_time()
            logger.info(f"Сохранение эксперимента в {current_time}")

            # Гарантируем, что parameters - валидный JSON
            parameters = experiment_data.get('parameters', {})
            if not isinstance(parameters, (dict, list)):
                logger.warning(f"Параметры не являются dict/list: {type(parameters)}. Преобразование в dict.")
                parameters = {}

            parameters_str = json.dumps(parameters, ensure_ascii=False, default=str)

            cursor.execute('''
                INSERT INTO experiments 
                (user_id, experiment_name, model_type, 
                 accuracy, training_samples, test_samples, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                experiment_data.get('experiment_name', 'Без названия'),
                experiment_data.get('model_type', 'perceptron'),
                float(experiment_data.get('accuracy', 0)),
                int(experiment_data.get('training_samples', 0)),
                int(experiment_data.get('test_samples', 0)),
                parameters_str,
                current_time
            ))

            experiment_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # Логируем действие
            self.log_action(user_id, "Сохранение эксперимента",
                          f"{experiment_data.get('experiment_name', 'Без названия')}")

            logger.info(f"Эксперимент сохранен с ID: {experiment_id}")
            return experiment_id

        except Exception as e:
            logger.error(f"Ошибка сохранения эксперимента: {e}", exc_info=True)
            return -1

    def save_reaction(self, user_id: int, reaction_data: Dict[str, Any]) -> int:
        """Сохранение реакции"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            current_time = self._get_current_time()
            logger.info(f"Сохранение реакции в {current_time} для пользователя {user_id}")

            # Гарантируем валидный JSON для всех полей
            substances = reaction_data.get('substances', {})
            if not isinstance(substances, (dict, list)):
                logger.warning(f"Вещества не являются dict/list: {type(substances)}")
                substances = {}

            concentrations = reaction_data.get('concentrations', {})
            if not isinstance(concentrations, dict):
                logger.warning(f"Концентрации не являются dict: {type(concentrations)}")
                concentrations = {}

            time_points = reaction_data.get('time_points', [])
            if not isinstance(time_points, list):
                logger.warning(f"Временные точки не являются list: {type(time_points)}")
                time_points = []

            prediction_result = reaction_data.get('prediction_result', {})
            if not isinstance(prediction_result, dict):
                logger.warning(f"Результат предсказания не является dict: {type(prediction_result)}")
                prediction_result = {}

            substances_str = json.dumps(substances, ensure_ascii=False, default=str)
            concentrations_str = json.dumps(concentrations, ensure_ascii=False, default=str)
            time_points_str = json.dumps(time_points, ensure_ascii=False, default=str)
            prediction_result_str = json.dumps(prediction_result, ensure_ascii=False, default=str)

            cursor.execute('''
                INSERT INTO reactions 
                (user_id, reaction_type, substances, concentrations, 
                 time_points, prediction_result, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                reaction_data.get('reaction_type', 'unknown'),
                substances_str,
                concentrations_str,
                time_points_str,
                prediction_result_str,
                float(reaction_data.get('confidence', 0.0)),
                current_time
            ))

            reaction_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # Логируем действие
            self.log_action(user_id, "Сохранение реакции",
                          f"Тип: {reaction_data.get('reaction_type', 'unknown')}, "
                          f"Уверенность: {reaction_data.get('confidence', 0):.2%}")

            logger.info(f"Сохранена реакция ID: {reaction_id} в {current_time}")

            # Проверяем что записалось
            self._check_last_record('reactions')

            return reaction_id

        except Exception as e:
            logger.error(f"Ошибка сохранения реакции: {e}", exc_info=True)
            return -1

    def get_experiments(self, user_id: int = None) -> List[Dict[str, Any]]:
        """Получение списка экспериментов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if user_id:
                cursor.execute('''
                    SELECT id, experiment_name, model_type, accuracy, created_at
                    FROM experiments
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                ''', (user_id,))
            else:
                cursor.execute('''
                    SELECT id, experiment_name, model_type, accuracy, created_at
                    FROM experiments
                    ORDER BY created_at DESC
                ''')

            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    'id': row[0],
                    'experiment_name': row[1],
                    'model_type': row[2],
                    'accuracy': row[3],
                    'timestamp': row[4]
                })

            conn.close()
            return experiments

        except Exception as e:
            logger.error(f"Ошибка получения экспериментов: {e}")
            return []

    def get_reactions(self, user_id: int = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение списка реакций"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if user_id:
                cursor.execute('''
                    SELECT id, reaction_type, confidence, created_at
                    FROM reactions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT id, reaction_type, confidence, created_at
                    FROM reactions
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))

            reactions = []
            for row in cursor.fetchall():
                reactions.append({
                    'id': row[0],
                    'predicted_type': row[1],
                    'confidence': row[2],
                    'timestamp': row[3]
                })

            conn.close()
            return reactions

        except Exception as e:
            logger.error(f"Ошибка получения реакций: {e}")
            return []

    def get_reaction_details(self, reaction_id: int) -> Optional[Dict[str, Any]]:
        """Получение детальной информации о реакции"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT reaction_type, substances, concentrations, 
                       time_points, prediction_result, confidence, created_at
                FROM reactions
                WHERE id = ?
            ''', (reaction_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'reaction_type': row[0],
                    'substances': self._safe_json_loads(row[1]),
                    'concentrations': self._safe_json_loads(row[2]),
                    'time_points': self._safe_json_loads(row[3]),
                    'prediction_result': self._safe_json_loads(row[4]),
                    'confidence': row[5],
                    'timestamp': row[6]
                }
            return None

        except Exception as e:
            logger.error(f"Ошибка получения деталей реакции: {e}")
            return None

    def get_experiment_details(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Получение детальной информации об эксперименте"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT experiment_name, model_type, accuracy, 
                       training_samples, test_samples, parameters, created_at
                FROM experiments
                WHERE id = ?
            ''', (experiment_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'experiment_name': row[0],
                    'model_type': row[1],
                    'accuracy': row[2],
                    'training_samples': row[3],
                    'test_samples': row[4],
                    'parameters': self._safe_json_loads(row[5]),
                    'timestamp': row[6]
                }
            return None

        except Exception as e:
            logger.error(f"Ошибка получения деталей эксперимента: {e}")
            return None

    def get_database_stats(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}

            # Количество записей в каждой таблице
            tables = ['users', 'experiments', 'reactions', 'actions']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Средняя точность экспериментов
            cursor.execute("SELECT AVG(accuracy) FROM experiments WHERE accuracy IS NOT NULL")
            stats['avg_accuracy'] = cursor.fetchone()[0] or 0

            # Средняя уверенность реакций
            cursor.execute("SELECT AVG(confidence) FROM reactions WHERE confidence IS NOT NULL")
            stats['avg_confidence'] = cursor.fetchone()[0] or 0

            # Последняя активность
            cursor.execute("""
                SELECT MAX(created_at) FROM (
                    SELECT created_at FROM experiments
                    UNION ALL
                    SELECT created_at FROM reactions
                    UNION ALL
                    SELECT created_at FROM actions
                )
            """)
            stats['last_activity'] = cursor.fetchone()[0]

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Ошибка получения статистики БД: {e}")
            return {}

    def check_database(self):
        """Проверка структуры базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Проверяем таблицы
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logger.info(f"Таблицы в БД: {tables}")

            # Проверяем структуру таблицы reactions
            cursor.execute("PRAGMA table_info(reactions)")
            columns = cursor.fetchall()
            logger.info("Структура таблицы reactions:")
            for col in columns:
                logger.info(f"  {col[1]} - {col[2]}")

            # Показываем статистику
            for table in ['users', 'experiments', 'reactions', 'actions']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"Записей в {table}: {count}")

            conn.close()

        except Exception as e:
            logger.error(f"Ошибка проверки БД: {e}")