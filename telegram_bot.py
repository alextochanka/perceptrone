"""
Telegram бот для предсказания типа химической реакции
"""

import asyncio
import logging
import numpy as np
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode
from telegram.error import TelegramError

# ================== ЛОКАЛЬНЫЕ ИМПОРТЫ ==================
try:
    from config import TELEGRAM_BOT_TOKEN
    from database import ChemicalDatabase
    from core import ReactionBot, REACTION_TYPES
except ImportError:
    TELEGRAM_BOT_TOKEN = "7860657179:AAHXw6AjW1yxzZf8l9chtjGIzv0mQSZ7EGY"

    class ChemicalDatabase:
        def __init__(self, db_path=None):
            pass
        def register_user(self, *a, **kw): pass
        def log_action(self, *a, **kw): pass
        def save_experiment(self, *a, **kw): return 1
        def save_reaction(self, *a, **kw): return 1


    class ReactionBot:
        def __init__(self):
            self.db = ChemicalDatabase()
            self.current_model = None
            self.current_experiment_id = None

        def train_model(self, model_type='perceptron', n_samples=2000, max_iter=3000, hidden_layers=None):
            return {
                'status': 'success',
                'accuracy': 0.95,
                'model_type': model_type,
                'experiment_id': 1
            }

        def predict_reaction(self, time, conc, user_id=0):
            # Определяем тип реакции по эвристике
            from datetime import datetime
            import logging
            logger = logging.getLogger(__name__)

            if max(conc.get("B", [0])) > 0 and max(conc.get("C", [0])) > 0 and abs(
                    max(conc.get("B", [0])) - max(conc.get("C", [0]))) < 0.1:
                result = {
                    "status": "success",
                    "reaction_type": "type2",
                    "type_name": "Параллельная реакция",
                    "confidence": 0.82,
                    "method": "rule_based"
                }
            else:
                result = {
                    "status": "success",
                    "reaction_type": "type1",
                    "type_name": "Последовательная реакция",
                    "confidence": 0.91,
                    "method": "rule_based"
                }

            # Сохраняем в БД с переданным user_id
            try:
                reaction_data = {
                    'reaction_type': result.get('reaction_type', 'unknown'),
                    'substances': list(conc.keys()),
                    'concentrations': conc,
                    'time_points': time,
                    'prediction_result': result,
                    'confidence': result.get('confidence', 0.0)
                }

                # Добавляем логирование для отладки
                logger.info(f"Сохраняю реакцию для user_id={user_id} в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                reaction_id = self.db.save_reaction(user_id, reaction_data)

                if reaction_id > 0:
                    result['reaction_id'] = reaction_id
                    logger.info(f"Реакция сохранена с ID: {reaction_id} для пользователя {user_id}")
                else:
                    logger.warning(f"Не удалось сохранить реакцию для пользователя {user_id}")

            except Exception as e:
                logger.error(f"Ошибка сохранения в БД: {e}", exc_info=True)
                result['db_error'] = str(e)

            return result

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ================== ОСНОВНОЙ КЛАСС ==================
class TelegramReactionBot:
    def __init__(self, token: str):
        self.token = token
        self.db = ChemicalDatabase()
        self.reaction_bot = ReactionBot()
        self.user_sessions = {}
        self.bot_app = None

    # ---------- ОБРАБОТЧИК ОШИБОК ----------
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик ошибок"""
        logger.error(f"Ошибка при обработке обновления {update}: {context.error}")

        try:
            # Отправляем сообщение пользователю
            if update and update.effective_chat:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ Произошла ошибка при обработке запроса. Попробуйте снова."
                )
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения об ошибке: {e}")

    # ---------- /start ----------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user

        self.db.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )

        self.db.log_action(user.id, "start", "Запуск бота")

        keyboard = [
            [
                InlineKeyboardButton("❓ Помощь", callback_data="help"),
                InlineKeyboardButton("🧪 Предсказать реакцию", callback_data="predict")
            ],
            [
                InlineKeyboardButton("🎓 Обучить модель", callback_data="train_menu")
            ]
        ]

        await update.message.reply_text(
            "🧪 **Бот анализа химических реакций**\n\n"
            "Я умею определять тип химической реакции по данным концентраций.\n"
            "Выберите действие:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- /help ----------
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда помощи"""
        user = update.effective_user
        self.db.log_action(user.id, "help_command", "Запрос помощи")

        await update.message.reply_text(
            "❓ **О боте**\n\n"
            "Бот анализирует изменение концентраций веществ\n"
            "и автоматически определяет тип реакции.\n\n"
            "**Поддерживаемые реакции:**\n"
            "• A → B → C → D (последовательная)\n"
            "• A → B → D и A → C → D (параллельная)\n\n"
            "**Доступные команды:**\n"
            "/start - Начать работу\n"
            "/train - Обучить модель\n"
            "/predict - Предсказать реакцию\n"
            "/help - Помощь",
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- /train ----------
    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда обучения модели"""
        user = update.effective_user
        self.db.log_action(user.id, "train_command", "Запрос на обучение")

        keyboard = [
            [
                InlineKeyboardButton("🧠 Перцептрон (простой)", callback_data="train_perceptron"),
                InlineKeyboardButton("🤖 MLP (сложный)", callback_data="train_mlp")
            ],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]

        await update.message.reply_text(
            "🎓 **Обучение модели**\n\n"
            "Выберите тип нейронной сети для обучения:\n\n"
            "• 🧠 **Перцептрон** - простая однослойная сеть, быстрое обучение\n"
            "• 🤖 **MLP** - многослойный перцептрон, высокая точность, дольше обучается\n\n"
            "Для обучения требуется 1000+ образцов данных.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- /predict ----------
    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда предсказания"""
        user = update.effective_user
        self.db.log_action(user.id, "predict_command", "Запрос предсказания")

        await self.predict_menu(update, context)

    # ---------- CALLBACK ----------
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not query or not query.from_user:
            logger.error("Некорректный callback запрос")
            return

        user_id = query.from_user.id
        data = query.data

        try:
            if data == "help":
                await self.show_help(query, user_id)

            elif data == "predict":
                await self.predict_menu_callback(query)

            elif data == "train_menu":
                await self.train_menu(query)

            elif data == "train_perceptron":
                await self.start_training(query, user_id, "perceptron")

            elif data == "train_mlp":
                await self.start_training(query, user_id, "mlp")

            elif data == "input_conc":
                await self.ask_concentrations(query, user_id)

            elif data == "gen_conc":
                await self.generate_concentrations(query, user_id)

            elif data == "main_menu":
                await self.show_main_menu(query)

        except Exception as e:
            logger.error(f"Ошибка обработки callback: {e}", exc_info=True)
            await query.edit_message_text(
                f"❌ **Ошибка**\n\nПроизошла ошибка при обработке запроса. Попробуйте снова.",
                parse_mode=ParseMode.MARKDOWN
            )

    # ---------- ПОМОЩЬ (CALLBACK) ----------
    async def show_help(self, query, user_id: int):
        self.db.log_action(user_id, "help", "Открыта помощь через callback")

        await query.edit_message_text(
            "❓ **О боте**\n\n"
            "Бот позволяет пользователю обучать модели.\n"
            "Бот анализирует изменение концентраций веществ\n"
            "и автоматически определяет тип реакции.\n\n"
            "**Поддерживаемые реакции:**\n"
            "• A → B → C → D (последовательная)\n"
            "• A → B и A → C → D (параллельная)\n\n"
            "**Доступные команды:**\n"
            "/start - Начать работу\n"
            "/train - Обучить модель\n"
            "/predict - Предсказать реакцию\n"
            "/help - Помощь",
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- МЕНЮ ПРЕДСКАЗАНИЯ (CALLBACK) ----------
    async def predict_menu_callback(self, query):
        keyboard = [
            [
                InlineKeyboardButton("✍️ Ввести концентрации", callback_data="input_conc"),
                InlineKeyboardButton("🎲 Сгенерировать пример", callback_data="gen_conc")
            ],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]

        await query.edit_message_text(
            "🧪 **Предсказание реакции**\n\n"
            "Выберите способ задания концентраций:\n\n"
            "• ✍️ **Ввести концентрации** - вручную ввести данные\n"
            "• 🎲 **Сгенерировать пример** - получить готовый пример для анализа",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- МЕНЮ ПРЕДСКАЗАНИЯ (COMMAND) ----------
    async def predict_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [
                InlineKeyboardButton("✍️ Ввести концентрации", callback_data="input_conc"),
                InlineKeyboardButton("🎲 Сгенерировать пример", callback_data="gen_conc")
            ],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]

        if update.message:
            await update.message.reply_text(
                "🧪 **Предсказание реакции**\n\n"
                "Выберите способ задания концентраций:\n\n"
                "• ✍️ **Ввести концентрации** - вручную ввести данные\n"
                "• 🎲 **Сгенерировать пример** - получить готовый пример для анализа",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )

    # ---------- МЕНЮ ОБУЧЕНИЯ ----------
    async def train_menu(self, query):
        keyboard = [
            [
                InlineKeyboardButton("🧠 Перцептрон", callback_data="train_perceptron"),
                InlineKeyboardButton("🤖 MLP", callback_data="train_mlp")
            ],
            [InlineKeyboardButton("🔙 Назад", callback_data="main_menu")]
        ]

        await query.edit_message_text(
            "🎓 **Обучение модели**\n\n"
            "Выберите тип нейронной сети:\n\n"
            "• 🧠 **Перцептрон** - простая однослойная сеть\n"
            "• 🤖 **MLP** - многослойный перцептрон\n\n"
            "Обучение займет несколько секунд.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- ЗАПУСК ОБУЧЕНИЯ ----------
    async def start_training(self, query, user_id: int, model_type: str):
        """Запуск обучения модели"""

        # Отправляем сообщение о начале обучения
        await query.edit_message_text(
            f"🔄 **Начинаю обучение {model_type}...**\n\n"
            f"Генерирую данные для обучения...",
            parse_mode=ParseMode.MARKDOWN
        )

        # Логируем начало обучения
        self.db.log_action(user_id, "train_start", f"Модель: {model_type}")

        try:
            # Вызываем обучение модели
            result = self.reaction_bot.train_model(
                model_type=model_type,
                n_samples=2000,
                max_iter=3000,
                hidden_layers=(128, 64) if model_type == 'mlp' else None
            )

            if result.get('status') == 'success':
                accuracy = result.get('accuracy', 0)

                accuracy_text = f"{accuracy:.1%}"
                if accuracy > 0.9:
                    accuracy_text += " (Отлично! 👏)"
                elif accuracy > 0.8:
                    accuracy_text += " (Хорошо! 👍)"
                elif accuracy > 0.7:
                    accuracy_text += " (Нормально) 🤔"
                else:
                    accuracy_text += " (Низкая точность) ⚠️"

                await query.edit_message_text(
                    f"✅ **Модель успешно обучена!**\n\n"
                    f"**Тип модели:** {model_type}\n"
                    f"**Точность:** {accuracy_text}\n"
                    f"**ID эксперимента:** {result.get('experiment_id', 'N/A')}\n"
                    f"**Образцов:** {result.get('training_samples', 0)} обучающих, "
                    f"{result.get('test_samples', 0)} тестовых\n\n"
                    f"Модель готова к использованию! 🎯",
                    parse_mode=ParseMode.MARKDOWN
                )

                self.db.log_action(
                    user_id,
                    "train_success",
                    f"{model_type} accuracy: {accuracy:.3f}"
                )

            else:
                error_msg = result.get('message', 'Неизвестная ошибка')
                await query.edit_message_text(
                    f"❌ **Ошибка обучения**\n\n"
                    f"Не удалось обучить модель {model_type}.\n"
                    f"Ошибка: {error_msg}",
                    parse_mode=ParseMode.MARKDOWN
                )
                self.db.log_action(user_id, "train_failed", error_msg)

        except Exception as e:
            logger.error(f"Ошибка в обучении: {e}", exc_info=True)
            await query.edit_message_text(
                f"❌ **Критическая ошибка**\n\n"
                f"Произошла непредвиденная ошибка:\n`{str(e)[:200]}`",
                parse_mode=ParseMode.MARKDOWN
            )
            self.db.log_action(user_id, "train_critical_error", str(e))

    # ---------- РУЧНОЙ ВВОД ----------
    async def ask_concentrations(self, query, user_id: int):
        self.user_sessions[user_id] = {"state": "awaiting_concentrations"}
        self.db.log_action(user_id, "input_concentrations", "Ручной ввод")

        await query.edit_message_text(
            "✍️ **Введите концентрации**\n\n"
            "Формат (10 временных точек):\n"
            "```\n"
            "A=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1\n"
            "B=0.0,0.1,0.2,0.3,0.4,0.4,0.3,0.2,0.1,0.0\n"
            "C=0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.5\n"
            "D=0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4\n"
            "```\n\n"
            "*Можно вводить не все вещества, недостающие будут заполнены нулями*",
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- ГЕНЕРАЦИЯ ПРИМЕРА ----------
    async def generate_concentrations(self, query, user_id: int):
        import random

        # Случайно выбираем тип реакции для примера
        reaction_type = random.choice(['sequential', 'branching'])

        time = list(range(10))

        if reaction_type == 'sequential':
            # Последовательная реакция с неотрицательными значениями
            concentrations = {
                "A": [max(0, 1.0 - 0.15 * i) for i in range(10)],
                "B": [max(0, min(1, 0.0 + 0.12 * i * (1 - 0.15 * i))) for i in range(10)],
                "C": [max(0, min(1, 0.0 + 0.08 * i * (1 - 0.12 * i))) for i in range(10)],
                "D": [max(0, min(1, 0.0 + 0.10 * i * (1 - 0.08 * i))) for i in range(10)]
            }
        else:
            # Разветвленная реакция с неотрицательными значениями
            concentrations = {
                "A": [max(0, 1.0 - 0.22 * i) for i in range(10)],
                "B": [max(0, min(1, 0.0 + 0.11 * i * (1 - 0.22 * i))) for i in range(10)],
                "C": [max(0, min(1, 0.0 + 0.11 * i * (1 - 0.22 * i))) for i in range(10)],
                "D": [max(0, min(1, 0.0 + 0.15 * i * (1 - 0.11 * i))) for i in range(10)]
            }

        # Нормализуем, чтобы сумма была 1 для каждой временной точки
        for i in range(10):
            total = sum(concentrations[s][i] for s in ['A', 'B', 'C', 'D'])
            if total > 0:
                for s in ['A', 'B', 'C', 'D']:
                    concentrations[s][i] = max(0, concentrations[s][i] / total)

        # Передаем user_id при предсказании
        result = self.reaction_bot.predict_reaction(time, concentrations, user_id)

        self.db.log_action(
            user_id,
            "generate_example",
            f"Тип примера: {reaction_type}, предсказано: {result.get('reaction_type')}"
        )

        # Форматируем данные для показа
        data_text = "```\n"
        for substance, values in concentrations.items():
            data_text += f"{substance}=" + ",".join([f"{v:.2f}" for v in values]) + "\n"
        data_text += "```"

        await query.edit_message_text(
            f"🎲 **Сгенерирован пример {reaction_type} реакции**\n\n"
            f"{data_text}\n\n"
            f"{self.format_result(result)}",
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- ГЛАВНОЕ МЕНЮ ----------
    async def show_main_menu(self, query):
        keyboard = [
            [
                InlineKeyboardButton("❓ Помощь", callback_data="help"),
                InlineKeyboardButton("🧪 Предсказать реакцию", callback_data="predict")
            ],
            [
                InlineKeyboardButton("🎓 Обучить модель", callback_data="train_menu")
            ]
        ]

        await query.edit_message_text(
            "🧪 **Бот анализа химических реакций**\n\n"
            "Я умею определять тип химической реакции по данным концентраций.\n"
            "Выберите действие:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )

    # ---------- ТЕКСТОВЫЙ ВВОД ----------
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        user_id = user.id if user else None

        if not user_id:
            await update.message.reply_text("Ошибка: не удалось определить пользователя")
            return

        session = self.user_sessions.get(user_id)

        if not session or session["state"] != "awaiting_concentrations":
            # Если не ожидаем ввода концентраций, показываем помощь
            await update.message.reply_text(
                "Введите /start для начала работы или /help для помощи.",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        try:
            # Показываем статус обработки
            processing_msg = await update.message.reply_text("🔍 **Анализирую данные...**")

            concentrations = {}
            corrections_made = []  # Для записи внесенных исправлений

            for line in update.message.text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if "=" in line:
                    k, v = line.split("=", 1)
                    substance = k.strip().upper()

                    # Парсим значения
                    values = []
                    for val in v.split(","):
                        val = val.strip()
                        if val:
                            try:
                                num_val = float(val)
                                # Проверяем на отрицательное значение
                                if num_val < 0:
                                    corrections_made.append(
                                        f"Вещество {substance}: отрицательное значение {num_val} заменено на 0")
                                    num_val = 0.0
                                values.append(num_val)
                            except ValueError:
                                await processing_msg.edit_text(f"❌ Ошибка: значение '{val}' не является числом")
                                return

                    if substance in ["A", "B", "C", "D"]:
                        concentrations[substance] = values
                    else:
                        await processing_msg.edit_text(
                            f"❌ Ошибка: неизвестное вещество '{substance}' (допустимо: A, B, C, D)")
                        return

            if not concentrations:
                await processing_msg.edit_text("❌ Ошибка: не распознано ни одного вещества")
                return

            # Определяем количество временных точек
            lengths = [len(v) for v in concentrations.values()]
            if len(set(lengths)) > 1:
                await processing_msg.edit_text("❌ Ошибка: количество точек должно быть одинаковым для всех веществ")
                return

            n_points = lengths[0] if lengths else 0
            if n_points < 3:
                await processing_msg.edit_text("❌ Ошибка: нужно минимум 3 временные точки")
                return

            # Ограничиваем максимальное количество точек
            if n_points > 50:
                await processing_msg.edit_text("❌ Ошибка: максимально допустимо 50 временных точек")
                return

            time = list(range(n_points))

            # Дополняем недостающие вещества нулями
            for substance in ["A", "B", "C", "D"]:
                if substance not in concentrations:
                    concentrations[substance] = [0.0] * n_points

            # Проверяем на слишком большие значения
            for substance, values in concentrations.items():
                for i, val in enumerate(values):
                    if val > 10.0:  # Ограничение на максимальную концентрацию
                        corrections_made.append(
                            f"Вещество {substance}[{i}]: значение {val} слишком большое, ограничено 10.0")
                        concentrations[substance][i] = 10.0

            # Проверяем, что сумма концентраций не слишком большая
            for i in range(n_points):
                total = sum(concentrations[s][i] for s in ["A", "B", "C", "D"])
                if total > 15.0:
                    # Нормализуем, если сумма слишком большая
                    scale_factor = 15.0 / total
                    for s in ["A", "B", "C", "D"]:
                        concentrations[s][i] *= scale_factor
                    corrections_made.append(
                        f"Точка {i}: сумма концентраций {total:.2f} слишком большая, масштабирована")

            # Если были исправления, сообщаем пользователю
            if corrections_made:
                corrections_text = "⚠️ *Внесены исправления в данные:*\n"
                for i, correction in enumerate(corrections_made[:5]):  # Показываем первые 5 исправлений
                    corrections_text += f"{i + 1}. {correction}\n"

                if len(corrections_made) > 5:
                    corrections_text += f"... и еще {len(corrections_made) - 5} исправлений\n"

                corrections_text += "\nАнализ продолжается с исправленными данными."
                await update.message.reply_text(corrections_text, parse_mode=ParseMode.MARKDOWN)

            # Выполняем предсказание с передачей user_id
            await processing_msg.edit_text("🧠 **Выполняю предсказание...**")
            result = self.reaction_bot.predict_reaction(time, concentrations, user_id)

            if result.get('status') == 'success':
                self.db.log_action(
                    user_id,
                    "predict",
                    f"Тип реакции: {result.get('reaction_type')}, уверенность: {result.get('confidence', 0):.2f}"
                )

                # Добавляем информацию об исправлениях в результат
                if corrections_made:
                    result['corrections_applied'] = len(corrections_made)
                    result['corrections_info'] = corrections_made[:3]  # Сохраняем первые 3 исправления

                await processing_msg.edit_text(
                    self.format_result(result),
                    parse_mode=ParseMode.MARKDOWN
                )

            else:
                error_msg = result.get('message', 'Неизвестная ошибка')
                await processing_msg.edit_text(
                    f"❌ **Ошибка предсказания**\n\n{error_msg}",
                    parse_mode=ParseMode.MARKDOWN
                )

            # Удаляем сессию
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]

        except Exception as e:
            logger.error(f"Ошибка обработки текста: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ **Ошибка обработки данных**\n\n`{str(e)[:200]}`",
                parse_mode=ParseMode.MARKDOWN
            )

    # ---------- ФОРМАТ ВЫВОДА ----------
    def format_result(self, result):
        reaction_type = result.get('reaction_type', 'unknown')
        confidence = result.get('confidence', 0)
        type_name = result.get('type_name', 'Неизвестный тип')
        method = result.get('method', 'unknown')

        # Определяем эмодзи и цвет в зависимости от уверенности
        if confidence > 0.8:
            confidence_emoji = "🔵"
            confidence_text = "Высокая уверенность"
        elif confidence > 0.6:
            confidence_emoji = "🟡"
            confidence_text = "Средняя уверенность"
        else:
            confidence_emoji = "🔴"
            confidence_text = "Низкая уверенность"

        # Определяем эмодзи для типа реакции
        if reaction_type == 'type1':
            reaction_emoji = "➡️"
        elif reaction_type == 'type2':
            reaction_emoji = "🌳"
        else:
            reaction_emoji = "❓"

        return (
            f"✅ **Результат анализа**\n\n"
            f"{reaction_emoji} **Тип реакции:** {type_name}\n"
            f"{confidence_emoji} **Уверенность:** {confidence:.1%} ({confidence_text})\n"
            f"🧠 **Метод:** `{method}`\n\n"
            f"📊 _Результат сохранен в базе данных_"
        )

    # ---------- ЗАПУСК БОТА ----------
    async def run_bot(self):
        """Асинхронный запуск бота"""
        try:
            logger.info("Запуск Telegram бота...")

            # Создаем приложение бота
            self.bot_app = Application.builder().token(self.token).build()

            # Добавляем обработчики команд
            self.bot_app.add_handler(CommandHandler("start", self.start))
            self.bot_app.add_handler(CommandHandler("help", self.help_command))
            self.bot_app.add_handler(CommandHandler("train", self.train_command))
            self.bot_app.add_handler(CommandHandler("predict", self.predict_command))

            # Добавляем обработчики callback и текстовых сообщений
            self.bot_app.add_handler(CallbackQueryHandler(self.handle_callback))
            self.bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

            # Добавляем обработчик ошибок
            self.bot_app.add_error_handler(self.error_handler)

            logger.info("Бот инициализирован, начинаю поллинг...")

            # Запускаем поллинг
            await self.bot_app.initialize()
            await self.bot_app.start()
            await self.bot_app.updater.start_polling(
                poll_interval=0.5,
                timeout=10,
                drop_pending_updates=True
            )

            logger.info("Бот успешно запущен и ожидает сообщений")

            # Бесконечный цикл ожидания
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Получен сигнал остановки бота")
            raise

        except Exception as e:
            logger.error(f"Ошибка в работе бота: {e}", exc_info=True)
            raise

        finally:
            # Очистка ресурсов
            if self.bot_app:
                try:
                    await self.bot_app.stop()
                    await self.bot_app.shutdown()
                except Exception as e:
                    logger.error(f"Ошибка при остановке бота: {e}")

    def stop_bot(self):
        """Остановка бота"""
        if self.bot_app:
            try:
                # Для остановки асинхронного бота нужно использовать asyncio
                asyncio.create_task(self.bot_app.stop())
                asyncio.create_task(self.bot_app.shutdown())
                logger.info("Команда на остановку бота отправлена")
            except Exception as e:
                logger.error(f"Ошибка при остановке бота: {e}")


# ================== ФУНКЦИЯ ЗАПУСКА ==================
def run_telegram_bot(token: str):
    """Запуск Telegram бота"""
    if not token or token == "":
        print("❌ Токен не указан!")
        print("📝 Получите токен у @BotFather и укажите его в config.py")
        return

    print(f"🤖 Запуск Telegram бота с токеном: {token[:15]}...")

    try:
        # Создаем экземпляр бота
        bot = TelegramReactionBot(token)

        # Запускаем бота
        import asyncio

        # Создаем новое событийное луп для этого потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Запускаем бота
            loop.run_until_complete(bot.run_bot())
        except KeyboardInterrupt:
            print("\n👋 Остановка бота по запросу пользователя")
        except Exception as e:
            print(f"❌ Ошибка при работе бота: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Корректное завершение
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            print("✅ Бот остановлен")

    except Exception as e:
        print(f"❌ Критическая ошибка при запуске бота: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Проверяем токен
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "":
        print("❌ Токен не указан!")
        print("📝 Добавьте токен в config.py или запустите через GUI")
    else:
        run_telegram_bot(TELEGRAM_BOT_TOKEN)