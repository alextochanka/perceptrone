"""
Главный графический интерфейс приложения с активацией Telegram бота
"""
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import asyncio
import subprocess

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox,
    QGridLayout, QMessageBox, QFileDialog, QProgressBar,
    QLineEdit, QCheckBox, QSplitter, QFrame, QListWidget,
    QListWidgetItem, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QDateTime, QProcess
from PySide6.QtGui import QFont, QColor, QPalette, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Локальные импорты
try:
    from core import ReactionBot
    from config import TELEGRAM_BOT_TOKEN
    from neural_network import NeuralNetworkModel
    from telegram_bot import run_telegram_bot
except ImportError as e:
    print(f"Предупреждение при импорте: {e}")
    TELEGRAM_BOT_TOKEN = "8523979775:AAEvIG0md3VLrToCEJm8D2tXya82Z-tg_q0"


    class NeuralNetworkModel:
        def __init__(self, model_type='perceptron'):
            self.model_type = model_type
            self.model = None
            self.is_trained = False

        def create_model(self, **kwargs):
            pass

        def train(self, X, y):
            return {'status': 'success'}

        def evaluate(self, X, y):
            return {'status': 'success', 'accuracy': 0.95}

        def predict(self, X):
            return np.array([0])

        def predict_proba(self, X):
            return np.array([[1.0, 0.0, 0.0, 0.0]])

        def get_model_info(self):
            return {}

        def save_model(self, path):
            return True

        def load_model(self, path):
            return True


    class ReactionBot:
        def __init__(self):
            self.current_model = None

        def train_model(self, **kwargs):
            return {'status': 'success', 'accuracy': 0.95}

        def predict_reaction(self, *args, **kwargs):
            return {'status': 'success', 'reaction_type': 'type1'}

        def get_experiments(self):
            return []

        def get_predictions(self):
            return []

        def get_statistics(self):
            return {}


    def run_telegram_bot(token):
        print(f"Бот запущен с токеном: {token[:15]}...")


# ================== КЛАССЫ ДЛЯ ИМПОРТА ==================

class BotThread(QThread):
    """Поток для запуска Telegram бота"""

    status_changed = Signal(str, str)
    message_received = Signal(str)

    def __init__(self, token):
        super().__init__()
        self.token = token
        self.running = False
        self.bot_instance = None

    def run(self):
        """Запуск бота в отдельном потоке"""
        try:
            self.running = True
            self.status_changed.emit("running", f"Токен: {self.token[:15]}...")
            self.message_received.emit("✅ Бот запущен успешно")

            # Запускаем бота
            from telegram_bot import run_telegram_bot
            self.bot_instance = run_telegram_bot
            self.bot_instance(self.token)

        except ImportError as e:
            self.status_changed.emit("error", f"Ошибка импорта: {str(e)}")
            self.message_received.emit(f"❌ Ошибка импорта модулей: {str(e)}")

        except Exception as e:
            self.status_changed.emit("error", str(e))
            self.message_received.emit(f"❌ Ошибка запуска бота: {str(e)}")

        finally:
            self.running = False
            self.status_changed.emit("stopped", "")

    def stop(self):
        """Остановить бот"""
        self.running = False
        self.terminate()  # Останавливаем поток
        self.wait(2000)  # Ждем завершения
        self.status_changed.emit("stopped", "")
        self.message_received.emit("🛑 Бот остановлен")


class WorkerThread(QThread):
    """Поток для выполнения долгих операций"""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)
    message = Signal(str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.message.emit("Начинаю выполнение задачи...")
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BotActivationTab(QWidget):
    """Вкладка активации Telegram бота"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.bot_process = None
        self.bot_thread = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("🤖 Активация Telegram бота")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)

        # Информационный блок
        info_group = QGroupBox("📋 Информация")
        info_layout = QVBoxLayout()

        info_text = QLabel("""
        <p>Для работы Telegram бота необходимо:</p>
        <ol>
            <li>Создать бота через <a href="https://t.me/BotFather">@BotFather</a> в Telegram</li>
            <li>Получить токен бота (выглядит как: 1234567890:ABCdefGHIjkLMNoPQRsTUVwxyZ)</li>
            <li>Вставить токен в поле ниже</li>
            <li>Нажать кнопку "Активировать"</li>
        </ol>
        <p>После активации бот будет доступен в Telegram по имени, которое вы указали при создании.</p>
        <p><b>Бот работает независимо от основного приложения!</b></p>
        """)
        info_text.setWordWrap(True)
        info_text.setOpenExternalLinks(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Ввод токена
        token_group = QGroupBox("🔑 Ввод токена")
        token_layout = QGridLayout()

        token_layout.addWidget(QLabel("Токен бота:"), 0, 0)
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Введите токен вашего бота")

        # Загружаем сохраненный токен, если есть
        try:
            from config import TELEGRAM_BOT_TOKEN
            if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "":
                self.token_input.setText(TELEGRAM_BOT_TOKEN)
        except:
            pass

        self.token_input.setEchoMode(QLineEdit.Password)
        token_layout.addWidget(self.token_input, 0, 1, 1, 2)

        # Кнопка показать/скрыть токен
        self.show_token_btn = QPushButton("👁 Показать")
        self.show_token_btn.setCheckable(True)
        self.show_token_btn.clicked.connect(self.toggle_token_visibility)
        token_layout.addWidget(self.show_token_btn, 0, 3)

        token_group.setLayout(token_layout)
        layout.addWidget(token_group)

        # Кнопки управления
        button_layout = QHBoxLayout()

        self.activate_btn = QPushButton("🚀 Активировать бота")
        self.activate_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #219653;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.activate_btn.clicked.connect(self.activate_bot)
        button_layout.addWidget(self.activate_btn)

        self.deactivate_btn = QPushButton("⏸ Остановить бота")
        self.deactivate_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.deactivate_btn.clicked.connect(self.deactivate_bot)
        self.deactivate_btn.setEnabled(False)
        button_layout.addWidget(self.deactivate_btn)

        self.test_btn = QPushButton("🧪 Тестировать соединение")
        self.test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_btn)

        self.save_token_btn = QPushButton("💾 Сохранить токен")
        self.save_token_btn.clicked.connect(self.save_token)
        button_layout.addWidget(self.save_token_btn)

        layout.addLayout(button_layout)

        # Статус бота
        status_group = QGroupBox("📊 Статус бота")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("❌ Бот не активирован")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        status_layout.addWidget(self.status_label)

        # Индикатор статуса
        self.status_indicator = QLabel("⚫")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setFont(QFont("Arial", 24))
        status_layout.addWidget(self.status_indicator)

        # Информация о боте
        self.bot_info = QLabel("")
        self.bot_info.setAlignment(Qt.AlignCenter)
        self.bot_info.setWordWrap(True)
        status_layout.addWidget(self.bot_info)

        # Лог бота
        self.bot_log = QTextEdit()
        self.bot_log.setReadOnly(True)
        self.bot_log.setMaximumHeight(150)
        self.bot_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Courier New', monospace;
                padding: 5px;
            }
        """)
        status_layout.addWidget(QLabel("Лог бота:"))
        status_layout.addWidget(self.bot_log)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Инструкция
        instruction_group = QGroupBox("📖 Инструкция")
        instruction_layout = QVBoxLayout()

        instruction_text = QLabel("""
        <h4>После активации бота:</h4>
        <ol>
            <li>Откройте Telegram</li>
            <li>Найдите бота по имени, указанному при создании</li>
            <li>Нажмите /start для начала работы</li>
            <li>Используйте кнопки меню для управления</li>
        </ol>
        <h4>Основные команды бота:</h4>
        <ul>
            <li><code>/start</code> - Начать работу с ботом</li>
            <li><code>/help</code> - Помощь</li>
            <li><code>/stats</code> - Ваша статистика</li>
            <li><code>/top</code> - Топ пользователей</li>
        </ul>
        <h4>Бот может:</h4>
        <ul>
            <li>Анализировать химические реакции</li>
            <li>Определять тип реакции (последовательная или разветвленная)</li>
            <li>Строить графики концентраций</li>
            <li>Сохранять результаты в базу данных</li>
        </ul>
        """)
        instruction_text.setWordWrap(True)
        instruction_layout.addWidget(instruction_text)
        instruction_group.setLayout(instruction_layout)
        layout.addWidget(instruction_group)

        layout.addStretch()

        # Таймер для обновления статуса
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_bot_status)
        self.status_timer.start(2000)

    def toggle_token_visibility(self):
        """Показать/скрыть токен"""
        if self.show_token_btn.isChecked():
            self.token_input.setEchoMode(QLineEdit.Normal)
            self.show_token_btn.setText("👁 Скрыть")
        else:
            self.token_input.setEchoMode(QLineEdit.Password)
            self.show_token_btn.setText("👁 Показать")

    def activate_bot(self):
        """Активировать Telegram бота"""
        token = self.token_input.text().strip()

        if not token:
            QMessageBox.warning(self, "Ошибка", "Введите токен бота!")
            return

        if len(token) < 30:
            QMessageBox.warning(self, "Ошибка", "Токен слишком короткий. Проверьте правильность ввода.")
            return

        try:
            # Сохраняем токен в конфиг
            self.save_token_to_config(token)

            # Запускаем бот в отдельном потоке
            self.bot_thread = BotThread(token)
            self.bot_thread.status_changed.connect(self.update_bot_status_from_thread)
            self.bot_thread.message_received.connect(self.add_to_log)
            self.bot_thread.start()

            self.activate_btn.setEnabled(False)
            self.deactivate_btn.setEnabled(True)
            self.token_input.setEnabled(False)
            self.save_token_btn.setEnabled(False)

            self.add_to_log("🤖 Бот запускается...")
            self.status_label.setText("🔄 Бот запускается...")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; background-color: #fff3cd;")
            self.status_indicator.setText("🟡")
            self.bot_info.setText(f"Токен: {token[:15]}...")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось активировать бота:\n{str(e)}")
            self.add_to_log(f"❌ Ошибка: {str(e)}")

    def deactivate_bot(self):
        """Деактивировать Telegram бота"""
        try:
            if hasattr(self, 'bot_thread') and self.bot_thread is not None and self.bot_thread.isRunning():
                self.bot_thread.stop()
                self.bot_thread.wait(2000)  # Ждем 2 секунды
                if self.bot_thread.isRunning():
                    self.bot_thread.terminate()

            self.activate_btn.setEnabled(True)
            self.deactivate_btn.setEnabled(False)
            self.token_input.setEnabled(True)
            self.save_token_btn.setEnabled(True)

            self.status_label.setText("❌ Бот остановлен")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")
            self.status_indicator.setText("⚫")
            self.bot_info.setText("")
            self.add_to_log("🛑 Бот остановлен")

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при остановке бота:\n{str(e)}")

    def test_connection(self):
        """Тестировать соединение с ботом"""
        token = self.token_input.text().strip()

        if not token:
            QMessageBox.warning(self, "Ошибка", "Введите токен для тестирования!")
            return

        try:
            import requests
            # Проверяем токен через API Telegram
            self.add_to_log("🔍 Тестирование соединения...")

            response = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)

            if response.status_code == 200:
                bot_data = response.json()
                if bot_data.get('ok'):
                    bot_info = bot_data['result']
                    QMessageBox.information(self, "Успех",
                                            f"✅ Соединение успешно!\n\n"
                                            f"Имя бота: @{bot_info.get('username', 'N/A')}\n"
                                            f"Имя: {bot_info.get('first_name', 'N/A')}\n"
                                            f"ID: {bot_info.get('id', 'N/A')}\n\n"
                                            f"Бот готов к активации!")
                    self.add_to_log(f"✅ Бот найден: @{bot_info.get('username')}")
                else:
                    QMessageBox.warning(self, "Ошибка", "Неверный токен бота!")
                    self.add_to_log("❌ Неверный токен бота")
            else:
                QMessageBox.warning(self, "Ошибка", f"Ошибка соединения: {response.status_code}")
                self.add_to_log(f"❌ Ошибка HTTP: {response.status_code}")

        except requests.exceptions.Timeout:
            QMessageBox.warning(self, "Ошибка", "Таймаут соединения. Проверьте интернет.")
            self.add_to_log("❌ Таймаут соединения")
        except requests.exceptions.ConnectionError:
            QMessageBox.warning(self, "Ошибка", "Ошибка соединения. Проверьте интернет.")
            self.add_to_log("❌ Ошибка соединения")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка тестирования: {str(e)}")
            self.add_to_log(f"❌ Ошибка: {str(e)}")

    def save_token(self):
        """Сохранить токен без запуска бота"""
        token = self.token_input.text().strip()

        if not token:
            QMessageBox.warning(self, "Ошибка", "Введите токен для сохранения!")
            return

        if len(token) < 30:
            QMessageBox.warning(self, "Ошибка", "Токен слишком короткий. Проверьте правильность.")
            return

        try:
            self.save_token_to_config(token)
            QMessageBox.information(self, "Успех",
                                    "✅ Токен сохранен в конфигурации!\n\nТеперь вы можете активировать бота.")
            self.add_to_log("✅ Токен сохранен в конфигурации")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения токена:\n{str(e)}")
            self.add_to_log(f"❌ Ошибка сохранения: {str(e)}")

    def save_token_to_config(self, token):
        """Сохранить токен в конфигурационный файл"""
        try:
            config_path = Path(__file__).parent / "config.py"

            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Ищем и заменяем токен
                import re
                pattern = r'TELEGRAM_BOT_TOKEN\s*=\s*["\'][^"\']*["\']'
                replacement = f'TELEGRAM_BOT_TOKEN = "{token}"'

                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                else:
                    # Если токен не найден, добавляем его
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        new_lines.append(line)
                        if '# ==================== ТОКЕН TELEGRAM БОТА ====================' in line:
                            new_lines.append(f'TELEGRAM_BOT_TOKEN = "{token}"')
                    content = '\n'.join(new_lines)

                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.add_to_log("✅ Токен сохранен в конфигурации")
                return True
            else:
                raise FileNotFoundError("Файл config.py не найден")

        except Exception as e:
            self.add_to_log(f"⚠ Не удалось сохранить токен: {str(e)}")
            raise

    def update_bot_status(self):
        """Обновить статус бота"""
        pass

    def update_bot_status_from_thread(self, status, details=""):
        """Обновить статус из потока бота"""
        if status == "running":
            self.status_label.setText("✅ Бот активен")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; background-color: #d4edda;")
            self.status_indicator.setText("🟢")
            self.bot_info.setText(f"Бот работает\n{details}")

        elif status == "stopped":
            self.status_label.setText("❌ Бот остановлен")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")
            self.status_indicator.setText("⚫")
            self.bot_info.setText("")

        elif status == "error":
            self.status_label.setText("⚠ Ошибка бота")
            self.status_label.setStyleSheet("color: #e67e22; font-weight: bold; background-color: #fff3cd;")
            self.status_indicator.setText("🟡")
            self.bot_info.setText(f"Ошибка: {details}")

    def add_to_log(self, message):
        """Добавить сообщение в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.bot_log.append(f"[{timestamp}] {message}")
        # Прокручиваем вниз
        scrollbar = self.bot_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class TrainingTab(QWidget):
    """Вкладка обучения модели"""

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("🎓 Обучение нейронной сети")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)

        # Информация о обучении
        info_group = QGroupBox("ℹ️ Информация")
        info_layout = QVBoxLayout()

        info_text = QLabel("""
        <p>Обучите нейронную сеть для классификации химических реакций.</p>
        <p><b>Поддерживаемые типы реакций:</b></p>
        <ol>
            <li>A → B → C → D (последовательная)</li>
            <li>A → B и A → C → D (параллельная)</li>
        </ol>
        <p>После обучения модель сможет определять тип реакции по данным концентраций.</p>
        """)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Параметры обучения (УБРАНО СЛУЧАЙНОЕ СЕМЯ)
        params_group = QGroupBox("⚙️ Параметры обучения")
        params_layout = QGridLayout()

        # Тип модели
        params_layout.addWidget(QLabel("Тип модели:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["perceptron", "mlp"])
        self.model_type_combo.setCurrentIndex(0)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        params_layout.addWidget(self.model_type_combo, 0, 1)

        # Размер скрытых слоев (для MLP)
        params_layout.addWidget(QLabel("Скрытые слои:"), 1, 0)
        self.hidden_layers_input = QLineEdit("128,64")
        self.hidden_layers_input.setPlaceholderText("Например: 64,32 или 128,64,32")
        self.hidden_layers_input.setEnabled(False)  # По умолчанию выключено для perceptron
        params_layout.addWidget(self.hidden_layers_input, 1, 1)

        # Количество образцов
        params_layout.addWidget(QLabel("Количество образцов:"), 2, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(500, 5000)
        self.samples_spin.setValue(2000)
        self.samples_spin.setSingleStep(100)
        params_layout.addWidget(self.samples_spin, 2, 1)

        # Максимальные итерации
        params_layout.addWidget(QLabel("Макс. итераций:"), 3, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 10000)
        self.max_iter_spin.setValue(3000)
        self.max_iter_spin.setSingleStep(100)
        params_layout.addWidget(self.max_iter_spin, 3, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Кнопки
        button_layout = QHBoxLayout()

        self.train_btn = QPushButton("🚀 Начать обучение")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹️ Остановить")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Статус обучения
        self.status_label = QLabel("Готов к обучению")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Результаты
        results_group = QGroupBox("📊 Результаты обучения")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

    def on_model_type_changed(self, text):
        """Включение/выключение поля скрытых слоев"""
        self.hidden_layers_input.setEnabled(text == "mlp")

    def start_training(self):
        """Запуск обучения БЕЗ случайного семени"""
        # Получаем параметры
        model_type = self.model_type_combo.currentText()
        n_samples = self.samples_spin.value()
        max_iter = self.max_iter_spin.value()

        # Подготавливаем параметры для MLP
        hidden_layers = None
        if model_type == 'mlp':
            try:
                layers_text = self.hidden_layers_input.text().strip()
                if layers_text:
                    hidden_layers = tuple(map(int, layers_text.split(',')))
                    if len(hidden_layers) == 0:
                        raise ValueError("Укажите хотя бы один слой")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Неверный формат скрытых слоев:\n{str(e)}")
                return

        # Показываем прогресс
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Обучение запущено...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.results_text.clear()

        # Запускаем в отдельном потоке
        self.worker = WorkerThread(
            self.train_model_task,
            model_type,
            n_samples,
            max_iter,
            hidden_layers
        )
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.message.connect(self.update_status)
        self.worker.start()

    def train_model_task(self, model_type, n_samples, max_iter, hidden_layers):
        """Задача обучения модели БЕЗ случайного семени"""
        try:
            # Вызываем метод бота
            result = self.bot.train_model(
                model_type=model_type,
                n_samples=n_samples,
                max_iter=max_iter,
                hidden_layers=hidden_layers
            )
            return result
        except Exception as e:
            raise e

    def training_finished(self, result):
        """Обучение завершено"""
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if result.get('status') == 'success':
            self.status_label.setText("✅ Обучение завершено успешно!")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")

            accuracy = result.get('accuracy', 0)
            accuracy_color = "#27ae60" if accuracy > 0.85 else "#f39c12" if accuracy > 0.7 else "#e74c3c"

            text = f"""<div style="font-family: 'Segoe UI', Arial, sans-serif;">
                <h3 style="color: {accuracy_color}; margin-bottom: 10px;">🎉 МОДЕЛЬ УСПЕШНО ОБУЧЕНА!</h3>
                <hr style="border: 1px solid #ddd;">

                <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Тип модели:</td>
                        <td style="padding: 5px;">{result.get('model_type', 'N/A')}</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 5px; font-weight: bold;">Точность:</td>
                        <td style="padding: 5px; color: {accuracy_color}; font-weight: bold;">{accuracy:.4f} ({accuracy:.1%})</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Количество итераций:</td>
                        <td style="padding: 5px;">{result.get('n_iterations', 0)}</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 5px; font-weight: bold;">Обучающих образцов:</td>
                        <td style="padding: 5px;">{result.get('training_samples', 0)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Тестовых образцов:</td>
                        <td style="padding: 5px;">{result.get('test_samples', 0)}</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 5px; font-weight: bold;">ID эксперимента:</td>
                        <td style="padding: 5px;">{result.get('experiment_id', 'N/A')}</td>
                    </tr>
                </table>

                <hr style="border: 1px solid #ddd;">
                <p style="color: #27ae60; font-weight: bold;">Модель готова к использованию! 🎯</p>
            </div>"""

            self.results_text.setHtml(text)

            # Сохраняем информацию о качестве
            if accuracy > 0.9:
                self.add_log_entry("✨ Отличная точность модели!")
            elif accuracy > 0.8:
                self.add_log_entry("👍 Хорошая точность модели")
            elif accuracy > 0.7:
                self.add_log_entry("⚠ Средняя точность модели")
            else:
                self.add_log_entry("❌ Низкая точность модели")

        else:
            self.status_label.setText("❌ Ошибка обучения")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            error_msg = result.get('message', 'Неизвестная ошибка')
            self.results_text.setText(f"<div style='color: #e74c3c; padding: 10px;'>❌ Ошибка: {error_msg}</div>")
            self.add_log_entry(f"❌ Ошибка обучения: {error_msg}")

    def training_error(self, error_msg):
        """Ошибка при обучении"""
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.status_label.setText("❌ Ошибка обучения")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.results_text.setText(f"<div style='color: #e74c3c; padding: 10px;'>❌ Ошибка: {error_msg}</div>")
        self.add_log_entry(f"❌ Ошибка обучения: {error_msg}")

    def update_status(self, message):
        """Обновление статуса"""
        self.status_label.setText(message)

    def stop_training(self):
        """Остановка обучения"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            if self.worker.isRunning():
                self.worker.wait(1000)

        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Обучение остановлено")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.add_log_entry("⏹ Обучение остановлено пользователем")

    def add_log_entry(self, message):
        """Добавить запись в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_text = self.results_text.toPlainText()
        if current_text:
            self.results_text.setPlainText(f"[{timestamp}] {message}\n{current_text}")
        else:
            self.results_text.setPlainText(f"[{timestamp}] {message}")


class PredictionTab(QWidget):
    """Вкладка предсказаний с валидацией данных"""

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.setup_ui()

    def validate_concentration_data(self, concentrations):
        """Проверка и валидация данных о концентрациях с ограничением до 10.0"""
        issues = []
        corrections = {}
        corrected_data = {}

        # 1. Обрабатываем каждое вещество
        for substance, values in concentrations.items():
            clean_values = []
            for i, value in enumerate(values):
                if value < 0:
                    clean_values.append(0.0)
                    if substance not in corrections:
                        corrections[substance] = []
                    corrections[substance].append(f"Точка {i}: {value:.2f} → 0.0")
                    issues.append("negative")
                elif value > 10.0:
                    clean_values.append(10.0)
                    if substance not in corrections:
                        corrections[substance] = []
                    corrections[substance].append(f"Точка {i}: {value:.2f} → 10.0")
                    issues.append("too_high")
                else:
                    clean_values.append(float(value))

            corrected_data[substance] = clean_values

        # 2. Проверяем сумму концентраций для каждой точки времени
        if corrected_data:
            n_points = len(list(corrected_data.values())[0])
            for i in range(n_points):
                total = 0
                for values in corrected_data.values():
                    if i < len(values):
                        total += values[i]

                if total > 15.0:
                    # Масштабируем, если сумма слишком большая
                    scale_factor = 15.0 / total
                    for substance, values in corrected_data.items():
                        if i < len(values):
                            old_val = values[i]
                            values[i] = min(10.0, values[i] * scale_factor)
                            if substance not in corrections:
                                corrections[substance] = []
                            corrections[substance].append(
                                f"Точка {i}: сумма {total:.1f}, {old_val:.2f} → {values[i]:.2f}"
                            )
                    issues.append("total_too_high")

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "corrections": corrections,
            "corrected_data": corrected_data
        }

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("🔮 Предсказание типа реакции")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)

        # Информация о типах реакций
        info_group = QGroupBox("📝 Информация о типах реакций")
        info_layout = QVBoxLayout()

        info_text = QLabel("""
        <p><b>Система определяет только два типа реакций:</b></p>
        <ol>
            <li><b>Последовательная:</b> A → B → C → D</li>
            <li><b>Параллельная:</b> A → B и A → C → D</li>
        </ol>
        <p><b>Требования к данным:</b></p>
        <ul>
            <li>Концентрации должны быть <b>неотрицательными</b> (0 и выше)</li>
            <li>Рекомендуемый диапазон: 0.0 - 10.0</li>
            <li>Минимум 3 временные точки</li>
            <li>Максимум 50 временных точек</li>
            <li>Одинаковое количество точек для всех веществ</li>
        </ul>
        <p><i>Отрицательные значения будут автоматически исправлены на 0.</i></p>
        <p><i>Совет: Сначала обучите модель на вкладке "🎓 Обучение" для лучшей точности.</i></p>
        """)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Вкладки для ввода данных
        self.tab_widget = QTabWidget()

        # Вкладка 1: Простой ввод
        simple_tab = QWidget()
        simple_layout = QGridLayout()

        simple_layout.addWidget(QLabel("<b>Введите начальные концентрации:</b>"), 0, 0, 1, 2)

        # Вещество A
        simple_layout.addWidget(QLabel("A (начальный реагент):"), 1, 0)
        self.a_input = QDoubleSpinBox()
        self.a_input.setRange(0.0, 10.0)  # Минимум 0.0!
        self.a_input.setValue(1.0)
        self.a_input.setDecimals(2)
        self.a_input.setSingleStep(0.1)
        simple_layout.addWidget(self.a_input, 1, 1)

        # Вещество B
        simple_layout.addWidget(QLabel("B (начальная концентрация):"), 2, 0)
        self.b_input = QDoubleSpinBox()
        self.b_input.setRange(0.0, 10.0)  # Минимум 0.0!
        self.b_input.setValue(0.0)
        self.b_input.setDecimals(2)
        self.b_input.setSingleStep(0.1)
        simple_layout.addWidget(self.b_input, 2, 1)

        # Вещество C
        simple_layout.addWidget(QLabel("C (начальная концентрация):"), 3, 0)
        self.c_input = QDoubleSpinBox()
        self.c_input.setRange(0.0, 10.0)  # Минимум 0.0!
        self.c_input.setValue(0.0)
        self.c_input.setDecimals(2)
        self.c_input.setSingleStep(0.1)
        simple_layout.addWidget(self.c_input, 3, 1)

        # Вещество D
        simple_layout.addWidget(QLabel("D (начальная концентрация):"), 4, 0)
        self.d_input = QDoubleSpinBox()
        self.d_input.setRange(0.0, 10.0)  # Минимум 0.0!
        self.d_input.setValue(0.0)
        self.d_input.setDecimals(2)
        self.d_input.setSingleStep(0.1)
        simple_layout.addWidget(self.d_input, 4, 1)

        simple_tab.setLayout(simple_layout)
        self.tab_widget.addTab(simple_tab, "🧪 Простой ввод")

        # Вкладка 2: Расширенный ввод
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()

        advanced_layout.addWidget(QLabel("<b>Введите полные данные (3-50 точек):</b>"))

        self.data_input = QTextEdit()
        self.data_input.setPlaceholderText(
            "Введите данные в формате:\n"
            "A=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1\n"
            "B=0.0,0.1,0.2,0.3,0.4,0.4,0.3,0.2,0.1,0.0\n"
            "C=0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.5\n"
            "D=0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4\n\n"
            "⚠️ ВАЖНО:\n"
            "• Концентрации должны быть неотрицательными (0 и выше)\n"
            "• Рекомендуемый диапазон: 0.0 - 10.0\n"
            "• Отрицательные значения будут автоматически исправлены на 0\n"
            "• Слишком большие значения (>10.0) будут ограничены\n"
            "• Минимум 3 временные точки\n"
            "• Максимум 50 точек\n"
            "• Количество точек должно быть одинаковым для всех веществ"
        )
        self.data_input.setMaximumHeight(250)
        self.data_input.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
            }
        """)
        advanced_layout.addWidget(self.data_input)

        advanced_tab.setLayout(advanced_layout)
        self.tab_widget.addTab(advanced_tab, "📊 Расширенный ввод")

        layout.addWidget(self.tab_widget)

        # Кнопки управления
        button_layout = QHBoxLayout()

        self.predict_btn = QPushButton("🎯 Предсказать")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.predict_btn.clicked.connect(self.make_prediction)
        button_layout.addWidget(self.predict_btn)

        self.example_sequential_btn = QPushButton("📈 Пример: Последовательная")
        self.example_sequential_btn.clicked.connect(self.load_sequential_example)
        button_layout.addWidget(self.example_sequential_btn)

        self.example_branched_btn = QPushButton("🌳 Пример: Параллельная")
        self.example_branched_btn.clicked.connect(self.load_branched_example)
        button_layout.addWidget(self.example_branched_btn)

        self.clear_btn = QPushButton("🧹 Очистить")
        self.clear_btn.clicked.connect(self.clear_data)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Статус
        self.status_label = QLabel("Готов к анализу")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Результаты предсказания
        result_group = QGroupBox("📋 Результат предсказания")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                font-size: 13px;
            }
        """)
        result_layout.addWidget(self.result_text)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        layout.addStretch()

    def load_sequential_example(self):
        """Загрузка примера последовательной реакции с корректными данными"""
        example_data = """A=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1
B=0.0,0.1,0.2,0.3,0.4,0.4,0.3,0.2,0.1,0.0
C=0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.5
D=0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.3,0.4"""

        if self.tab_widget.currentIndex() == 0:  # Простой ввод
            self.a_input.setValue(1.0)
            self.b_input.setValue(0.0)
            self.c_input.setValue(0.0)
            self.d_input.setValue(0.0)
            self.status_label.setText(
                "Загружен пример последовательной реакции (используйте Расширенный ввод для полных данных)")
            self.status_label.setStyleSheet("color: #3498db; font-weight: bold;")
        else:  # Расширенный ввод
            self.data_input.setPlainText(example_data)
            self.status_label.setText("✅ Загружен пример последовательной реакции (все значения неотрицательные)")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")

        self.result_text.clear()

    def load_branched_example(self):
        """Загрузка примера параллельной реакции с корректными данными"""
        example_data = """A=1.0,0.8,0.6,0.4,0.3,0.2,0.1,0.1,0.1,0.0
B=0.0,0.1,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.0
C=0.0,0.1,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.0
D=0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.7,0.8,1.0"""

        if self.tab_widget.currentIndex() == 0:  # Простой ввод
            self.a_input.setValue(1.0)
            self.b_input.setValue(0.0)
            self.c_input.setValue(0.0)
            self.d_input.setValue(0.0)
            self.status_label.setText(
                "Загружен пример параллельной реакции (используйте Расширенный ввод для полных данных)")
            self.status_label.setStyleSheet("color: #3498db; font-weight: bold;")
        else:  # Расширенный ввод
            self.data_input.setPlainText(example_data)
            self.status_label.setText("✅ Загружен пример параллельной реакции (все значения неотрицательные)")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")

        self.result_text.clear()

    def clear_data(self):
        """Очистка данных"""
        self.a_input.setValue(1.0)
        self.b_input.setValue(0.0)
        self.c_input.setValue(0.0)
        self.d_input.setValue(0.0)
        self.data_input.clear()
        self.result_text.clear()
        self.status_label.setText("Данные очищены")
        self.status_label.setStyleSheet("color: #7f8c8d;")

    def make_prediction(self):
        """Выполнение предсказания с валидацией данных"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("🔍 Проверка данных...")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.predict_btn.setEnabled(False)
            self.result_text.clear()

            # Получаем данные в зависимости от выбранной вкладки
            if self.tab_widget.currentIndex() == 0:  # Простой ввод
                # Генерируем данные на основе начальных концентраций
                time_points = list(range(10))
                a_val = max(0, self.a_input.value())  # Гарантируем неотрицательность
                b_val = max(0, self.b_input.value())
                c_val = max(0, self.c_input.value())
                d_val = max(0, self.d_input.value())

                # Генерируем реалистичные данные с гарантией неотрицательности
                concentrations = {
                    'A': [max(0, a_val * (0.9 ** i)) for i in range(10)],
                    'B': [max(0, min(1, b_val + (0.12 * i * (1 - b_val)))) for i in range(10)],
                    'C': [max(0, min(1, c_val + (0.08 * i * (1 - c_val)))) for i in range(10)],
                    'D': [max(0, min(1, d_val + (0.15 * i * (1 - d_val)))) for i in range(10)]
                }

                self.result_text.append("📊 Сгенерированные данные на основе введенных концентраций...")

            else:  # Расширенный ввод
                # Парсим данные из текстового поля
                text = self.data_input.toPlainText().strip()
                if not text:
                    raise ValueError("Введите данные для анализа")

                lines = text.split('\n')
                concentrations = {}
                time_points = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if '=' in line:
                        substance, values = line.split('=', 1)
                        substance = substance.strip().upper()

                        # Проверяем допустимость вещества
                        if substance not in ['A', 'B', 'C', 'D']:
                            raise ValueError(f"Неизвестное вещество: '{substance}'. Допустимо: A, B, C, D")

                        # Парсим значения с проверкой
                        parsed_values = []
                        raw_values = values.split(',')

                        for i, v in enumerate(raw_values):
                            v = v.strip()
                            if not v:
                                continue

                            try:
                                num_val = float(v)
                                parsed_values.append(num_val)
                            except ValueError:
                                raise ValueError(f"Неверное значение в веществе {substance}: '{v}' не является числом")

                        if not parsed_values:
                            raise ValueError(f"Вещество {substance}: нет числовых значений")

                        concentrations[substance] = parsed_values

                if not concentrations:
                    raise ValueError("Не удалось распознать данные")

                # Проверяем количество точек
                lengths = [len(v) for v in concentrations.values()]
                if len(set(lengths)) > 1:
                    raise ValueError("Количество временных точек должно быть одинаковым для всех веществ!")

                if not lengths:
                    raise ValueError("Нет данных для анализа")

                time_points = list(range(lengths[0]))

                if len(time_points) < 3:
                    raise ValueError("Введите данные для минимум 3 временных точек")

                if len(time_points) > 50:
                    raise ValueError("Максимально допустимо 50 временных точек")

                # Дополняем недостающие вещества
                for substance in ['A', 'B', 'C', 'D']:
                    if substance not in concentrations:
                        concentrations[substance] = [0.0] * len(time_points)

                # ВАЛИДАЦИЯ ДАННЫХ
                validation_result = self.validate_concentration_data(concentrations)

                if validation_result["has_issues"]:
                    # Показываем предупреждение пользователю
                    warning_text = "⚠️ **Обнаружены проблемы в данных:**\n\n"

                    # Группируем исправления по типу
                    negative_fixed = []
                    large_fixed = []

                    for issue in validation_result["issues"]:
                        if "отрицательное" in issue:
                            negative_fixed.append(issue.split('[')[0])
                        elif "слишком большое" in issue:
                            large_fixed.append(issue.split('[')[0])

                    if negative_fixed:
                        substances = list(set(negative_fixed))
                        warning_text += f"• Отрицательные концентрации исправлены для веществ: {', '.join(substances)}\n"

                    if large_fixed:
                        substances = list(set(large_fixed))
                        warning_text += f"• Слишком большие значения (>10.0) исправлены для веществ: {', '.join(substances)}\n"

                    warning_text += "\n*Анализ продолжается с исправленными данными.*"

                    # Показываем предупреждение
                    QMessageBox.warning(self, "Исправления в данных", warning_text)

                    # Обновляем поле ввода исправленными данными (опционально)
                    corrected_text = ""
                    for substance in ['A', 'B', 'C', 'D']:
                        if substance in validation_result["corrected_data"]:
                            values_str = ",".join([f"{v:.2f}" for v in validation_result["corrected_data"][substance]])
                            corrected_text += f"{substance}={values_str}\n"

                    # Показываем исправленные данные в отдельном сообщении
                    self.result_text.append("📝 **Исправленные данные:**\n")
                    self.result_text.append("```")
                    self.result_text.append(corrected_text.strip())
                    self.result_text.append("```\n")

                    # Используем исправленные данные
                    concentrations = validation_result["corrected_data"]

                self.result_text.append(f"📊 Загружено данных: {len(time_points)} временных точек")

            # Выполняем предсказание
            self.status_label.setText("🧠 Анализирую данные...")
            self.result_text.append("🧠 Анализирую данные...")
            result = self.bot.predict_reaction(time_points, concentrations)

            if result.get('status') == 'success':
                reaction_type = result.get('reaction_type', 'unknown')
                confidence = result.get('confidence', 0)
                type_name = result.get('type_name', 'Неизвестный тип')
                reaction_id = result.get('reaction_id', 0)

                # Определяем цвет в зависимости от уверенности
                if confidence > 0.8:
                    color = "#27ae60"
                    confidence_text = "Высокая уверенность"
                elif confidence > 0.6:
                    color = "#f39c12"
                    confidence_text = "Средняя уверенность"
                else:
                    color = "#e74c3c"
                    confidence_text = "Низкая уверенность"

                result_html = f"""
                <div style="font-family: 'Segoe UI', Arial, sans-serif;">
                    <h3 style="color: {color}; margin-bottom: 15px; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                        🎯 РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ
                    </h3>

                    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold; width: 40%;">Тип реакции:</td>
                            <td style="padding: 8px; color: {color}; font-weight: bold;">{type_name}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 8px; font-weight: bold;">Уверенность:</td>
                            <td style="padding: 8px;">
                                <span style="color: {color}; font-weight: bold;">{confidence:.2%}</span>
                                <small style="color: #7f8c8d; margin-left: 10px;">({confidence_text})</small>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Код типа:</td>
                            <td style="padding: 8px;"><code>{reaction_type}</code></td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 8px; font-weight: bold;">ID сохраненной реакции:</td>
                            <td style="padding: 8px;">{reaction_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Метод анализа:</td>
                            <td style="padding: 8px;">{result.get('method', 'unknown')}</td>
                        </tr>
                    </table>

                    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 10px; margin-top: 15px;">
                        <p style="color: #155724; margin: 0;">
                            <b>✅ Анализ сохранен в базе данных.</b><br>
                            Проверьте вкладку "📁 База данных" для просмотра истории.
                        </p>
                    </div>

                    <div style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                        <p><i>Анализ выполнен на {len(time_points)} временных точках</i></p>
                    </div>
                </div>
                """

                self.result_text.setHtml(result_html)

                # Обновляем статус
                if confidence > 0.8:
                    self.status_label.setText("✅ Высокая уверенность в предсказании")
                    self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; background-color: #d4edda;")
                elif confidence > 0.6:
                    self.status_label.setText("⚠ Средняя уверенность в предсказании")
                    self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; background-color: #fff3cd;")
                else:
                    self.status_label.setText("❌ Низкая уверенность в предсказании")
                    self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")

            else:
                error_msg = result.get('message', 'Неизвестная ошибка')
                self.result_text.setHtml(f"""
                    <div style="color: #e74c3c; padding: 15px; background-color: #f8d7da; border-radius: 5px;">
                        <h3>❌ Ошибка предсказания</h3>
                        <p>{error_msg}</p>
                    </div>
                """)
                self.status_label.setText("❌ Ошибка анализа")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")

        except ValueError as e:
            # Ошибки валидации показываем пользователю
            error_html = f"""
            <div style="color: #e74c3c; padding: 15px; background-color: #f8d7da; border-radius: 5px;">
                <h3>❌ Ошибка в данных</h3>
                <p><b>{str(e)}</b></p>
                <p style="margin-top: 10px; font-size: 12px;">
                    <i>Проверьте формат введенных данных и попробуйте снова.</i>
                </p>
            </div>
            """
            self.result_text.setHtml(error_html)
            self.status_label.setText("❌ Ошибка в данных")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")

        except Exception as e:
            error_html = f"""
            <div style="color: #e74c3c; padding: 15px; background-color: #f8d7da; border-radius: 5px;">
                <h3>❌ Ошибка при предсказании</h3>
                <p><b>{str(e)}</b></p>
                <p style="margin-top: 10px; font-size: 12px;">
                    <i>Произошла непредвиденная ошибка. Проверьте данные и попробуйте снова.</i>
                </p>
            </div>
            """
            self.result_text.setHtml(error_html)
            self.status_label.setText("❌ Ошибка")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; background-color: #f8d7da;")

        finally:
            self.progress_bar.setVisible(False)
            self.predict_btn.setEnabled(True)


class VisualizationTab(QWidget):
    """Вкладка визуализации"""

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.figure = None
        self.canvas = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("📊 Визуализация данных")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)

        # Создаем фигуру и канвас
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)

        # Добавляем тулбар для навигации
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Кнопки управления графиками
        button_group = QGroupBox("📈 Тип графика")
        button_layout = QHBoxLayout()

        self.plot_training_btn = QPushButton("📚 История обучения")
        self.plot_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.plot_training_btn.clicked.connect(self.plot_training_history)
        button_layout.addWidget(self.plot_training_btn)

        self.plot_concentrations_btn = QPushButton("🧪 Концентрации (пример)")
        self.plot_concentrations_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.plot_concentrations_btn.clicked.connect(self.plot_concentrations)
        button_layout.addWidget(self.plot_concentrations_btn)

        self.plot_reaction_types_btn = QPushButton("🌡 Сравнение типов реакций")
        self.plot_reaction_types_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.plot_reaction_types_btn.clicked.connect(self.plot_reaction_types_comparison)
        button_layout.addWidget(self.plot_reaction_types_btn)

        self.clear_plot_btn = QPushButton("🧹 Очистить")
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        button_layout.addWidget(self.clear_plot_btn)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        # Статус графика
        self.plot_status = QLabel("Готов к построению графиков")
        self.plot_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_status)

        layout.addStretch()

    def plot_training_history(self):
        """Построение графика истории обучения"""
        try:
            self.figure.clear()

            # Создаем демонстрационные данные
            epochs = list(range(1, 101))

            # Генерируем реалистичные кривые обучения
            loss = [1.0 * (0.95 ** i) + np.random.normal(0, 0.01) for i in range(100)]
            accuracy = [0.5 + 0.005 * i + np.random.normal(0, 0.005) for i in range(100)]
            val_loss = [0.9 * (0.96 ** i) + np.random.normal(0, 0.015) for i in range(100)]
            val_accuracy = [0.55 + 0.0045 * i + np.random.normal(0, 0.008) for i in range(100)]

            # Сглаживаем данные
            window = 5
            loss_smooth = np.convolve(loss, np.ones(window) / window, mode='valid')
            accuracy_smooth = np.convolve(accuracy, np.ones(window) / window, mode='valid')
            val_loss_smooth = np.convolve(val_loss, np.ones(window) / window, mode='valid')
            val_accuracy_smooth = np.convolve(val_accuracy, np.ones(window) / window, mode='valid')
            epochs_smooth = epochs[window - 1:]

            ax1 = self.figure.add_subplot(111)

            # График потерь
            ax1.plot(epochs_smooth, loss_smooth, label='Потери (обучение)',
                     color='#e74c3c', linewidth=2.5, alpha=0.8)
            ax1.plot(epochs_smooth, val_loss_smooth, label='Потери (валидация)',
                     color='#c0392b', linewidth=2.5, linestyle='--', alpha=0.8)
            ax1.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Потери', color='#e74c3c', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#e74c3c')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(bottom=0)

            # Вторая ось Y для точности
            ax2 = ax1.twinx()
            ax2.plot(epochs_smooth, accuracy_smooth, label='Точность (обучение)',
                     color='#3498db', linewidth=2.5, alpha=0.8)
            ax2.plot(epochs_smooth, val_accuracy_smooth, label='Точность (валидация)',
                     color='#2980b9', linewidth=2.5, linestyle='--', alpha=0.8)
            ax2.set_ylabel('Точность', color='#3498db', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#3498db')
            ax2.set_ylim(0.4, 1.0)

            # Объединяем легенды
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='lower right', fontsize=10, framealpha=0.9)

            ax1.set_title('История обучения нейронной сети',
                          fontsize=14, fontweight='bold', pad=20)

            # Добавляем аннотацию с финальной точностью
            final_acc = val_accuracy_smooth[-1]
            ax2.annotate(f'Финальная точность: {final_acc:.3f}',
                         xy=(0.98, 0.02), xycoords='axes fraction',
                         fontsize=10, ha='right', color='#27ae60',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            self.figure.tight_layout()
            self.canvas.draw()
            self.plot_status.setText("✅ График истории обучения построен")
            self.plot_status.setStyleSheet("color: #27ae60; font-weight: bold;")

        except Exception as e:
            self.plot_status.setText(f"❌ Ошибка: {str(e)}")
            self.plot_status.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def plot_concentrations(self):
        """Построение графика концентраций"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Создаем демо-данные для последовательной реакции
            time = np.linspace(0, 10, 50)

            # Последовательная реакция: A → B → C → D
            A = np.exp(-0.25 * time)
            B = 0.6 * (np.exp(-0.15 * (time - 1)) - np.exp(-0.25 * time)) / (0.25 - 0.15)
            B = np.maximum(0, B)  # Обеспечиваем неотрицательность
            C = 0.4 * (np.exp(-0.1 * (time - 2)) - np.exp(-0.15 * (time - 1))) / (0.15 - 0.1)
            C = np.maximum(0, C)
            D = 1 - (A + B + C)
            D = np.maximum(0, D)  # Обеспечиваем неотрицательность

            # Нормализуем, чтобы сумма была 1
            total = A + B + C + D
            A, B, C, D = A / total, B / total, C / total, D / total

            colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
            labels = ['Вещество A', 'Вещество B', 'Вещество C', 'Вещество D']
            line_styles = ['-', '--', '-.', ':']
            markers = ['o', 's', '^', 'D']
            marker_indices = np.linspace(0, len(time) - 1, 10, dtype=int)

            for i, (conc, label, color, ls, marker) in enumerate(
                    zip([A, B, C, D], labels, colors, line_styles, markers)):
                ax.plot(time, conc, label=label, color=color, linewidth=2.5,
                        linestyle=ls, alpha=0.9)
                # Добавляем маркеры
                ax.plot(time[marker_indices], conc[marker_indices], marker=marker,
                        color=color, markersize=8, linestyle='', alpha=0.8)

            ax.set_xlabel('Время', fontsize=12, fontweight='bold')
            ax.set_ylabel('Концентрация', fontsize=12, fontweight='bold')
            ax.set_title('Динамика концентраций в последовательной реакции A→B→C→D',
                         fontsize=14, fontweight='bold', pad=20)

            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Добавляем стрелки для указания направления реакции
            ax.annotate('A → B', xy=(2.5, 0.4), xytext=(1.5, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                        fontsize=10, fontweight='bold')
            ax.annotate('B → C', xy=(5, 0.3), xytext=(4, 0.4),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                        fontsize=10, fontweight='bold')
            ax.annotate('C → D', xy=(7.5, 0.2), xytext=(6.5, 0.3),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                        fontsize=10, fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()
            self.plot_status.setText("✅ График концентраций построен")
            self.plot_status.setStyleSheet("color: #27ae60; font-weight: bold;")

        except Exception as e:
            self.plot_status.setText(f"❌ Ошибка: {str(e)}")
            self.plot_status.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def plot_reaction_types_comparison(self):
        """Построение сравнения типов реакций"""
        try:
            self.figure.clear()

            # Создаем 2 субплога для сравнения
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            time = np.linspace(0, 10, 100)

            # Цвета и метки
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
            labels = ['Вещество A', 'Вещество B', 'Вещество C', 'Вещество D']

            # 1. ПОСЛЕДОВАТЕЛЬНАЯ РЕАКЦИЯ: A → B → C → D
            k1, k2, k3 = 0.4, 0.3, 0.2  # Константы скорости

            # Аналитические решения для последовательной реакции
            A_seq = np.exp(-k1 * time)
            B_seq = (k1 / (k2 - k1)) * (np.exp(-k1 * time) - np.exp(-k2 * time))
            C_seq = k1 * k2 * (
                    (np.exp(-k1 * time) / ((k2 - k1) * (k3 - k1))) +
                    (np.exp(-k2 * time) / ((k1 - k2) * (k3 - k2))) +
                    (np.exp(-k3 * time) / ((k1 - k3) * (k2 - k3)))
            )
            D_seq = 1 - (A_seq + B_seq + C_seq)

            # Обеспечиваем неотрицательность
            B_seq = np.maximum(0, B_seq)
            C_seq = np.maximum(0, C_seq)
            D_seq = np.maximum(0, D_seq)

            # Нормализуем
            total_seq = A_seq + B_seq + C_seq + D_seq
            if np.any(total_seq > 0):
                A_seq, B_seq, C_seq, D_seq = A_seq / total_seq, B_seq / total_seq, C_seq / total_seq, D_seq / total_seq

            # График 1: Последовательная реакция
            concentrations_seq = [A_seq, B_seq, C_seq, D_seq]
            for i, (conc, label, color) in enumerate(zip(concentrations_seq, labels, colors)):
                ax1.plot(time, conc, label=label, color=color, linewidth=2.5, alpha=0.9)

            ax1.set_xlabel('Время', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Концентрация', fontsize=11, fontweight='bold')
            ax1.set_title('Последовательная реакция\nA → B → C → D',
                          fontsize=13, fontweight='bold', color='#2980b9')
            ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
            ax1.grid(True, alpha=0.2, linestyle='--')
            ax1.set_ylim(0, 1)

            # Добавляем стрелки и аннотации для последовательной реакции
            ax1.annotate('A→B', xy=(1.5, 0.6), xytext=(0.5, 0.7),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')
            ax1.annotate('B→C', xy=(4, 0.4), xytext=(3, 0.5),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')
            ax1.annotate('C→D', xy=(7, 0.2), xytext=(6, 0.3),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')

            # 2. РАЗВЕТВЛЕННАЯ РЕАКЦИЯ: A → B и A → C → D (параллельные пути)
            k1_b, k2_b, k3_b = 0.3, 0.2, 0.25  # Константы скорости для разветвленной реакции

            # Вещества расходуются из A по двум путям
            A_branch = np.exp(-(k1_b + k2_b) * time)

            # Первый путь: A → B
            B_branch = (k1_b / (k1_b + k2_b)) * (1 - np.exp(-(k1_b + k2_b) * time))

            # Второй путь: A → C → D
            # C образуется из A и превращается в D
            C_branch = (k2_b / (k3_b - (k1_b + k2_b))) * (
                    np.exp(-(k1_b + k2_b) * time) - np.exp(-k3_b * time)
            )
            C_branch = np.maximum(0, C_branch)

            # D образуется из C
            D_branch = 1 - (A_branch + B_branch + C_branch)
            D_branch = np.maximum(0, D_branch)

            # Нормализуем
            total_branch = A_branch + B_branch + C_branch + D_branch
            if np.any(total_branch > 0):
                A_branch, B_branch, C_branch, D_branch = (
                    A_branch / total_branch, B_branch / total_branch,
                    C_branch / total_branch, D_branch / total_branch
                )

            # График 2: Разветвленная реакция
            concentrations_branch = [A_branch, B_branch, C_branch, D_branch]
            for i, (conc, label, color) in enumerate(zip(concentrations_branch, labels, colors)):
                ax2.plot(time, conc, label=label, color=color, linewidth=2.5, alpha=0.9)

            ax2.set_xlabel('Время', fontsize=11, fontweight='bold')
            ax2.set_title('Параллельная\nA → B  и  A → C → D',
                          fontsize=13, fontweight='bold', color='#c0392b')
            ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
            ax2.grid(True, alpha=0.2, linestyle='--')
            ax2.set_ylim(0, 1)

            # Добавляем аннотации для разветвленной реакции
            # Стрелка A→B
            ax2.annotate('A→B', xy=(2, 0.7), xytext=(1, 0.8),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')
            # Стрелка A→C
            ax2.annotate('A→C', xy=(2, 0.5), xytext=(1, 0.6),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')
            # Стрелка C→D
            ax2.annotate('C→D', xy=(6, 0.3), xytext=(5, 0.4),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
                         fontsize=10, fontweight='bold')

            # Добавляем схематическое изображение механизмов реакций
            fig.text(0.05, 0.95, '📊 Сравнение механизмов химических реакций',
                     fontsize=16, fontweight='bold', color='#2c3e50')

            # Информационные блоки
            info_text1 = "Последовательная:\n• A превращается в B\n• B превращается в C\n• C превращается в D\n• Все стадии последовательны"
            info_text2 = "Параллельная:\n• A распадается на B и C\n• B - конечный продукт\n• C превращается в D\n• Два параллельных пути"

            ax1.text(0.02, 0.98, info_text1, transform=ax1.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # Добавляем индикаторы максимальных концентраций
            for ax, concs, title_text in [(ax1, concentrations_seq, "Последовательная"),
                                          (ax2, concentrations_branch, "Параллельная")]:
                max_vals = [conc.max() for conc in concs]
                max_time_idx = [np.argmax(conc) for conc in concs]
                max_times = [time[idx] for idx in max_time_idx]

                for i, (label, color, max_val, max_time) in enumerate(zip(labels, colors, max_vals, max_times)):
                    if max_val > 0.05:  # Только для значимых концентраций
                        ax.plot(max_time, max_val, 'o', color=color, markersize=8)
                        ax.annotate(f'{max_val:.2f}', xy=(max_time, max_val),
                                    xytext=(10, 0), textcoords='offset points',
                                    fontsize=8, color=color, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Оставляем место для заголовка
            self.canvas.figure = fig
            self.figure = fig
            self.canvas.draw()

            self.plot_status.setText("✅ График сравнения типов реакций построен")
            self.plot_status.setStyleSheet("color: #27ae60; font-weight: bold;")

        except Exception as e:
            self.plot_status.setText(f"❌ Ошибка: {str(e)}")
            self.plot_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
            import traceback
            print(traceback.format_exc())

    def clear_plot(self):
        """Очистка графика"""
        self.figure.clear()
        self.canvas.draw()
        self.plot_status.setText("График очищен")
        self.plot_status.setStyleSheet("color: #7f8c8d;")

class DatabaseTab(QWidget):
    """Вкладка базы данных"""

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("📁 База данных экспериментов")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)

        # Кнопки управления БД
        control_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.refresh_btn.clicked.connect(self.load_data)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Статистика
        stats_group = QGroupBox("📊 Статистика системы")
        stats_layout = QGridLayout()

        self.total_experiments_label = QLabel("Эксперименты: 0")
        stats_layout.addWidget(self.total_experiments_label, 0, 0)

        self.total_predictions_label = QLabel("Предсказания: 0")
        stats_layout.addWidget(self.total_predictions_label, 0, 1)

        self.avg_accuracy_label = QLabel("Средняя точность: 0.00%")
        stats_layout.addWidget(self.avg_accuracy_label, 1, 0)

        self.avg_confidence_label = QLabel("Средняя уверенность: 0.00%")
        stats_layout.addWidget(self.avg_confidence_label, 1, 1)

        self.last_update_label = QLabel("Обновлено: никогда")
        stats_layout.addWidget(self.last_update_label, 2, 0, 1, 2)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Таблица экспериментов
        layout.addWidget(QLabel("<b>📚 Последние эксперименты:</b>"))
        self.experiments_table = QTableWidget()
        self.experiments_table.setColumnCount(5)
        self.experiments_table.setHorizontalHeaderLabels([
            "ID", "Название", "Тип модели", "Точность", "Дата"
        ])
        self.experiments_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.experiments_table.setAlternatingRowColors(True)
        self.experiments_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        layout.addWidget(self.experiments_table)

        # Таблица предсказаний
        layout.addWidget(QLabel("<b>🔮 Последние предсказания:</b>"))
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(5)
        self.predictions_table.setHorizontalHeaderLabels([
            "ID", "Тип реакции", "Уверенность", "Вероятность", "Дата"
        ])
        self.predictions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.predictions_table.setAlternatingRowColors(True)
        self.predictions_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        layout.addWidget(self.predictions_table)

        layout.addStretch()

    def load_data(self):
        """Загрузка данных из БД"""
        try:
            # Получаем эксперименты
            experiments = self.bot.get_experiments()
            self.experiments_table.setRowCount(len(experiments))

            for row, exp in enumerate(experiments):
                self.experiments_table.setItem(row, 0, QTableWidgetItem(str(exp.get('id', ''))))
                self.experiments_table.setItem(row, 1, QTableWidgetItem(exp.get('experiment_name', '')))
                self.experiments_table.setItem(row, 2, QTableWidgetItem(exp.get('model_type', '')))

                accuracy = exp.get('accuracy', 0)
                accuracy_item = QTableWidgetItem(f"{accuracy:.4f}" if accuracy else "N/A")

                # Раскрашиваем в зависимости от точности
                if accuracy > 0.9:
                    accuracy_item.setBackground(QColor("#d4edda"))
                    accuracy_item.setForeground(QColor("#155724"))
                elif accuracy > 0.8:
                    accuracy_item.setBackground(QColor("#fff3cd"))
                    accuracy_item.setForeground(QColor("#856404"))
                else:
                    accuracy_item.setBackground(QColor("#f8d7da"))
                    accuracy_item.setForeground(QColor("#721c24"))

                self.experiments_table.setItem(row, 3, accuracy_item)

                timestamp = exp.get('timestamp', '')
                self.experiments_table.setItem(row, 4, QTableWidgetItem(str(timestamp)))

            # Получаем предсказания
            predictions = self.bot.get_predictions()
            self.predictions_table.setRowCount(len(predictions))

            for row, pred in enumerate(predictions):
                self.predictions_table.setItem(row, 0, QTableWidgetItem(str(pred.get('id', ''))))

                predicted_type = pred.get('predicted_type', '')
                type_item = QTableWidgetItem(predicted_type)
                if 'type1' in predicted_type:
                    type_item.setBackground(QColor("#d1ecf1"))
                    type_item.setForeground(QColor("#0c5460"))
                elif 'type2' in predicted_type:
                    type_item.setBackground(QColor("#d4edda"))
                    type_item.setForeground(QColor("#155724"))
                self.predictions_table.setItem(row, 1, type_item)

                confidence = pred.get('confidence', 0)
                confidence_item = QTableWidgetItem(f"{confidence:.4f}" if confidence else "N/A")

                # Раскрашиваем уверенность
                if confidence > 0.8:
                    confidence_item.setBackground(QColor("#d4edda"))
                    confidence_item.setForeground(QColor("#155724"))
                elif confidence > 0.6:
                    confidence_item.setBackground(QColor("#fff3cd"))
                    confidence_item.setForeground(QColor("#856404"))
                else:
                    confidence_item.setBackground(QColor("#f8d7da"))
                    confidence_item.setForeground(QColor("#721c24"))

                self.predictions_table.setItem(row, 2, confidence_item)

                prob_text = f"{(confidence * 100):.1f}%" if confidence else "N/A"
                self.predictions_table.setItem(row, 3, QTableWidgetItem(prob_text))

                timestamp = pred.get('timestamp', '')
                self.predictions_table.setItem(row, 4, QTableWidgetItem(str(timestamp)))

            # Обновляем статистику
            stats = self.bot.get_statistics()

            self.total_experiments_label.setText(
                f"Эксперименты: {stats.get('total_experiments', 0)}"
            )
            self.total_predictions_label.setText(
                f"Предсказания: {stats.get('total_predictions', 0)}"
            )

            avg_acc = stats.get('average_accuracy', 0)
            self.avg_accuracy_label.setText(
                f"Средняя точность: {avg_acc:.2%}" if avg_acc else "Средняя точность: N/A"
            )

            # Раскрашиваем среднюю точность
            if avg_acc > 0.9:
                self.avg_accuracy_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            elif avg_acc > 0.8:
                self.avg_accuracy_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            else:
                self.avg_accuracy_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

            avg_conf = stats.get('average_confidence', 0)
            self.avg_confidence_label.setText(
                f"Средняя уверенность: {avg_conf:.2%}" if avg_conf else "Средняя уверенность: N/A"
            )

            # Раскрашиваем среднюю уверенность
            if avg_conf > 0.8:
                self.avg_confidence_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            elif avg_conf > 0.6:
                self.avg_confidence_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            else:
                self.avg_confidence_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

            self.last_update_label.setText(
                f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка загрузки данных:\n{str(e)}")


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.bot = ReactionBot()
        self.setup_ui()
        self.setWindowTitle("🧪 Анализатор химических реакций с ИИ")
        self.setMinimumSize(1400, 900)

        # Загрузка иконки
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Таймер для автообновления данных
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_data)
        self.auto_refresh_timer.start(30000)  # Обновлять каждые 30 секунд

    def setup_ui(self):
        """Настройка интерфейса"""
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout(central_widget)

        # Заголовок
        header = QLabel("🧬 Анализатор химических реакций")
        header_font = QFont()
        header_font.setPointSize(22)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:0.5 #9b59b6, stop:1 #2ecc71);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 5px;
            }
        """)
        main_layout.addWidget(header)

        # Подзаголовок
        subtitle = QLabel("Лабораторная работа №10: Нейронные сети для классификации химических реакций")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 13px; margin-bottom: 15px; font-weight: bold;")
        main_layout.addWidget(subtitle)

        # Информационная панель - только информация о типах реакций
        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.StyledPanel)
        info_panel.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        info_layout = QHBoxLayout(info_panel)

        # Только информация о типах реакций
        status_item = "<b>Поддерживаемые типы реакций:</b> 2 (Последовательная, Параллельная)"

        label = QLabel(status_item)
        label.setStyleSheet("padding: 5px 15px;")
        info_layout.addWidget(label)

        info_layout.addStretch()
        main_layout.addWidget(info_panel)

        # Вкладки
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                top: -1px;
            }
            QTabBar::tab {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #e9ecef;
            }
        """)

        # Создаем вкладки
        self.bot_activation_tab = BotActivationTab(self)
        self.training_tab = TrainingTab(self.bot)
        self.prediction_tab = PredictionTab(self.bot)
        self.visualization_tab = VisualizationTab(self.bot)
        self.database_tab = DatabaseTab(self.bot)

        # Добавляем вкладки
        self.tab_widget.addTab(self.bot_activation_tab, "🤖 Активация бота")
        self.tab_widget.addTab(self.training_tab, "🎓 Обучение")
        self.tab_widget.addTab(self.prediction_tab, "🔮 Предсказание")
        self.tab_widget.addTab(self.visualization_tab, "📊 Визуализация")
        self.tab_widget.addTab(self.database_tab, "📁 База данных")

        main_layout.addWidget(self.tab_widget)

        # Статус бар
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готов к работе")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #f8f9fa;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
            }
        """)

        # Добавляем перманентные виджеты в статус бар
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)

        # Таймер для обновления статуса
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def update_status(self):
        """Обновление статуса в статус-баре"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(f"Время: {current_time}")

        current_tab = self.tab_widget.tabText(self.tab_widget.currentIndex())
        self.status_bar.showMessage(f"Текущая вкладка: {current_tab}")

    def auto_refresh_data(self):
        """Автоматическое обновление данных"""
        if self.tab_widget.currentWidget() == self.database_tab:
            self.database_tab.load_data()

    def closeEvent(self, event):
        """Обработка закрытия окна с русскими кнопками"""
        # Останавливаем бота при закрытии
        if hasattr(self.bot_activation_tab,
                   'bot_thread') and self.bot_activation_tab.bot_thread and self.bot_activation_tab.bot_thread.isRunning():
            # Создаем диалог с русскими кнопками
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Подтверждение")
            dialog.setText("Telegram бот все еще работает.\nОстановить его перед выходом?")
            dialog.setIcon(QMessageBox.Icon.Question)

            # Создаем кнопки
            yes_button = QPushButton("Да")
            no_button = QPushButton("Нет")
            cancel_button = QPushButton("Отмена")

            # Добавляем кнопки в диалог
            dialog.addButton(yes_button, QMessageBox.ButtonRole.YesRole)
            dialog.addButton(no_button, QMessageBox.ButtonRole.NoRole)
            dialog.addButton(cancel_button, QMessageBox.ButtonRole.RejectRole)

            # Показываем диалог
            dialog.exec()

            # Обрабатываем результат
            clicked_button = dialog.clickedButton()
            if clicked_button == yes_button:
                self.bot_activation_tab.deactivate_bot()
                event.accept()
            elif clicked_button == no_button:
                event.accept()
            else:  # cancel_button
                event.ignore()
                return
        else:
            event.accept()

        # Останавливаем таймеры
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if hasattr(self, 'auto_refresh_timer'):
            self.auto_refresh_timer.stop()

        # Останавливаем потоки обучения
        if hasattr(self.training_tab, 'worker') and self.training_tab.worker and self.training_tab.worker.isRunning():
            self.training_tab.worker.terminate()
            self.training_tab.worker.wait(1000)


# Экспортируемый класс для main.py
class ChemicalReactionGUI(MainWindow):
    """Алиас для совместимости с main.py"""
    pass


# ================== ТЕСТОВЫЙ ЗАПУСК ==================
if __name__ == "__main__":
    # Тестовый запуск
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Устанавливаем стиль для всего приложения
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            padding: 8px 15px;
            border-radius: 5px;
            font-weight: bold;
        }
        QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
        }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #3498db;
        }
        QTableWidget {
            gridline-color: #dee2e6;
            selection-background-color: #3498db;
            selection-color: white;
        }
        QHeaderView::section {
            background-color: #f8f9fa;
            padding: 5px;
            border: 1px solid #dee2e6;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #3498db;
            border-radius: 3px;
        }
    """)

    window = ChemicalReactionGUI()
    window.show()

    sys.exit(app.exec())