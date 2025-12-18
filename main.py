"""
Главный файл запуска приложения химических реакций
"""
import sys
import logging
from pathlib import Path
import threading
import asyncio

# Импортируем только конфигурацию
try:
    from config import LOG_DIR, LOG_CONFIG, TELEGRAM_BOT_TOKEN
except ImportError:
    BASE_DIR = Path(__file__).parent.absolute()
    LOG_DIR = BASE_DIR / 'logs'
    TELEGRAM_BOT_TOKEN = ""
    LOG_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'console_output': True
    }

# Создание директорий
LOG_DIR.mkdir(exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    datefmt=LOG_CONFIG['date_format'],
    handlers=[
        logging.FileHandler(LOG_DIR / 'chemical_reactions.log', encoding='utf-8'),
        logging.StreamHandler() if LOG_CONFIG['console_output'] else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_bot_in_thread(token: str):
    """Запуск бота в отдельном потоке"""
    try:
        from telegram_bot import run_telegram_bot
        run_telegram_bot(token)
    except Exception as e:
        logger.error(f"Ошибка запуска бота в потоке: {e}")


def run_gui_with_bot(token: str = None):
    """Запуск графического интерфейса с возможностью активации бота"""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        from gui import ChemicalReactionGUI

        app = QApplication(sys.argv)
        app.setApplicationName("Chemical Reactions Analyzer")
        app.setApplicationVersion("1.0")

        # Установка иконки
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

        window = ChemicalReactionGUI()
        window.show()

        logger.info("Графический интерфейс запущен успешно")
        print("✅ Графический интерфейс запущен!")
        print("💡 Для активации Telegram бота перейдите на вкладку '🤖 Активация бота'")
        print("🧪 Система определяет только 2 типа реакций:")
        print("   1. A → B → C → D (последовательная)")
        print("   2. A → B → D и A → C → D (разветвленная)")

        return app.exec()

    except ImportError as e:
        logger.critical(f"Ошибка импорта GUI: {e}", exc_info=True)
        print(f"\n❌ Ошибка импорта GUI: {e}")
        print("📦 Установите зависимости командой:")
        print("pip install PySide6 matplotlib numpy scikit-learn python-telegram-bot")
        return 1

    except Exception as e:
        logger.critical(f"Ошибка запуска GUI: {e}", exc_info=True)
        print(f"\n❌ Ошибка запуска GUI: {e}")
        return 1


def main():
    """Основная функция запуска"""
    try:
        print("\n" + "=" * 70)
        print("🧪 СИСТЕМА АНАЛИЗА ХИМИЧЕСКИХ РЕАКЦИЙ")
        print("=" * 70)
        print("Лабораторная работа №10")
        print("Нейронные сети для классификации химических реакций")
        print("=" * 70)

        # Запускаем сразу графический интерфейс
        print("\n🚀 Запуск графического интерфейса...")
        return run_gui_with_bot(TELEGRAM_BOT_TOKEN)

    except KeyboardInterrupt:
        print("\n\n👋 Приложение остановлено пользователем")
        return 0

    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())