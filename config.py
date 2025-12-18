"""
Конфигурационный файл приложения химических реакций
"""
from pathlib import Path

# ==================== ПУТИ И ДИРЕКТОРИИ ====================
BASE_DIR = Path(__file__).parent.absolute()
LOG_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'

# Создание директорий
for directory in [LOG_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== ТОКЕН TELEGRAM БОТА ====================
TELEGRAM_BOT_TOKEN = "8523979775:AAEvIG0md3VLrToCEJm8D2tXya82Z-tg_q0"

# ==================== НАСТРОЙКИ МОДЕЛЕЙ ====================
MODEL_CONFIG = {
    'perceptron': {
        'hidden_layer_sizes': (),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 3000,
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
        'batch_size': 'auto'
    },
    'mlp': {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 5000,
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
        'batch_size': 'auto'
    }
}

# ==================== НАСТРОЙКИ ГЕНЕРАЦИИ ДАННЫХ ====================
GENERATION_CONFIG = {
    'default_samples': 2000,
    'test_size': 0.2,
    'time_points': 50,
    't_max': 10.0
}

# ==================== НАСТРОЙКИ ЛОГИРОВАНИЯ ====================
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'console_output': True
}

# ==================== ТИПЫ ХИМИЧЕСКИХ РЕАКЦИЙ ====================
REACTION_TYPES = {
    'type1': {
        'name': 'A → B → C → D (последовательная)',
        'description': 'Последовательная цепочка превращений'
    },
    'type2': {
        'name': 'A → B → D и A → C → D (разветвленная)',
        'description': 'Разветвленная реакция с двумя путями'
    }
}

# ==================== ЦВЕТА ДЛЯ ГРАФИКОВ ====================
COLORS = {
    'A': '#3498db',  # Синий
    'B': '#e74c3c',  # Красный
    'C': '#2ecc71',  # Зеленый
    'D': '#9b59b6',  # Фиолетовый
    'loss': '#e67e22',  # Оранжевый
    'accuracy': '#27ae60'  # Темно-зеленый
}