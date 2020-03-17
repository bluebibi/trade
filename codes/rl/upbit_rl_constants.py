import os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"

BUY_AMOUNT = 100000
MAX_EPISODES = 10
GAMMA = 0.98
LEARNING_RATE = 0.001
EPSILON_START = 0.20
REPLAY_MEMORY_THRESHOLD_FOR_TRAIN = 100
TRAIN_INTERVAL = 100
TRAIN_BATCH_SIZE_PERCENT = 10.0
TRAIN_REPEATS = 5
QNET_COPY_TO_TARGET_QNET_INTERVAL = 100

PERFORMANCE_FIGURE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'performance.png')
BUYER_MODEL_SAVE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'parameters_buyer_model.pth')
SELLER_MODEL_SAVE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'parameters_seller_model.pth')
