import enum
import os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"

FEATURES = ["daily_base_timestamp",
        "ask_price_0", "ask_size_0", "ask_price_1", "ask_size_1", "ask_price_2", "ask_size_2", "ask_price_3",
        "ask_size_3", "ask_price_4", "ask_size_4",
        "bid_price_0", "bid_size_0", "bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2", "bid_price_3",
        "bid_size_3", "bid_price_4", "bid_size_4"]

FEATURES_WITHOUT_VOLUME = ["daily_base_timestamp",
        "ask_price_0", "ask_price_1", "ask_price_2", "ask_price_3", "ask_price_4",
        "bid_price_0", "bid_price_1", "bid_price_2", "bid_price_3", "bid_price_4"]

OHLCV_FEATURES = ["daily_base_timestamp", "open", "high", "low", "final", "volume"]

OHLCV_FEATURES_WITHOUT_VOLUME = ["daily_base_timestamp", "open", "high", "low", "final"]

SIZE_OF_FEATURE = len(FEATURES)
SIZE_OF_FEATURE_WITHOUT_VOLUME = len(FEATURES_WITHOUT_VOLUME)

SIZE_OF_OHLCV_FEATURE = len(OHLCV_FEATURES)
SIZE_OF_OHLCV_FEATURE_WITHOUT_VOLUME = len(OHLCV_FEATURES_WITHOUT_VOLUME)

INITIAL_TOTAL_KRW = 1000000
BUY_AMOUNT = 100000
COMMISSION_RATE = 0.0015
SLIPPAGE_RATE = 0.001  # 10000000 + 10000 = 10010000

MAX_EPISODES = 10000
GAMMA = 1.0
LEARNING_RATE = 0.001
EPSILON_START = 0.20
EPSILON_FINAL = 0.005
REPLAY_MEMORY_THRESHOLD_FOR_TRAIN = 100

TRAIN_BATCH_SIZE_PERCENT = 10.0
PERFORMANCE_GRAPH_DRAW_INTERVAL = 500
SAVE_MODEL_INTERVAL = 500
REPLAY_MEMORY_SIZE = 10000000

TRAIN_BATCH_MIN_SIZE_STEPS = 512
TRAIN_REPEATS_STEPS = 2
TRAIN_INTERVAL_STEPS = 100
QNET_COPY_TO_TARGET_QNET_INTERVAL_STEPS = 1000 # STEPS

TRAIN_BATCH_MIN_SIZE = 1024
TRAIN_REPEATS = 10
QNET_COPY_TO_TARGET_QNET_INTERVAL_EPISODES = 2 # EPISODES

PLAY_SAVE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'play')
PLAY_FIGURE_PATH = os.path.join(PLAY_SAVE_PATH, "play.png")

PERFORMANCE_SAVE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'performance')
PERFORMANCE_FIGURE_PATH = os.path.join(PERFORMANCE_SAVE_PATH, 'performance.png')

MODEL_SAVE_PATH = os.path.join(PROJECT_HOME, 'codes', 'rl', 'policy_models')
BUYER_MODEL_FILE_NAME = 'buyer_model_{0}_{1}_W{2}_F{3}_{4}.pth'
BUYER_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, BUYER_MODEL_FILE_NAME)
SELLER_MODEL_FILE_NAME = 'seller_model_{0}_{1}_W{2}_F{3}_{4}.pth'
SELLER_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, SELLER_MODEL_FILE_NAME)

S3_BUCKET_NAME = 'invest-thinkonweb'

VERBOSE_STEP = False


class TimeUnit(enum.Enum):
    TEN_MINUTES = "10_MIN"
    ONE_HOUR = "1_HOUR"
    ONE_DAY = "1_DAY"