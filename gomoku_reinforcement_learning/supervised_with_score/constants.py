RANDOM_SEED = None

# board constants
BOARD_WIDTH = 15
BOARD_HEIGHT = 15
BOARD_SIZE = 15
N_IN_ROW = 5
P_X = 1
P_O = -1
P_E = 0

# train constants
LEARN_RATE = 2e-3
LR_MULTIPLIER = 1.0
TEMPERATURE = 1.0
NUM_PLAYOUT = 1000
C_PUCT = 5.0
BUFFER_SIZE = 10000
BATCH_SIZE = 1024

KL_TARGET = 0.02
CHECK_FREQ = 50
GAME_BATCH_NUM = 500
BEST_WIN_RATIO = 0.0
NUM_PURE_MCTS_PLAYOUT = 1000
PLAY_BATCH_SIZE = 1
EPOCHS = 5

# tree node
VIRTUAL_LOSS = 1
NUM_INSTANCE = 10

TRAIN_MODEL_NAME = 'train_model'
BEST_MODEL_NAME = 'best_model'

TRAIN_MODEL_PATH = './model/train_model'
BEST_MODEL_PATH = './model/best_model'
CACHE_FILE_PATH = './model/cache.txt'