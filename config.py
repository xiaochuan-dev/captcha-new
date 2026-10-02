# 词表：仅数字 + 小写字母（大写在数据侧映射为小写）
CHARSET = "0123456789+-*=?"
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK = 0

# 模型输入尺寸（与原 model 一致）
IMG_H = 32
IMG_W = 128
CHANNELS = 1

# 训练
BATCH_SIZE = 128
EPOCHS = 200
LR = 3e-4
WEIGHT_DECAY = 0.05
VAL_RATIO = 0.1
NUM_WORKERS = 2

# 数据：Hugging Face 数据集
HF_REPO = "xiaochuan-dev/captcha-new"
HF_FILENAME = "new_xinanjiaotong.parquet"
LOCAL_PARQUET = "./data/new_xinanjiaotong.parquet"

# 模型结构参数（与原 model 一致）
MODEL_DIM = 256
MODEL_DEPTH = 6
MODEL_HEADS = 4
MODEL_DROPOUT = 0.2
