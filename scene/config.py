
RANDOM_OVERLAP = False
SPLIT_NEW = False


# 掩码相关常量
UNSEEN_VALUE = -1
MAX_VIEWS = 240
MAX_POINTS = 180_000
MIN_MASK_POINTS = 100
MIN_IOU_THRESHOLD = 0.05

# 文件路径相关
# DEFAULT_SAM_FOLDER = "sam"
DEFAULT_SAM_FOLDER = "sam"
# DEFAULT_SAM_FOLDER = "gd_sam"
UNION_FOLDER = "union"
ORIGIN_FOLDER = "origin"

if not SPLIT_NEW:
    COMPARE_FOLDER = "compare"
    SPLIT_FOLDER = "split"
    SPLIT_MASK_FILE_PATTERN = "{:05d}_mask.npy"
else:
    COMPARE_FOLDER = "compare_new"
    SPLIT_FOLDER = "split_new"
    SPLIT_MASK_FILE_PATTERN = "rgb_{}.npy"

# 文件格式
MASK_FILE_PATTERN = "{:05d}_masks_sam.npy"
RESULT_FILE_PATTERN = "{:05d}.png"
