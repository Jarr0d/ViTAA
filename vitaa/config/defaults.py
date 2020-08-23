import os

from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.IMS_PER_ID = 4


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN =[0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.CROP_SIZE = 32
_C.INPUT.DOWNSAMPLE_RATIO = 1/8
_C.INPUT.PADDING = 10


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.VISUAL_MODEL = "resnet50"
_C.MODEL.TEXTUAL_MODEL = "bilstm"
_C.MODEL.NUM_CLASSES = 11003
_C.MODEL.NUM_PARTS = 5
_C.MODEL.FREEZE = False
_C.MODEL.WEIGHT = "imagenet"


# -----------------------------------------------------------------------------
# LSTM
# -----------------------------------------------------------------------------
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.ONEHOT = True
_C.MODEL.LSTM.EMBEDDING_SIZE = 512
_C.MODEL.LSTM.NUM_UNITS = 512
_C.MODEL.LSTM.VOCABULARY_SIZE = 12000
_C.MODEL.LSTM.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.LSTM.MAX_LENGTH = 100


# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
_C.MODEL.EMBEDDING = CN()
_C.MODEL.EMBEDDING.EMBED_HEAD = "simple"
_C.MODEL.EMBEDDING.FEATURE_SIZE = 512
_C.MODEL.EMBEDDING.DROPOUT_PROB = 0.3


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999

_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500, )

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

