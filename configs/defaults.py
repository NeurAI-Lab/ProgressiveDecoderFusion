# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN

from configs.dense_configs import dense_options

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'uninet'
_C.MODEL.BACKBONE_NAME = 'resnet50'
_C.MODEL.NORM_LAYER = 'bn'
# model loading args..
_C.MODEL.PRETRAINED_PATH = ""
_C.MODEL.IS_FULL_MODEL = False
_C.MODEL.LOAD_BACKBONE = False
_C.MODEL.BACKBONE_LOAD_NAME = 'backbone'
_C.MODEL.HEAD_LOAD_NAME = 'head'

# -----------------------------------------------------------------------------
# INPUT options
# -----------------------------------------------------------------------------
_C.INPUT = CN()


# --------------------------------------------------------------------------- #
# Dataloader Options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.ANNOTATION_FOLDER = 'gtFine/annotations_coco_format_v1'
_C.DATALOADER.IMG_FOLDER = 'leftImg8bit'
_C.DATALOADER.ANN_FILE_FORMAT = 'instances_%s.json'
# ImageNet mean and standard deviation..
_C.DATALOADER.MEAN = [.485, .456, .406]
_C.DATALOADER.STD = [.229, .224, .225]
_C.DATALOADER.TRAIN_TRANSFORMS = ['Expand', 'RandomSampleCrop',
                                  'ResizeMultiScale', 'HorizontalFlip', 'ColorJitter', 'AddIgnoreRegions',
                                  'ConvertFromInts', 'ToTensor', 'Normalize']
_C.DATALOADER.VAL_TRANSFORMS = ['Resize', 'ToTensor', 'Normalize']
_C.DATALOADER.TRAIN_BATCH_TRANSFORMS = []
# Multi scale augmentation defaults..
_C.DATALOADER.MS_MULTISCALE_MODE = 'value'
_C.DATALOADER.MS_RATIO_RANGE = [0.75, 1]
_C.DATALOADER.MIN_DEPTH = 1e-3
_C.DATALOADER.MAX_DEPTH = 80.
_C.DATALOADER.SIZE_DIVISIBILITY = 128


# --------------------------------------------------------------------------- #
# Backbone and encoder Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.ENCODER_TYPE = 'resnet'
_C.MODEL.ENCODER.NUM_EN_FEATURES = 6
_C.MODEL.ENCODER.OUT_CHANNELS_BEFORE_EXPANSION = 512
_C.MODEL.ENCODER.FEAT_CHANNELS = [2048, 2048, 2048, 2048]
_C.MODEL.ENCODER.STRIDE = 32


# --------------------------------------------------------------------------- #
# Miscellaneous Options
# ---------------------------------------------------------------------------- #

_C.MISC = CN()
_C.MISC.AUX_TASKS = ['sem_cont', 'sur_nor']
_C.MISC.SEM_CONT_MULTICLASS = False
_C.MISC.SEM_CONT_POS_WEIGHT = 0.95
_C.MISC.SIMILARITY_TYPE = 'CKA'
_C.MISC.GENERIC_EVALS = False


# --------------------------------------------------------------------------- #
# Collect all Options
# ---------------------------------------------------------------------------- #

_C = dense_options(_C)
