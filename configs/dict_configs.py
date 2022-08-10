# --------------------------------------------------------------------------- #
# Dict config
# ---------------------------------------------------------------------------- #

BACKBONE_KWARGS = dict(
    # resnet args...
    dcn=None, dilate_only_last_layer=False, dilated=False, multi_grid=True, root='~/.encoding/models',
    # resnet for classifier args
    low_res=True)

TRANSFORMS_KWARGS = dict()

TASKS_DICT = dict(detect=True, segment=True, depth=True, inst_depth=True,
                  inst_seg=True, sem_cont=False, sur_nor=False, inst_seg_dense=False)
TASK_TO_LOSS_NAME = dict(detect='default', segment='default', depth='default',
                         inst_depth='default', inst_seg='default',
                         sem_cont='default', sur_nor='default',
                         inst_seg_dense='default')
TASK_TO_LOSS_ARGS = dict()
TASK_TO_LOSS_KWARGS = dict()
TASK_TO_LOSS_CALL_KWARGS = dict(segment=dict(ignore_index=-1))
TASK_TO_MIN_OR_MAX = dict(detect=1, segment=1, depth=-1, inst_depth=-1, inst_seg=1, ae=-1)
LOSS_INIT_WEIGHTS = dict(detect_cls_loss=1., detect_reg_loss=1.,
                         detect_centerness_loss=1., segment_loss=1.,
                         depth_loss=8., inst_depth_l1_loss=0.05, inst_seg_loss=1.,
                         ae_pca_error=0.1, ae_proj_error=0.1, ae_att_loss=0.0002)
LOSS_START_EPOCH = dict(detect_cls_loss=1, detect_reg_loss=1,
                        detect_centerness_loss=1, segment_loss=1,
                        depth_loss=1, inst_depth_l1_loss=1, inst_seg_loss=1)

LR_SCHEDULER_ARGS = dict()
OPTIMIZER_ARGS = dict()

#  Hierarchy options
TASK_TO_EN_BLOCK = dict(segment=6, depth=6, sem_cont=6, sur_nor=6, ae=6)
TASK_TO_EN_GROUPS = dict(uninet=['segment', 'depth', 'sem_cont', 'sur_nor', 'ae', 6])
TASK_TO_DE_GROUPS = dict(uninet=['segment', 'depth', 'sem_cont', 'sur_nor', 'ae', 5])
