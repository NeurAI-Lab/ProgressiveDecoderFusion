from .segment_losses.seg_losses_all import cross_entropy_loss, \
    ClassBalancedSegmentationLosses
from .depth_losses import RMSE, DepthL1Loss
from .aux_losses import SemanticContLoss, NormalsCosineLoss, \
    BalancedBinaryCrossEntropyLoss, NormalsL1Loss
from .ae_losses import AELossModule


task_to_loss_fn = {
    'segment': {
        'default': ClassBalancedSegmentationLosses,
        'balanced': ClassBalancedSegmentationLosses,
        'cross_entropy': cross_entropy_loss},
    'depth': {
        'default': RMSE, 'rmse': RMSE, 'l1_loss': DepthL1Loss},
    'sem_cont': {
        'default': BalancedBinaryCrossEntropyLoss,
        'binary_ce': BalancedBinaryCrossEntropyLoss,
        'sem_cont_loss': SemanticContLoss},
    'sur_nor': {
        'default': NormalsL1Loss, 'l1_loss': NormalsL1Loss,
        'consine_loss': NormalsCosineLoss},
    'ae': {
        'default': AELossModule}}
