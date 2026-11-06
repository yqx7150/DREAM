from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .metric_util import calculate, calculate_rmse
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate', 'calculate_rmse', 'calculate_psnr', 'calculate_ssim', 'calculate_niqe']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
