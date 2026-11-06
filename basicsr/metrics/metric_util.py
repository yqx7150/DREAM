import numpy as np

try:  # scikit-image >= 0.16
    from skimage.metrics import mean_squared_error as _mse
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
except ImportError:  # pragma: no cover - legacy scikit-image
    from skimage.measure import compare_mse as _mse
    from skimage.measure import compare_psnr as _psnr
    from skimage.measure import compare_ssim as _ssim

from basicsr.utils import bgr2ycbcr


def calculate(gt_img, pred_img, data_range=1.0):
    """Return ``(PSNR, SSIM, MSE)`` between a reference and a prediction.

    Used for both sinogram- and image-domain evaluation: the arrays are
    ``float`` and already normalised to ``[0, data_range]``.
    """
    psnr = _psnr(gt_img, pred_img, data_range=data_range)
    ssim = _ssim(gt_img, pred_img, data_range=data_range)
    mse = _mse(gt_img, pred_img)
    return psnr, ssim, mse


def calculate_rmse(gt_img, pred_img):
    """Root mean squared error between two normalised arrays."""
    return float(np.sqrt(_mse(gt_img, pred_img)))


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.
