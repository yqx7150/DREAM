try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import mean_squared_error as compare_mse
except:
    from skimage.measure import compare_psnr,compare_ssim, compare_mse
    
import numpy as np
import torch
def calculate(gt_img, lq_img, data_range):
    psnr = compute_PSNR(gt_img, lq_img, data_range)
    ssim = compute_SSIM(gt_img, lq_img, data_range)
    # rmse = compute_RMSE(gt_img, lq_img)
    mse = compute_MSE(gt_img, lq_img)
    return psnr, ssim, mse

def compute_MSE(gt_img, lq_img):
    
    return compare_mse(gt_img,lq_img)


def compute_RMSE(gt_img, lq_img):
    # img1 = img1 * 2000 / 255 - 1000
    # img2 = img2 * 2000 / 255 - 1000
    # if type(img1) == torch.Tensor:
    #     return torch.sqrt(compute_MSE(img1, img2)).item()
    # else:
    return np.sqrt(compute_MSE(gt_img, lq_img))


def compute_PSNR(gt_img, lq_img, data_range):
    # if type(img1) == torch.Tensor:
    #     mse_ = compute_MSE(img1, img2)
    #     return 10 * torch.log10((data_range ** 2) / mse_).item()
    # else:
    #     mse_ = compute_MSE(img1, img2)
    #     return 10 * np.log10((data_range ** 2) / mse_)
    return compare_psnr(gt_img,lq_img,data_range=data_range)

def compute_SSIM(gt_img, lq_img, data_range):
    return compare_ssim(gt_img,lq_img,data_range=data_range)


def dingliang(target, input):
    if type(target) == torch.Tensor:
        mask = torch.where(target > 0.01 ,1 ,0)  # mask
        data = mask*target
        S = torch.sum(data)
        if type(input) == torch.Tensor:
            input_data = mask * input
            S_in = torch.sum(input_data)
        else:
            input_data = mask.cpu().numpy() * input
            S_in = np.sum(input_data)

    else:
        mask = np.where(target > 0.01 ,1 ,0)
        data = mask*target
        S = np.sum(data)
        if type(input) == torch.Tensor:
            input_data = torch.from_numpy(mask) * input
            S_in = torch.sum(input_data)
        else:
            input_data = mask * input
            S_in = np.sum(input_data)


    coefficine = S / S_in

    return input * coefficine

'''

def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.shape) == 2:
        h, w = img1.shape
        if type(img1) == torch.Tensor:
            img1 = img1.view(1, 1, h, w)
            img2 = img2.view(1, 1, h, w)
        else:
            img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :])
            img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :])
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    # if size_average:
    #     return ssim_map.mean().item()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1).item()
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()
'''