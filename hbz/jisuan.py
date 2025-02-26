
from calculate import calculate
import os
from scipy.io import loadmat, savemat
import cv2
import numpy as np
from hbz_waigua import setup_logger
import logging
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

# get_lq_path_root = '/home/b109/Desktop/hbz/DiffIR-demotionblur_two/datasets/60_net_data/img/test'
get_lq_path_root = '/home/b109/Desktop/hbz/select_50/lq/mat'



# img_path_root = '/home/b109/Desktop/hbz/DiffIR-demotionblur_mask/result_mat/mask10_patch_test20'
# img_path_root = '/home/b109/Desktop/hbz/DiffIR-demotionblur_two/result_mat/nomin_nomin/mat'
# img_path_root = '/home/b109/Desktop/hbz/duibi/DenoisingDiffusionProbabilityModel-ddpm-/result/mlem/mat'
img_path_root = '/home/b109/Desktop/hbz/duibi/ncsn++/result/pet/mlem/mat'
# img_path_root = '/home/b109/Desktop/hbz/duibi/U-Vit_pet/result/mlem/mat'

log_path = os.path.join(img_path_root,os.path.pardir)
setup_logger(
    "base",
    log_path,
    "result",
    level=logging.INFO,
    screen=True,
    tofile=True,
)

mat_list = os.listdir(img_path_root)
test_psnr_sum = 0
test_psnr_zf_sum = 0
test_ssim_sum = 0
test_ssim_zf_sum = 0
test_mse_sum = 0
test_mse_zf_sum = 0
idx = 0
logger = logging.getLogger("base")
for i in sorted(mat_list):
    
    mat_path = os.path.join(img_path_root, i)
    get_lq_mat_path = os.path.join(get_lq_path_root, i)
    if os.path.isdir(mat_path):
        continue

    gt = loadmat(mat_path)['gt']
    GT_img = np.reshape(gt, [256, 256], order = 'F')
    GT_img = GT_img / np.max(GT_img)

    rec = loadmat(mat_path)['rec']
    rec_img = np.reshape(rec, [256, 256], order = 'F')
    rec_img = rec_img / np.max(rec_img)

    lq = loadmat(get_lq_mat_path)['lq']
    lq_img = np.reshape(lq, [256, 256], order = 'F')
    lq_img = lq_img / np.max(lq_img)



    idx += 1

    # calgt = gt_img[0,:,:]
    # callq = lq_img[0,:,:]
    # calrec = (sr_img[0,:,:] + sr_img[1,:,:] + sr_img[2,:,:]) / 3




    psnr, ssim, mse = calculate(GT_img, rec_img, 1)
    psnr_zf, ssim_zf, mse_zf = calculate(GT_img, lq_img, 1)
    logger.info(
      "img:{:15s} - PSNR: {:.8f} dB; SSIM: {:.8f}; MSE: {:.8f} #### [lq-gt]PSNR: {:.8f} dB; SSIM: {:.8f}; MSE: {:.8f}".format(
          i, psnr, ssim, mse, psnr_zf, ssim_zf, mse_zf
      )
    )

    test_psnr_sum += psnr
    test_psnr_zf_sum += psnr_zf
    test_ssim_sum += ssim
    test_ssim_zf_sum += ssim_zf
    test_mse_sum += mse
    test_mse_zf_sum += mse_zf

    write_images(rec_img*0.8, os.path.join('/home/b109/Desktop/hbz/duibi/ncsn++/result/pet/mlem/img', f'{i}_rec.png'))
    write_images(GT_img*0.8, os.path.join('/home/b109/Desktop/hbz/duibi/ncsn++/result/pet/mlem/img', f'{i}_gt.png'))
    write_images(lq_img*0.8, os.path.join('/home/b109/Desktop/hbz/duibi/ncsn++/result/pet/mlem/img', f'{i}_lq.png'))

# print('avg_psnr:', test_psnr_sum / idx, 'avg_ssim:', test_ssim_sum / idx, 'avg_mse:', test_mse_sum / idx)
# logger.info(
#     "----Average PSNR/SSIM results----\n\tPSNR: {:.2f} dB; SSIM: {:.4f}; MSE: {:.8f}; NMSE: {:.8f}*****  零填充: PSNR: {:.2f} dB; SSIM: {:.4f}; MSE: {:.8f}; NMSE: {:.8f} ***** Average_time: {:.4f}s\n".format(
#         ave_psnr, ave_ssim, ave_mse, ave_nmse, ave_psnr_zf, ave_ssim_zf, ave_mse_zf, ave_nmse_zf, ave_time
#     )
# )
print(idx)
logger.info(
    "----Average PSNR/SSIM results----\n\tPSNR: {:.8f} dB; SSIM: {:.8f}; MSE: {:.8f} ####### [lq-gt]PSNR: {:.8f} dB; SSIM: {:.8f}; MSE: {:.8f}\n".format(
        test_psnr_sum / idx, test_ssim_sum / idx, test_mse_sum / idx,
        test_psnr_zf_sum / idx, test_ssim_zf_sum / idx, test_mse_zf_sum / idx
    )
)
