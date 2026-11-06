import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import numpy as np
from scipy.io import savemat
from basicsr.metrics.metric_util import calculate
import time
@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    @staticmethod
    def _to_uint8(x):
        """Rescale an array to ``uint8`` for visualisation."""
        x = np.asarray(x, dtype=np.float32)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        return (x * 255.0).round().astype(np.uint8)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def test(self, mode=None):
        """Generic forward pass (overridden by DREAMHM_S1Model / DREAMHM_S2Model)."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, opt=None):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, opt=opt)
    def saveMich(self,data,path):
        mich = np.round(data).astype(np.uint16)
        mich.tofile(path)
    def nor(self, data):
        size = data.shape
        # assert len(size) == 2
        if isinstance(data, torch.Tensor):
            minValue = torch.min(data)
            maxValue = torch.max(data)
            out = (data - minValue) / (maxValue - minValue)
        else:
            minValue = np.min(data)
            maxValue = np.max(data)
            out = (data - minValue) / (maxValue - minValue)
        return {'nor':out, 'max':maxValue, 'min':minValue}
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_numpy(x):
        """Return ``x`` as a ``numpy`` array (accepts tensors and lists)."""
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _fill_fidelity_channel(self, block, lq_block, locations):
        """Restore the hidden positions of the masked channels.

        The third channel of the 3-D sinogram block is never masked, so its
        prediction is used to fill the positions covered by the masks.  For
        DREAM-RM the masked block coordinates are provided by the dataset
        (``locations``); for DREAM-HM they are recovered from the zero pattern
        of the input block.
        """
        block = block.copy()
        if locations:
            for c, key in enumerate(('0', '1')):
                if key not in locations:
                    continue
                loc = locations[key]
                x_idx = self._to_numpy(loc[0]).reshape(-1)
                y_idx = self._to_numpy(loc[1]).reshape(-1)
                for xi, yi in zip(x_idx, y_idx):
                    xi, yi = int(xi), int(yi)
                    block[c, yi:yi + 1, xi:xi + 1] = block[2, yi:yi + 1, xi:xi + 1]
        elif lq_block is not None and lq_block.shape[0] >= 3:
            for c in (0, 1):
                hidden = lq_block[c] <= 0
                block[c][hidden] = block[2][hidden]
        return block

    @staticmethod
    def _fuse_channels(block):
        """Fuse the 3-D block back to a 2-D sinogram.

        DREAM-RM (masked channels + fidelity channel): average of the two
        masked-channel predictions.  DREAM-HM (complementary cold / hot
        channels): the two predictions are summed, because
        ``S = S * M_cold + S * M_hot``.
        """
        if block.shape[0] >= 3:
            return (block[0] + block[1]) / 2.0
        return block.sum(axis=0)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, opt=None):
        """Validation (``opt is None``) and test / reconstruction (``opt`` given).

        In test mode the reconstructed sinogram of every sample is written to
        ``path.visualization/<image>.mat`` under the key ``sino_net1`` so that
        the MATLAB OSEM / MLEM script (``reconstruction/``) can be applied.
        """
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        pad_w = int((opt or dataloader.dataset.opt).get('pad_w', 32))
        is_sino = os.environ.get('data_type', 'sinogram') == 'sinogram'

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        psnr_sum = psnr_zf_sum = ssim_sum = ssim_zf_sum = mse_sum = mse_zf_sum = 0.0
        time_sum = 0.0
        time_records = []
        n = 0
        idx = -1
        logger = get_root_logger()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            if opt is not None:  # test / reconstruction
                start_time = time.time()
                # the block is rescaled by its peak before entering the network
                max_value = torch.max(val_data['lq'])
                nordata = {'lq': val_data['lq'] / max_value,
                           'gt': val_data['gt'] / torch.max(val_data['gt'])}
                self.feed_data(nordata)
                self.test(opt['phase'])
                visuals = self.get_current_visuals()

                sr_img = visuals['result'].squeeze(0).clamp(0, 1).numpy()
                gt_img = visuals['gt'].squeeze(0).numpy()
                lq_img = visuals['lq'].squeeze(0).numpy()

                if is_sino:
                    if sr_img.shape[0] >= 3:
                        sr_img = self._fill_fidelity_channel(sr_img, lq_img, val_data.get('lct', {}))
                    # drop the zero padding of the detector dimension
                    sl = slice(pad_w, sr_img.shape[2] - pad_w)
                    sr_img, gt_img, lq_img = sr_img[:, :, sl], gt_img[:, :, sl], lq_img[:, :, sl]

                if is_sino and gt_img.shape[0] >= 3:
                    calgt, callq = gt_img[2], lq_img[2]
                else:
                    calgt, callq = self._fuse_channels(gt_img), self._fuse_channels(lq_img)
                calrec = self._fuse_channels(sr_img)

                psnr, ssim, mse = calculate(calgt, calrec, 1)
                psnr_zf, ssim_zf, mse_zf = calculate(calgt, callq, 1)
                psnr_sum += psnr
                psnr_zf_sum += psnr_zf
                ssim_sum += ssim
                ssim_zf_sum += ssim_zf
                mse_sum += mse
                mse_zf_sum += mse_zf
                n += 1

                if is_sino:
                    savedata = np.reshape(calrec, [-1, 1], order='F') * max_value.item()
                    savemat(osp.join(self.opt['path']['visualization'], img_name + '.mat'),
                            {'sino_net1': savedata})
                    elapsed = time.time() - start_time
                    time_sum += elapsed
                    time_records.append({'name': img_name, 'time': elapsed})
                    logger.info(
                        'img:{:15s} - Time: {:.4f}s - PSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}'
                        '*****  LQ: PSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}'.format(
                            img_name, elapsed, psnr, ssim, mse, psnr_zf, ssim_zf, mse_zf))
                else:
                    imwrite(self._to_uint8(calrec),
                            osp.join(self.opt['path']['visualization'], img_name + '.png'))
                    logger.info(
                        'img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}'
                        '*****  LQ: PSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}'.format(
                            img_name, psnr, ssim, mse, psnr_zf, ssim_zf, mse_zf))

            else:  # validation during training
                self.feed_data(val_data)
                self.test()
                visuals = self.get_current_visuals()

                sr_img = visuals['result'].squeeze(0).clamp(0, 1).numpy()
                gt_img = visuals['gt'].squeeze(0).numpy()
                lq_img = visuals['lq'].squeeze(0).numpy()

                if is_sino:
                    if sr_img.shape[0] >= 3:
                        sr_img = self._fill_fidelity_channel(sr_img, lq_img, val_data.get('lct', {}))
                    sl = slice(pad_w, sr_img.shape[2] - pad_w)
                    sr_img, gt_img = sr_img[:, :, sl], gt_img[:, :, sl]

                del self.gt, self.lq, self.output
                torch.cuda.empty_cache()

                if save_img or with_metrics:
                    if is_sino and gt_img.shape[0] >= 3:
                        calgt = gt_img[2]
                    else:
                        calgt = self._fuse_channels(gt_img)
                    calsr = self._fuse_channels(sr_img)

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        suffix = self.opt['val']['suffix'] or self.opt['name']
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{suffix}.png')
                    imwrite(self._to_uint8(calsr), save_img_path)

                if with_metrics:
                    psnr, ssim, mse = calculate(calgt, calsr, 1)
                    metrics = {'calculate_psnr': psnr, 'calculate_ssim': ssim, 'calculate_mse': mse}
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += metrics[opt_['type']]

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if opt is not None:
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}'
                '*****  LQ: PSNR: {:.6f} dB; SSIM: {:.6f}; MSE: {:.8f}\n'.format(
                    dataset_name, psnr_sum / n, ssim_sum / n, mse_sum / n,
                    psnr_zf_sum / n, ssim_zf_sum / n, mse_zf_sum / n))
            if is_sino:
                time_avg = time_sum / n if n > 0 else 0.0
                logger.info(
                    '----Reconstruction Time Statistics for {}----\n\tTotal images: {}\n'
                    '\tTotal time: {:.4f}s\n\tAverage time per image: {:.4f}s\n\tFPS: {:.2f}\n'.format(
                        dataset_name, n, time_sum, time_avg, 1.0 / time_avg if time_avg > 0 else 0.0))
                time_log_path = osp.join(self.opt['path']['visualization'], 'reconstruction_time.log')
                with open(time_log_path, 'w') as f:
                    f.write('Image Name\t\tReconstruction Time(s)\n')
                    f.write('=' * 60 + '\n')
                    for record in time_records:
                        f.write('{:30s}\t{:.4f}\n'.format(record['name'], record['time']))
                    f.write('=' * 60 + '\n')
                    f.write('Total images: {}\n'.format(n))
                    f.write('Total time: {:.4f}s\n'.format(time_sum))
                    f.write('Average time: {:.4f}s\n'.format(time_avg))
                    f.write('FPS: {:.2f}\n'.format(1.0 / time_avg if time_avg > 0 else 0.0))
                logger.info('Detailed time records saved to: {}'.format(time_log_path))
        else:
            if use_pbar:
                pbar.close()
            if with_metrics and idx >= 0:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= (idx + 1)
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
