% clear
% A = load('A.mat').data;
% B = load('AA.mat').data;
% A = squeeze(A);
% B = squeeze(B);
% X = double(A); 
% Y = double(B);
%  
% maxvalue1=max(max(A));
% minvalue1=min(min(A));
% A = (A - minvalue1) / (maxvalue1 - minvalue1);
% maxvalue1=max(max(B));
% minvalue1=min(min(B));
% B = (B - minvalue1) / (maxvalue1 - minvalue1);
%  
% [psnr,mse]= psnr_zl(A,B,1)
% ssim1 = ssim_zl(A,B)

function F = calculate
F.psnr = @psnr_zl;
F.ssim = @ssim_zl;
end

function [PSNR, MSE] = psnr_zl(X, Y, datarange) % 计算峰值信噪比PSNR  X: gt, Y:lq
MSE = mean(mean((X-Y).^2));
PSNR = 10 * log10((datarange^2) / MSE);
end
 
function [mssim, ssim_map] = ssim_zl(img1, img2, K, window, L)  % L:datarange, 在这里L强制是1
%前两个输入是两张图片，K，L是计算均值、标准差、C1、C2需要的参数，windows是滤波器窗口大小
if (nargin < 2 || nargin > 5)
%    ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end
 
if (size(img1) ~= size(img2))
%    ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end
 
[M N] = size(img1);
 
if (nargin == 2)
   if ((M < 11) || (N < 11))   % 图像大小过小，则没有意义。
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);        % 参数一个标准偏差1.5，11*11的高斯低通滤波。
   K(1) = 0.01;                                                                      % default settings
   K(2) = 0.03;                                                                      %
   L = 255;                                  %
end
 
if (nargin == 3)
   if ((M < 11) || (N < 11))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end
 
if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end
 
if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end
 
L=1;
C1 = (K(1)*L)^2;    % 计算C1参数，给亮度L（x，y）用。
C2 = (K(2)*L)^2;    % 计算C2参数，给对比度C（x，y）用。
window = window/sum(sum(window)); %滤波器归一化操作。
img1 = double(img1); 
img2 = double(img2);
 
mu1   = filter2(window, img1, 'valid');  % 对图像进行滤波因子加权
mu2   = filter2(window, img2, 'valid');  % 对图像进行滤波因子加权
 
mu1_sq = mu1.*mu1;     % 计算出Ux平方值。
mu2_sq = mu2.*mu2;     % 计算出Uy平方值。
mu1_mu2 = mu1.*mu2;    % 计算Ux*Uy值。
 
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;  % 计算sigmax （标准差）
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;  % 计算sigmay （标准差）
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;   % 计算sigmaxy（标准差）
 
if (C1 > 0 && C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
   denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end
 
mssim = mean2(ssim_map);
 
return
end
