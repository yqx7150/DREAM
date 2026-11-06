% This is a MATLAB demo for the Kernel MLAA reconstruction described in 
%
%     Wang GB.
%     PET-enabled Dual-Energy CT: Image Reconstruction and A Proof-of-Concept 
%     Computer Simulation Study.
%     Physics in Medicine and Biology, 65(24): 245028, 2020
%
% gbwang@ucdavis.edu, 12-12-2020
% Last updated: 03-15-2021
%

clc; clear;
run('../../KER_v0.3/setup');  

%% load system matrix
% ******************************************************************************
% ****    CREDIT to Dr. JIAN ZHOU who generated these TOF system matrices   ****
% ******************************************************************************

% system matrix - attenuation
load('GE690_attn_sysmat_180x3.27mm.mat');
Gopt.imgsiz = [180 180];
Gopt.prjsiz = [281 288];

% system matrix - emission
load('GE690_tof2d_sysmat_11tbin_180x3.27mm.mat');
Popt.imgsiz = [180 180];
numbin      = 11;  % number of TOF bins
Popt.prjsiz = [281 288 numbin];

%% PET projection data

% load the image data
load('psct_xcat');

% noise-free projection data
count = 5e6; % count level
proj  = proj_forw(P, Popt, x0); 
a0 = exp(- proj_forw(G, Gopt, u0/10) );  % attenuation factor
ai = repmat(a0,[1 numbin]);
ri = mean(ai(:).*proj(:)) * 0.4 * ones(size(proj));  % 40% uniform background
y0 = ai(:).*proj + ri; % noiseless projection

% noisy measurements
cs = count / sum(y0(:));
y0 = cs * y0;
ri = cs * ri;
ni = ones(size(y0))*cs; % (uniform) normalization factor
yi = poissrnd(y0); % noisy projection

%% Initial estimates

% mask 
Gopt.mask = mask; 
imgsiz = size(mask);

% Initial estimate from x-ray CT
uinit = CT2LAC(CT,'120','bilinear')/10;

%% Standard reconstruction

% maximum iteration number
maxit = 600;

% other settings
Popt.disp     = 1;
Popt.savestep = 100;

% MLEM reconstruction (with x-ray CT-derived attenuation map)
a0 = exp(- proj_forw(G, Gopt, uinit) );  % attenuation factor
ai = repmat(a0,[1 Popt.prjsiz(3)]);
disp('TOF MLEM reconstruction for initial x estimate ...');
x_mlem = eml_kem(yi, ni.*ai(:), P, Popt, [], ri, 100);
    
% Stadard MLAA (WARNING: it may take a while)
xinit = x_mlem;
disp('------ Standard MLAA reconstruction ...')
tic;
[u, x, out] = psct_kmlaa(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit);
toc;
X{1} = out.uest;

%% Kernel MLAA

% Features from x-ray CT
imgsiz_CT = size(CT);
R = buildNbhd(imgsiz_CT, 'clique', 1); % Extract features using a 3x3 patch 
I = [[1:prod(imgsiz_CT)]' R.N]; 
F = CT(I);
F = F * diag(1./std(F,1)); % normalization

% building kernel matrix K 
sigma = 1;
[N, W] = buildKernel(imgsiz, 'knn', 50, F, 'radial', sigma);
K = buildSparseK(N, W);

% Kernel MLAA
disp('------ Kernel MLAA reconstruction ...')
tic;
[u, x, out] = psct_kmlaa(yi, ni, G, Gopt, uinit(:), P, Popt, xinit(:), ri, maxit, K);
out.uest = K * out.uest;
toc;
X{2} = out.uest;

%% Display
lname = {'Standard MLAA', 'Kernel MLAA'};
for k = 1:length(X)
    xk = X{k}*10;
    for j = 1:size(out.uest,2)
        uj = reshape(xk(:,j),imgsiz);
        mse{k}(j) = 10*log10(sum((uj(mask)-u0(mask)).^2)/sum(u0(mask).^2));
    end
end

% image MSE
it = 1:7;
itt = (it-1)*100;
ltype = {'rx-', 'bo-'};
figure; hold on; box on;
for k = 1:length(X)
    plot(itt, mse{k}(it), ltype{k}, 'LineWidth', 2);
end
xlabel('iteration number');
ylabel('MSE (dB)');
legend(lname,'Location','NorthEast');

% images
imgsiz = [180, 180];
dispwd = [30,150];
dispwdx = [75 110]; dispwdy = [95 130];
maxv = 0.15;
for k = 1:length(X)
    u = X{k}(:,end)*10;
    mse_k = 10*log10(sum((u(mask)-u0(mask)).^2)/sum(u0(mask).^2));
    figure, imagesc(reshape(u,imgsiz),[0 maxv])
    axis off; axis image; colormap(gray); colorbar;
    xlim(dispwd); ylim(dispwd);
    set(gca,'FontSize',14);
    title(sprintf('%s, MSE=%3.1fdB',lname{k},mse_k));
end
