% This is a MATLAB demo showing how to use the kernel method for dynamic 
% PET image reconstruction. The kernel method is described in 
%
%   G. Wang and J. Qi, PET image reconstruction using kernel method, IEEE
%   Transactions on Medical Imaging, 2015, 34(1): 61-71;
%
% Guobao Wang @ UC Davis (02-03-2015)
%
clear; clc;

% add path
run('../../irt/setup');  
run('../../KER_v0.3/setup');  

% image size
imgsiz = [111 111]; 

%% system matrix
sysflag = 1;  % 1: using Fessler's IRT; 
              % 0: using your own system matrix G 
disp('--- Generating system matrix G ...')
if sysflag
    % require Fessler's IRT matlab toolbox to generate a system matrix
    ig = image_geom('nx', imgsiz(1), 'ny', imgsiz(2), 'fov', 33.3);

    % field of view
    ig.mask = ig.circ(16, 16) > 0;

    % system matrix G
    prjsiz = [249 210];
    sg = sino_geom('par', 'nb', prjsiz(1), 'na', prjsiz(2), 'dr', 70/prjsiz(1), 'strip_width', 'dr');
    G  = Gtomo2_strip(sg, ig, 'single', 1);
    Gopt.mtype  = 'fessler';
    Gopt.ig     = ig;
    Gopt.imgsiz = imgsiz;
else
    Gfold = ''; % where you store your system matrix G;
    load([Gfold, 'G']);	% load matrix G
    Gopt.mtype  = 'matlab';
    Gopt.imgsiz = imgsiz;
end
Gopt.disp = 0; % no display of iterations

%% Simulate dynamic PET sinograms
disp('--- Generating dynamic images ...')

% phantom
load('zubal_data')

% frame durations
dt = (scant(:,2)-scant(:,1))/60;
numfrm = length(dt);

% dynamic images of activity
X0 = zeros(prod(size(model)),numfrm);
ID = [0 32 64 128 196 255];
for m = 1:numfrm
    for n = 1:length(ID)
        X0(model==ID(n),m) = TAC(m,n)*dt(m);
    end
end

% noise-free geometric projection
for m = 1:size(X0,2)
    proj(:,m) = proj_forw(G, Gopt, X0(:,m));
end

% attenuation factor
ai = exp(-repmat(proj_forw(G, Gopt, u(:)), [1 numfrm]));  

% background (randoms and scatters)
ri = repmat(mean(ai.*proj,1)*0.2,[size(proj,1) 1]);  % 20% uniform background

% total noiseless projection
y0 = ai.*proj + ri; 

% count level
count = 8e6; % a total of 8 million events

% normalized sinograms
cs = count / sum(y0(:));
y0 = cs * y0;
ri = cs * ri;
yi = poissrnd(y0); % noisy projection
ni = ai*cs; % multiplicative factor

% initial estimate
xinit = [];

%% Kernelized EM 
disp('--- Building image prior and kernel matrix ...')

% build image prior using 3 composite images
M = {[1:16], [17:20], [21:24]};
for i = 1:length(M)
    y_i = sum(yi(:,M{i}),2);
    n_i = sum(ni(:,M{i}),2);
    r_i = sum(ri(:,M{i}),2);
    [x, out] = eml_kem(y_i, n_i, G, Gopt, xinit, r_i, 100); 
    x = gaussfilt(x,imgsiz,3);
    U(:,i) = x(:) / sum(dt(M{i}));
end
U = U * diag(1./std(U,1)); % normalization
            
% build the kernel matrix K using k nearest neighbors
tic;
sigma = 1;
[N, W] = buildKernel(imgsiz, 'knn', 48, U, 'radial', sigma);
K = buildSparseK(N, W);
toc;           

disp('--- Doing kernel-based reconstruction ...')

% kernel-based reconstruction
mm = 1:numfrm; % select frames to reconstruct
maxit = 60;
tic;
for i = 1:length(mm)
    m = mm(i);
    [a, out] = eml_kem(yi(:,m), ni(:,m), G, Gopt, xinit, ri(:,m), maxit, K); 
    X{1}(:,i) = K * a; % NOTE: output of eml_kem.m is the kernel coefficient a 
end
toc;

%% Conventional reconstruction - MLEM (i.e. K = I)
disp('--- Doing conventional MLEM reconstruction ...')

tic;
for i = 1:length(mm)
    m = mm(i);
    [x, out] = eml_kem(yi(:,m), ni(:,m), G, Gopt, xinit, ri(:,m), maxit); 
    X{2}(:,i) = x;
end
toc;

%% display
maxv = [40 40 270];
mm = [2 12 22]; % selecting a few frames to display
ftitle = {'KEM','EM'};
for i = 1:length(X)
    for j = 1:length(mm)
        x0 = X0(:,mm(j));
        x = X{i}(:,mm(j));
        figure,imagesc(reshape(x,imgsiz),[0 maxv(j)]); axis image;
        set(gca,'FontSize',16); axis off; colormap(jet)
        snr = -10*log10(sum((x-x0).^2)/sum(x0.^2));
        title(sprintf('Frame %d, %s, SNR=%3.1fdB',mm(j),ftitle{i},snr)); 
    end
end


