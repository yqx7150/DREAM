function recon_from_mat(sino_dir, out_dir, varargin)
%RECON_FROM_MAT  PET image reconstruction of DREAM-reconstructed sinograms.
%
%   recon_from_mat(SINO_DIR, OUT_DIR, 'Name', Value, ...)
%
%   SINO_DIR  folder with the .mat files written by dream_hm/test.py, each
%             containing the variable `sino_net1` (256*192 values, column-major)
%   OUT_DIR   folder for the reconstructed images
%
%   Name/Value pairs
%     'method'  'osem' (default) or 'mlem'
%     'nblock'  number of subsets for OSEM (default 12)
%     'niter'   number of iterations (default 6 for OSEM, 60 for MLEM)
%     'fwhm'    width of the Gaussian post-smoothing in pixels (default 7,
%               pass 0 to disable)
%     'ni'      normalisation sinogram (default: ones)
%     'ri'      additive term (randoms + scatter, default: zeros)
%     'suffix'  filename suffix of the inputs (default '')
%
%   The reconstruction geometry (256 x 192 parallel-beam sinogram, 32 cm FOV,
%   256 x 256 image) matches the simulations in the paper.

if nargin < 2
    error('recon_from_mat:usage', 'recon_from_mat(sino_dir, out_dir, ...)');
end

opt = parse_opts(varargin{:});
if isempty(opt.niter)
    if strcmpi(opt.method, 'osem')
        opt.niter = 6;
    else
        opt.niter = 60;
    end
end

% ---- KER / IRT toolbox ------------------------------------------------
here = fileparts(mfilename('fullpath'));
ker = fullfile(here, '..', 'third_party', 'KER_v0.3');
if exist(fullfile(ker, 'irt', 'setup.m'), 'file') ~= 2
    error('recon_from_mat:ker', 'KER_v0.3 toolbox not found in %s', ker);
end
old = pwd;
cd(fullfile(ker, 'irt')); run(fullfile(ker, 'irt', 'setup.m'));
cd(ker); run(fullfile(ker, 'setup.m'));
cd(old);

% ---- geometry (identical to the paper) -------------------------------
imgsiz = [256 256];
ig = image_geom('nx', imgsiz(1), 'ny', imgsiz(2), 'fov', 32);
ig.mask = ig.circ(10, 10) > 0;

prjsiz = [256 192];
dr = 32 / 256;
sg = sino_geom('par', 'nb', prjsiz(1), 'na', prjsiz(2), 'dr', dr, 'strip_width', 'dr');
G = Gtomo2_strip(sg, ig, 'single', 1);
Gopt.mtype = 'fessler';
Gopt.ig = ig;
Gopt.imgsiz = imgsiz;

if strcmpi(opt.method, 'osem')
    Gb = Gblock(G, opt.nblock);
end

% ---- loop over the reconstructed sinograms ---------------------------
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(sino_dir, '*.mat'));
fprintf('recon_from_mat: %d sinograms in %s\n', numel(files), sino_dir);

for k = 1:numel(files)
    name = files(k).name;
    [~, stem] = fileparts(name);
    S = load(fullfile(sino_dir, name));
    if isfield(S, 'sino_net1')
        y = S.sino_net1;
    else
        error('recon_from_mat:variable', '%s has no `sino_net1` variable', name);
    end
    y = double(reshape(y, prjsiz(1), prjsiz(2)));

    ni = opt.ni;
    ri = opt.ri;
    if isempty(ni), ni = ones(size(y)); end
    if isempty(ri), ri = zeros(size(y)); end

    tic;
    xinit = ones(sum(ig.mask(:)), 1);   % constant initial image inside the mask
    if strcmpi(opt.method, 'osem')
        x = eml_osem(xinit, Gb, y, ni, ri, ...
            'niter', opt.niter, 'isave', 'last', 'precon', 'classic');
        x = ig.embed(x);
    else
        x = eml_em(xinit, G, y(:), ni(:), ri(:), 'niter', opt.niter, 'isave', 'last');
        x = ig.embed(x);
    end
    if opt.fwhm > 0
        x = gaussfilt(x, imgsiz, opt.fwhm);
    end
    elapsed = toc;

    rec = x;
    save(fullfile(out_dir, [stem '_rec.mat']), 'rec');
    imwrite(rescale_img(rec), fullfile(out_dir, [stem '_rec.png']));
    fprintf('  %-24s %6.2f s\n', stem, elapsed);
end
fprintf('done: %s\n', out_dir);
end

% -------------------------------------------------------------------------
function img = rescale_img(x)
x = double(x);
x = x - min(x(:));
m = max(x(:));
if m > 0, x = x / m; end
img = uint8(round(255 * x));
end

function opt = parse_opts(varargin)
opt = struct('method', 'osem', 'nblock', 12, 'niter', [], 'fwhm', 7, ...
    'ni', [], 'ri', [], 'suffix', '');
if mod(numel(varargin), 2) ~= 0
    error('recon_from_mat:args', 'Name/Value pairs expected');
end
for i = 1:2:numel(varargin)
    key = varargin{i};
    if ~isfield(opt, key)
        error('recon_from_mat:args', 'unknown option %s', key);
    end
    opt.(key) = varargin{i + 1};
end
end