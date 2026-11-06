function [x, out, Phi] = eml_kem(yi, ni, G, Gopt, x0, ri, maxit, K)
%--------------------------------------------------------------------------
% Kernelized expectation-maximization (KEM) algorithm for PET image 
% reconstruction. The details of the KEM are described in 
%
%   G. Wang and J. Qi, PET image reconstruction using kernel method, IEEE
%   Transactions on Medical Imaging, 2015, 34(1): 61-71;
%
%--------------------------------------------------------------------------
% INPUT:
%   yi      sinogram in vector
%   ni      multiplicative factor (normalization, attenuation), can be empty
%   G       system matrix
%   Gopt    option set for G, can be empty if G is a matlab sparse matrix
%   ri      addtive factor (randoms and scatters), can be empty
%   x0      initial image estimate, can be empty
%   maxit   maximum iteration number
%   K       kernel matrix accounting for image piror
%
% OUTPUT
%   x       image estimate in vector
%   out     output
%   Phi     objective function value
%
%--------------------------------------------------------------------------
% Programmer: Guobao Wang @ UC Davis, Qi Lab, 02-01-2012
% Last Modified: 02-03-2015
%--------------------------------------------------------------------------

%% check inputs
imgsiz = Gopt.imgsiz;
numpix = prod(imgsiz);
if isempty(x0)
    x0 = ones(numpix,1);
end
[yi, ri, ni] = sino_preprocess(yi, ri, ni);
if nargin<8
    K = speye(numpix);
end
ktype = 'org';

% set Gopt
Gopt = setGopt(ni, G, Gopt);
Gopt.sens = ker_back(K,Gopt.sens,ktype); % MUST
if isempty(maxit)
    maxit = 10;
end

% initialization
x    = max(mean(x0(:))*1e-9,x0(:)); x(~Gopt.mask) = 0;
yeps = mean(yi(:))*1e-9;
wx   = Gopt.sens;

% output
if nargin>1
    out = []; Phi = [];
end
out.xest = zeros(length(x(:)), min(maxit,ceil(maxit/Gopt.savestep)+1));
t1 = cputime;

%% iterative loop
for it = 1:maxit     
    
    % save data
    if Gopt.disp
        disp(sprintf('iteration %d',it));
    end
    if nargout>1 & ( it==1 | rem(it,Gopt.savestep)==0 )
        itt = min(it, floor(it/Gopt.savestep) + 1);
        out.step(itt)   = it;
        out.time(:,itt) = cputime - t1;        
        out.xest(:,itt) = x;
    end
    
    % EM update
    z  = ker_forw(K,x,ktype);
    yb = ni.*proj_forw(G, Gopt, z) + ri;
    yy = yi./(yb+yeps);
    yy(yb==0&yi==0) = 1;
    zb = proj_back(G, Gopt, ni.*yy);
    xb = ker_back(K,zb,ktype);
    x  = x ./ wx .* xb;
    x(~Gopt.mask) = 0;
        
    % objective function value
    if nargout>2
        iy = yb>0;
        Phi(it) = sum(yi(iy).*log(yb(iy))-yb(iy));
        if it>1 & Phi(it)<Phi(it-1)
            warning('Objective function is not increasing')
        end
    end
    
end

proj_clear(Gopt);

% ------------------------------------------------------------------------
% kernel forward projection
% ------------------------------------------------------------------------
function y = ker_forw(K, x, type)
switch type
    case 'org'
        y = K * x;
    case 'psd'
        y = K'*(K*x);
end

% ------------------------------------------------------------------------
% kernel back projection
% ------------------------------------------------------------------------
function x = ker_back(K, y, type)
switch type
    case 'org'
        x = K' * y;
    case 'psd'
        x = K'*(K*y);
end


    
