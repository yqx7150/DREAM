function [x, out, Phi] = eml_kem4D(yi, ni, G, Gopt, x0, ri, maxit, KS, KT)
%--------------------------------------------------------------------------
% The spatiotemporal kernel method for HTR dynamic PET image reconstruction.  
% It is implemented using the expectation maximization algorithm. Detail of
% the method is described in 
%
%   G. Wang, High temporal-resolution dynamic PET image reconstruction using 
%   a new spatiotemporal kernel method. IEEE Transactions on Medical Imaging, 
%   38(3): 664 – 674, 2019.
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
%   KS      spatial kernel matrix accounting for image piror
%   KT	    temporal kernel matrix
%
% OUTPUT
%   x       kernel coefficient image estimate in vector
%   out     output
%   Phi     objective function value
%
%--------------------------------------------------------------------------
% Programmer: Guobao Wang @ UC Davis Medical Center, 02-01-2017
% Last Modified: 09-03-2018
%--------------------------------------------------------------------------

%% check inputs
imgsiz = Gopt.imgsiz;
numpix = prod(imgsiz);
numfrm = size(yi,2);
if isempty(x0)
    x0 = ones(numpix,numfrm);
end
[yi, ri, ni] = sino_preprocess(yi, ri, ni, numfrm);
if nargin<8
    KS = speye(numpix);
end
if nargin<9
    KT = speye(numfrm);
end

% set Gopt
Gopt = setGopt(ni, G, Gopt);
Gopt.sens = ker_back(Gopt.sens,KS,KT); % MUST
if isempty(maxit)
    maxit = 10;
end

% initialization
x    = max(mean(x0(:))*1e-9,x0); x(~Gopt.mask) = 0;
yeps = mean(yi(:))*1e-9;
wx   = Gopt.sens;

% output
if nargin>1
    out = []; Phi = [];
end
out.xest = zeros(numpix, numfrm, min(maxit,ceil(maxit/Gopt.savestep)+1));
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
        out.xest(:,:,itt) = x;
    end
    
    % EM update
    z  = ker_forw(x,KS,KT);
    yb = ni.*proj_forw(G, Gopt, z, numfrm) + ri;
    yy = yi./(yb+yeps);
    yy(yb==0&yi==0) = 1;
    zb = proj_back(G, Gopt, ni.*yy, numfrm);
    xb = ker_back(zb,KS,KT);
    x  = x ./ wx .* xb;
    x(~Gopt.mask) = 0;
        
    % objective function value
    if nargout>2
        iy = yb(:)>0;
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
function Y = ker_forw(X, KS, KT)
Y = KS*X*KT';

% ------------------------------------------------------------------------
% kernel back projection
% ------------------------------------------------------------------------
function X = ker_back(Y, KS, KT)
X = KS'*Y*KT;


    
