function [N, W] = buildKernelT_New(numfrm, nbrtyp, nbrpar, X, kertyp, kerpar, normflag, midflag)
%
% Get the index and weight of neighboring voxes in a neighborhood
%
% gbwang@ucdavis.edu (4-17-2018)
%

%% check input variables
if nargin<2 | isempty(nbrtyp)
    nbrtyp = 'cube';
end
if nargin<3 | isempty(nbrpar)
    nbrpar = 3;
end
if nargin<4
    X = [];
end
if nargin<5 | isempty(kertyp)
    kertyp = 'invdist';
end
if nargin<6 | isempty(kerpar)
    kerpar = 1;
end
if isscalar(kerpar) 
    kerpar = kerpar*ones(numfrm,1);
end
if nargin<7 | isempty(normflag)
    normflag = 1;
end
if nargin<8 | isempty(midflag)
    midflag = 1;
end
J = [1:numfrm]';

%% neighborhood type
switch nbrtyp
            
        
    case 'cube'
        
        % sizes of neighborhood window
        wlen = 2*floor(nbrpar/2); % length of neighborhood window
        xidx = -wlen(1)/2:wlen(1)/2; 
                
        % weight
        h = fspecial('gaussian', [nbrpar(1) 1], nbrpar(1)/(4*sqrt(2*log(2)))); 
        
        % index and distance
        N = ones(numfrm, length(h(:)));
        W = zeros(numfrm, length(h(:)));
        l = 1; n = 1;
        for x = xidx
            Xnew = setBoundary1(J+x,numfrm,J);
            idx  = Xnew>0; N(idx,l) = Xnew(idx); W(idx,l) = h(n);
            % N(:,l) = Xnew; W(:,l) = h(n);
            l = l + 1;
            n = n + 1;
        end
        
        if l<=size(N,2)
            N(:,l:end) = [];
            W(:,l:end) = [];
        end
        if not(isempty(X))
            W = W.*calc_wgt(X, N, kertyp, kerpar);
        end
        i = ceil(size(N,2)/2);
        if ~midflag
            N(:,i) = [];
            W(:,i) = [];
        end        
    case 'knn'
        k = nbrpar(1);
        [N, D] = knnsearch(X, X, 'dist', 'seuclidean', 'k', k);
        if ~midflag
            N = N(:,2:end);
        end
        W = calc_wgt(X, N, kertyp, kerpar);
end

% normalized to 1 and output
if normflag
    sumw = repmat(sum(W,2),[1 size(W,2)]);
    W = W./sumw;
    W(sumw==0) = 0;
end


%% sub functions

%--------------------------------------------------------------------------
function x = setBoundary1(x, N, J)
%--------------------------------------------------------------------------
if nargin<3
    J = [];
end
if N==1
    x = x(:).^0;
else
    idx = x(:)>N;
    if any(idx)
        x(idx) = 0; %J(idx); %N - (x(idx)-N); 
    end
    idx = x(:)<1;
    x(idx) = 0; %J(idx);%1 + (1-x(idx)); 
    x = x(:);
end

%--------------------------------------------------------------------------
function W = calc_wgt(X, N, kertyp, kerpar)
%--------------------------------------------------------------------------
J  = [1:size(N,1)]';
D  = zeros(size(N));
for i = 1:size(N,2)
    D(:,i) = sqrt(mean(((X(J,:)-X(N(:,i),:))).^2,2));
end
s = mean(D(:))./mean(D,2);
S = std(D(:));
D = diag(1./(S)) * D;

switch kertyp
    case 'invdist'
        W = 1./D;
        
    case 'radial' % radial Gaussian
        W = exp(-diag(1./(2*kerpar))*D.^2);
        
    case 'poly' % polynomial
        for i = 1:size(N,2)
            W(:,i) = ( mean(X(J,:).*X(N(:,i),:),2) + kerpar ).^2;
        end
    case 'nnlle'
        for j = 1:size(N,1)
            w = lsqnonneg(X(N(j,:),:)',X(j,:)');
            if any(isnan(w))
                w = zeros(length(w),1);
            end
            W(j,:) = w;
        end
        
    otherwise
        error('unknown kernel type')
end 
