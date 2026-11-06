function [yi, ri, ni] = sino_preprocess(yi, ri, ni, numfrm)
% 
% pre-processing sinograms
%
% gbwang@ucdavis.edu (01-09-2013)
%

if isempty(ri)
    ri = zeros(size(yi));
end
if isempty(ni)
    ni = ones(size(yi));
end
if nargin<4 
    numfrm = 1;
end

% shift
minri = min(ri);
if minri<0
    ri = ri - minri;
    yi = yi - minri;
end
yi(yi<0) = 0;


% vectors
N  = length(yi(:));
yi = reshape(yi,[N/numfrm numfrm]);
ri = reshape(ri,[N/numfrm numfrm]);
ni = reshape(ni,[N/numfrm numfrm]);
