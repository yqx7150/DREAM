function K = buildSparseK(N, W, J, numvox, flag)
%--------------------------------------------------------------------------
% generate a sparse kernel matrix K
%
% gbwang@ucdavis.edu (05-20-2013)
%

% check input variables
if nargin<3 | isempty(J)
    J = [1:size(N,1)]';
end
if nargin<4 | isempty(numvox)
    numvox = size(N,1);
end
if nargin<5
    flag = 1;
end

% create K
K = sparse(J, J(N(:,1)), W(:,1), numvox, numvox);
for n = 2:size(N,2)
    Kn = sparse(J, J(N(:,n)), W(:,n), numvox, numvox);
    K  = K + Kn;
end

% prevent any all-zeros rows
sumK = full(sum(K,2));
I = find(sumK==0);
for i = 1:length(I) 
    K(I(i),I(i)) = 1;
end

% make symmetric
if flag
    K = ( K' + K )/2;
    K = make_sym(K);
end
