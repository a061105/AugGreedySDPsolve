function [A,b,C] = generate_SDP_syn_data(n,m,seed)
rng(seed);
rmat = randn(n);
C = rmat'*rmat;
rmat = randn(n);
X = rmat'*rmat;
A = zeros(n,n,m);
b = zeros(m,1);
for i=1:m
    rmat = randn(n);
    A(:,:,i) = rmat'*rmat;
end

b = squeeze(sum(sum(bsxfun(@times,A,X))));