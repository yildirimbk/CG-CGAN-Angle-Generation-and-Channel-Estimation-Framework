
function [ x_hat, T,y_w , Me, MSE ] = OMP_weight( M_w, D_w, y_w, epsilon, Nfft, Ntrain, Lr, t )
%Orthogonal Matching Pursuit using target sparsity factor(m)
% this was made for simulation and by no means an efficient implementation
% A - Sensing Matrix
% v - data vector
% m - sparsity level of x
%epsilon -var_n
%d = length(v);

[measures, size_d]=size(M_w);
r = reshape(y_w, size(y_w,2,3));
y_w=reshape(y_w, size(y_w,2,3));

Me=[];
x= zeros(size(M_w, 2), 1);
MSE=norm(y_w(:,1) - M_w*x);
it=0;%
for i=1:t%
%while MSE > epsilon
    c_w=M_w'*r;
    [maxInnerProduct, indexSet] = max(sum(abs(c_w),2));
    Omega =indexSet(1);
    if it==0
        T = Omega;
    else
        T = union(Omega, T_last); 
    end
   
    
    M_wT=M_w(:,T);
    % Me=(M_wT'*M_wT)\M_wT';
    Me=pinv(M_wT);
    
    x_T=zeros(length(T),Nfft);
    sum_R=0;
    for k=1:Nfft
    x_T(:,k)=Me*y_w(:,k);
    r(:,k) = y_w(:,k) - M_wT*x_T(:,k);
    
    sum_R=sum_R+r(:,k)'*r(:,k);
    end
    it=it+1;
    
    MSE=(1/(Nfft*Ntrain*Lr))*sum_R;
    
    T_last = T;

    
end

% return the estimation vector
x_hat = zeros(size_d,Nfft);
x_hat(T,:) = x_T;



