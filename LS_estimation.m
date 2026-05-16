function [ x_hat, y_w , Me_diag] = LS_estimation( M_w, y_w, Nfft, Ntrain, Lr, N_paths)

[measures, size_d]=size(M_w);
r = reshape(y_w, size(y_w,2,3));
y_w=reshape(y_w, size(y_w,2,3));

diag_indices = 1 : (N_paths+1) : (N_paths*N_paths); % Indices of diagonal elements

A_diag = M_w(:, diag_indices);  % Only the columns contributing to the diagonal of x


% Me=(M_wT'*M_wT)\M_wT';

Me_diag = pinv(A_diag);          % Size: (N_paths x L)

x_diag_T = zeros(N_paths, Nfft);                         % Estimated diagonal components
x_hat     = zeros(N_paths * N_paths, Nfft);              % Final full vector (with zeros off-diagonal)


% sum_R=0;
for k=1:Nfft
  x_diag_T(:,k) = Me_diag * y_w(:,k);                       % Diagonal estimate only
  % r(:,k)        = y_w(:,k) - A_diag * x_diag_T(:,k);        % Residual
  % sum_R         = sum_R + r(:,k)' * r(:,k);                 % Accumulate MSE
end

% return the estimation vecotr
x_hat(diag_indices, :) = x_diag_T;

% MSE = (1 / (Nfft * Ntrain * Lr)) * sum_R;
