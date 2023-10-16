function [theta_star, LL] = omp_mGLM_cv(n1_star, X1, n2_star, X2, s, alpha, GD_iter)

theta_star_prev = zeros( size(X1,2), size(n1_star,1) );
LL = zeros(s,1);
S_mk = [ 1:size(n1_star,1) ];
S_cov = ones(size(S_mk));
r0 = length(S_mk);

lambda_star = exp( X1 * theta_star_prev );
lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));

del = n1_star' - lambda_star;
grad = X1'*del;

for r=r0:s    
   for itr=1:GD_iter
       lambda_tmp = exp( X1 * theta_star_prev );
       lambda_tmp = lambda_tmp ./ (1+sum( lambda_tmp,2 ));
       for ii=1:r
           gr = X1(:,S_cov(ii))' * (n1_star(S_mk(ii),:)' - lambda_tmp(:,S_mk(ii)));
           theta_star_prev(S_cov(ii),S_mk(ii)) = theta_star_prev(S_cov(ii),S_mk(ii)) + alpha*gr;           
       end
   end
   
   lambda_star = exp( X1 * theta_star_prev );
   lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));
   del = n1_star' - lambda_star;
   grad = X1'*del; grad_tmp = grad;
   for ii=1:r
       grad_tmp(S_cov(ii), S_mk(ii)) = 0;
   end
   [~, j] = max(abs(grad_tmp(:)));
   [j_cov, j_mk] = ind2sub(size(grad),j);
   S_cov = [S_cov j_cov];
   S_mk = [S_mk, j_mk];

   lambda_star2 = exp( X2 * theta_star_prev );
   lambda_star2 = lambda_star2 ./ (1+sum( lambda_star2,2 ));
   lambda_g = sum( lambda_star2,2 );   
   LL(r) = sum( log(1-lambda_g) ) + sum( sum( (n2_star').*log(lambda_star2./(1-lambda_g)) ) );
   
end
theta_star = theta_star_prev;
LL( 1:(r0-1) ) = -Inf;
end