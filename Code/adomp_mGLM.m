function [theta_star, uk, Bk, ak, Pk, LLk, Biask] = adomp_mGLM(nk_star, Xk, uk, Bk, ak, Pk, s, alpha, beta, GD_iter, UnsuppMks, UnsuppCovs, theta_star_init)

if isempty(theta_star_init)
    theta_star_prev = zeros( size(Xk,2), size(nk_star,1) );
else
    theta_star_prev = theta_star_init;
end

if isempty(UnsuppMks) && isempty(UnsuppCovs)
%%% Full Model
    S_mk = [ 1:size(nk_star,1) ];
    S_cov = ones(size(S_mk));
    r0 = length(S_mk);

    lambda_star = exp( Xk * theta_star_prev );
    lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));
    
    del = nk_star' - lambda_star;
    grad = Xk'*del + beta*uk;
    for m=1:size(nk_star,1)
        grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
    end

    for r=r0:s
        for itr=1:GD_iter
            for ii=1:r
                gr = grad(S_cov(ii),S_mk(ii));                
                theta_star_prev(S_cov(ii),S_mk(ii)) = theta_star_prev(S_cov(ii),S_mk(ii)) + alpha*gr;
            end
            
            lambda_tmp = exp( Xk * theta_star_prev );
            lambda_tmp = lambda_tmp ./ (1+sum( lambda_tmp,2 ));
        	grad = Xk' * (nk_star' - lambda_tmp) + beta*uk;
            for m=1:size(nk_star,1)
                grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
            end            
        end        
        
        lambda_star = exp( Xk * theta_star_prev );
        lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));        
        del = nk_star' - lambda_star;
        grad = Xk'*del + beta*uk;
        for m=1:size(nk_star,1)
            grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
        end
        grad_tmp = grad;
        for ii=1:r
            grad_tmp(S_cov(ii),S_mk(ii)) = 0;
        end
        
        [~, j] = max(abs(grad_tmp(:)));
        [j_cov, j_mk] = ind2sub(size(grad),j);
        S_cov = [S_cov j_cov];
        S_mk = [S_mk, j_mk];
        
    end
    theta_star = theta_star_prev;
    
elseif ~isempty(UnsuppMks) && isempty(UnsuppCovs)
%%% Reduced Model - Exclude History in the UnsuppMks
    S_mk = [ 1:size(nk_star,1) ];
    S_cov = ones(size(S_mk));
    r0 = length(S_mk);

    lambda_star = exp( Xk * theta_star_prev );
    lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));
    
    del = nk_star' - lambda_star;
    grad = Xk'*del + beta*uk;
    for m=1:size(nk_star,1)
        grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
    end
    grad(2:end, UnsuppMks) = 0;
    
    for r=r0:s
        for itr=1:GD_iter
            for ii=1:r
                gr = grad(S_cov(ii),S_mk(ii));                
                theta_star_prev(S_cov(ii),S_mk(ii)) = theta_star_prev(S_cov(ii),S_mk(ii)) + alpha*gr;
            end
            
            lambda_tmp = exp( Xk * theta_star_prev );
            lambda_tmp = lambda_tmp ./ (1+sum( lambda_tmp,2 ));
        	grad = Xk' * (nk_star' - lambda_tmp) + beta*uk;
            for m=1:size(nk_star,1)
                grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
            end      
        end        
        lambda_star = exp( Xk * theta_star_prev );
        lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));        
        del = nk_star' - lambda_star;
        grad = Xk'*del + beta*uk;
        for m=1:size(nk_star,1)
            grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
        end
        grad_tmp = grad;
        for ii=1:r
            grad_tmp(S_cov(ii),S_mk(ii)) = 0;
        end
        
        [~, j] = max(abs(grad_tmp(:)));
        [j_cov, j_mk] = ind2sub(size(grad),j);
        S_cov = [S_cov j_cov];
        S_mk = [S_mk, j_mk];
        
    end
    theta_star = theta_star_prev;    
    
elseif ~isempty(UnsuppMks) && ~isempty(UnsuppCovs)
%%% Reduced Model - Exclude params [UnsuppMks(ii), UnsuppCovs(ii)]
    S_mk = [ 1:size(nk_star,1) ]; S_mk( UnsuppMks( UnsuppCovs==1 ) ) = [];
    S_cov = ones(size(S_mk));
    r0 = length(S_mk);

    lambda_star = exp( Xk * theta_star_prev );
    lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));
    
    del = nk_star' - lambda_star;
    grad = Xk'*del + beta*uk;
    for m=1:size(nk_star,1)
        grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
    end
    for ii=1:length(UnsuppMks)
        grad(UnsuppCovs(ii), UnsuppMks(ii)) = 0;
    end

    for r=r0:s
        for itr=1:GD_iter
            for ii=1:r
                gr = grad(S_cov(ii),S_mk(ii));                
                theta_star_prev(S_cov(ii),S_mk(ii)) = theta_star_prev(S_cov(ii),S_mk(ii)) + alpha*gr;
            end
            
            lambda_tmp = exp( Xk * theta_star_prev );
            lambda_tmp = lambda_tmp ./ (1+sum( lambda_tmp,2 ));
        	grad = Xk' * (nk_star' - lambda_tmp) + beta*uk;
            for m=1:size(nk_star,1)
                grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
            end   
        end        
        lambda_star = exp( Xk * theta_star_prev );
        lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));        
        del = nk_star' - lambda_star;
        grad = Xk'*del + beta*uk;
        for m=1:size(nk_star,1)
            grad(:, m) = grad(:, m) - beta * Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) * theta_star_prev(:,m);
        end
        for ii=1:length(UnsuppMks)
            grad(UnsuppCovs(ii), UnsuppMks(ii)) = 0;
        end
        grad_tmp = grad;
        for ii=1:r
            grad_tmp(S_cov(ii),S_mk(ii)) = 0;
        end
        
        [~, j] = max(abs(grad_tmp(:)));
        [j_cov, j_mk] = ind2sub(size(grad),j);
        S_cov = [S_cov j_cov];
        S_mk = [S_mk, j_mk];
        
    end
    theta_star = theta_star_prev;
end

LLk = 0;
Biask = 0;
lambda_star = exp( Xk * theta_star );
lambda_star = lambda_star ./ (1+sum( lambda_star,2 ));
lambda_g = sum( lambda_star , 2 ); 
lambda_g = min(lambda_g, 1-eps); lambda_g = max(lambda_g, eps);
llk = sum( log( 1-lambda_g ) ) + sum( sum( (nk_star').*log(lambda_star ./( 1-lambda_g )) ) );

ak = beta*ak + llk;
uk = beta*uk;
for m=1:size(nk_star,1)
	Lambda_star = diag( lambda_star(:,m) .* ( 1-lambda_star(:,m) ) );

    if (~isempty(UnsuppMks) && ~isempty(UnsuppCovs)) && ismember(m, UnsuppMks)
        mk_Cov = UnsuppCovs( find(UnsuppMks==m) );
        Xk_red = Xk; Xk_red(:,UnsuppCovs(mk_Cov)) = 0;
        
        ak = ak - 0.5 * theta_star(:,m)' * (Xk_red' * Lambda_star * Xk_red) * theta_star(:,m)...
                - sum( theta_star(:,m)' * (Xk_red' * (nk_star(m,:)' - lambda_star(:,m))) );
        uk(:,m) = uk(:, m) + Xk_red' * Lambda_star * Xk_red * theta_star(:,m)...
                           + Xk_red' * ( nk_star(m,:)' - lambda_star(:,m) );
        
        Bk( [1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , : ) = beta*Bk( [1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , : ) + Xk_red' * Lambda_star * Xk_red;
    
        LLk = LLk - theta_star(:,m)' * Bk([1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , :) * theta_star(:,m);
    
        grad = uk(:,m) - Bk([1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , :) * theta_star(:,m);
        grad(UnsuppCovs(mk_Cov)) = 0;
        
        if beta<0.975
        Pk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) = pinv( Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) );
        else
        Xk_red = Xk; Xk_red(:,UnsuppCovs(mk_Cov)) = 0;
        Yk = Pk([1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , :) * Xk_red';
        TMP = max(diag(Lambda_star),eps); TMP = diag(TMP);
        TMP2 = ones(1,size(Xk_red,2)); TMP2(UnsuppCovs(mk_Cov)) = 0; TMP2 = diag(TMP2);
        Pk([1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , :) = (TMP2 * Pk([1:size(Xk_red,2)] + (m-1)*size(Xk_red,2) , :) * TMP2 ...
                                                    - (Yk/(beta*eye(length(Lambda_star))/TMP + Xk_red*Yk))*Yk')/beta;
        end
    else
        ak = ak - 0.5 * theta_star(:,m)' * (Xk' * Lambda_star * Xk) * theta_star(:,m)...
                - sum( theta_star(:,m)' * (Xk' * (nk_star(m,:)' - lambda_star(:,m))) );
        uk(:,m) = uk(:, m) + Xk' * Lambda_star * Xk * theta_star(:,m)...
                           + Xk' * ( nk_star(m,:)' - lambda_star(:,m) );
        
        Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) = beta*Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) + Xk' * Lambda_star * Xk;
    
        LLk = LLk - theta_star(:,m)' * Bk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :) * theta_star(:,m);
    
        grad = uk(:,m) - Bk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :) * theta_star(:,m);
        
        if beta<0.975
        Pk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) = pinv( Bk( [1:size(Xk,2)] + (m-1)*size(Xk,2) , : ) );
        else
        Xk_tmp = Xk;
        Yk = Pk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :) * Xk_tmp';
        TMP = max(diag(Lambda_star),eps); TMP = diag(TMP);
        Pk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :) = (Pk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :)...
                                                    - (Yk/(beta*eye(length(Lambda_star))/TMP + Xk_tmp*Yk))*Yk')/beta;
        end
    end
    Biask = Biask + grad'*Pk([1:size(Xk,2)] + (m-1)*size(Xk,2) , :)*grad;
end
LLk = (0.5)*LLk + ak + sum( sum( theta_star.*uk ) );

end