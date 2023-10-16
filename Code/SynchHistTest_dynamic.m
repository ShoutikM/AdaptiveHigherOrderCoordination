function [Dev, Md, nu, nukFilt, nukSmth,...
    theta_star_null, lambda_star_null, lambda_g_null,...
    LLk_red, Biask_red, gamma, gamma_KF, gamma_FB] = SynchHistTest_dynamic(n_star, X, beta, ordRidx, theta_star,...
                                                                           LLk_full, Biask_full, odds, ncx2OPTS)

Md = length(ordRidx);
K = size(theta_star,1);
T = size(n_star, 2);
W = T/K;

lambda_star_null = zeros(T, size(n_star,1));
lambda_g_null = zeros(T,1);

LLk_red = zeros(K,1);
Biask_red = zeros(K,1);
ll_diff = zeros(K,1);
Bias_diff = zeros(K,1);
Dev = zeros(K,1);
ex_in = zeros(K,1);
gamma = zeros(K, size(n_star,1));

ak_null = 0;
bk_null = zeros( size(X,2), size(n_star,1) );
Bk_null = zeros(size(X,2)*size(n_star,1), size(X,2));
Pk = [];
for mm=1:size(n_star,1)
    Pk = cat(1, Pk, (1-beta)*eye(size(X,2)));
end

for k=1:K
    Xk = X( (k-1)*W+1 : k*W , :);
    for mm=ordRidx'
        gamma(k,mm) = mean( odds([1:W] + (k-1)*W, mm) - Xk * squeeze(theta_star(k, mm, :)) );
    end
    ex_in(k) = -sign(sum(gamma(k,:)));
end

Nem1 = ncx2OPTS.Nem_g;
rho = ncx2OPTS.rho_g;
sig2w = (ncx2OPTS.w_g)*var(gamma); sig2w(sig2w==0) = 1;
sig2v = (ncx2OPTS.v_g)*var(gamma); sig2v(sig2v==0) = 1;
[gamma_KF, gamma_FB, sig2w, sig2v] = KalmanSmoothingFB(gamma, sig2w, sig2v, Nem1, rho);
gamma_Sm = gamma_FB;

theta_star_null = theta_star;
for k=1:K

    Xk = X( (k-1)*W+1 : k*W , :);
    nk_star = n_star(:, (k-1)*W+1 : k*W );

    tht_star = squeeze(theta_star(k,:,:))';

    [tht_star_null, LLk_red(k), Biask_red(k),...
                    ak_null, bk_null, Bk_null, Pk] = ReducedModel(tht_star, gamma_Sm(k,:), ordRidx, ...
                    beta, nk_star, Xk, ak_null, bk_null, Bk_null, Pk, ncx2OPTS.GD_iter, ncx2OPTS.alpha);

    theta_star_null(k,:,:) = tht_star_null';

    %%% Reduced Model's lambdas
    lambda_star_null((k-1)*W+1 : k*W , :) = exp( Xk* tht_star_null);
    lambda_star_null((k-1)*W+1 : k*W , :) = lambda_star_null((k-1)*W+1 : k*W , :) ./ (1+sum( lambda_star_null((k-1)*W+1 : k*W , :) , 2 ));
    lambda_g_null((k-1)*W+1 : k*W) = sum( lambda_star_null((k-1)*W+1 : k*W , :) , 2 );
    
    %%% Deviance
    ll_diff(k) = LLk_full(k) - LLk_red(k);
    Bias_diff(k) = (Biask_full(k) - Biask_red(k));
    Dev(k) = (1+beta)*( 2*ll_diff(k)-Bias_diff(k) );
    
end

    TmpNu = Dev-Md;
    sz0 = (1e-3)/var(TmpNu);
    
    if exist('ncx2OPTS','var')
        sz0 = ncx2OPTS.initF / var(TmpNu);
        rho = ncx2OPTS.rho;
        NN = ncx2OPTS.NN;
        Nem = ncx2OPTS.Nem;
    else
        rho = 1;
        NN = 20;
        Nem = 5;
    end
    [nukSmth,nukSmthU,nukSmthL,nukFilt,nukFiltU,nukFiltL] = NoncentChi2FiltSmooth(Dev,Md,sz0,rho,NN,Nem);
    nu = nukFilt;
    nu = nukSmth;

end

function [tht_star_null, LLk_red, Biask_red,...
          ak_null, uk_null, Bk_null, Pk] = ReducedModel(tht_star, gamma_Sm, ordRidx,...
                                                        beta, nk_star, Xk, ak_null, uk_null, Bk_null, Pk, GD_iter, alpha)
%%% Fix r-th order mk rates and adjust other mks' rates (initialization)

tht_star_null = 0*tht_star;
for mm=ordRidx'
	tht_star_null(1,mm) = tht_star(1,mm) + gamma_Sm(mm);
end
    
%%% Estimate Reduced Model and its Likelihood
UnsuppMks = ordRidx'; s = length(find(tht_star~=0)) - length(ordRidx);
UnsuppCovs = ones(size(UnsuppMks));

[tht_star_null, uk_null, Bk_null, ak_null, Pk, LLk_red, Biask_red] = adomp_mGLM(nk_star, Xk, uk_null, Bk_null, ak_null, Pk,...
                                                                                s, alpha, beta, GD_iter, UnsuppMks, UnsuppCovs, tht_star_null);

end

function [xkk, xkK, sig2w, sig2v] = KalmanSmoothingFB(yk, sig2w, sig2v, Nem, rho)
K = size(yk,1);
M = size(yk,2);

if ~exist('rho', 'var') || isempty(rho)
    rho = 1;
end
rho2=rho^2;
for EMiter=1:Nem
    %%% KF -- forward step
    xkk = zeros(K, M);
    sig2kk = zeros(K, M); sig2kk(1,:) = 1;   
    for k=1:K
        xkk(k+1,:) = rho*xkk(k,:);
        sig2kk(k+1,:) = rho2*sig2kk(k,:) + sig2w;
        
        xkk(k+1,:) = xkk(k+1,:) + ( yk(k,:) - xkk(k+1,:) ).*( sig2kk(k+1,:) ./( sig2v + sig2kk(k+1,:) ) );
        sig2kk(k+1,:) = sig2kk(k+1,:) .* ( sig2v ./( sig2v + sig2kk(k+1,:) ) );
    end
    
    %%% Fixed Interval Smoothing -- backward step
    xkK = xkk;
    sig2kK = sig2kk;    
    for k=K:-1:1
        xkK(k,:) = xkk(k,:) + (xkK(k+1,:) - rho*xkk(k,:) ) .* ( rho*sig2kk(k,:)./( sig2kk(k,:) + sig2w ) );
        sig2kK(k,:) = sig2kk(k,:) + (sig2kK(k+1,:) - sig2kk(k,:) - sig2w ) .* ( rho*sig2kk(k,:)./( sig2kk(k,:) + sig2w ) ).^2;
    end
    
    %%% EM update of sig2w and sig2v
    Ew = 0;
    Ev = 0;
    for k=1:K
        Ew = Ew + (xkK(k+1,:) - rho*xkK(k,:)).^2 + ( 1-2*( rho*sig2kk(k,:)./(sig2kk(k,:)+sig2w) ) ) .* sig2kK(k+1,:) + sig2kK(k,:);
        Ev = Ev + (yk(k) - xkK(k+1)).^2 + sig2kK(k+1,:);
    end
    
    sig2w = (1/K)*Ew;
	sig2v = (1/K)*Ev;
    
end

xkk = xkk(2:end,:);
xkK = xkK(2:end,:);

end
