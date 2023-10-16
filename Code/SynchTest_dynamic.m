function [Dev, Md, nu, gamma] = SynchTest_dynamic(theta_star, X_star, W, odds, ordRidx, alpha, beta, ncx2OPTS)
    M = size(theta_star, 1);
    K = size(theta_star, 2);
    
    ex_in = zeros(K,1);
    gamma = zeros(K, M);
    
    Md = length( ordRidx );
    Mr = M - Md;

%% Reduced Model Estimation
    SrCompidx = 1:M; SrCompidx(ordRidx) = [];
    
    XR_star = X_star(:, SrCompidx);
	thetaR_star = zeros(Mr, K);
    
    Xb_star = zeros(K, Mr);
    for k=1:K
        Xb_star(k,:) = sum( XR_star((k-1)*W+1:k*W,:), 1)/W;
    end

    X = zeros(size(Xb_star(1,:)))';
    for k=1:K
        X = beta*X + Xb_star(k,:)';
        for gditer=1:2000
            grad = W * ( X - ((1-beta^k)/(1-beta))*( exp(thetaR_star(:,k)) )/( 1+sum(exp(thetaR_star(:,k)))+sum(odds(ordRidx,k)) ) );
            thetaR_star(:,k) = thetaR_star(:,k) + alpha*grad;
        end
    end
    thetaR_star_tmp = thetaR_star;
    thetaR_star = log(odds);
    thetaR_star(SrCompidx, :) = thetaR_star_tmp;

%% Deviance Difference
    ll_diff = zeros(K,1);
    Xb_star = zeros(K, M);
    for k=1:K
        Xb_star(k,:) = sum( X_star((k-1)*W+1:k*W,:), 1)/W;
    end
    
    X = zeros(size(Xb_star(1,:)))';
    for k=1:K
        X = beta*X + Xb_star(k,:)';
        ll_diff(k) = W * (  (theta_star(:,k) - thetaR_star(:,k))' * X - ...
                    ( log( 1+sum(exp(theta_star(:,k))) ) - log( 1+sum(exp(thetaR_star(:,k))) ) )  );
        gamma(k,ordRidx) = ( thetaR_star(ordRidx,k)-theta_star(ordRidx,k) )';
        ex_in(k) = -sign( sum( gamma(k,:) ) );
    end
    
    Dev = (1+beta)*2*ll_diff;

%% Non-Centrality Parameter Estimation
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
    nu = nukSmth;
    
end
