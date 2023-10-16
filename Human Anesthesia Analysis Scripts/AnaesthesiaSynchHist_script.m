close all;
clear all;
clc

addpath('HOMEDIR/Code/');

%%
load('mg29rasta.mat');
spks = Y; clear Y;

neur_num = find(sum(spks, 2)>900);

F = 50;
tmp = [zeros(numel(neur_num), 1e6 - length(spks)), spks(neur_num, :)];
n = tmp(:, 1:F:end);
for ii=2:F
    n = n + tmp(:, ii:F:end);
end

n = double(n>0);

L = size(n,1);
L_star = 2^L-1;
Nthr = 15;
T = size(n,2);
GD_iter = 750;
alpha = 1e-4; % Learning Rate
sigma = 0.01; % Size of Significance Test

hKern = [1,3,9];
p = length(hKern);
M = 1 + L*p;

W=10;
K=T/W;
beta = 0.99;

if mod(K,1)
    T1 = ceil(K) * W;
    Yhat = [zeros(size(n,1), T1-T) n];
    
    n=Yhat;
    T=T1;
    K = ceil(K);
end

%% Data partition for cross-validation

if ~exist('incl_idx','var')
incl_idx = [];
X_star = [];
id_list = 1:L_star;
for idx = id_list
    bi_idx = de2bi(idx,L);
    idx0 = find(bi_idx==0); idx1 = find(bi_idx==1);
    
    tmp1 = prod(n(idx1,:),1);
    if sum(tmp1) == 0
        id_gr = id_list(id_list>=idx);
        for ii=1:numel(idx1)
            id_gr_bits(ii,1:numel(id_gr)) = bitget(id_gr, idx1(ii));
        end
        elim_id = find(sum(id_gr_bits)==numel(idx1));
        id_list( ismember(id_list, elim_id + idx-1) ) = [];
    else
        tmp = max(0, prod(n(idx1,:),1)-double(sum(n(idx0,:),1)>0))';
        if sum(tmp)>Nthr
            X_star = [X_star tmp];
            incl_idx = [incl_idx idx];
        end        
    end
end
end

ord_idx = sum( de2bi(incl_idx, L) ,2);

if ~exist('n_star','var')
    n_star = X_star';
end
if ~exist('ng','var')
    ng = sum(n_star, 1);
end

n_star = X_star';
ng = sum(n_star, 1);

%%
n1_star = n_star(:,1:2:end);
n2_star = n_star(:,2:2:end);
n1 = n(:,1:2:end);
n2 = n(:,2:2:end);

X = getDesMat(L, n, p, hKern);
X1 = getDesMat(L, n1, p, hKern);
X2 = getDesMat(L, n2, p, hKern);

%% Dynamic Analysis
disp('Cross-Validating...')
%%% Cross validate for sparsity level
cv_max = ceil((M * numel(incl_idx))/4);
[~, LL1] = omp_mGLM_cv(n1_star, X1, n2_star, X2, cv_max, alpha, GD_iter);
[~, LL2] = omp_mGLM_cv(n2_star, X2, n1_star, X1, cv_max, alpha, GD_iter);
[~, s] = max(LL1+LL2); s = s*2;

disp('Adaptive Filtering...')
%%% AdOMP
ak = 0;
uk = zeros( size(X,2), size(n_star,1) );
Bk = zeros( size(n_star,1) * size(X,2), size(X,2) );
Pk = [];
for mm=1:size(n_star,1)
    Pk = cat(1, Pk, (1-beta)*eye(size(X,2)));
end
Bias = zeros(K,1);
LL = zeros(K,1);
theta_star = zeros(K, size(n_star,1), size(X,2));
lambda_star = zeros(T, size(n_star,1));
lambda_g = zeros(T,1);
for k=1:K
    Xk = X( (k-1)*W+1 : k*W , :);
    nk_star = n_star(:, (k-1)*W+1 : k*W );
    
    [tht_star, uk, Bk, ak, Pk, LLk, Biask] = adomp_mGLM(nk_star, Xk, uk, Bk, ak, Pk, s, alpha, beta, GD_iter,[],[],[]);
    theta_star(k, :, :) = tht_star';
    Bias(k) = Biask;
    LL(k) = LLk;
    
    lambda_star((k-1)*W+1 : k*W , :) = exp( Xk*tht_star );
    lambda_star((k-1)*W+1 : k*W , :) = lambda_star((k-1)*W+1 : k*W , :) ./ (1+sum( lambda_star((k-1)*W+1 : k*W , :) , 2 ));
    lambda_g((k-1)*W+1 : k*W) = sum( lambda_star((k-1)*W+1 : k*W , :) , 2 );
end

%
%%% Numerical correction for log-odds computational stability
for t=1:T
    tmp = lambda_star(t,:);
    IDX_ones = find(tmp == 1);
    IDX_others = 1:length(tmp); IDX_others(IDX_ones) = [];
    if ~isempty(IDX_ones)
        tmp(IDX_ones) = 1-eps*length(IDX_others);
        tmp(IDX_others) = tmp(IDX_others) + eps;
    end
    lambda_star(t,:) = tmp;
end
%}

%% Model Goodness-of-Fit
% % for l=1:numel(incl_idx)
% for l=find(ord_idx==1)'
%     generate_KS_ACF_test([], n_star(l,:), [], 1500, lambda_star(:,l)' );
% %     generate_KS_ACF_test([], n_star(l,:), [], 500, Lambda_star_null{2}(:,l)' );
% end

%% Dynamic Statistical Test of rth order Synch
lambda = zeros(T,L);
bi_incl_idx = de2bi(incl_idx, L);
for l=1:L
    tmp = find( bi_incl_idx(:,l) );
    lambda(:, l) = sum( lambda_star(:, tmp), 2 );
end

odds = zeros(T, numel(incl_idx));
for l=1:numel(incl_idx)
    k_tmp = find( bi_incl_idx(l,:) );
    odds(:,l) = log( prod( lambda(:,k_tmp), 2 ) ./ prod( 1 - lambda(:,k_tmp), 2 ) );
end

ord_idx = sum(de2bi(incl_idx, L),2);
ord = 1:L;

ncx2OPTS.GD_iter = GD_iter;
ncx2OPTS.alpha = alpha;

ncx2OPTS.rho_g = 3/4;
ncx2OPTS.w_g = 5e1;
ncx2OPTS.v_g = 5e1;
ncx2OPTS.Nem_g = 10;

ncx2OPTS.initF = 1e-1;
ncx2OPTS.rho = 1;
ncx2OPTS.NN = 20;
ncx2OPTS.Nem = 5;

Jstat_ord = cell(L,1);
h_ord = cell(L,1);
Ms = zeros(L,1);
Devs = cell(L,1);
nus = cell(L,1);
nusFilt = cell(L,1);
nusSmth = cell(L,1);

Tht_star_null = cell(L,1);
Lambda_star_null = cell(L,1);
Lambda_g_null = cell(L,1);
LLk_null = cell(L,1);
Biask_null = cell(L,1);
Gamma = cell(L,1);
GammaKF = cell(L,1);
GammaFB = cell(L,1);

disp('Statistical Inference...')
parfor r = ord
    ordRidx = find(ord_idx==r);
    if isempty(ordRidx) || r==1
        Jstat_ord{r} = zeros(K,1);
        h_ord{r} = zeros(K,1);
        Ms(r) = length(ordRidx) * (M-1);
        Devs{r} = zeros(K,1);
        nus{r} = zeros(K,1);
        nusFilt{r} = zeros(K,1);
        nusSmth{r} = zeros(K,1);
        
        Tht_star_null{r} = zeros(size(theta_star));
        Lambda_star_null{r} = zeros(size(lambda_star));
        Lambda_g_null{r} = zeros(size(lambda_g));
        LLk_null{r} = zeros(K,1);
        Biask_null{r} = zeros(K,1);
        Gamma{r} = zeros(K,1);
        GammaKF{r} = zeros(K,1);
        GammaFB{r} = zeros(K,1);
        
        continue;
    end
    
    %%% Reduced Model, Deviance Difference and Non-Centrality Parameter
    [Dev, Md, nu, nukFilt, nukSmth,...
    theta_star_null, lambda_star_null, lambda_g_null,...
    LLk_red, Biask_red, gamma, gamma_KF, gamma_FB] = SynchHistTest_dynamic(n_star, X, beta, ordRidx, theta_star,...
                                                                           LL, Bias, odds, ncx2OPTS);

    %%% Hypothesis Test Outcome and J-Statistic
    h = and(Md>0, (1 - sigma) < chi2cdf(Dev, Md));    
    Jstat = zeros(K,1);
    for k=1:K
        Jstat(k) = ( 1 - sigma - ncx2cdf( chi2inv( 1-sigma, Md), Md, nu(k) ) );
    end
    ex_in = -sign(sum(gamma_FB,2));
    Jstat = ex_in.*(h.*Jstat);
    
    Jstat_ord{r} = Jstat;
	h_ord{r} = h;
	Ms(r) = Md;
	Devs{r} = Dev;
	nus{r} = nu;
    nusFilt{r} = nukFilt;
    nusSmth{r} = nukSmth;
    
    Tht_star_null{r} = theta_star_null;
    Lambda_star_null{r} = lambda_star_null;
    Lambda_g_null{r} = lambda_g_null;
    LLk_null{r} = LLk_red;
    Biask_null{r} = Biask_red;
    Gamma{r} = gamma;
    GammaKF{r} = gamma_KF;
    GammaFB{r} = gamma_FB;
    
end

%%
disp('Plotting...')
Jstat_im = zeros(L, K);
for r=1:L
    Jstat_im(r,:) = Jstat_ord{r}';
end
h_im = zeros(L, K);
for r=1:L
    h_im(r,:) = h_ord{r}';
end

%%
n_Rords = zeros(L, size(n,2));
for r = 1:L
    tmp = sort(find(ord_idx == r), 'ascend');

    n_Rords(r,:) = sum( n_star(tmp,:) , 1 );

end

figure; set(gcf, 'Position', [75, 360, 1500, 600]);

subplot(6,1,5:6)
imagesc(kron(Jstat_im(2:end,:), ones(1,1)), [-1 1]); colormap redblue;
yticks('');
xticks([0:T/6:T]/W); xticklabels('');

tmp = n_Rords;
hspacing = 5; %>=1, integer
vspacing = 1.5; %>=1
subplot(6,1,3:4)
hold on;
for ii=1:size(tmp,2)
    for jj=2:size(tmp,1)
        if tmp(jj,ii)
            set(line, 'XData', [hspacing,hspacing]*ii, 'YData', (size(tmp,1) - jj)*vspacing+[-0.5, 0.5]+1, 'Color', 'k');
        end
    end
end
hold off;
ylim([vspacing/2 vspacing*(size(tmp,1))-vspacing/2]); yticklabels(''); yticks('');
xlim([0, T*hspacing]); xticks([0:T/6:T]*hspacing); xticklabels('');

tmp = n(:, :);
hspacing = 5; %>=1, integer
vspacing = 1.5; %>=1
subplot(6,1,1:2)
hold on;
for ii=1:size(tmp,2)
    for jj=1:size(tmp,1)
        if tmp(jj,ii)
            set(line, 'XData', [hspacing,hspacing]*ii, 'YData', jj*vspacing+[-0.5, 0.5]+1, 'Color', 'k');
        end
    end
end
hold off;
ylim([vspacing/2 vspacing*(size(tmp,1))+2+vspacing/2]); yticklabels(''); yticks('');
xlim([0, T*hspacing]); xticks([0:T/6:T]*hspacing); xticklabels('');
