close all;
clear all;
clc

% addpath('HOMEDIR/Code/');

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
alpha = 1e-3; % Learning Rate
sigma = 0.01; % Size of Significance Test

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
        
%% Dynamic Analysis
M = length(incl_idx);

% Gradient Descent
theta_star = zeros(M,K);
lambdastar_est = zeros(M,K);
lambdagstar_est = zeros(1,K);

Xb_star = zeros(K, M);
for k=1:K
	Xb_star(k,:) = sum( X_star((k-1)*W+1:k*W,:), 1)/W;
end

X = zeros(size(Xb_star(1,:)))';
for k=1:K
	X = beta*X + Xb_star(k,:)';
    for gditer=1:2000
        grad = W * ( X - ((1-beta^k)/(1-beta))*( exp(theta_star(:,k)) )/( 1+sum(exp(theta_star(:,k))) ) );
        theta_star(:,k) = theta_star(:,k) + alpha*grad;
    end
    
	lambdastar_est(:,k) = exp(theta_star(:,k))/( 1+sum( exp(theta_star(:,k)) ) );
    lambdagstar_est(k) = sum(lambdastar_est(:,k));
end

%% KS/ACF Tests
% % for l=1:M
% for l=2
%     generate_KS_ACF_test([], n_star(l,:), [], 500, kron( lambdastar_est(l,:), ones(1,W) ) );
% end

%% Statistical Inference (Dynamic)
% Set null nalues for out-of-support parameters
lambda_est = zeros(L,K);
bi_incl_idx = de2bi(incl_idx, L);
for l=1:L
    tmp = find( bi_incl_idx(:,l) );
    lambda_est(l,:) = sum( lambdastar_est(tmp, :) );
end

bi_incl_idx = de2bi(incl_idx, L);
odds = zeros(M, K);
for l=1:M
    k_tmp = find( bi_incl_idx(l,:) );
    odds(l,:) = prod( lambda_est(k_tmp,:), 1 ) / prod( 1 - lambda_est(k_tmp,:), 1 );
end

ord = 1:L;

ncx2OPTS.initF = 1e-1;
ncx2OPTS.rho = 1;
ncx2OPTS.NN = 20;
ncx2OPTS.Nem = 5;

Jstat_ord = cell(numel(ord),1);
h_ord = cell(numel(ord),1);
Ms = zeros(numel(ord),1);
Devs = cell(numel(ord), 1);
nus = cell(numel(ord), 1);

ordRidx = [];
for r = ord
    ordRidx = find(ord_idx==r);
    if isempty(ordRidx) || r==1
        Jstat_ord{r} = zeros(K,1);
        h_ord{r} = zeros(K,1);
        Ms(r) = numel(ordRidx);
        Devs{r} = zeros(K,1);
        nus{r} = zeros(K,1);
        continue;
    end
    
    [Dev, Md, nu, gamma] = SynchTest_dynamic(theta_star, X_star, W, odds, ordRidx, alpha, beta, ncx2OPTS);
    
	h = and(Md>0, (1 - sigma) < chi2cdf(Dev, Md));
    Jstat = zeros(K,1);
    for k=1:K
        Jstat(k) = ( 1 - sigma - ncx2cdf( chi2inv( 1-sigma, Md), Md, nu(k) ) );
    end
    ex_in = -sign(sum(gamma,2));
    Jstat = ex_in.*(h.*Jstat);
    
    Jstat_ord{r} = Jstat;
    h_ord{r} = h;
    Ms(r) = Md;
    Devs{r} = Dev;
    nus{r} = nu;
end

%%
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
