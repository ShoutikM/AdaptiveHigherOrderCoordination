function [] = generate_KS_ACF_test(X, n, theta, p, lambda_est, conf, fsave)
% This function returns the quantiles for the KS test and the ACF values
% and lags. The function also generates the KS and ACF plots. Note that
% this is for a single model.
%
% Inputs:
%            X - Matrix of Covariates
%            n - Spike recordings
%        theta - Model Parameters
%            p - Number of History Terms for one neuron
%   lambda_est - Pre-computed CIFs (optional - doesn't require X or theta)
%         conf - conf% confidence interal shown (optional - default is 95%)
%        fsave - name of .fig file to save plots (optional)
%
% Shoutik Mukherjee

%% KS Test
if nargin<5
    lambda_est = zeros(size(n));
    for i=1:length(n)
        lambda_est(i) = exp(X(i,:)*theta(:,i))/(1+exp(X(i,:)*theta(:,i))); 
    end
end

% Analytic Correction
index = find(n);
q = -log(1-lambda_est);
r = rand(1);
ksi = sum(q(1:index(1)-1))-log(1-r*(1-exp(-q(index(1)))));
for i=1:length(index)-1
   r = rand(1);
   ksi = [ksi, sum(q(index(i)+1:index(i+1)-1))-log(1-r*(1-exp(-q(index(i+1)))))];
end
z = 1 - exp(-ksi); z(z==1)=1-eps; z(z==0)=eps;
z_exp = sort(z);
% Uniform Quantiles
y_k = [];
for i=1:length(z_exp)
   y_k = [y_k,(i-.5)/length(z_exp)];
end

%% ACF Test
gauss = norminv(z);
[acf, lag] = xcorr(gauss,'unbiased');
acf = acf(lag>0);
lag = lag(lag>0);

%% Plot Results
if ~exist('conf', 'var') || isempty(conf)
    conf = 0.95;
end

ks_conf = erfinv(conf);
acf_conf = sqrt(2)*erfinv(conf);

if exist('fsave','var')
    figure('Visible','off');
    subplot(1,2,1);
    plot(y_k, z_exp,'k');
    hold on; plot(y_k,y_k,'b'); plot(y_k, y_k+ks_conf/sqrt(length(z_exp)),'r-');
    plot(y_k, y_k-ks_conf/sqrt(length(z_exp)),'r-');hold off;
    xlim([0,1]);ylim([0,1]);
    ylabel('Empirical CDF'); xlabel('Uniform Quantiles');
    title('KS Plot');

    subplot(1,2,2);
    plot(lag,acf,'*');
    hold on;
    plot(lag, acf_conf/sqrt(length(z))*ones([1,length(lag)]), 'r-');
    plot(lag, -acf_conf/sqrt(length(z))*ones([1,length(lag)]), 'r-');
    hold off;
    title('ACF'); xlabel('Lag');
    axis([0 p -5*acf_conf/sqrt(length(z)) 5*acf_conf/sqrt(length(z))]);
    
    set(gcf, 'Visible', 'off', 'CreateFcn', 'set(gcf,''Visible'',''on'')')
    savefig(fsave);
else
    figure;
    subplot(1,2,1);
    plot(y_k, z_exp,'k');
    hold on; plot(y_k,y_k,'b'); plot(y_k, y_k+ks_conf/sqrt(length(z_exp)),'r-');
    plot(y_k, y_k-ks_conf/sqrt(length(z_exp)),'r-');hold off;
    xlim([0,1]);ylim([0,1]);
    ylabel('Empirical CDF'); xlabel('Uniform Quantiles');
    title('KS Plot');

    subplot(1,2,2);
    plot(lag,acf,'*');
    hold on;
    plot(lag, acf_conf/sqrt(length(z))*ones([1,length(lag)]), 'r-');
    plot(lag, -acf_conf/sqrt(length(z))*ones([1,length(lag)]), 'r-');
    hold off;
    title('ACF'); xlabel('Lag');
    axis([0 p -5*acf_conf/sqrt(length(z)) 5*acf_conf/sqrt(length(z))]);
end

end