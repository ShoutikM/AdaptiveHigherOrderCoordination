function [rhoMean, CI_UP_Mean, CI_LO_Mean, rho, CI_upper, CI_lower] = SynchCorrelation(n, W, alpha, showfig)

if ~exist('showfig','var')
    showfig = 0;
end

L = size(n,1);
T = size(n,2);

K=T/W;

%%% Smooth Spike Trains
n_smooth = 0*n;
for ll=1:L
    rate = mean(n(ll,:)); sig2 = 1/rate;
    SmKernel = fspecial('gaussian', [1,500], sig2); SmKernel = SmKernel./max(SmKernel);
    n_smooth(ll,:) = conv(n(ll,:),SmKernel, 'same');
end

%%% Synchronous Output -- for single-trials, this is an estimate of the
%%% correlation at each time sample
rho = zeros(nchoosek(L,2),K);
CI_upper = zeros(nchoosek(L,2),K);
CI_lower = zeros(nchoosek(L,2),K);
cnt=0;
for ii=1:L
    for jj=ii+1:L
        cnt=cnt+1;
        for kk=1:K
            [R, P, RLO, RUP] = corrcoef(n_smooth(ii,[1:W]+(kk-1)*W), n_smooth(jj,[1:W]+(kk-1)*W));
            rho(cnt, kk) = R(1,2);
            CI_lower(cnt,kk) = RLO(1,2);
            CI_upper(cnt,kk) = RUP(1,2);
            
            if isnan(rho(cnt,kk))
                rho(cnt,kk)=0;
                CI_lower(cnt,kk) = ( exp( 2*(-norminv(1-alpha/2)/sqrt(W-3)) )-1 )/( exp( 2*(-norminv(1-alpha/2)/sqrt(W-3)) )+1 );
                CI_upper(cnt,kk) = ( exp( 2*(norminv(1-alpha/2)/sqrt(W-3)) )-1 )/( exp( 2*(norminv(1-alpha/2)/sqrt(W-3)) )+1 );
            end
            
        end
    end
end

rhoMean = nanmean(rho);
CI_UP_Mean = nanmean(CI_upper);
CI_LO_Mean = nanmean(CI_lower);

if showfig
    figure;
    plot(rhoMean,'r');
    hold on;
    fill([1:K, fliplr(1:K)], [CI_UP_Mean, fliplr(CI_LO_Mean)], 'r', 'FaceAlpha',0.2, 'EdgeColor', 'none')
    plot(1:K, zeros(1,K), 'k--');
    hold off;
    xlim([1,K]); ylim([-1, 1]);
end

end