function [coeffVarMean, coeffVar, ISI] = SpikeRegularity(n, W, showfig)

if ~exist('showfig','var')
    showfig = 0;
end

L = size(n,1);
T = size(n,2);

K=T/W;

ISI = cell(L,K);
coeffVar = zeros(L,K);
for ll=1:L
    tLast = 0;
    for kk = 1:K
        nk = n(ll, (kk-1)*W+[1:W]);
        ts = find(nk);
        if ~isempty(ts)
%             ISI{ll,kk} = [(kk-1)*W+ts(1) - tLast, diff(ts)];
            ISI{ll,kk} = [diff(ts)];
            tLast = (kk-1)*W+ts(end);
        end
        
        coeffVar(ll,kk) = std(ISI{ll,kk})/mean(ISI{ll,kk});
    end
end

coeffVarMean = nanmean(coeffVar);
coeffVarStd = nanstd(coeffVar,[],1);
CI_UP_Mean = coeffVarMean + 2*coeffVarStd/sqrt(L);
CI_LO_Mean = coeffVarMean - 2*coeffVarStd/sqrt(L);

if showfig
    figure;
    plot(coeffVarMean,'r');
    hold on;
    fill([1:K, fliplr(1:K)], [CI_UP_Mean, fliplr(CI_LO_Mean)], 'r', 'FaceAlpha',0.2, 'EdgeColor', 'none')
    plot(1:K, ones(1,K), 'k--');
    hold off;
    xlim([1,K]); %ylim([0, 5]);
end

end