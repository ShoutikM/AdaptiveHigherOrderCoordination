function [rhoR, rhoR_CI_UP, rhoR_CI_LO, rho] = SynchCIF(lambda_star, lambda, incl_idx, ord_idx, showfig)

if ~exist('showfig','var')
    showfig = 0;
end

T = size(lambda_star,1);
L_star = size(lambda_star,2);
L = size(lambda,2);

lambdaComp = 1 - lambda;
normalization = 1;
% normalization = sqrt( prod(lambda,2) .* prod(lambdaComp,2) );

bi_incl_idx = de2bi(incl_idx, L);

rho = zeros(T,L_star);
for mm=1:L_star
    idx = find(bi_incl_idx(mm,:)==1);
    idxComp = find(bi_incl_idx(mm,:)==0);
    
    IndepCIF = prod(lambda(:,idx),2) .* prod(lambdaComp(:,idxComp),2);
    rho(:,mm) = ( lambda_star(:,mm) - IndepCIF )./normalization; 
end

rhoR = cell(L,1);
rhoR_CI_UP = cell(L,1);
rhoR_CI_LO = cell(L,1);

for r = 1:L
    ordRidx = find(ord_idx==r);
    if isempty(ordRidx) || r==1
        rhoR{r} = zeros(T,1);
        rhoR_CI_UP{r} = zeros(T,1);
        rhoR_CI_LO{r} = zeros(T,1);
        
        continue;
    end
    
    rhoR{r} = nanmean(rho(:,ordRidx),2);
    rhoR_CI_UP{r} = rhoR{r} + 2 * nanstd(rho(:,ordRidx),[],2)/sqrt(length(ordRidx));
    rhoR_CI_LO{r} = rhoR{r} - 2 * nanstd(rho(:,ordRidx),[],2)/sqrt(length(ordRidx));
    
end

if showfig
    figure;
    for ii=2:size(rhoR,1)
        if sum(abs(rhoR{ii}))==0
            continue;
        end
        subplot(size(rhoR,1)-1,1,ii-1);
        plot(rhoR{ii},'r');
        hold on;
        fill([1:T, fliplr(1:T)], [rhoR_CI_UP{ii}; fliplr(rhoR_CI_LO{ii})], 'r', 'FaceAlpha',0.2, 'EdgeColor', 'none')
        plot(1:T, zeros(1,T), 'k--');
        hold off;
        xlim([1,T]); ylim([-1.1, 1.1] * max( max(abs(rhoR_CI_UP{ii})), max(abs(rhoR_CI_LO{ii})) ));
    end
end

end