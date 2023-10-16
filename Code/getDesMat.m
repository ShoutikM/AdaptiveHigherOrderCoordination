function [X, D] = getDesMat(L, n, p, histKern, stim, q, stimKern, G, dataBuff)
T = size(n,2);

H = [];
S = [];

if (~exist('dataBuff','var') || isempty(dataBuff))
    dataBuff = 1; 
    % 1: zero-pad data for initial covariate vectors
    % 0: pre-buffered data -- DON'T zero-pad
end

if ~( isempty(p) || isempty(histKern) ) % Population History
    if iscell(histKern) %different self- and cross-history kernels
        chistKern = histKern{1};
        shistKern = histKern{2};
        pCH = length(chistKern);
        pSH = length(shistKern);
        MaxBuff = max(sum(chistKern),sum(shistKern));
        
        l_self = p; %neuron to which self-history corresponds
        lls = 1:L; lls(l_self)=[];

        H = zeros( (L-1)*pCH + pSH , T );
        nhat = [ zeros( L , MaxBuff ) , n ];
        
        if dataBuff
            trange = [1:T];
        else
            trange = [MaxBuff+1:T];
        end
        for i=trange
            for idx=1:pSH
                    if idx==1
                        H(idx, i) = sum(nhat(l_self,i-shistKern(idx)+MaxBuff:i-1+MaxBuff));
                    else
                        H(idx, i) = sum(nhat(l_self,i-sum(shistKern(1:idx))+MaxBuff:i-1-sum(shistKern(1:idx-1))+MaxBuff));
                    end
            end
            
            llcnt = 0;
            for ll=lls
                llcnt = llcnt+1;
                for idx=1:pCH
                    if idx==1
                        H(pSH + (llcnt-1)*pCH+idx, i) = sum(nhat(ll,i-chistKern(idx)+MaxBuff:i-1+MaxBuff));
                    else
                        H(pSH + (llcnt-1)*pCH+idx, i) = sum(nhat(ll,i-sum(chistKern(1:idx))+MaxBuff:i-1-sum(chistKern(1:idx-1))+MaxBuff));
                    end
                end
            end
            
        end
        if ~dataBuff
            H(:,1:MaxBuff) = [];
        end
        
    else %same self- and cross-history kernels
        H = zeros(p*L,T);
        nhat = [zeros(L,sum(histKern)), n];
        
        if dataBuff
            trange = [1:T];
        else
            trange = [sum(histKern)+1:T];
        end
        for i=trange
            for l=1:L
                for idx=1:p
                    if idx==1
                        H((l-1)*p+idx, i) = sum(nhat(l,i-histKern(idx)+sum(histKern):i-1+sum(histKern)));
                    else
                        H((l-1)*p+idx, i) = sum(nhat(l,i-sum(histKern(1:idx))+sum(histKern):i-1-sum(histKern(1:idx-1))+sum(histKern)));
                    end
                end
            end
        end
        if ~dataBuff
            H(:,1:sum(histKern)) = [];
        end
    end
    
end

if ~( nargin<5 || isempty(q) || isempty(stimKern) || isempty(stim) ) % Stimulus
    M = size(stim,1);
    S = zeros(q*M, T);
    stimhat = [zeros(M,sum(stimKern)), stim];
    if dataBuff
        trange = [1:T];
    else
        trange = [sum(stimKern)+1:T];
    end
    for i=trange
        for idx=1:q
            if idx==1
                S([1:M] + (idx-1)*M, i) = sum(stimhat(:, i-stimKern(idx)+sum(stimKern):i-1+sum(stimKern)),2);                
            else
                S([1:M] + (idx-1)*M, i) = sum(stimhat(:, i-sum(stimKern(1:idx))+sum(stimKern):i-1-sum(stimKern(1:idx-1))+sum(stimKern)),2);
            end
        end
    end
	if ~dataBuff
        S(:,1:sum(stimKern)) = [];
	end
    
end

Teff = length(trange);
if (~exist('G', 'var') || isempty(G))
    X = [ones(Teff,1), S', H'];
else
    X = [ones(Teff,1), (S' * G), H'];
end
d=[1 std(X(:,2:end))]; d(d==0)=1;
D=diag(1./d);
X=(X)*D;

end