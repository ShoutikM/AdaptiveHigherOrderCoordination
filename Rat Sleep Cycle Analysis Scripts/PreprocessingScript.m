%% Preprocessing script

%% Load Full duration of spiking data at downsampled rate fs
if ~exist('basename','var')
    basename = 'BWRat19_032513';
end

if ~exist('fs', 'var')
    fs = 100; % Hz -- downsampled rate or 1/binsize
end

if ~exist('hKern', 'var')
    Buff = 0;
else
    Buff = sum(hKern);
end

if ~exist('W', 'var')
    W = 1;
end

load([basename '\' basename '_BasicMetaData.mat']);
fs_data = bmd.Par.SampleRate;
MaxT = max(bmd.RecordingFileIntervals);
clear bmd;

load([basename '\' basename '_CellIDs.mat']);
load([basename '\' basename '_SStable.mat']); clear S_TsdArrayFormat shank cellIx;

WSRestrictedIntervals = load([basename '\' basename '_WSRestrictedIntervals.mat']);
WkTime = WSRestrictedIntervals.WakeTimePairFormat;
SlpTime = WSRestrictedIntervals.SleepTimePairFormat;
REMTime = WSRestrictedIntervals.REMEpisodeTimePairFormat;
SWSTime = WSRestrictedIntervals.SWSEpisodeTimePairFormat;

SpkTimes = S_CellFormat; clear S_CellFormat;
for ii=1:length(SpkTimes)
    SpkTimes{ii} = unique( ceil( SpkTimes{ii}*fs ) ); % Bin indices of spike times (binSize = 1/fs sec)
end
MaxT = ceil(MaxT*fs);
if MaxT==Inf
    MaxT = ceil(SlpTime(2)*fs);
end

SpkTimes_pE = SpkTimes(CellIDs.EAll); NEcells = length(SpkTimes_pE);
SpkTimes_pI = SpkTimes(CellIDs.IAll); NIcells = length(SpkTimes_pI);

nE = zeros(NEcells, MaxT);
nI = zeros(NIcells, MaxT);

for ii=1:NEcells
    nE(ii,SpkTimes_pE{ii}) = 1;
end
for ii=1:NIcells
    nI(ii,SpkTimes_pI{ii}) = 1;
end

nAll = [nE; nI];

%% Select response window

%%% Wake-Sleep-Wake changes
BiasBuff = 50 / (fs/100);
WSWCycle = 2;
TnREM = 40;
T0 = (SWSTime(WSWCycle,2)*fs)+1-TnREM*fs-BiasBuff*fs-Buff;
T1 = (SWSTime(WSWCycle+1,1)*fs+TnREM*fs);
TimeInterval = [ T0 : T1 ];
REMon = Buff + TnREM*fs + BiasBuff*fs + 1;
REMoff = length(TimeInterval) - TnREM*fs + 1;

nE = nE(:,TimeInterval);
nI = nI(:,TimeInterval);
nAll = nAll(:,TimeInterval);

[~, Eidx] = sort(sum(nE,2), 'descend');
nE = nE(Eidx(1:size(nI,1)),:);

save([basename '_PreProcessed_fs' num2str(fs)], 'nE', 'nI', 'nAll', 'REMon', 'REMoff', 'TnREM');

%% Spiking Raster
%{
tmp = n(:,:);
T = length(tmp);
hspacing = 5; %>=1, integer
vspacing = 1.5; %>=1
figure;
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
xlim([0, T*hspacing]); xticks([0:T/10:T]*hspacing); xticklabels('');
%}