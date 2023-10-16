%% Controls for Synchrony Analysis
close all;
clear all;
clc

% addpath('HOMEDIR/Code/');

%%
basename = 'BWRat19_032513';
fs = 200;

PreprocessingScript

clearvars -except basename fs hKern BiasBuff;

%%
load([basename '_PreProcessed_fs' num2str(fs) '.mat']);

% n = nE;
n = nI;

L = size(n,1);
L_star = 2^L-1;
Nthr = 10;
T = size(n,2);
alpha = 1e-4; % Learning Rate
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

BiasBuffT = fs*BiasBuff;
BiasBuffK = BiasBuffT/W;

L = size(n,1);
T = size(n,2);

%% Pearson Correlation and Spiking Irregularity Measure (Canada, PLoS 2021)
W=200;
K = T/W;
alpha = 0.05;
showfig = 1;
[rhoMean, CI_UP_Mean, CI_LO_Mean, rho, CI_upper, CI_lower] = SynchCorrelation(n, W, alpha, showfig);
[coeffVarMean, coeffVar, ISI] = SpikeRegularity(n, W, showfig);
