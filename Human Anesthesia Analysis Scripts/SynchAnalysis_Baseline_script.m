%% Controls for Synchrony Analysis
close all;
clear all;
clc

% addpath('HOMEDIR/Code/');

%% Simulated Network
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
T = size(n,2);

%% Pearson Correlation and Spiking Irregularity Measure (Canada, PLoS 2021)
W=200;
K = T/W;
alpha = 0.05;
showfig = 1;
[rhoMean, CI_UP_Mean, CI_LO_Mean, rho, CI_upper, CI_lower] = SynchCorrelation(n, W, alpha, showfig);
[coeffVarMean, coeffVar, ISI] = SpikeRegularity(n, W, showfig);
