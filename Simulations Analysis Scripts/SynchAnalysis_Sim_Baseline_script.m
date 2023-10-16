%% Pairwise Correlations of Smoothed Spike Trains
close all;
clear all;
clc

% addpath('HOMEDIR/Code/');

%% Load Data
load('SyncHist_SimData1.mat');
load('SyncHist_SimData2.mat');

L = size(n,1);
T = size(n,2);

%%
W=200;
K=T/W;

alpha = 0.05;
showfig = 1;
[rhoMean, CI_UP_Mean, CI_LO_Mean, rho, CI_upper, CI_lower] = SynchCorrelation(n, W, alpha, showfig);
[coeffVarMean, coeffVar, ISI] = SpikeRegularity(n, W, showfig);

%%
clear all;
clc;

load('SyncHist_SimData_Analyzed.mat');

clearvars -except lambda_star lambda incl_idx ord_idx

showfig = 1;
[rhoR, rhoR_CI_UP, rhoR_CI_LO, rho] = SynchCIF(lambda_star, lambda, incl_idx, ord_idx, showfig);
