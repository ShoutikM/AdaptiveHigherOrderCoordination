This repository contains the analysis scripts and codes for the results reported in: 
Adaptive modeling and inference of higher-order coordination in neuronal assemblies: a dynamic greedy approach

Contact information:
Shoutik Mukherjee: smukher2@umd.edu
Behtash Babadi: behtash@umd.edu


The following data and scripts were used to generate the simulated results shown in Fig. 3 and Fig. 4
Simulations Analysis Scripts:
	SyncHist_SimData1.mat:							Simulated spiking data (Figure 3)
	SyncHist_SimData2.mat:							Simulated spiking data (Figure 4)
	SyncHist_SimData1_Analyzed.mat:					History-dependent analysis output for SimData1, used for Baseline Comparisons
	SyncHist_SimData2_Analyzed.mat:					History-dependent analysis output for SimData2, used for Baseline Comparisons
	SyncAnalysis_Hist_Sim_script.m:					Script performing history-dependent analysis
	SyncAnalysis_dynamic_Sim_script.m:				Script performing history-independent analysis
	SyncAnalysis_Baseline_script.m:					Script performing analyses with Pearson correlation, coefficient of variation, and avg. CIF difference


The following scripts were used to analyze spiking activity of human cortical neurons during propofol-induced anesthesia (Fig. 5)
The data were originally published in:(Lewis et al., PNAS, 2012; https://doi.org/10.1073/pnas.1210907109).
We thank Emery N. Brown and Patrick L. Purdon for providing access to the anonymized data.

Human Anesthesia Analysis Scripts:
	Anesthesia_Synch_script.m:						Script performing history-independent analysis
	Anesthesia_SynchHist_script.m:					Script performing history-dependent analysis
	SynchAnalysis_Baseline_script.m:				Script performing analyses with Pearson correlation and coefficient of variation


The following scripts were used to analyze spiking activity of excitatory and inhibitory neurons in rat motor cortex during 1 sleep cycle (Fig. 6)
The data were originally published in:(Watson et al., Neuron, 2016; doi:https://doi.org/10.1016/j.neuron.2016.03.036).
The data are available openly, and were accessed at:(Watson et al., CRCNSorg, 2016; doi:http://dx.doi.org/10.6080/K02N506Q).

Rat Sleep Cycle Analysis Scripts:
	PreprocessingScript.m:							Script that extracts spiking activity during one sleep cycle from a specified experiment
	RatSleep_Sync_script.m:							Script performing history-independent analysis
	RatSleep_SyncHist_script.m:						Script performing history-dependent analysis
	SynchAnalysis_Baseline_script.m:				Script performing analyses with Pearson correlation and coefficient of variation
	
	
Codes:
	adomp_mGLM.m:					Implementation of AdOMP for dynamic history-dependent discretized MkPP model
	omp_mGLM_cv.m:					Cross-validate for sparsity level using static log-likelihood for discretized MkPP model
	getDesMat.m:					Construct set of history covariates for history-dependent analysis
	
	SynchHistTest_dynamic.m:			Statistical inference of rth-order coordinated spiking in history-dependent model
	SynchTest_dynamic.m:				Statistical inference of rth-order coordinated spiking in history-independent model
	NoncentChi2FiltSmooth.m:			State-space dynamic estimation of non-centrality parameter for characterizing limiting distribution of alt. hypothesis
	
	SynchCorrelation.m:				Avg. Pearson correlation as a single-trial measure of coordinated spiking
	SpikeRegularity.m:				Avg. Coefficient of variation in interspike intervals as a single-trial measure of higher-order coordinated spiking
	SynchCIF.m:						Avg. difference between rth-order mark CIFs and rate of independent rth-order interactions as a single-trial measure 
										of coordinated spiking
										
Codes were developed and tested using MATLAB R2017b.
