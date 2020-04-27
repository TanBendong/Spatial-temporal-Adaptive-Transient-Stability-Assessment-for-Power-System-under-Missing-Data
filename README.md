**Copyright notice**: These codes can be used only for reproducing our unpublished paper "Spatial-temporal Adaptive Transient Stability Assessment for Power System under Missing Data" submitted to **International Journal of Electrical Power & Energy Systems**, and other usages are not allowed!

# Spatial-temporal-Adaptive-Transient-Stability-Assessment-for-Power-System-under-Missing-Data
These codes are for our paper "Spatial-temporal Adaptive Transient Stability Assessment for Power System under Missing Data", and the following introduction provides usage for them. As the capacity of github is limited, we apologize that the datasets cannot be uploaded, but readers can gernerate them according to data generation method in our paper. Enjoy it!
## Environment Requirement
* Python 3.x
* Matlab
## 1.Relief_FT
* main.m: The main program of feature importance calculation. Its input is training data, and output is importance of each temporal features.
* Relief_FT: Program of feature importance calculation, called by main.m.
* weights.mat: The results of main.m

## 2.Optimal PMU Clusters Searching Model
* optimal_featureset.m: The main program of the optimal PMU clusters searching model. Its input is temporal feature importance, and output is optimal PMU clusters.
* fitness.m: The objective function of the optimal PMU clusters searching model.
* circlecon.m: The constraints of the optimal PMU clusters searching model.
* PMU_Place.m: The power system observability description algorithm.
* 17.csv: The results of the optimal PMU clusters.

## 3. Spatial-temporal Adaptive TSA
* Spatial_temporal_adaptive_TSA.py: The main program of spatial-temporal adaptive TSA. Its input is testing data, and output is average response time and average accuracy of TSA under missing data.
* Ensemble_LSTM.py: This is ensemble LSTM model. Its input is validation data, and output is weight for each LSTM. Then ensemble LSTM model is created by integrating each LSTM with corresponding weight.
* DynamicLSTM.py: The training program for single LSTM.
