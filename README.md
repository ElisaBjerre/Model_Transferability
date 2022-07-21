# Model_Transferability
Evaluating spatial transferability of a Random Forest metamodel beyond its training data.

This repository contains the python code for the transferability method presented in Bjerre et al. (2022), https://doi.org/10.1016/j.jhydrol.2022.128177. 
A pandas dataframe (in script named "dataframe_Storaa") with all covariates and target variable must be created by the user and used as input to the Random Forest regression and to compute the histogram distance metrics. 

**covariates_histograms.py:**
Plots histograms and computes histogram distance metrics for all training and test sets, across all continuous covariates. 
  
**covariates_categorical.py:** Generates barplots and computes distance metrics for all training and test sets, across all categorical covariates. 
  
**define_spatial_subsets.py:** Functions used to divide dataset into 10 training and test sets (1 randomly sampled, 9 spatial subsets). 

**train_RFmodels.py:** Trains Random Forest models, makes predictions on hold-out datasets, evaluates model performance (R2). Computes individual covariate importances used for weighting in the transferability assessment. 

**transferability_assessment.py:** For each distance metric, the script plots and evaluates the linear correlation between weighted distance and model performance. 

**mask_lu_s.npy:** Mask of the Storaa catchment, where roads, urban areas etc. are removed.

**Storaa_subbasins_10.tif:** Sub-basins generated in Q.GIS.

Cite as: Bjerre, E., Fienen, M. N., Schneider, R., Koch, J., & Anker, L. H. (2022). Assessing spatial transferability of a random forest metamodel for predicting drainage fraction, 612, 128177. https://doi.org/10.1016/j.jhydrol.2022.128177
