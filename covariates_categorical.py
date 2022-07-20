# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:03:29 2020

@author: ebj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:41:08 2020

This script reads the covariates from the dataframe and computes histograms
and histogram metrics for all covariates in a comparison between the six
hold-out data sets: random and 5 spatial subsets. 

@author: ebj
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import define_spatial_subsets as defsub

#%% load paths
dir_path = os.path.dirname(os.path.realpath(__file__)) #Directory of the module
modelfolder = os.path.dirname(dir_path)                # mike she model folder
parentfolder=os.path.dirname(os.path.dirname(modelfolder)) # parent folder 
    
#make folder for saving outputs 
resultfolder = dir_path+"/Categorical_covariates/"
directory = os.path.dirname(resultfolder)
if not os.path.exists(directory):
    os.makedirs(directory)

#%%-------------Load and prepare dataframe for ML algorithms-------------------
#Load dataframe
dataframe = pd.read_csv('dataframe_Storaa')

#Target variable
y = dataframe[['dfrac']]
#Dataframe 
X=dataframe 
del dataframe

#Select categorical variables 
headers=X.columns.to_series()
# X_cat = X.drop(headers[~headers.str.contains('T lay|Kx lay|Kz lay|Kz_eff|Sand ratio|Root depth|soil type|land use|soil class|x index|y index')], axis=1)
X_cat = X.drop(headers[~headers.str.contains('soil type|land use|soil class|x index|y index')], axis=1)

del X
#%%-----------Split dataset into training and validation sets------------------
hold_out=0.20 #20% hold out for testing

#Hydrological sub-basins
X_train, X_test, _, _, _, _ = defsub.spatial_subbasins_9(X_cat,y, hold_out)
del X_cat
del y

#%% 
dpi=400
features_cat = ['soil type', 'land use','soil class']

d_cat={} #save categorical distances

for sub in X_train.keys(): #loop over subsets

    d_feat={} #save distances for feature
    for feat in features_cat: #loop over categorical features 
        varclass=headers[headers.str.contains(feat)]
        var_ratio_test={}
        var_ratio_train={}
        
        d_class={} #save distances for each class in feature
        for var in varclass: #loop over classes within each feature
            var_ratio_train[var]=(np.sum(X_train[sub][var])/len(X_train[sub][var]))
            var_ratio_test[var]=(np.sum(X_test[sub][var])/len(X_test[sub][var]))
    
        d_class['test']  =  var_ratio_test
        d_class['train'] =  var_ratio_train
        df_class = pd.DataFrame(data=d_class)
        
        plt.figure(figsize=(3,2))
        plt.rcParams['font.size'] = '12'
        ax = df_class.plot.bar(rot=90)
        ax.set_ylabel('Class ratio')
        ax.legend( loc='best')
        plt.tight_layout()
        plt.savefig(resultfolder+'barplot_'+str(feat)+'_'+str(sub), dpi=dpi);plt.close()

        df_class['dif']=  abs(df_class['train']- df_class['test'] )

        d_feat[feat] = df_class['dif'].sum()
    d_cat[sub]=d_feat  
    
    df_cat = pd.DataFrame(data=d_cat)
    df_cat.T.to_csv(resultfolder+'df_Dist_cat', index=True)
