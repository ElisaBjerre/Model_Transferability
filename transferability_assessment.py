# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:30:14 2020

@author: ebj

This script calculates the weighted distance between training and test data based on 
the distance metrics calculated from covariates_histograms.py, which must be run
first to produce the result files loaded in this script.

"""
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import scipy.stats as sc
import seaborn as sns
# from math import ceil
import copy
from matplotlib.pyplot import cm
import itertools
from sklearn.linear_model import LinearRegression

#%% load and define paths 
dir_path = os.path.dirname(os.path.realpath(__file__)) #Directory of the module

#make folder for saving outputs 
modelfolder = dir_path+'/RF_results/'
histfolder = dir_path+'/Histogram_distances/'
catfolder = dir_path+'/Categorical_covariates/'

#create subfolder for saving output
resultfolder = dir_path+"/Transferability/"
directory = os.path.dirname(resultfolder)
if not os.path.exists(directory):
    os.makedirs(directory)
    
#%% load data     
#read metamodel performance 
df_r2 = pd.read_csv(modelfolder +'/df_Performance_R2', index_col =0)

#read distance metrics and save to dictionary 
dist_all={}
temp= pd.read_csv(histfolder+'df_Intersection_stdzn_A1', index_col =0)
#transform intersection into negative distance metric by multiplying by -1: 
dist_all['Intersection'] = temp*-1
dist_all['Min dif pair'] = pd.read_csv(histfolder+'df_Dist_MinDiffPair_stdzn_A1', index_col =0)
dist_all['City-block'] = pd.read_csv(histfolder+'df_Dist_Cityblock_stdzn_A1', index_col =0)
dist_all['Euclidian'] = pd.read_csv(histfolder+'df_Dist_Euclidian_stdzn_A1', index_col =0)

dist_cat={}
dist_cat['Categorical'] = pd.read_csv(catfolder+'df_Dist_cat', index_col =0)

#%%read calculated feature importances 
I_idv = pd.read_csv(modelfolder+'df_feature_importance_idv_', index_col =0, delimiter=',').T
I_idv_con = I_idv.drop(['land use', 'soil type', 'soil class'], axis=1)

#%%remove distances for covariates not included in RF model (load from importance df)
for i in dist_all.keys():
    headers=dist_all[i].columns.to_series()
    headers2 = I_idv_con.columns.to_series()
    dist_all[i] = dist_all[i].drop(headers[headers.ne(headers2)], axis=1)

I_cat=I_idv.drop(headers2, axis=1)

#%% Plot settings 
dpi=400

#%% functions

def compute_WD(dict_dist, df_weight):
    '''Computes weighted distances 
    dict_dist: dictionary of metrics each containing a dataframe of distances 
    for each subset and feature
    df_weight:dataframe of weights for all combinations of subsets and features
    returns dictionary of metrics each containing dataframe of weighted distances'''
    WD_all= {} # weighted distance for individual features
    for metric in dict_dist.keys(): #loop over metrics
        WD_feat = {}
        # for feat in df_weight.keys(): #loop over features
        for feat in dict_dist[metric].keys(): #loop over features    
            I_feat=df_weight[feat]
            dist_feat = dict_dist[metric][feat]
            WD_feat[feat]=I_feat*dist_feat
            WD_feat = pd.DataFrame(data=WD_feat)
        WD_all[metric]=WD_feat
    return WD_all

def subplot_linear_com(dict_dist_in, dist_cat, df_perf, weight, weight_cat):
    '''creates figure for each (weighted) distance metric with scatterplots of R2 vs (weighted distance) for all features separately
    reads dictionary of metrics, with dataframe of distances for all features and subsets
    and dataframe of performance metric (e.g. R2 or RMSE) for all subsets 
    if weight=True it is the weighte ddataframe is loaded, then the weighted distance is computed 
    https://seaborn.pydata.org/generated/seaborn.regplot.html''' 
    
    
    
    # dist_cat=copy.deepcopy(dist_cat)
    # dict_dist=copy.deepcopy(dist_all)
    # df_perf =copy.deepcopy(df_r2)
    # weight=copy.deepcopy(I_idv_con)
    # weight_cat=copy.deepcopy(I_cat)
    
    dist_cat=copy.deepcopy(dist_cat)
    dict_dist=copy.deepcopy(dict_dist_in)
    df_perf =copy.deepcopy(df_perf)
    weight=copy.deepcopy(weight)
    weight_cat=copy.deepcopy(weight_cat)
    
    dict_WD=compute_WD(dict_dist, weight) #compute weighted distance
    WD_cat=compute_WD(dist_cat, weight_cat)
    dict_WD.update(WD_cat)
    
    nocols=len(dict_WD.keys())
    norows=1
    fig, axes = plt.subplots(nrows=norows, ncols=nocols, figsize=(8,2), dpi=dpi, sharey=True)
    plt.rcParams['font.size'] = '8'
    fig.subplots_adjust(hspace=0.6, wspace=0.2, bottom=0.4, top=0.9)
    fig.text(0.05, 0.65, '$R^{2}$', va='center', rotation='vertical')    
    
    #dictionary for saving linear combination of WD
    WD_lin={} 
    for ax, metric in zip(axes.flatten(),dict_WD.keys()): #loop over metrics
        wd_metric=dict_WD[metric]
        df_perf=df_perf.loc[wd_metric.T.keys()]
        
        #add linear combination of distances to df
        meandist=np.mean(pd.DataFrame(data=wd_metric), axis=1)
        WD_lin[metric]=meandist
        
        y=df_perf['Hold-out']
        x = WD_lin[metric]
        x=x.sort_index(ascending=True)
        y=y.sort_index(ascending=True)
            
        n=len(x)
        color2=cm.rainbow(np.linspace(0,1,n))
        marker = itertools.cycle(('x', '+', 'o', '*', 'v', '^', 'p', 'd', 'X', '*')) 
        sns.regplot(x,y, ax=ax, ci=95, scatter_kws={'s':1}, color='red')
        namelist=['rdm', 'sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9']
        for i,c,name in zip(x.index, color2, namelist):
            print(i)
            ax.scatter(x[i],y[i],color=c, label=name, marker=next(marker), s=70)
        
        pearR,p=sc.pearsonr(x,y)

        ax.text(0.5, 0.9, r'$\rho_{P} = $'+ str('{:.2f}'.format(pearR)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(metric, fontsize=8)
        ax.set_ylabel('')    
        ax.set_xlabel('')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.21), loc='upper center', ncol=5)
    fig.text(0.5, 0.23, 'Weighted Distance', ha='center')
    fig.savefig(resultfolder+'Lin_wght_dist');

    plt.close()
    
    return WD_lin

#%%Linear combinations
linear_dist = subplot_linear_com(dist_all, dist_cat, df_r2, I_idv_con, I_cat)

#%%example for City-block
x = linear_dist['City-block']
y=df_r2['Hold-out']
r2_pred={} #predicted r2

   
for index, value in x.items(): #loop over hold-outs 
    # Split the data into training/testing sets
    x_train = x.drop(index)
    x_test = x.loc[index]
    
    x_train=np.array(x_train).reshape(-1,1) #reshape to 1 feature for regression function
    x_test=np.array(x_test).reshape(-1,1)
    
    # Split the targets into training/testing sets
    y_train = y.drop(index)
    y_test = y[index]
    
    # Create linear regression object
    regr = LinearRegression()
    
    # Train the model using the training sets
    regr.fit(x_train, y_train)
    
    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    
    r2_pred[index]=y_pred
    
df_r2_pred=pd.DataFrame(data=r2_pred).T

#%%    
plt.rcParams['font.size'] = '14'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5))

n=len(df_r2_pred)
color2=cm.rainbow(np.linspace(0,1,n))
marker = itertools.cycle(('x', '+', 'o', '*', 'v', '^', 'p', 'd', 'X', '*')) 
x=df_r2_pred
namelist=['rdm', 'sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9']
y=df_r2['Hold-out']
for i,c, name in zip(x.index, color2, namelist):
    ax.scatter(x.loc[i],y.loc[i],color=c, label=name, marker=next(marker), s=200)
ax.plot([0, 1],[0,1], 'k--', linewidth=0.9)
ax.set_ylabel('$R^{2}$ metamodel spatial CV')    
ax.set_xlabel('$R^{2}$ prediction (transferability)')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.92, 0.83), loc='upper left', ncol=1)

# plt.tight_layout()
fig.savefig(resultfolder+'Transferability_2', dpi=400, bbox_inches='tight', pad_inches=0.02)
plt.show()