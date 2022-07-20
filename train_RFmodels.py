# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:44:37 2020

Train and test metamodels with Random Forest out of bag on selected covariates. 

10 training and test sets are included, 1 randomly sampled and 9 spatial subsets. 

Outputs are dataframes with model performance (R2 and RMSE) and individual 
covariate importances

@author: ebj
"""
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import ensemble
import os
import pandas as pd
import rfpimp as rfp
import define_spatial_subsets_v3 as defsub
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
from math import ceil

#%% Model setup 
covmap=True  #True for mappable parameters only, False for all covariates
subbas=True #true for sub-basins, False for subsets

#%% load and define paths 
dir_path = os.path.dirname(os.path.realpath(__file__)) #Directory of the module
modelfolder = os.path.dirname(dir_path)                # mike she model folder
parentfolder=os.path.dirname(os.path.dirname(modelfolder)) # parent folder 
temp=dir_path[len(modelfolder)+1:]
mname=temp[:-19]#name of mike she model 
    
#make folder for saving outputs 
resultfolder = dir_path+"/RF_results/"
directory = os.path.dirname(resultfolder)
if not os.path.exists(directory):
    os.makedirs(directory)

#%%'-------------Load and prepare dataframe for ML algorithms----------------'''
#Load dataframe of covariates 
data = pd.read_csv('dataframe_Storaa')

headers=data.columns.to_series()

#Target variable
y = data[['dfrac']]  

#define predictor variables               
if covmap==True:
    #Drop non-mappable parameters
    X=data.drop(headers[headers.str.contains('dfrac|T lay|Kx lay|Kz lay|Kz_eff|Sand ratio')], axis=1)
else:
    #all covarriates
    X=data.drop(headers[headers.str.contains('dfrac')], axis=1)
    
del data  

#save dataframes to be able to restore results later
filename = resultfolder+'X_dataframe.sav'
pickle.dump(X, open(filename, 'wb'))

filename = resultfolder+'y_dataframe.sav'
pickle.dump(y, open(filename, 'wb'))  
#%%-----------Split dataset into training set and validation set------------'''
hold_out=0.20 #20% hold out for testing

if subbas==True:
    """Divide hydrological sub-basins"""
    X_train, X_test, y_train, y_test, idx_train, idx_test = defsub.spatial_subbasins_9(X,y, hold_out)
    
X=X.drop(['x index'], axis=1)
X=X.drop(['y index'], axis=1)

#%%--RF oob training and testing -------------------------------------------'''
'''OBS insert optimized values from gridsearch'''
n=500
max_depth = 30
min_samples_leaf = 5
min_samples_split = 5
max_features = 0.33

#%%Fit model to training data set 
rf_oob={}
for i in X_train.keys():
    print('Training '+i)
    rf_oob[i]=ensemble.RandomForestRegressor(criterion='mse',n_estimators=n,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_features=max_features,bootstrap=True,oob_score=True,random_state=99,n_jobs=15)
    rf_oob[i].fit(X_train[i],y_train[i].values.ravel())
    # save the model to resultfolder
    filename = resultfolder+'finalized_model_'+i+'.sav'
    pickle.dump(rf_oob[i], open(filename, 'wb'))

#%%predict on test data set
rf_oob_predict={}
for i in X_train.keys():
    print('Predicting hold-out '+i)
    rf_oob_predict[i] = rf_oob[i].predict(X_test[i])

#%%predict on training data
rf_oob_predict_train={}
for i in X_train.keys():
    print('Predicting training data '+i)
    rf_oob_predict_train[i] = rf_oob[i].predict(X_train[i])

#%%'''Evaluation of scores'''
r2_ho={}
r2_oob = {}
r2_train = {}
rmse_ho = {}
rmse_train = {}

for i in X_train.keys():
    r2_ho[i]=r2_score(y_test[i], rf_oob_predict[i])
    r2_oob[i]=rf_oob[i].oob_score_
    r2_train[i]=r2_score(y_train[i], rf_oob_predict_train[i])
    rmse_ho[i]=sqrt(mean_squared_error(y_test[i], rf_oob_predict[i]))
    rmse_train[i]=sqrt(mean_squared_error(y_train[i], rf_oob_predict_train[i]))
    
#%% Save performance metrics to dataframes 

d_perf={} 
d_perf['Training'] = r2_train
d_perf['OOB'] = r2_oob
d_perf['Hold-out'] = r2_ho
d_perf2={}
d_perf2['Training'] = rmse_train
d_perf2['Hold-out'] = rmse_ho

df_perf_r2 = pd.DataFrame(data=d_perf)
df_perf_r2.to_csv(resultfolder+'df_Performance_R2', index=True)
df_perf_rmse = pd.DataFrame(data=d_perf2)
df_perf_rmse.to_csv(resultfolder+'df_Performance_RMSE', index=True)

#%%save metadata and results to txt file
f = open(resultfolder+"/Metadata.txt","w+")

f.write("Hold-out test data: %0.2f\r\n" % hold_out)
f.write(str(rf_oob['ran'])+"\n")
f.close()

f2 = open(resultfolder+"/Covariates.txt","w+")
f2.write(str(X.keys())+"\n")

#%%-----------------------------Plotting------------------------------------'''
mask = np.load('mask_lu_s.npy')
bins = 50 #histogram bins 
dpi=400

#%% Barplot of model performance R2
plt.figure(figsize=(3,2))
plt.rcParams['font.size'] = '12'
ax = df_perf_r2.plot.bar(rot=90)
ax.set_ylabel('$R^{2}$')
ax.legend( loc='lower left')
plt.tight_layout()
plt.savefig(resultfolder+'barplot_R2', dpi=dpi);plt.close()
    
#%% all densityplots in one figure
X_temp = dict(X_train)
del X_temp['rdm']

nosubplots=len(X_temp.keys())
nocols=3
norows=ceil(nosubplots/nocols)
fig, axes = plt.subplots(nrows=norows, ncols=nocols, figsize=(3*nocols, 2*norows), dpi=dpi, sharex=True)
fig.subplots_adjust(hspace=0.4, wspace=0.3)
plt.rcParams['font.size'] = '8'
nobins=20

names=['sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9']

for ax, sub, name in zip(axes.flatten(), X_temp.keys(), names):
    x= np.array(y_test[sub]['dfrac'])
    y= rf_oob_predict[sub]
    n=len(x) #number of datapoints 
    xbuffer=(x.max()-x.min())*0.02 #add 5% to axes 
    xmin = x.min()-xbuffer
    xmax = x.max()+xbuffer
    ybuffer=(y.max()-y.min())*0.02 
    ymin = y.min()-ybuffer
    ymax = y.max()+ybuffer
    
    r2=r2_score(x,y)
    rmse=sqrt(mean_squared_error(x, y))
    
    #density plot
    hb = ax.hexbin(x,y, bins='log', cmap='inferno_r', mincnt=1)
    
    #prepare boxplot 
    df = pd.DataFrame({sub:x,'dfrac':y })
    # Classify each entry according to the bin they belong to, from min to max of feature
    bins = np.linspace(np.min(x), np.max(x), num=nobins)
    width = np.diff(bins)[0] 
    df['interval_index'] = np.digitize(df[sub], bins)
    # Get middle value for each bin, for example, if bin is (40-45), middle value would be 42.5
    middle_value = bins[:-1] + width/2       
    stats = [df['dfrac'][df['interval_index'] == i].values for i in range(1, len(bins))]
    medianprops=dict(color='blue')
    boxprops=dict(linewidth=0.5)
    whiskerprops = dict(linewidth=0.3, color='black')
    capprops=dict(linewidth=0.3)
    ax.boxplot(stats, positions=middle_value, widths=width*0.6, manage_ticks=False, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    
    ax.plot([0, 1],[0,1], 'k--', linewidth=0.7)
    # ax.text(0.01, (ymax-0.3), r'$R^2 = $'+ str('{:.2f}'.format(r2))+'\n'+ r'$RMSE = $'+ str('{:.2f}'.format(rmse))+'\n'+ r'$n = $'+ str('{:.0f}'.format(n)), fontsize=8)
    ax.text((xmax*0.01), (ymax*0.8), r'$R^2 = $'+ str('{:.2f}'.format(r2))+'\n'+ r'$n = $'+ str('{:.0f}'.format(n)), fontsize=8)
    ax.axis([xmin, xmax, ymin, ymax])
    cb = fig.colorbar(hb, ax=ax)
    ax.set(title=name)
    ax.tick_params(axis='both', which='major', labelsize=8)

#remove empty subplots 
delsub=(norows*nocols)-nosubplots
if delsub>=0:
    for i in range(delsub):
        fig.delaxes(axes.flat[-(i+1)])
    
fig.text(0.01, 0.5, 'Predicted drainage fraction', va='center', rotation='vertical')
fig.text(0.98, 0.5, 'log10(N)', va='center', rotation='vertical')
fig.text(0.5, 0.01, 'Mike SHE drainge fraction', ha='center')    
plt.tight_layout(rect=(0.02,0.02,0.98,0.98))
plt.savefig(resultfolder+'plot_density_RFoob', dpi=dpi); plt.close()

#%% Feature permutation importances 1: individual continuous covariates, categorical grouped for each category 

I_folder = resultfolder+"/Importance_plots/" #make folder for saving importance plots
directory = os.path.dirname(I_folder)
if not os.path.exists(directory):
    os.makedirs(directory)
    
headers=X_train['rdm'].columns.to_series()

#define continuous variables 
headers_con = headers[~headers.str.contains('soil type code|land use|soil class')]

#group categorical variables
headers_st = headers[headers.str.contains('soil type code')]
headers_lu = headers[headers.str.contains('land use')]
headers_sc = headers[headers.str.contains('soil class')]

#merge continuous and grouped variables
list_headers=list(headers_con)
list_headers.append(list(headers_lu))
list_headers.append(list(headers_st))
list_headers.append(list(headers_sc))

features_map1 = list_headers
    
#%% Individual feature performance
I_idv={}
        
n_samples=-1 #all datapoints in validation set is used for importance calculation. This ensures reproducability
random_seed=42 #choose random seed for reproducability 

for i in X_train.keys(): 
    I_map = rfp.importances(rf_oob[i], X_test[i], y_test[i], features=features_map1, n_samples=n_samples, random_seed=random_seed)
    #define new feature names for plotting land use, soil type and soil class
    lu=I_map[I_map.index.str.contains('land use')==True]
    st=I_map[I_map.index.str.contains('soil type')==True]
    sc=I_map[I_map.index.str.contains('soil class')==True]
    I_map=I_map.rename(index={lu.index[0]: "land use", st.index[0]: "soil type", sc.index[0]: "soil class"})
    
    rfp.plot_importances(I_map)
    plt.savefig(I_folder+'group1_Imp_test_'+i, dpi=dpi); plt.close()
    
    I_map2 = rfp.importances(rf_oob[i], X_train[i], y_train[i], features=features_map1)
    #define new feature names for plotting land use, soil type and soil class
    lu=I_map2[I_map2.index.str.contains('land use')==True]
    st=I_map2[I_map2.index.str.contains('soil type')==True]
    sc=I_map2[I_map2.index.str.contains('soil class')==True]
    I_map2=I_map2.rename(index={lu.index[0]: "land use", st.index[0]: "soil type", sc.index[0]: "soil class"})
    
    I_idv[i]=I_map2.to_dict()['Importance']
    
    rfp.plot_importances(I_map2)
    plt.savefig(I_folder+'group1_Imp_train_'+i, dpi=dpi); plt.close()

df = pd.DataFrame(data=I_idv)
df.to_csv(resultfolder+'df_feature_importance_idv_')

del I_map
del I_map2
del df