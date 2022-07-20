# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:15:33 2020

Divides the dataset into 10 training and test sets: 
 - 1 randomly sampled, 80% training and 20% test data
 - 9 spatially defined, based on a tif file of catchment subbasins.  

@author: ebj
"""
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
import os

#%% load paths
dir_path = os.path.dirname(os.path.realpath(__file__)) #Directory of the module

#%% functions
    
def loadmask():
    #specify here which mask to use
    mask = np.load('mask_lu_s.npy')
    return mask

mask = loadmask()

def getFlat_Masked_Array(array):
    '''This function reads a 2D array as input and masks out all cells 
    where there is no drainage fraction data and returns flattened 1D array'''
    array[mask==0]=np.nan
    flat = np.ravel(array)
    flat_array = flat[~np.isnan(flat)]  
    return flat_array
    
def spatial_subbasins_9(X,y, hold_out): 
    '''Reads dataframes X and y (training and test sets) and splits them
    into hydrological subbasins as defined by the .tif file
    Saves dictionaries with dataframe of all covariates for each subbasin'''
    
    #load tif file of hydrological basins
    with rasterio.open(dir_path+'\Storaa_subbasins_10.tif') as ds:
        sub = ds.read() ##reading all raster values
    
    sub = sub[0,:,:]
    sub = np.flipud(sub) #flipud to match X and y format
    subbas_flat = getFlat_Masked_Array(sub) #masks nan and flattens array 
    
    #Mask out grids that contain nan or -999 (nodata) in the X dataframe training set
    boolX1=np.isnan(X) #true for nan
    y=y[~np.any(boolX1,axis=1)]
    X=X[~np.any(boolX1,axis=1)]
    subbas_flat=subbas_flat[~np.any(boolX1,axis=1)]
    
    boolX2=(X==-999) #true for -999
    y=y[~np.any(boolX2,axis=1)]
    X=X[~np.any(boolX2,axis=1)]
    subbas_flat=subbas_flat[~np.any(boolX2,axis=1)]    
    
    #mask out nodatavalues 
    nodataval = ds.nodatavals # no data value in tif file
    y=y[subbas_flat!=nodataval]
    X=X[subbas_flat!=nodataval] 
    subbas_flat = subbas_flat[subbas_flat!=nodataval] 
    
    #create dictionaries for saving output    
    X_test = {}
    X_train = {}
    y_train = {}
    y_test = {}
    idx_train = {}
    idx_test = {}
    
    idx=X[['x index', 'y index']]
    X=X.drop(['x index'], axis=1)
    X=X.drop(['y index'], axis=1)
    
    #Create random train-test set
    X_train_ran, X_test_ran, y_train_ran, y_test_ran, idx_train_ran, idx_test_ran = train_test_split(X, y, idx, test_size=hold_out, random_state=0)
    X_train['rdm']=X_train_ran
    X_test['rdm']  = X_test_ran
    y_train['rdm'] = y_train_ran
    y_test['rdm'] = y_test_ran
    idx_train['rdm'] = idx_train_ran
    idx_test['rdm'] = idx_test_ran
    
    #Create hydrological sub-basin train test sets    
    no_name=1 
    subno = np.unique(subbas_flat) #subbasin numbers, loop through, create df for each subbas
    for no in subno:
        if no != 4: #exclude sub-basin number 4 and rename accordingly 
            print('sb'+str(int(no_name)))
            X_test['sb'+str(int(no_name))] = X[subbas_flat==no]
            X_train['sb'+str(int(no_name))] = X[subbas_flat!=no]  
            y_test['sb'+str(int(no_name))] = y[subbas_flat==no]
            y_train['sb'+str(int(no_name))] = y[subbas_flat!=no]  
            idx_test['sb'+str(int(no_name))] = idx[subbas_flat==no]
            idx_train['sb'+str(int(no_name))] = idx[subbas_flat!=no] 
            no_name=no_name+1
            
    return X_train, X_test, y_train, y_test, idx_train, idx_test