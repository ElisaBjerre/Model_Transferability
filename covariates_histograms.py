# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:41:08 2020

This script reads the covariates from the dataframe and computes histograms
and histogram distances for all covariates in a comparison between random and 
spatial hold-out data sets. 

The outputs are histogram plots and dataframes with distances measures, 
which can be read by hist_metrics_performance.py. 

@author: Elisa Bjerre (elisabjerre@gmail.com)
"""
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import date
import define_spatial_subsets_v3 as defsub
import plotting_ML_v2 as p
from scipy.spatial import distance as dist
import sys
from math import ceil

#%% load paths
dir_path = os.path.dirname(os.path.realpath(__file__)) #Directory of the module
modelfolder = os.path.dirname(dir_path)                # mike she model folder
parentfolder=os.path.dirname(os.path.dirname(modelfolder)) # parent folder 

dateprint = date.today().strftime("%b-%d-%Y")

outfolder = dir_path+"/ML_output_files/"
directory = os.path.dirname(outfolder)
if not os.path.exists(directory):
    os.makedirs(directory)
    
#make folder for saving outputs 
resultfolder = dir_path+"/ML_output_files/Histogram_distances_subbasins_v3/"
directory = os.path.dirname(resultfolder)
if not os.path.exists(directory):
    os.makedirs(directory)

#%%-------------Load and prepare dataframe for ML algorithms-------------------
#Load dataframe
dataframe = pd.read_csv(outfolder+'dataframe_Storaa_100m_DKM18_invA_May-06-2021')

#Target variable
y = dataframe[['dfrac']]

#Dataframe 
# X=dataframe.drop(['dfrac'], axis=1)
X=dataframe 
'''In this specific case, we do not delete dfrac from X
as we wish to compute histograms and distances for this parameter together
with the covariates'''

del dataframe

#Only mappable parameters are relevant for transferability, remove non-mappable
#exclude categorical variables for a start 
#excluded covariates: flow accumulation
headers=X.columns.to_series()
X = X.drop(headers[headers.str.contains('flow accumulation|T lay|Kx lay|Kz lay|Kz_eff|Sand ratio|Root depth|soil type|land use|soil class')], axis=1)

#%%-----------Split dataset into training and validation sets------------------
hold_out=0.20 #20% hold out for testing

#Hydrological sub-basins
X_train, X_test, _, _, _, _ = defsub.spatial_subbasins(X,y, hold_out)

del X
del y

del X_train['sub-basin 4']
del X_test['sub-basin 4']

#%%----------------------Functions---------------------------------------------
def return_intersection(hist_1, hist_2):
    '''The two histograms must have same number of bins'''
    if len(hist_1)==len(hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection
    else: 
        print('The two histograms must have the same number of bins')
        
def findMinDifference(A, B): 
    '''function to calculate minimum difference between two elements in two arrays'''
    # Sort both arrays using sort function 
    A.sort() 
    B.sort() 
    a = 0 #pointers at beginning of a and b 
    b = 0
    m = len(A) #end of loop
    n = len(B) 
    # Initialize result as max value 
    result = sys.maxsize 
    # Scan Both Arrays upto size of the Arrays 
    while (a < m and b < n):
        if (abs(A[a] - B[b]) < result): 
            result = abs(A[a] - B[b]) #store as intermediate minimum
            indexa = a
            indexb = b
        # Move Smaller Value 
        if (A[a] < B[b]): 
            a += 1
        else: 
            b += 1         
    # return final min result 
    return result, indexa, indexb

def findMinDiffPairs(x1,x2):
    '''Returns minimum difference of pair assignments (Cha et al. 2002) of two
    1d arrays x1 and x2 (same length)'''
    min_pairs = []
    end=len(x1)
    while (len(min_pairs) < end): #loop until each array element is paired.
        dif, ind1,ind2 = findMinDifference(x1,x2) #find minimum difference instance 
        min_pairs.append(dif) #assign to list of min pairs
        x1=np.delete(x1,ind1) #delete paired array elements and repeat on reduced arrays. 
        x2=np.delete(x2,ind2)
    mindifpairs=np.sum(min_pairs) #sum of the minimum difference of pairs. 
    return mindifpairs

def subplot_hist_feat3(xtest, xtrain, con, density, stdzn):
    '''Creates figure for each feature with subplots of train-test histograms for all subsets
    input: dictionaries of train and test data for all subsets for a single feature
    saves distance mesures to dataframes. 
    con=True means only continuous variables
    density=True means standardization of histograms to Area=1.
    stard=True means standardization of each train-test set
    '''
    # Plot settings 
    dpi=400
    clr1='cornflowerblue'
    clr2='sandybrown'  
    bins = 50 #number of histogram bins

    d_intsec={} #dictionary to hold distances
    d_mindifpair={} #Minimum difference of pair assignments
    d_eucl={} #Euclidian distance
    d_city={} #City block
    d_cheb={} #chebyshev
    
    if con==True: #continuous variables (remove categorical variables)
        headers=xtest[list(xtest.keys())[0]].keys() 
        features = headers[~headers.str.contains('soil type code')]
        features = features[~features.str.contains('land use')]
        features = features[~features.str.contains('soil class')]
    else:
        features=xtest[list(xtest.keys())[0]].keys() 
        
    if density==True: 
        n2='_A1'
    else:
        n2=''
    
    features = features[~features.str.contains('subno')]    
    for feat in features: #loop over features   
        nosubplots=len(xtest.keys())
        norows=ceil(nosubplots/2)
        nocols=2
        fig, axes = plt.subplots(nrows=norows, ncols=nocols, figsize=(8, 2*norows), dpi=dpi)
        fig.subplots_adjust(hspace=0.5)
        plt.rcParams['font.size'] = '8'
        fig.suptitle(feat)
        label1='train'
        label2='test'
        
        if feat=='dfrac': #logscale for dfrac 
            log=True
        else:
            log=False
        
        #temporary list containers 
        temp_intsec={}
        temp_mindifpair={} #Minimum difference of pair assignments
        temp_eucl={} #Euclidian distance
        temp_city={} #City block
        temp_cheb={} #chebyshev
        
        for ax, sub in zip(axes.flatten(), xtrain.keys()): #loop over subsets 
            #remove -999 (nan values) from data
            x1 = list(filter(lambda x: x!= -999, xtrain[sub][feat]))
            x2 = list(filter(lambda x: x!= -999, xtest[sub][feat]))            
            
            if stdzn == True: #standardize histograms for multivariate comparison 
                mean=sum(x1+x2) / len(x1 + x2) #mean of combined train and test set
                std=np.std(x1+x2) #standard deviation of combined train and test set
                x1=(np.array(x1)-mean)/std
                x2=(np.array(x2)-mean)/std
                n1='_stdzn' #for standardized values 
            else:
                n1=''    

            #define min and max of histograms
            xrange = (min(min(x1),min(x2)), max(max(x1),max(x2)))
            
            #compute histograms 
            a, binsa, _ = ax.hist(x1, bins=bins, label=label1, range=xrange, log=log, color=clr1, alpha=0.5, density=density)
            b, binsb, _ = ax.hist(x2, bins=bins, label=label2, range=xrange, log=log, color=clr2, alpha=0.5, density=density)
            ax.set(title=sub)
            
            #compute distances 
            intsec = return_intersection(a,b)
            mindifpair=findMinDiffPairs(a,b)
            mean1=np.mean(x1)
            mean2=np.mean(x2)
            med1=np.median(x1)
            med2=np.median(x2)
            eucl=dist.euclidean(a,b)
            city=dist.cityblock(a,b)
            cheb=dist.chebyshev(a,b)
            
            #save to temporary lists
            temp_intsec[sub]=intsec
            temp_mindifpair[sub]=mindifpair
            temp_eucl[sub]=eucl
            temp_city[sub]=city
            temp_cheb[sub]=cheb
            
            #plot settings 
            ax.text(0.7, 0.9, r'Intersection={:.2f}'.format(intsec), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.7, 0.8, r'Min_dif_pair={:.2f}'.format(mindifpair), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.7, 0.7, r'd_eucl={:.2f}'.format(eucl), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.7, 0.6, r'd_city={:.2f}'.format(city), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            # ax.text(0.7, 0.5, r'd_cheb={:.2f}'.format(cheb), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.axvline(mean1, color=clr1, linestyle='dashed', linewidth=2, label=label1+' mean')
            ax.axvline(mean2, color=clr2, linestyle='dashed', linewidth=2, label=label2+' mean')
            ax.axvline(med1, color=clr1, linestyle='dotted', linewidth=2, label=label1+' median')
            ax.axvline(med2, color=clr2, linestyle='dotted', linewidth=2, label=label2+' median')

        #remove empty subplots 
        delsub=(norows*nocols)-nosubplots
        if delsub>=0:
            for i in range(delsub):
                fig.delaxes(axes.flat[-(i+1)])
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.55, 0.05), loc='lower left')

        plt.tight_layout(rect=(0,0,1,0.95))
        # plt.tight_layout(rect=(0.02,0.02,0.98,0.98))
        fig.savefig(resultfolder+'hist_'+str(feat)+n1+n2)   
        plt.close("all")
        
        #save to dictionaries 
        d_intsec[feat]=temp_intsec
        d_mindifpair[feat]=temp_mindifpair
        d_eucl[feat]=temp_eucl
        d_city[feat]=temp_city
        d_cheb[feat]=temp_cheb
    
    #save as dataframes to csv files     
    df_intsec = pd.DataFrame(data=d_intsec)
    df_intsec.to_csv(resultfolder+'df_Intersection'+n1+n2, index=True)
    df_eucl = pd.DataFrame(data=d_eucl)
    df_eucl.to_csv(resultfolder+'df_Dist_Euclidian'+n1+n2, index=True)
    df_city = pd.DataFrame(data=d_city)
    df_city.to_csv(resultfolder+'df_Dist_Cityblock'+n1+n2, index=True)
    df_cheb = pd.DataFrame(data=d_cheb)
    df_cheb.to_csv(resultfolder+'df_Dist_Chebyshev'+n1+n2, index=True)
    df_mindifpair = pd.DataFrame(data=d_mindifpair)
    df_mindifpair.to_csv(resultfolder+'df_Dist_MinDiffPair'+n1+n2, index=True)
        
    return df_intsec, df_eucl, df_city, df_cheb, df_mindifpair

#%% compute histograms and distances
    
#histograms normalized to A=1 and standardized 
subplot_hist_feat3(xtest=X_test, xtrain=X_train, con=True, density=True, stdzn=True) 

# subplot_hist_feat3(xtest=X_test, xtrain=X_train, con=True, density=True, stdzn=False) 

# del X_test
# del X_train
