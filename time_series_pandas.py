import numpy as np
#import matplotlib.pyplot as plt
#from astropy.io import ascii
#from astropy.table import Table
#from astropy.modeling import models,fitting
from math import sqrt
import random,sys
import pandas as pd
from extract_features import Features
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def split(arr,cond):			#split an array at a given condition
	return [arr[cond],arr[~cond]]

AllVar = pd.read_csv('AllVar.phot',header=None, sep=',')
AllVar.columns = ['Numerical_ID','MJD','Mag','Mag_err','RA','DEC']
cat_vars = pd.read_csv('CatalinaVars.tbl',header = 0,delim_whitespace=True)

grouped = AllVar.groupby('Numerical_ID')

#merged = pd.merge(AllVar, cat_vars, how='outer', on='Numerical_ID')
#grouped = merged.groupby('Numerical_ID')

#plt.ion()
#plt.figure()
feature_vars = list()
feature_space = list()
i=0
for key, group in grouped:
    group = group.sort_values(by = 'MJD')
    f = Features(group, mag_err=True)
    f.call_all_features()
    feature_space.append(f.feature_list)
    Catalina_object = cat_vars.loc[cat_vars['Numerical_ID'] == key]
    if Catalina_object.empty == False:
        feature_vars.append((key, f.feature_list, Catalina_object['Var_Type'].values[0], Catalina_object['Period_(days)'].values[0]))
    else:
        feature_vars.append((key, f.feature_list))
#     plt.title( str(Catalina_object['Var_Type']) + ' ' + str(key) )
#     plt.gca().invert_yaxis()
#     plt.xlabel('MJD')
#     plt.ylabel('Mag')
#     plt.scatter(group['MJD'], group['Mag'], marker = 'o', color = 'black')
#     plt.pause(2)
#     plt.clf()
    i+=1
    print i
		
feature_space_df = pd.DataFrame(feature_space)
#feature_space_df.describe()
feature_space = feature_space_df.as_matrix()

feature_l2norm = normalize(feature_space, axis = 0)
pca = PCA()
feature_l2norm_pca = pca.fit_transform(feature_l2norm)


feature_1varnorm = np.zeros([46821, 23])

for i in range(0,23):
    feature_1varnorm[:,i] = np.float64(feature_space[:,i] - np.mean(feature_space[:,i]))/np.std(feature_space[:,i])
    
pca_1 = PCA()
feature_1varnorm_pca = pca_1.fit_transform(feature_1varnorm)

