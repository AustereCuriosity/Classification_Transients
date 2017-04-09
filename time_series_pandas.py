import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from astropy.modeling import models,fitting
from math import sqrt
import random,sys
import pandas as pd
from extract_features import Features

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
feature_vars = []
i=0
for key, group in grouped:
    f =Features(group, mag_err=True)
    f.call_all_features()
    Catalina_object = cat_vars.loc[cat_vars['Numerical_ID'] == key]
    if Catalina_object.empty == False:
        feature_vars.append((key, f.feature_list, Catalina_object['Var_Type']))
    else:
        feature_vars.append((key, f.feature_list))
    plt.title( str(Catalina_object['Var_Type']) + ' ' + str(key) )
	plt.gca().invert_yaxis()
	plt.xlabel('MJD')
	plt.ylabel('Mag')
	plt.scatter(group['MJD'], group['Mag'], marker = 'o', color = 'black')
	plt.pause(2)
	plt.clf()
    i+=1
    print i
		
print feature_vars