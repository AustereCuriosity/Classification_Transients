import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from astropy.modeling import models,fitting
from math import sqrt
import itertools as it
import random,sys
import time
from itertools import groupby
import pandas as pd

def split(arr,cond):			#split an array at a given condition
	return [arr[cond],arr[~cond]]

# cut -d , -f 1,1 AllVar.phot > newfile.txt
Obj_ID_list = ascii.read('newfile.txt', format='fast_basic')
cat_vars = ascii.read('CatalinaVars.tbl', format='fast_basic')


Obj_struct = np.asarray(Obj_ID_list)

grouped = [(k, len(list(g))) for k, g in groupby(Obj_struct['Obj_ID'])]

catalina_vars = np.asarray(cat_vars)
catalina_vars.sort(kind = 'quicksort', order = 'Numerical_ID' )
# time_series = list()
# Obj_ID_curr = Obj_struct['Obj_ID'][0]
# 
# while(True):
# 	time_series_split = split(Obj_struct,Obj_struct['Obj_ID'] == Obj_ID_curr)
# 	time_series.append((Obj_ID_curr, time_series_split[0].size))
# 	print Obj_ID_curr
# 	if(time_series_split[1].size == 0):
# 		break
# 	try:
# 		Obj_struct = time_series_split[1]
# 		Obj_ID_curr = time_series_split[1][0]['Obj_ID']
# 	except:
# 		break
# 
# with open('AllVar.phot') as f:
# 	plots = []
# 	i = 0
# 	while True:
# 		series = list(it.islice(f, time_series[i][1])) #islice returns an iterator ,so you convert it to list here.
# 		i += 1
#		if series:
# 			series = ascii.read(series,delimiter = ',')
# 			plt.scatter(series['col2'], series['col3'], marker = 'o', color = 'black')
# 			plots.append(series)       # may be store it 
# 		else:
# 			break
#	
plt.ion()
plt.figure()
with open('AllVar.phot') as f:
	plots = []
	i = 0
	while True:
		series = list(it.islice(f, grouped[i][1])) #islice returns an iterator ,so you convert it to list here.
		if series:
			key = grouped[i][0]
			variable_index = np.searchsorted(catalina_vars['Numerical_ID'], key)
			Catalina_object = catalina_vars[variable_index]
			if Catalina_object['Numerical_ID'] == key:
					plt.title( str(Catalina_object['Var_Type']) + ' ' + str(key) )
			series = ascii.read(series,delimiter = ',')
			series = np.asarray(series)
			#dtype=[('ID', np.int64), ('MJD', np.float64), ('Mag', np.float64), ('Mag_err', np.float64), ('RA', np.float64), ('DEC', np.float64)]
			plt.gca().invert_yaxis()
			plt.xlabel('MJD')
			plt.ylabel('Mag')
			plt.scatter(series['col2'], series['col3'], marker = 'o', color = 'black')
			plt.pause(2)
			plots.append((key, series, Catalina_object))       # store it
			#series.write('{0}{1}.fits'.format(key,Catalina_object))
			print key, Catalina_object
			plt.clf()
			
			if i%1000 == 0 and i!=0:
				plots = []
		else:
			break
		i += 1
