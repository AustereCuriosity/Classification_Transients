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
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

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
    #if key in (1123015019241, 1123015024731, 1157022019001, 1157020021365, 1155054039020, 1155054020923, 1155025011884, 1155022019510, 1152056026024, 1152030012731, 1152028028497, 1152028025177, 1152027059311, 1152026061985, 1152026027692, 1152026012877, 1152025066519, 1152025059518, 1152025030558, 1152025011108, 1152024067005, 1152024066684, 1152024047266, 1149064033598, 1149064020637, 1149035039958, 1149032034124, 1149032032696, 1149032011655, 1149028066683, 1149028054794, 1149027054524, 1149027050411, 1149027025335, 1149026081544, 1149026075858, 1149025041253, 1146069052359, 1146068091844, 1146068047489, 1146063006940, 1146061047409, 1146042005469, 1146033042569, 1146031032626, 1146030057280, 1146030009612, 1146029057640, 1146029044328, 1146029003829, 1146028048409, 1143070066639, 1143070043229, 1143070036314, 1143069073133, 1143069010376, 1143060039607, 1143047005482, 1143044048404, 1143039028116, 1143039015331, 1143037055708, 1143037042845, 1143035022220, 1143033061153, 1143033059299, 1143033028826, 1143033016539, 1143032044258, 1143032026639, 1143031019687, 1143030069283, 1143030022581, 1143030017400, 1143029075892, 1143029068650, 1143029051729, 1143029043243, 1143029041137, 1143029037426, 1143029014323, 1143028063911, 1143028018472, 1140098051678, 1140098025593):
    #    plot = True
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
    print i, key
		
feature_space_df = pd.DataFrame(feature_space)
#feature_space_df.describe()
feature_space = feature_space_df.as_matrix()
target = [x[2] for x in feature_vars]
target = np.asarray(target)

feature_l2norm = normalize(feature_space, axis = 0)
pca = PCA()
feature_l2norm_pca = pca.fit_transform(feature_l2norm)


feature_1varnorm = np.zeros([46821, 23])

for i in range(0,23):
    feature_1varnorm[:,i] = np.float64(feature_space[:,i] - np.mean(feature_space[:,i]))/np.std(feature_space[:,i])
    
pca_1 = PCA()
feature_1varnorm_pca = pca_1.fit_transform(feature_1varnorm)

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(feature_1varnorm, target)

