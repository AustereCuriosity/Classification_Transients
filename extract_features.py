import numpy as np
from astropy.io import ascii
from scipy import stats
import pandas as pd
import math
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt


#np.max(plots[0][1]['col3'] magnitude

class Features():

    def __init__(self, series, **kwargs):
        self.magnitude = series['Mag'].as_matrix()
        self.time = series['MJD'].as_matrix()
        self.flux = (10**(-self.magnitude*0.4)) * 363.1
        error = kwargs.pop('mag_err', None)
        if error is not None:
        	self.mag_err=series['Mag_err'].as_matrix()
        
        self.features = dict()
        
        #plot original light curve    
        #plt.scatter(self.time, self.magnitude, color = 'green')
        #plt.pause(2)
        
        #mag_err outliers
        outlier_indices = np.where(self.mag_err > 1)
        if float(len(outlier_indices)) / len(self.magnitude) > 0.05:
            outlier_indices = np.argpartition(self.mag_err, -int(math.ceil(0.05 * self.n)))[-int(math.ceil(0.05 * self.n)):]
        self.mag_err = np.delete(self.mag_err, outlier_indices)
        self.magnitude = np.delete(self.magnitude, outlier_indices)
        self.time = np.delete(self.time, outlier_indices)
        
        #plot light curve excluding mag_err outliers
        #plt.scatter(self.time, self.magnitude, color = 'red')
        #plt.pause(2)
        #plt.clf()
        
        #interquartile range
        quartiles = np.percentile(self.magnitude, [25, 75])
        IQR = quartiles[1] - quartiles[0]
        q1 = np.where(self.magnitude < quartiles[0] - 3*IQR)
        q3 = np.where(self.magnitude > quartiles[1] + 3*IQR)
        
        #modified Z-scores
        median = np.median(self.magnitude)
        mad = np.median(np.abs(self.magnitude - median))
        m_z_score = 0.6745*(self.magnitude - median)/ mad
        
        #modified Z-scores with an absolute value greater than 3.5 be labeled as potential outliers
        indices = np.where(np.abs(m_z_score) > 3.5)
        
        #Z-scores
        mean = np.mean(self.magnitude)
        std = np.sqrt((1.0 / (len(self.magnitude) - 1)) * sum((self.magnitude - mean) ** 2))
        z_score = (self.magnitude - mean)/ std

        #Z-scores with an absolute value greater than 3 be labeled as potential outliers
        lower = np.asarray(np.where(z_score < -3)[0])
        higher = np.asarray(np.where(z_score > 3)[0])
        #check for consecutive observations in outliers i.e. if neighbouring points for an outlier are also outliers, they probably signify a pattern
        if len(lower) > 2:
            first, length = self.longestConsecutive(lower)
            while length > 2:
                index = np.argwhere(lower == first)
                indices = range(index, index + length)
                lower = np.delete(lower, indices)
                if len(lower) < 3:
                    break
                first, length = self.longestConsecutive(lower)
        if len(higher) > 2:
            first, length = self.longestConsecutive(higher)
            while length > 2:
                index = np.argwhere(higher == first)
                indices = range(index, index + length)
                higher = np.delete(higher, indices)
                if len(higher) < 3:
                    break
                first, length = self.longestConsecutive(higher)
        
        #remove outliers
        print lower, higher
        self.mag_err = np.delete(self.mag_err, lower)
        self.magnitude = np.delete(self.magnitude, lower)
        self.time = np.delete(self.time, lower)
        self.mag_err = np.delete(self.mag_err, higher)
        self.magnitude = np.delete(self.magnitude, higher)
        self.time = np.delete(self.time, higher)
        
        #plot light curve without outliers
        #plt.scatter(self.time, self.magnitude, color = 'green')
        #plt.pause(2)
        #plt.clf()
        
        self.n  = len(self.magnitude)
        self.mean = np.mean(self.magnitude)
        self.median = np.median(self.magnitude)
        self.std = np.sqrt((1.0 / (self.n - 1)) * sum((self.magnitude - self.mean) ** 2))
        nlags = kwargs.pop('nlags', None)
        if nlags is not None:
            self.nlags = nlags
        else:
            self.nlags = 100

    def longestConsecutive(self, num):
        maxrun, maxend = -1, -1
        rl = {}
        for x in num:
            run = rl[x] = rl.get(x-1, 0) + 1
            if run > maxrun:
                maxend, maxrun = x, run
        return maxend-maxrun+1, maxrun

    def call_all_features(self):
    	self.amplitude()
    	self.beyond_1_std()
    	self.linear_trend()
    	self.flux_percentile_ratio()
    	self.skew()
    	self.max_slope()
    	self.median_abs_dev()
    	self.percent_amplitude()
    	self.small_kurtosis()
    	self.med_buffer_range_percentage()
    	self.pdfp()
    	self.pair_slope_trend()
    	self.STD()
    	self.above_median()
    	self.rcorbor()
    	self.magratio()
    	self.RCS()
    	self.StetsonK()
    	self.Autocor_length()

    def amplitude(self):
        #self.features['amplitude'] = 0.5* (np.max(self.magnitude) - np.min(self.magnitude))
        sorted_mag = np.sort(self.magnitude)
        self.features['amplitude'] = (np.median(sorted_mag[-int(math.ceil(0.05 * self.n)):]) - np.median(sorted_mag[0:int(math.ceil(0.05 * self.n))])) / 2.0   

    def beyond_1_std(self):
        #unique, counts = np.unique(beyond_std, return_counts=True)
        #map_count = dict(zip(unique,counts))
        #if True in unique:
        #    beyond_1_std = float(map_count[True])/len(self.magnitude)*100.0
        #else:
        #    beyond_1_std = 0
        #self.features['beyond_1_std'] = beyond_1_std
        weighted_mean = np.average(self.magnitude, weights=1 / self.mag_err ** 2)
        var = (1.0 / (self.n - 1)) * sum((self.magnitude - weighted_mean) ** 2)
        std = np.sqrt(var)
        beyond_std_count = np.sum(np.abs(self.magnitude - weighted_mean) > std)
        self.features['beyond_1_std'] = float(beyond_std_count) / self.n
        
    def linear_trend(self):
        self.regression = stats.linregress(self.time,self.magnitude)
        self.features['linear trend'] = self.regression.slope 

    def flux_percentile_ratio(self):#flux
        self.percentiles = np.percentile(self.magnitude, [95, 5, 60, 40, 67.5, 32.5, 75, 25, 82.5, 17.5, 90, 10])
        for i in range(2,11,2):
            self.features['flux_percentile_ratio_%d' %i] = float(self.percentiles[i] - self.percentiles[i+1]) / (self.percentiles[0] - self.percentiles[1])
    
    def skew(self):
        self.features['skew'] = stats.skew(self.magnitude)
    
    def max_slope(self):
        time = np.ediff1d(self.time)
        magnitude = np.ediff1d(self.magnitude)
        indices = np.where(time == 0.0)
        time = np.delete(time, indices)
        magnitude = np.delete(magnitude, indices)
        self.features['max_slope'] = np.max(np.abs(np.float64(magnitude) / time))
        
    def median_abs_dev(self):#flux
        #self.flux_median = np.median(self.flux)
        self.features['median_abs_dev'] = np.median(np.abs(self.magnitude - self.median))
        
    def percent_amplitude(self):#flux
        self.features['percent_amplitude'] = max([ np.abs(np.max(self.magnitude) - self.median), np.abs(np.min(self.magnitude) - self.median) ])
        
    def small_kurtosis(self):#sample statistics
        #self.features['small_kurtosis'] = stats.kurtosis(self.magnitude)
        S = np.sum(((self.magnitude - self.mean) / self.std) ** 4)

        c1 = float(self.n * (self.n + 1)) / ((self.n - 1) * (self.n - 2) * (self.n - 3))
        c2 = float(3 * (self.n - 1) ** 2) / ((self.n - 2) * (self.n - 3))

        self.features['small_kurtosis'] = c1 * S - c2
    
    def med_buffer_range_percentage(self):#flux
        mbrp_count = np.sum(np.abs(self.magnitude - self.median) < (0.1 * self.median))
        self.features['med_buffer_range_percentage'] = float(mbrp_count) / self.n
    
    def pdfp(self):#flux
        if self.percentiles is None:
            self.percentiles = np.percentile(self.magnitude, [95,5])
        self.features['pdfp'] = (self.percentiles[0] - self.percentiles[1]) / self.median
    
    def pair_slope_trend(self):#flux
        #pos_30_slope = []
        if self.n > 30:
            #for i in range (self.n -31, self.n -1):
            #    pos_30_slope.append((self.flux[i+1] - self.flux[i]) > 0)
            pair_30_slope_count = np.sum(np.ediff1d(self.magnitude[-31:]) > 0)
            total_count = 30
        else:
            #for i in range (0, self.n -1):
            #    pos_30_slope.append((self.flux[i+1] - self.flux[i]) > 0)
            pair_30_slope_count = np.sum(np.ediff1d(self.magnitude) > 0)
            total_count = self.n - 1
        self.features['pair_slope_trend'] = float(pair_30_slope_count) / total_count
        
    def STD(self):
        self.features['std'] = self.std

    def above_median(self):
        max = np.max(self.magnitude)
        self.features['above_median'] = (max - self.median)/(max - np.min(self.magnitude))
    
    def rcorbor(self):#p(mag > (magmed + 1.5))
        rcorbor_count = np.sum(self.magnitude > (self.median + 1.5))
        self.features['rcorbor'] = float(rcorbor_count) / self.n
        
    def magratio(self):#p(mag > magmed)
        magratio_count = np.sum(self.magnitude > self.median)
        self.features['magratio'] = float(magratio_count) / self.n

    def RCS(self):
        s = np.cumsum(self.magnitude - self.mean) * 1.0 / (self.n * self.std)
        self.features['RCS']  = np.max(s) - np.min(s)
        
    def StetsonK(self):
        mean_mag = (np.sum(self.magnitude/(self.mag_err*self.mag_err)) /
                    np.sum(1.0 / (self.mag_err*self.mag_err)))

        delta = (np.sqrt(self.n * 1.0 / (self.n - 1)) * (self.magnitude - mean_mag) / self.mag_err)

        self.features['stetsonK']  = (1 / np.sqrt(self.n * 1.0) * np.sum(np.abs(delta)) / np.sqrt(np.sum(delta ** 2)))

    def Autocor_length(self):
        AC = stattools.acf(self.magnitude, nlags=self.nlags)
        k = next((index for index, value in enumerate(AC) if value < np.exp(-1)), None)

        while k is None:
            self.nlags = self.nlags + 100
            AC = stattools.acf(self.magnitude, nlags=self.nlags)
            k = next((index for index, value in enumerate(AC) if value < np.exp(-1)), None)

        self.features['autocor_len'] = k

    #Lomb-Scargle Periodogram

    @property
    def feature_list(self):
        return self.features

if __name__=='__main__':
    
    #data_object = raw_input()
    #print data_object
	#fits.getdata('{0}{1}.fits'.format(key,series))
    #series  = ascii.read('sample_series/FILE1109065028988',delimiter = ',')
    series = pd.read_csv('sample_series/FILE1109066003963',header=None, sep=',')
    series.columns = ['Numerical_ID','MJD','Mag','Mag_err','RA','DEC']
    f = Features(series, mag_err=True)
    f.call_all_features()
    print f.feature_list
