import numpy as np
from astropy.io import ascii
from scipy import stats

#np.max(plots[0][1]['col3'] magnitude

class Features():

    def __init__(self, series, **kwargs):
        self.magnitude = series['Mag']
        self.time = series['MJD']
        self.series = series
        self.flux = (10**(-self.magnitude*0.4)) * 363.1
        error = kwargs.pop('mag_err', None)
        if error is not None:
        	self.mag_error=series['Mag_err']
        
        self.features = dict()
    
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
    	
    def amplitude(self):
        self.features['amplitude'] = 0.5* (np.max(self.magnitude) - np.min(self.magnitude))
        
    def beyond_1_std(self):
        beyond_std = np.abs(self.magnitude - np.mean(self.magnitude)) > np.std(self.magnitude)
        unique, counts = np.unique(beyond_std, return_counts=True)
        map_count = dict(zip(unique,counts))
        beyond_1_std = float(map_count[True])/(map_count[True]+map_count[False])*100.0
        self.features['beyond_1_std'] = beyond_1_std
        
    def linear_trend(self):
        self.features['linear trend'] = stats.linregress(self.time,self.magnitude)
        
    def flux_percentile_ratio(self):
        self.percentiles = np.percentile(self.flux, [95, 5, 60, 40, 67.5, 32.5, 75, 25, 82.5, 17.5, 90, 10])
        for i in range(2,11,2):
            self.features['flux_percentile_ratio_%d' %i] = (self.percentiles[i]- self.percentiles[i+1]) / (self.percentiles[0]- self.percentiles[1])
    
    def skew(self):
        self.features['skew'] = stats.skew(self.magnitude)
    
    def max_slope(self):
        #self.series.sort(order = 'time')
        self.features['max_slope'] = np.max(np.abs(np.ediff1d(self.magnitude) / np.ediff1d(self.time)))
        
    def median_abs_dev(self):
        self.features['median_abs_dev'] = np.median(self.flux - np.median(self.flux))
        
    def percent_amplitude(self):
        flux_median = np.median(self.flux)
        self.features['percent_amplitude'] = max([ np.abs(np.max(self.flux) - flux_median), np.abs(np.min(self.flux) - flux_median) ])
        
    def small_kurtosis(self):
        self.features['small_kurtosis'] = stats.kurtosis(self.magnitude)
    
    def med_buffer_range_percentage(self):
        mbrp = np.abs(self.flux - np.median(self.flux)) < (0.1 * np.median(self.flux))
        unique, counts = np.unique(mbrp, return_counts=True)
        map_count = dict(zip(unique,counts))
        mbrp  = float(map_count[True])/(map_count[True]+map_count[False])*100.0
        self.features['med_buffer_range_percentage'] = mbrp
    
    def pdfp(self):
        if self.percentiles is not None:
            self.percentiles = np.percentile(self.flux, [95,5])

        self.features['pdfp'] = (self.percentiles[0] - self.percentiles[1]) / np.median(self.flux)
    
    def pair_slope_trend(self):
        n = len(self.flux)
        pos_30_slope = []
        for i in range (n-30, n):
            pos_30_slope.append((self.flux[i+1] - self.flux[i]) > 0 )
        unique, counts = np.unique(pos_30_slope, return_counts=True)
        map_count = dict(zip(unique,counts))
        pair_slope_trend = float(map_count[True])/(map_count[True]+map_count[False])*100.0
        self.features['beyond_1_std'] = pair_slope_trend
        
    def std(self):
        self.features['std'] = np.std(self.magnitude)

    def 
    @property
    def feature_list(self):
        return self.features

if __name__=='__main__':
    
    #data_object = raw_input()
    #print data_object
	#fits.getdata('{0}{1}.fits'.format(key,series))
    series  = ascii.read('sample_series/FILE1109065026725',delimiter = ',')
    f =Features(series)
    f.call_all_features()
