from subprocess import Popen,PIPE
import psutil

plots_generator = Popen(["python","time_series.py"],stdin = PIPE ,stdout = PIPE,stderr = PIPE,bufsize = 0)
features_extractor = Popen(["python","extract_features.py"],stdin = PIPE,stdout = PIPE,stderr = PIPE,bufsize = 0)

while True:
	ID = plots_generator.stdout.readline()
	features_extractor.stdin.write('{0}\n'.format(ID))
	
	features = features_extractor.stdout.readline()
	print features
