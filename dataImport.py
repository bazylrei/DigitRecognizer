import scipy
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats.distributions import randint
from sklearn.decomposition import PCA


if __name__  == "__main__":
	data = pd.DataFrame(pd.read_csv('train.csv', sep=',', header=0))
	header = data.columns.values

	print data.shape
	print data.head()
	print header

	dataArray = np.array(data)
	print dataArray.shape
	 
	 
	# MAKE SURE Inc_IND column is last
	X = dataArray[:, 1:].astype(float)
	y = dataArray[: ,0]
	print X.shape
	print y.shape
	 
	yFreq = scipy.stats.itemfreq(y)
	print yFreq

	pca = PCA(n_components = 784)
	pca.fit(X)
	x = pca.explained_variance_ratio_

	var1 = np.cumsum(np.round(x, decimals = 4) * 100)
	plt.plot(var1)
	plt.show()

	
