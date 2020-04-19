import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
fs=1200 # Sampling rate of the signal

position_data= pd.read_csv('AB_13_SC_30 - Report1.txt', sep="\t",skiprows=8)
#position_data.describe()
#position_data=(position_data.dropna()).to_numpy()
X=position_data.iloc[:,3:6]*100
#X=(X.fillna(X.mean())).to_numpy()
X=(X.fillna(0)).to_numpy()
time=position_data.iloc[:,2][1:]

t=np.linspace(0.1 ,len(X)/fs,len(X))

def testGauss(X, fs):
    b = gaussian(300, 25)
    gaX = filters.convolve1d(X[:,0][0:] , b/b.sum())
    gaY = filters.convolve1d(X[:,1][0:], b/b.sum())
    gaZ = filters.convolve1d(X[:,2][0:], b/b.sum())
    
    #print(ga)
    #plt.plot(t, ga)
    return gaX,gaY,gaZ
g_smth=testGauss(X,fs) # Window size: 250 ms or 300 samples
g_smth=np.asarray(g_smth).transpose()
#plt.figure('gaussWindow')
#plt.plot(t,g_smth)
#plt.title('gaussWindow ,std=25,size=300')
#plt.xlabel('time(s)')
#plt.ylabel('position(cm)')

#X=pd.to_numeric(data)
#K=np.multiply(X,100)


# proportion of variance
X_std = (g_smth - np.mean(g_smth,axis=0))/np.std(g_smth,axis=0)


#covr_matrix = np.cov(X_std , rowvar = False)
#covr_matrix1 = np.cov(X_std , rowvar = True)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_std)

#print("covr_matrix.shape > "+str(covr_matrix.shape))
#print("covr_matrix.shape1 > "+str(covr_matrix1.shape))
#print("covariance of the matrix is : \n%s" %covr_matrix1)


def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)
pca_components=pca2(g_smth,pc_count = None)
scalar = preprocessing.StandardScaler()
standardized_data = scalar.fit_transform(g_smth)
# n_components = numbers of dimenstions you want to retain
pca = decomposition.PCA(n_components=2)
# This line takes care of calculating co-variance matrix, eigen values, eigen vectors and multiplying top 2 eigen vectors with data-matrix X.
pca_data = pca.fit_transform(g_smth)






#In general a good idea is to scale the data
#scaler = StandardScaler()
#scaler.fit(X)
#X=scaler.transform(X)    
#
#pca = PCA()
#x_new = pca.fit_transform(X)
#
#def myplot(score,coeff,labels=None):
#    xs = score[:,0]
#    ys = score[:,1]
#    n = coeff.shape[0]
#    scalex = (1.0)/(xs.max() - xs.min())
#    scaley = (1.0)/(ys.max() - ys.min())
#    #plt.scatter(xs * scalex,ys * scaley, c = y)
#    for i in range(n):
#        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#        if labels is None:
#            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#        else:
#            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
#            return xs,ys
#
#my_plot=myplot(score,coeff,labels=None)
#
#plt.xlim(-1,1)
#plt.ylim(-1,1)
#plt.xlabel("PC{}".format(1))
#plt.ylabel("PC{}".format(2))
#plt.grid()
#
##Call the function. Use only the 2 PCs.
#myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
#plt.show()
