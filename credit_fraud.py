import numpy as np 
import pandas as pd 
from scipy.stats import norm
import matplotlib.pyplot as plt


def data_normalize(x_train):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    x_train = (x_train-mu)/sigma
    return x_train
	
def lograthmic_scale(x_train) :
    x_train = np.log10(np.power(x_train,2))
    return x_train
    
credit = pd.read_csv('creditcard.csv')
credit_val = credit.values

credit_val = lograthmic_scale(credit_val)


v1 = credit_val[:,1]
onev = np.ones(np.size(v1))
v1nor = ((v1-np.mean(v1))/(np.std(v1)))
plt.hist(np.log(np.cbrt(np.power(v1,2))))
plt.hist(np.log(np.add((np.cbrt(np.power(v1,2))),onev)))

v2 = credit_val[:,2]
plt.xlim(-6,6)
plt.hist(np.log10(np.power(v2,2)),bins = 40)

v3 = credit_val[:,3]
plt.hist(np.log10(np.cbrt(np.power(v3,2))),bins=40)

v4 = credit_val[:,4]
plt.hist(v4,bins=50)

v5 = credit_val[:,4]

plt.hist(np.log10(np.power(v5,2)),bins = 40)

v29nor = (v29-np.mean(v29,axis=0))/np.std(v29)
plt.hist(np.log10(np.power(v29nor,2)),bins=40)


x_train = credit_val[:,1:30]
y_train = credit_val[:,30]

x_train = data_normalize(x_train)
x_train = lograthmic_scale(x_train)
x_mean = np.mean(x_train,axis=0)
x_std = np.std(x_train,axis=0)
x_mean.shape = (1,29)
x_std.shape = (1,29)
x_prob = norm.pdf(x_train ,loc=x_mean,scale=x_std)
x_scal_prob = np.prod(x_prob,axis=1)
