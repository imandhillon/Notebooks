import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from matplotlib import pylab

np.random.seed(0)
'''
import itertools as itt
from collections import Counter
n = 3
xy = ("x", "y") # list of variables may be extended indefinitely
poly = '+'.join(itt.starmap(lambda u, t: u+"*"+t if t else u, zip(map(lambda v: "C["+str(v)+"]", itt.count()),map(lambda z: "*".join(z), map(lambda x: tuple(map(lambda y: "**".join(map(str, filter(lambda w: w!=1, y))), x)), map(dict.items, (map(Counter, itt.chain.from_iterable(itt.combinations_with_replacement(xy, i) for i in range(n+1))))))))))
'''


# Generate Gaussian datasets (2/high dimensional)
mean = (3,3)
cov = [[2,0],[0,2]]
gauss2d = np.random.multivariate_normal(mean, cov, 1000).T
print(gauss2d.size)

plt.scatter(gauss2d[0,:],gauss2d[1,:])
plt.show()

meanhighD = np.random.randint(10, size=50)
covhighD = np.identity(50, dtype=int)
gausshighD = np.random.multivariate_normal(meanhighD, covhighD, 1000)

##################
# Generate non-Gaussian datasets (2/high dimensional)
points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
x = points[:,0]
y = points[:,1]

z = np.polyfit(x, y, 3)
f = np.poly1d(z)

x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)
polynom2d = np.vstack((x_new, y_new))

#plt.plot(x,y,'o', x_new, y_new)
#plt.show()

# Now add noise


# Generate high dim signal

sin = np.zeros((50, 1000))
for i in range(50):
    # dec[0] is for (high/low) amplitude
    # dec[1] is for (high/low) noise variance
    # dec[2] is for (high/low) frequency
    # where < means low and >= means high
    dec = np.random.uniform(-1, 1, size=3)

    a = np.arange(1000, dtype=float) #.reshape((1000/50),50)

    if dec[0] < 0.3:
        amp = np.random.normal(0.5, 1.3)
    elif dec[0] >= 0.3:
        amp = np.random.normal(1.2, 2.0)

    if dec[1] < 0:
        noise = np.random.normal(-0.4, 0.3, size=1000)
    elif dec[1] >= 0:
        noise = np.random.normal(0.4, 1.1, size=1000)

    if dec[2] < 0.45:
        radinterval = np.random.normal(0.5, 1.3)
    elif dec[2] >= 0.45:
        radinterval = np.random.normal(1.3, 2.1)
    
    a *= radinterval
    rad = np.radians(a)

    sin[i] = (amp * np.sin(rad)) + noise

    # Will make label vector soon


# Run PCA on data
pca = PCA()
pca.fit(gausshighD)
#print('Explained')
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)
print(pca.mean_.shape)
print(pca.components_.shape)
reduceddata = np.dot(gausshighD - pca.mean_, pca.components_.T)
reproduction = np.dot(reduceddata, pca.components_) + pca.mean_
print(reduceddata.shape)
print(reproduction.shape)
plt.scatter(reproduction[0,:],reproduction[1,:])
plt.show()


#print('Singular Values')
#print(pca.singular_values_) 

kpca = KernelPCA(n_components=13)
kpca.fit(gausshighD)
print('kpca lambdas')
print(kpca.lambdas_)

kpca_explained = np.zeros(13)
for i in range(13):
    kpca_explained[i] = kpca.lambdas_[i]/np.sum(kpca.lambdas_)
print(kpca_explained)

#kpca.inverse_transform()