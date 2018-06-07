import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from matplotlib import pylab

np.random.seed(0)


def my_pca(data, svd=False, whiten=False):

    # Zero out mean for all dimensions
    for i in range(data.shape[0]):
        data[i,:] = data[i,:] - np.mean(data[i,:])

    mu = data.mean(axis=1)
    sigma = np.cov(data, rowvar=True)/data.shape[0]
    
    if svd is True:
        U,S,V = np.linalg.svd(sigma)
        if whiten is True:
            epsilon = 1e-9
            zca = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    else:
        evals, evecs = np.linalg.eigh(sigma)
        if whiten is True:
            epsilon = 1e-9
            D = np.diag(1. / np.sqrt(evals+epsilon))
            zca = np.dot(np.dot(evecs, D), evecs.T)
    print(evecs)
    projection = np.dot(evecs, data)
    print('evals: ', evals)
    sigma_proj = projection.std(axis=0).mean()

    fig, ax = plt.subplots()
    ax.scatter(projection[0,:], projection[1,:])
    for axis in evecs:
        start, end = mu, mu + sigma_proj * axis
        ax.annotate(
            '', xy=end, xycoords='data',
            xytext=start, textcoords='data',
            arrowprops=dict(facecolor='red', width=2.0))
    ax.set_aspect('equal')

    plt.show()
    loss = ((data - projection) **2).mean()
    print('mypca loss:', loss)


# Generate Gaussian datasets (2/high dimensional)
mean = (3,4)
cov = [[4,0],[0,4]]#2*np.eye(2)[[2,0],[0,2]]
gauss2d = np.random.multivariate_normal(mean, cov, 1000).T
mean2 = (5,18)
cov2 = [[10,2],[2,10]]
g2 = np.random.multivariate_normal(mean2, cov2, 1000).T
gauss2d = np.hstack((gauss2d, g2))

# Zero out mean for all dimensions
for i in range(gauss2d.shape[0]):
    gauss2d[i,:] = gauss2d[i,:] - np.mean(gauss2d[i,:])

plt.scatter(gauss2d[0,:],gauss2d[1,:])
plt.show()
my_pca(gauss2d)


# High D Gaussian
meanhighD = np.random.randint(10, size=50)
covhighD = np.identity(50, dtype=float)
gausshighD = np.random.multivariate_normal(meanhighD, covhighD, 1000)
#my_pca(gausshighD)

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
pca.fit(gauss2d.T)
print('Explained (sklearn)')
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

#reduceddata = np.dot(gauss2d - pca.mean_, pca.components_.T)
#reproduction = np.dot(reduceddata, pca.components_) + pca.mean_
transdata = pca.transform(gauss2d.T)
reproduction = pca.inverse_transform(transdata)
print(reproduction.shape)
plt.scatter(reproduction[:,0],reproduction[:,1])
plt.show()

skloss = ((gauss2d.T - reproduction) **2).mean()
print('skloss:', skloss)


# Plot sklearn projection with arrows for eigenvectors
mu = gauss2d.mean(axis=1)
sigma_proj = reproduction.std(axis=0).mean()

fig, ax = plt.subplots()
ax.scatter(reproduction.T[0,:], reproduction.T[1,:])
for axis in pca.components_:
    start, end = mu, mu + sigma_proj * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')

plt.show()


'''
kpca = KernelPCA(n_components=13)
kpca.fit(gausshighD)
print('kpca lambdas')
print(kpca.lambdas_)

kpca_explained = np.zeros(13)
for i in range(13):
    kpca_explained[i] = kpca.lambdas_[i]/np.sum(kpca.lambdas_)
print(kpca_explained)

#kpca.inverse_transform()
'''