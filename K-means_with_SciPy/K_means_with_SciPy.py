#import k means
from scipy.cluster.vq import kmeans, vq, whiten

#data generation
from numpy import vstack, array
from numpy.random import rand

#data generating with three feature
data = vstack((rand(100, 3) + array([.5,.5,.5]), rand(100, 3)))

#whiten the data
data = whiten(data)

#compute kmeans with three cluster
centroids,_ = kmeans(data, 3)
print(centroids)

#assign each sample to centroids
clx,_ = vq(data, centroids)
print(clx)
    

