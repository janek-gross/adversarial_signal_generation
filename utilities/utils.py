import numpy as np
from utilities.config import *

def create_sine_dataset(n_samples = 100000, n_steps = Config.n_steps):
    x = np.linspace(0,2*np.pi, n_steps)
    x = np.tile(x, (n_samples, 1))
    a = np.random.rand(n_samples)
    b = 2 * np.random.rand(n_samples)-1
    c = np.random.rand(n_samples)
    d = np.pi * (2 * np.random.rand(n_samples) - 1)
    x = a[:,None] * np.sin(x/(c[:,None]+ 0.2) + d[:,None])+b[:,None]
    return x

def categorical_noise():
    """
    returns a categorical randomvector z = (z_c,z_n) where
    z_c is a one-hot vector for the category and z_n is an n-dimensional normaldistributed.
    z.shape = (batch_size, n_steps, n_clusters + dim_clusters)
    """
    
    z_c = np.zeros((Config.batch_size, Config.n_steps, Config.n_clusters)).astype("float32")
    c = np.random.choice(Config.n_clusters, size = (Config.batch_size, Config.n_steps), replace = True)
    x,y = np.ogrid[0:Config.batch_size,0:Config.n_steps]
    z_c[x,y,c] = 1
    z_n = np.random.normal(0, Config.sigma_clusters, (Config.batch_size, Config.n_steps, Config.dim_clusters)).astype("float32")
    norm = np.sum(np.abs(z_n)**2,axis=2)**(1./2)
    norm = norm + 1e-10
    z_n = (z_n/norm[:,:,None])
    
    return z_c,z_n

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val