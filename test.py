import numpy as np
im = np.load('./Moffett Field/Y.npy')
edm = np.load('./Moffett Field/E.npy')

print(im.shape)
print(edm.shape)


im = np.load('./Synthetic data/Y.npy')
edm = np.load('./Synthetic data/E.npy')

print(im.shape)
print(edm.shape)