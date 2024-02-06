import os
import scipy.io
import numpy as np

labelled_mat = os.path.abspath('./labelled_mat') 

C = scipy.io.loadmat(labelled_mat + '/AR1a_C.mat')
C = C['csi_buff']
C = C[:1000,:]
E = scipy.io.loadmat(labelled_mat + '/AR1a_E.mat')
E = E['csi_buff']
E = E[:1000,:]
J = scipy.io.loadmat(labelled_mat + '/AR1a_J1.mat')
J = J['csi_buff']
J = J[:1000,:]
L = scipy.io.loadmat(labelled_mat + '/AR1a_L.mat')
L = L['csi_buff']
L = L[:1000,:]
R = scipy.io.loadmat(labelled_mat + '/AR1a_R.mat')
R = R['csi_buff']
R = R[:1000,:]
S = scipy.io.loadmat(labelled_mat + '/AR1a_S.mat')
S = S['csi_buff']
S = S[:1000,:]
W = scipy.io.loadmat(labelled_mat + '/AR1a_W.mat')
W = W['csi_buff']
W = W[:1000,:]
H = scipy.io.loadmat(labelled_mat + '/AR1a_H.mat')
H = H['csi_buff']
H = H[:1000,:]

dataset = np.vstack([C, E, J, L, R, S, W, H])

scipy.io.savemat('dataset/dataset.mat', {'csi_buff':dataset})