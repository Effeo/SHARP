import scipy.io
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(['E','W','R','J','L','S','C','H'])

polars_mat = os.path.abspath('./polars_mat')
mat_files = os.listdir(polars_mat)

for mat_file in mat_files:
    mat = scipy.io.loadmat(polars_mat + '/' + mat_file)
    mat = mat['csi_buff']
    label = mat_file[-6:-4]
    if label.startswith('_'):
        label = label[1]
    else:
        label = label[0]
    label = le.transform([label])[0]
    n = np.shape(mat)[0]
    labels = np.full((n, 1), label)
    mat = np.append(mat, labels, axis = 1)
    scipy.io.savemat('labelled_mat/' + mat_file, {'csi_buff':mat})