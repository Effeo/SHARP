import scipy.io
import numpy as np
import os

input_folder = os.path.abspath('../input_files')

mat_files = os.listdir(input_folder)

for mat_file in mat_files:
    mat = scipy.io.loadmat(input_folder + '/' + mat_file)
    mat = mat['csi_buff']
    radius_mat = np.abs(mat)
    angles_mat = np.angle(mat)
    n = np.shape(mat)[0]
    m = np.shape(mat)[1]
    processed_mat = np.zeros((n, 2*m))
    for i in range(m):
        processed_mat[:, 2*i] = radius_mat[:, i]
        processed_mat[:, 2*i + 1] = angles_mat[:, i]
    scipy.io.savemat('polars_mat/' + mat_file, {'csi_buff':processed_mat})