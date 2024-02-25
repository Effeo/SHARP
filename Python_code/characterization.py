import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd

def module_to_decibel(x):
    if x == 0:
        return 0
    return (10 * math.log10(x))

def plot_f1_score(cfr_mean, cfr_var, hatches, x, activity):
    fig, ax = plt.subplots(layout='constrained')
    print(cfr_mean[31])
    graph = ax.bar(x, [cfr_mean[31], cfr_mean[63], cfr_mean[95], cfr_mean[127], cfr_mean[159], cfr_mean[191], cfr_mean[223]]
               ,yerr=[cfr_var[31], cfr_var[63], cfr_var[95], cfr_var[127], cfr_var[159], cfr_var[191], cfr_mean[223]],
                 align='center', alpha=0.5, ecolor='black', capsize=10, hatch = hatches, color = colors_list) # color = colors_list


    coord_x = x = np.arange(7)
    width = 0.25  
    multiplier = 0

    ax.set_ylabel('Percentage')
    ax.set_xticks(x)
    ax.set_xticklabels([32, 64, 96, 128, 160, 192, 224], ha='right')
    ax.set_title('Means and variance of F1 score by ofdm: Activity ' + activity)
    ax.yaxis.grid(True)

    fig.savefig('models_outputs/characterization_' + activity +  '.png')

dataset = scipy.io.loadmat('./dataset/dataset.mat')
dataset = dataset['csi_buff']
dataset = np.delete(dataset, -1, 1)
n = np.shape(dataset)[0]
m = int((np.shape(dataset)[1])/2)
ds = np.zeros((n, m))
for i in range(0, m-1):
    ds[:, i] = dataset[:, 2*i]


vfunc = np.vectorize(module_to_decibel)
ds = vfunc(ds)

cfr_mean_C = np.mean(ds[:1000,:], axis = 0)
cfr_mean_E = np.mean(ds[1000:2000,:], axis = 0)
cfr_mean_G = np.mean(ds[2000:3000,:], axis = 0)
cfr_mean_J = np.mean(ds[3000:4000,:], axis = 0)
cfr_mean_L = np.mean(ds[4000:5000,:], axis = 0)
cfr_mean_R = np.mean(ds[5000:6000,:], axis = 0)
cfr_mean_S = np.mean(ds[6000:7000,:], axis = 0)
cfr_mean_W = np.mean(ds[7000:,:], axis = 0)


cfr_var_C = np.var(ds[:1000,:], axis = 0)
cfr_var_E = np.var(ds[1000:2000,:], axis = 0)
cfr_var_G = np.var(ds[2000:3000,:], axis = 0)
cfr_var_J = np.var(ds[3000:4000,:], axis = 0)
cfr_var_L = np.var(ds[4000:5000,:], axis = 0)
cfr_var_R = np.var(ds[5000:6000,:], axis = 0)
cfr_var_S = np.var(ds[6000:7000,:], axis = 0)
cfr_var_W = np.var(ds[7000:,:], axis = 0)


fig, ax = plt.subplots(layout='constrained')
colors_list = ['Red', 'Orange', 'Blue', 'Purple', 'Green', 'Yellow']
hatches = ['/', '\\', '|', '-', '+', 'x', 'o']

coord_x = x = np.arange(7)
print(cfr_mean_C[31])
plot_f1_score(cfr_mean_C, cfr_var_C, hatches, x, 'C')
plot_f1_score(cfr_mean_E, cfr_var_E, hatches, x, 'E')
plot_f1_score(cfr_mean_G, cfr_var_G, hatches, x, 'G')
plot_f1_score(cfr_mean_J, cfr_var_J, hatches, x, 'J')
plot_f1_score(cfr_mean_L, cfr_var_L, hatches, x, 'L')
plot_f1_score(cfr_mean_R, cfr_var_R, hatches, x, 'R')
plot_f1_score(cfr_mean_S, cfr_var_S, hatches, x, 'S')
plot_f1_score(cfr_mean_W, cfr_var_W, hatches, x, 'W')
