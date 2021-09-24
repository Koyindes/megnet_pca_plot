import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import json
from math import log
from scipy import integrate

def pq(x, i):
    global X_mean
    
    mean = X_mean[i]
    n_std = X_std[i]
    
    return 1 / (np.sqrt(2*np.pi) * n_std) * np.exp(-(x-n_std)**2 / (2*n_std**2))
 
def KL_divergence(i, j):   
    f = lambda x, a, b: pq(x,a) * log(pq(x,a) / pq(x,b))
    return integrate.quad(f, 0, 1, args=(i, j, ))[0]
    
# df_name = [['S', 'H', 'G']]
df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH', 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

pca_dict = {}

with open('pca.json', encoding='utf-8') as f:
    pca_dict = json.load(f)
    f.close()

X_list = []
X_mean = []
X_std = []

for dim in range(3):
    X = np.array([])
    X_count = np.zeros(101)
    for name in df_name[0]:
        if len(pca_dict[name]) != 0:
            X = np.append(X, np.array([i[dim] for i in pca_dict[name]]))
            
    mean = np.average(X)
    X_mean.append(mean)
    
    n_std = np.std(X)
    X_std.append(n_std)

col = range(3)
row = ['  0  ', '  1  ', '  2  ']
value = np.zeros((3, 3))

for p in range(len(X_mean)):
    for q in range(len(X_mean)):
        value[p][q] = KL_divergence(p, q)

plt.figure()
tab = plt.table(cellText = value, 
              colLabels = col, 
              rowLabels = row,
              loc = 'center', 
              cellLoc = 'center',
              rowLoc = 'center')
tab.set_fontsize(50)
tab.scale(1,1.5) 
plt.subplots_adjust(bottom=0.000, right=0.990, left=0.055, top=1.000)
plt.axis('off')
plt.show()
