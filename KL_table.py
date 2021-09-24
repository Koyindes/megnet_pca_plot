import numpy as np
import matplotlib.pyplot as plt
import json
from math import log

def KL_divergence(dim, p, q):
    global X_count_list
    
    P = X_count_list[dim][p]
    Q = X_count_list[dim][q]
    
    s = 0
    for i in range(len(Q)):
        px = P[i] / sum(P)
        qx = Q[i] / sum(Q)
        if px != 0 and qx != 0:
            s += px * log(px / qx)
    return s
    
# df_name = [['S', 'H', 'G']]
df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH', 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

pca_dict = {}

with open('pca.json', encoding='utf-8') as f:
    pca_dict = json.load(f)
    f.close()

X_list = []
X_count_list = []

for dim in range(3):
    X = []
    for name in df_name[0]:
        if len(pca_dict[name]) != 0:
            X.append([i[dim] for i in pca_dict[name]])
        else:
            X.append([])
    
    X_count = []
    for i in range(len(df_name[0])):
        count = np.zeros(11)
        for x in X[i]:
            count[int(x/0.1)] += 1
        X_count.append(count)
        
    X_list.append(X)
    X_count_list.append(X_count)

col = ['P-S', 'P-H', 'P-G', 'S-H', 'S-G', 'H-G']
row = [' pca_0 ', ' pca_1 ', ' pca_2 ', ' average ']
value = np.zeros((4, 6))
for r in range(len(row)-1):
    c = 0
    for i in range(len(df_name[0])):
        for j in range(i+1, len(df_name[0])):
            value[r][c] = (KL_divergence(r, i, j) + KL_divergence(r, j, i)) / 2
            c += 1
            
for i in range(len(col)):
    value[3][i] = (value[0][i] + value[1][i] + value[2][i])/3

plt.figure(figsize=(12,6))
tab = plt.table(cellText = value, 
              colLabels = col, 
              rowLabels = row,
              loc = 'center', 
              cellLoc = 'center',
              rowLoc = 'center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
# tab.scale(1,1.5) 
plt.subplots_adjust(bottom=0.000, right=0.995, left=0.070, top=1.000)
plt.axis('off')
plt.show()
