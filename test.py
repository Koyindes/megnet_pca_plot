import numpy as np
import matplotlib.pyplot as plt
import json
from math import log

def KL_divergence(p, q):
    kl = []
    for dim in range(3):
        P = [x[dim] for x in p]
        Q = [x[dim] for x in q]
        
        P_count = np.zeros(101)
        for x in P:
            P_count[int(x/0.01)] += 1
        P_count = P_count.tolist()
        
        Q_count = np.zeros(101)
        for x in Q:
            Q_count[int(x/0.01)] += 1
        Q_count = Q_count.tolist()
            
        s = 0
        for i in range(len(Q_count)):
            px = P_count[i] / sum(P_count)
            qx = Q_count[i] / sum(Q_count)
            if px != 0 and qx != 0:
                print(px * log(px / qx))
                s += px * log(px / qx)
        kl.append(s)
        print(s, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(kl)
    return kl
    
df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH',	 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

pca_dict = {}

with open('pca.json', encoding='utf-8') as f:
    pca_dict = json.load(f)
    f.close()

col = ['P-SHG', 'S-PHG', 'H-PSG', 'G-PSH']
row = [' pca_0 ', ' pca_1 ', ' pca_2 ']
value = np.zeros((3, 4))

for i, left in enumerate(df_name[0]):
    p = pca_dict[left]
    q = []
    for key in pca_dict.keys():
        if left != key:
            q.extend(pca_dict[key])
    value[0][i], value[1][i], value[2][i] = KL_divergence(p, q)
    
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
plt.subplots_adjust(bottom=0.040, right=0.995, left=0.055, top=1.000)
plt.axis('off')
plt.show()
