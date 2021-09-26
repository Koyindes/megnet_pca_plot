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
                s += px * log(px / qx)
        kl.append(s)
    return kl

def KL_3d(p, q):
    kl = 0
    arr_size = 11
    p_distr = np.zeros((arr_size, arr_size, arr_size))
    q_distr = np.zeros((arr_size, arr_size, arr_size))
    arr_size -= 1
    for p_item in p:
        p_distr[(round(p_item[0]*arr_size))][(round(p_item[1]*arr_size))][(round(p_item[2]*arr_size))] += 1
    for q_item in q:
        q_distr[(round(q_item[0]*arr_size))][(round(q_item[1]*arr_size))][(round(q_item[2]*arr_size))] += 1
    arr_size += 1
    for i in range(arr_size):
        for j in range(arr_size):
            for k in range(arr_size):
                px = p_distr[i][j][k] / np.sum(p_distr)
                qx = q_distr[i][j][k] / np.sum(q_distr)
                if px != 0 and qx != 0:
                    kl += px * log(px / qx)
    return kl

def KL_3d_fast(p, q, arr_size):
    kl = 0
    p_dict = {}
    q_dict = {}
    arr_size -= 1
    for p_item in p:
        i = round(p_item[0]*arr_size)
        j = round(p_item[1]*arr_size)
        k = round(p_item[2]*arr_size)
        if (i, j, k) in p_dict.keys():
            p_dict[(i, j, k)] += 1
        else:
            p_dict[(i, j, k)] = 1
    
    for q_item in q:
        i = round(q_item[0]*arr_size)
        j = round(q_item[1]*arr_size)
        k = round(q_item[2]*arr_size)
        if (i, j, k) in q_dict.keys():
            q_dict[(i, j, k)] += 1
        else:
            q_dict[(i, j, k)] = 1
    sum_p = 0
    sum_q = 0
    for key in p_dict.keys():
        sum_p += p_dict[key]
    for key in q_dict.keys():
        sum_q += q_dict[key]
    
    for key in p_dict.keys():
        px = p_dict[key] / sum_p
        if key in q_dict.keys():
            qx = q_dict[key] / sum_q
        else:
            qx = 0

        if px != 0 and qx != 0:
            kl += px * log(px / qx)
    return kl

df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH',	 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

pca_dict = {}

with open('pca.json', encoding='utf-8') as f:
    pca_dict = json.load(f)
    f.close()

col = ['P-SHG', 'S-PHG', 'H-PSG', 'G-PSH']
row = [' pca_0 ', ' pca_1 ', ' pca_2 ']

max_rows = 100
value = np.zeros((max_rows, 4))
max_kl = np.zeros(4)

print(pca_dict.keys())

for i, left in enumerate(df_name[0]):
    p = pca_dict[left]
    print(left, len(p))
    q = []
    for key in pca_dict.keys():
        if left != key:
            q.extend(pca_dict[key])
    # value[0][i], value[1][i], value[2][i] = KL_divergence(p, q)
    # value[0][i] = KL_3d(p, q)

    for row_idx in range(max_rows-2):
        arr_size = row_idx + 1
        value[row_idx][i] = KL_3d_fast(p, q, arr_size)
        max_kl[i] = max(max_kl[i], KL_3d_fast(p, q, arr_size))
    value[-2][i] = KL_3d_fast(p, q, 1001)
    value[-1][i] = KL_3d_fast(p, q, 10001)
print(value)
print(max_kl)

# plt.figure(figsize=(12,6))
# tab = plt.table(cellText = value, 
#               colLabels = col, 
#               rowLabels = row,
#               loc = 'center', 
#               cellLoc = 'center',
#               rowLoc = 'center')
# tab.auto_set_font_size(False)
# tab.set_fontsize(10)
# # tab.scale(1,1.5) 
# plt.subplots_adjust(bottom=0.040, right=0.995, left=0.055, top=1.000)
# plt.axis('off')
# plt.show()
