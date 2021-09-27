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
        
    for key in q_dict.keys():
        if key in p_dict.keys():
            px = p_dict[key] / sum_p
            qx = q_dict[key] / sum_q
            kl += px * log(px / qx)
            
    if len(q_dict.keys()) == (arr_size+1)**3 and len(p_dict.keys()) == (arr_size+1)**3:
        return [kl, 1]
    else:
        return [kl, 0]

df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH', 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

pca_dict = {}

with open('all_pca.json', encoding='utf-8') as f:
    pca_dict = json.load(f)
    f.close()

col = ['P-SHG', 'S-PHG', 'H-PSG', 'G-PSH']
col_e = ['E-P', 'E-S', 'E-H', 'E-G']
# row = [' pca_0 ', ' pca_1 ', ' pca_2 ']
row = [' norm-1 ', ' norm-2 ']

max_rows = 400
value = np.zeros((max_rows, 4))
max_kl = np.zeros(4)
max_kl_e = np.zeros(4)

print(pca_dict.keys())

effevtive_line = -1
for i, left in enumerate(df_name[0]):
    # p = pca_dict[left]
    p = []
    for key in pca_dict.keys():
        if left in key:
            p.extend(pca_dict[key])
    print(left, len(p))
    q = []
    for key in pca_dict.keys():
        if left != key and key != 'E':
            q.extend(pca_dict[key])
    # value[0][i], value[1][i], value[2][i] = KL_divergence(p, q)
    # value[0][i] = KL_3d(p, q)
    
    check = 1
    for row_idx in range(max_rows-2):
        arr_size = row_idx + 1
        value[row_idx][i], check = KL_3d_fast(p, q, arr_size)
        if check == 0 and effevtive_line == -1:
            effevtive_line = row_idx
        max_kl[i] = max(max_kl[i], value[row_idx][i])
    value[-2][i], check = KL_3d_fast(p, q, 1001)
    value[-1][i], check = KL_3d_fast(p, q, 10001)
# print(value)
print(max_kl)

effevtive_line_e = -1
value_e = np.zeros((max_rows, 4))
for i, right in enumerate(df_name[0]):
    p = pca_dict['E']
    q = []
    for key in pca_dict.keys():
        if right in key:
            q.extend(pca_dict[key])
    print(right, len(q))
    # value[0][i], value[1][i], value[2][i] = KL_divergence(p, q)
    # value[0][i] = KL_3d(p, q)
    
    check = 1
    for row_idx in range(max_rows-2):
        arr_size = row_idx + 1
        value_e[row_idx][i], check = KL_3d_fast(p, q, arr_size)
        if check == 0 and effevtive_line_e == -1:
            effevtive_line_e = row_idx
        max_kl_e[i] = max(max_kl_e[i], value_e[row_idx][i])
    value_e[-2][i], check = KL_3d_fast(p, q, 1001)
    value_e[-1][i], check = KL_3d_fast(p, q, 10001)
print(max_kl_e)
    
colors = ['green', 'orange', 'red', 'blue', 'gray', 'purple']
l = np.zeros((2, 4))

fig = plt.figure(figsize=(12,8))
for i in range(len(df_name[0])):
    y = [a[i] for a in value]
    l[0][i] = np.linalg.norm(y,ord=1)
    l[1][i] = np.linalg.norm(y,ord=2)
    ax = plt.plot(range(1, max_rows+1), y, marker='.', c=colors[i], label=col[i])
plt.axvline(effevtive_line)
plt.subplots_adjust(top=0.995, bottom=0.03, left=0.02, right=0.995)
plt.legend()

fig = plt.figure(figsize=(12,8))
for i in range(len(df_name[0])):
    y = [a[i] for a in value_e]
    ax = plt.plot(range(1, max_rows+1), y, marker='.', c=colors[i], label=col_e[i])
plt.axvline(effevtive_line_e)
plt.subplots_adjust(top=0.995, bottom=0.03, left=0.03, right=0.995)
plt.legend()

plt.figure(figsize=(12,6))
tab = plt.table(cellText = l, 
              colLabels = col, 
              rowLabels = row,
              loc = 'center', 
              cellLoc = 'center',
              rowLoc = 'center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
# tab.scale(1,1.5) 
plt.subplots_adjust(top=1.0, bottom=0.03, left=0.065, right=0.995,)
plt.axis('off')
plt.show()
