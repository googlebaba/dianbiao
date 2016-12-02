# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage


data = pd.read_csv('./data/1102DE.csv', dtype={"QUIP_ID":object})

#data['SPEC_CODE'] = data['SPEC_CODE'].astype("category")
#data['ORG_NO_5'] = data['ORG_NO_5'].astype("category")
#data['MANUFACTURER'] = data['MANUFACTURER'].astype("category")
#data['INST_DATE_QUARTER'] = data['INST_DATE_QUARTER'].astype("category")
#data['FAULT_3'] = data['FAULT_3'].astype("category")


def plot_pivo(name1, name2):
    fault_num1 = data.groupby([name1, name2], as_index=False)[data.columns[0]].count()
    data_piv1 = fault_num1.pivot(name1, name2, data.columns[0])
    g1 = sns.clustermap(data_piv1)
    plt.title("fault number")
    g1.savefig("./result/%svs%s.png" %(name1,name2))


def plot_uni(name):
    m_c = pd.get_dummies(data[name]).corr(method='pearson')
    l = linkage(m_c, 'ward')
    m_g = sns.clustermap(m_c, linewidths=0, cmap=plt.get_cmap('RdBu'),
                     vmax=1, vmin=-1, figsize=(14, 14), row_linkage=l, col_linkage=l)
    plt.title(name)

def plot_cor(name1, name2):
    m_c = data[[name1, name2]].corr(method='pearson')
    l = linkage(m_c, 'ward')
    m_g = sns.clustermap(m_c, linewidths=0, cmap=plt.get_cmap('RdBu'),
                     vmax=1, vmin=-1, figsize=(14, 14), row_linkage=l, col_linkage=l)
    plt.title(name1+"vs"+name2)





if __name__ == '__main__':
#    plot_uni('SPEC_CODE')
#    plot_uni('MANUFACTURER')
#    plot_uni('ORG_NO_5')
#    plot_uni('INST_DATE_QUARTER')
#    plot_uni('FAULT_3')

#    plot_pivo('FAULT_3', 'SPEC_CODE')
#
#    plot_pivo('FAULT_3', 'MANUFACTURER')

#    plot_pivo('FAULT_3', 'ORG_NO_5')

#    plot_pivo('FAULT_3', 'INST_DATE_QUARTER')

    plot_pivo('FAULT_TYPE', 'FAULT_QUARTER')

    plt.show()
