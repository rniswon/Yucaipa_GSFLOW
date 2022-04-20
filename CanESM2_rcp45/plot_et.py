import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

statvar = open('./output2/statvar_prms.dat','r')
line = statvar.readline()
cols = ['id', 'year', 'month', 'day', 'f1', 'f2', 'f3']
numcols = int(line.strip())
for i in range(numcols):
    line = statvar.readline()
    a = line.split()
    cols.append(a[0])
statvar.close()
df_statvar = pd.read_csv('./output2/statvar_prms.dat', skiprows=numcols + 1, names=cols, delim_whitespace=True,
                         index_col='id')
df_statvar_month = df_statvar.groupby(['year'])['basin_ppt', 'basin_potet', 'basin_actet'].sum().reset_index()
df_statvar_month['PEP'] = df_statvar_month['basin_potet']/df_statvar_month['basin_ppt']
df_statvar_month['AEP'] = df_statvar_month['basin_actet']/df_statvar_month['basin_ppt']
sns.relplot(data=df_statvar_month, x='PEP', y='AEP')
#plt.plot(df_statvar_month['PEP'].values,df_statvar_month['AEP'].values, lw=0)
plt.axhline(1.0)
plt.axvline(1.0)
plt.show()
print('done.')