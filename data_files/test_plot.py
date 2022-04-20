def getdf(name):
    df = pd.read_csv(name, header=0)
    df['Date']=pd.to_datetime(df['Date'])
    df['month']=pd.DatetimeIndex(df['Date']).month
    df['year']=pd.DatetimeIndex(df['Date']).year
    return df


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

cat = 'StreamExchng2Sat_Q'

df_data = pd.read_csv('Yucaipa_CanESM2_rcp45.dataset.csv', header=0)
df_45 = getdf('gsflow_CanESM2_rcp45.csv')
df_45 = df_45.loc[df_45['year']>2014]
df_85 = getdf('gsflow_CanESM2_rcp85.csv')
df_all = df_45.merge(df_85, left_on='Date', right_on='Date', suffixes=['_45', '_85'])
df_month = df_all.groupby(['year_45', 'month_45'])[cat + '_45', cat + '_85'].sum().reset_index()
##################################
# base model
df_base = getdf('gsflow_base.csv')
df_base = df_base.groupby(['year', 'month'])[cat].sum().reset_index()
mean_ppt = df_base[cat].mean()   # this is the historical mean monthly precip
##################################
# get the departure from historical mean
df_month[cat + '_45_d'] = df_month[cat + '_45']-mean_ppt
df_month[cat + '_85_d'] = df_month[cat + '_85']-mean_ppt
print('mean of {} 45 = {}'.format(cat, df_month[cat + '_45_d'].mean()))
print('mean of {} 85 = {}'.format(cat, df_month[cat + '_85_d'].mean()))
ar_ppt = np.array(df_month[cat + '_45_d'].values, dtype=float)
ar_ppt = np.append(ar_ppt, df_month[cat + '_85_d'].values)
#n =
ar_model = np.zeros(df_month[cat + '_45_d'].count(), dtype=int)
ar_model = np.append(ar_model, ar_model+1)
df_plot = pd.DataFrame()
df_plot[cat] = ar_ppt
df_plot['model'] = ar_model
#sns.violinplot(y=cat, hue='model', data=df_plot, split=True)
#sns.relplot(x=cat + '_45_d', y=)
sns.jointplot(data = df_all, x=cat + '_45', y=cat + '_85_d')
plt.title('comparison of GCM CanESM2_rcp45 monthly {}'.format(cat))
plt.show()
sns.histplot(data=df_month, x='ppt')
plt.show()

