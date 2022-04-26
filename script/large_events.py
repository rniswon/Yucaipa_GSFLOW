import numpy as np


def assign_wy(row):
    if row.Date.month>=10:
        return(pd.datetime(row.Date.year+1,1,1).year)
    else:
        return(pd.datetime(row.Date.year,1,1).year)

def getdatafile(data_name, gage_name, startyear, endyear, label):
    ##################
    # get precipitation from statvar data, which is in inches from the PRMS model run without GSFLOW
    df_data = pd.read_csv(data_name, names=data_col, skiprows=6, delim_whitespace=True)   # historical period model for stats
    df_data = df_data.loc[(df_data['year']>=startyear) & (df_data['year']<=endyear)]
    #df_data['Date'] = pd.to_datetime(df_data[['year', 'month', 'day']])
    #df_data['WY'] = df_data.apply(lambda x: assign_wy(x), axis=1)
    #ar_ppt = df_data['ppt'].values
    df_sf = pd.read_csv(gage_name, skiprows=2, names=sg_col, delim_whitespace=True)
    #fig, ax = plt.subplots()
    #startday = 365
    #endday = 1825
    #ax.plot(df_sf['time'].values[startday:endday], df_sf['flow'].values[startday:endday], lw=0.5)
    #ax.set_xlim(startday, endday)
    #ar_sf = df_sf['flow'].values
    #plt.show()
    df_data['model']=label
    df_sf['model']=label
    return df_data, df_sf

import pandas as pd
from scipy.signal import find_peaks
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


data_col = ['year', 'month', 'day', 'f1', 'f2', 'f3', 'tmx', 'tm1', 'tm2', 'tm3', 'tmn', 'tmn1', 'tmn2', 'tmn3', 'ppt',
            'sr1', 'sr2', 'sr3', 'sr4']
cols = ['id', 'year', 'month', 'day', 'f1', 'f2', 'f3', 'basin_ppt', 'basin_ssflow_cfs', 'basin_sroff',
        'basin_recharge', 'basin_potet', 'basin_actet']
sg_col = ['time', 'stage', 'flow', 'depth', 'width', 'midpt_flow', 'ppt', 'ET', 'sfr_runoff', 'uzf_runoff']
#modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green'), ('HadGEM2ES', 'red'), ('MIROC5', 'magenta')]
modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green')]
scenlist = [('rcp45', 's', '-'), ('rcp85', '+', '--')]
df_ppt, df_sf = getdatafile('../data_files/Yucaipa_CNRMCM5_rcp45.data',
                                       '../data_files/gages/gauge_3_S3601A.go', 1947, 2014, 'base')
peak_ppt_base, peak_ppt_ht_base = find_peaks(df_ppt['ppt'].values, 1.0)

df_peaks = pd.DataFrame(peak_ppt_ht_base)
df_peaks['model']='base'
#sns.histplot(x=peak_ppt_ht_base['peak_heights'])
h = np.histogram(peak_ppt_ht_base['peak_heights'], density=True)
plt.hist(h, bins=np.arange(1,5))
plt.xlabel('precipitation event size in inches')
plt.title('Historical period')
plt.show()
#peak_sf_base, peak_sf_ht_base = find_peaks(df_sf['flow'].values, 10000)
#sns.histplot(x=peak_sf_ht_base['peak_heights'])
#plt.show()
for mod in modlist:
    for scen in scenlist:
        #sns.histplot(x=peak_ppt_ht_base['peak_heights'])
        print('loading {} {}'.format(mod[0], scen[0]))
        data_file = '../data_files/Yucaipa_{}_{}.data'.format(mod[0], scen[0])
        gage_file = '../data_files/gages/gauge_{}_{}_1_S3608A.go'.format(mod[0], scen[0])
        df_ppt_gcm, df_sf_gcm = getdatafile(data_file, gage_file, 2015, 2099, mod[0]+'_'+scen[0])
        #df_ppt = df_ppt.append(df_ppt_gcm)
        #df_sf = df_sf.append(df_sf_gcm)
        peak_ppt, peak_ppt_ht = find_peaks(df_ppt_gcm['ppt'].values, 1.0)
        h = np.histogram(peak_ppt_ht['peak_heights'],bins=np.arange(1, 5), density=True)
        #df = pd.DataFrame(peak_ppt_ht)
        #df['model'] = mod[0]+'_'+scen[0]
        #df_peaks = df_peaks.append(df)
        #sns.histplot(x=peak_ppt_ht['peak_heights'])
        plt.hist(h)
        plt.title(mod[0]+'_'+scen[0])
        plt.xlabel('precipitation event size in inches')
        plt.savefig('./plots/{}_{}_ppt_events.png'.format(mod[0], scen[0]))
        plt.show()
#fig, ax = plt.subplots()
#ax.hist(df_peaks['peak_heights'], n_bins=20)
print('here')