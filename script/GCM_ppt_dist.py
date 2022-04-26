def getdatafile(gage_name, label, start, end):
    ##################
    df_sf = pd.read_csv(gage_name, skiprows=1, names=sg_col, delim_whitespace=True)
    df_sf['flow_cfs'] = df_sf['flow']/86440
    df_sf['flow_cfs'].replace(0, 0.01)
    startdate = datetime(1974, 1, 1)
    df_sf['date'] = startdate+pd.to_timedelta(df_sf['time'].astype(int).astype(str)+' days 00:00:00')
    df_sf['model']=label
    df_sf = df_sf.loc[(df_sf['date']>=start) & (df_sf['date']<=end)]
    return df_sf

import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


data_col = ['year', 'month', 'day', 'f1', 'f2', 'f3', 'tmx', 'tm1', 'tm2', 'tm3', 'tmn', 'tmn1', 'tmn2', 'tmn3', 'ppt',
            'sr1', 'sr2', 'sr3', 'sr4']
cols = ['id', 'year', 'month', 'day', 'f1', 'f2', 'f3', 'basin_ppt', 'basin_ssflow_cfs', 'basin_sroff',
        'basin_recharge', 'basin_potet', 'basin_actet']
sg_col = ['time', 'stage', 'flow', 'depth', 'width', 'midpt_flow', 'ppt', 'ET', 'sfr_runoff', 'uzf_runoff']
modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green'), ('HadGEM2ES', 'red'), ('MIROC5', 'magenta')]
#modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green')]
scenlist = [('rcp45', 's', '-'), ('rcp85', '+', '--')]
############################
gage = '3_S3601A'
flow_cutoff = 100.0
bins = 15
dense = False
############################
df_sf_base = getdatafile('../data_files/gages/gauge_{}.go'.format(gage), 'base', datetime(1974, 1, 1),
                         datetime(2014, 12, 31))
peak_sf_base, peak_sf_ht_base = find_peaks(df_sf_base['flow_cfs'].values, 100.0)

fig, ax = plt.subplots(2,1, figsize=(9,9))
ax[0].plot(df_sf_base['date'].values, df_sf_base['flow_cfs'], lw=0.5, color='blue')
ax[0].set_yscale('log')
ax[0].set_ylabel('streamflow in CFS')
ax[0].set_title('Historical simulated streamflow, gage {}'.format(gage))
ax[0].set_xlabel('date')
ax[1].hist(peak_sf_ht_base['peak_heights'], bins=bins, density=dense)
ax[1].set_ylabel('number of days flow > {} CFS'.format(flow_cutoff))
ax[1].set_xlabel('streamflow in CFS')
if dense:
    ax[1].set_ylim(0,0.0015)
plt.savefig('./plots/hydrograph_{}_hist.png'.format(gage))
plt.show()

for mod in modlist:
    for scen in scenlist:
        model = mod[0]+'_'+scen[0]
        print('loading {} {}'.format(mod[0], scen[0]))
        gage_file = '../data_files/gages/gauge_{}_{}_{}.go'.format(mod[0], scen[0], gage)
        df_sf_gcm = getdatafile(gage_file, model, datetime(2015, 1, 1), datetime(2099, 12, 31))
        peak_sf, peak_sf_ht = find_peaks(df_sf_gcm['flow_cfs'].values, flow_cutoff)
        ######################################
        fig, ax = plt.subplots(2,1, figsize=(9,9))
        ax[0].plot(df_sf_gcm['date'].values, df_sf_gcm['flow_cfs'], lw=0.5, color='blue')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('streamflow in CFS')
        ax[0].set_xlabel('date')
        ax[0].set_title('Future simulated streamflow, gage {}, model {}'.format(gage, model))
        ax[1].hist(peak_sf_ht['peak_heights'], bins=bins, density=dense)
        if dense:
            ax[1].set_ylim(0,0.0012)
        ax[1].set_ylabel('number of days flow > {} CFS'.format(flow_cutoff))
        ax[1].set_xlabel('streamflow in CFS')
        plt.savefig('./plots/hydrograph_{}_{}.png'.format(gage, model))
        #plt.show()
