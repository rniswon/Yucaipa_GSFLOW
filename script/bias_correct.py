def getannual(df, fld):
    startdate = df['date'].min()
    startyear = startdate.year
    enddate = df['date'].max()
    endyear = enddate.year
    ar_annual = np.array([], dtype=float)
    ar_yrs = np.array([], dtype=int)
    for yr in range(startyear, endyear+1):
        startdate = datetime(yr,1,1)
        enddate = datetime(yr,12,31)
        df_yr = df.loc[(df['date']>=startdate)&(df['date']<=enddate)]
        if fld.find('ppt')>0 or fld=='ppt':
            ar_annual = np.append(ar_annual, df_yr[fld].sum())
        else:
            ar_annual = np.append(ar_annual, df_yr[fld].mean())
        ar_yrs = np.append(ar_yrs, yr)
    return ar_yrs, ar_annual

import numpy as np
import pandas as pd
from datetime import datetime
from bias_correction import BiasCorrection, XBiasCorrection
import xarray as xr
from matplotlib import pyplot as plt


modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
paramlist = ['tmn', 'tmx', 'ppt']
scenlist = ['rcp45', 'rcp85']

######################################
#modindex = 3
######################################
# read in the historial model data
df_mod_hist = pd.read_csv('gcm_hist_table.csv', header=0)
df_mod_hist['date'] = df_mod_hist['date'].astype('datetime64')
# read in observed data
df_obs = pd.read_csv('data_file_2014.csv', header=0)
df_obs['date'] = pd.to_datetime(df_obs[['year', 'month', 'day']])
# merge with the model data
df_mod_obs = df_obs.merge(df_mod_hist, how='inner', left_on='date', right_on='date')
# get the date range of the common data
startdate = df_mod_obs['date'].min()
enddate = df_mod_obs['date'].max()
##########################################
paramindex = 1
##########################################
# pull out the observed data
obs_data = df_mod_obs[paramlist[paramindex]].values
obs_data_s = obs_data[:, np.newaxis, np.newaxis]
# read in the future data
df_sce = pd.read_csv('gcm_table.csv', header=0)
df_sce['date'] = pd.to_datetime(df_sce['date'])
startdatesce = df_sce['date'].min()
strt_sce = datetime.isoformat(startdatesce)[:10]
enddatesce = df_sce['date'].max()
enddt_sce = datetime.isoformat(enddatesce)[:10]
#rowcount = df_sce['date'].count()
for modindex in range(4):
    fld = '{}_hst_{}'.format(modlist[modindex], paramlist[paramindex])

    # pull out the model historical data
    model_data = df_mod_obs[fld].values
    model_data_s = model_data[:, np.newaxis, np.newaxis]

    fld = '{}_{}_{}'.format(modlist[modindex], scenlist[0], paramlist[paramindex])
    sce_data = df_sce[fld].values
    sce_data_s = sce_data[:, np.newaxis, np.newaxis]
    lat = range(1)
    lon = range(1)
    # pull out precipitation data
    strt = datetime.isoformat(startdate)[:10]
    enddt = datetime.isoformat(enddate)[:10]

    xobs_data = xr.DataArray(obs_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
    xmodel_data = xr.DataArray(model_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
    xmod_fix = xr.DataArray(model_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])

    # build the BiasCorrection object with just the historical period and correct the GCM historical data
    #ds = xr.Dataset({'obs_data':xobs_data, 'model_data':xmodel_data, 'sce_data': xsce_data})
    #bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
    bc = XBiasCorrection(xobs_data, xmodel_data, xmod_fix)
    if paramindex==2:
        df2 = bc.correct(method='basic_quantile')
    else:
        #df2 = bc.correct(method='gamma_mapping')
        df2 = bc.correct(method='basic_quantile')
        #df2 = bc.correct(method='modified_quantile')
    correct_values = df2.values[0][0]
    #df_sce_out[fld] = correct_values
    df = pd.DataFrame()
    df['date'] = df_mod_obs['date']
    df[fld] = model_data
    df[fld + '_cor'] = correct_values
    df[paramlist[paramindex]] = df_mod_obs[paramlist[paramindex]]
    ar_yr, ar_uncor = getannual(df, fld)
    ar_yr, ar_corr = getannual(df, fld + '_cor')
    ar_yr, ar_obs = getannual(df, paramlist[paramindex])

    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
    ax.plot(df_mod_obs['date'], np.cumsum(model_data), 'b-', lw=0.5, label='uncorrected')
    #plt.axhline(np.mean(ar_uncor), color='blue', lw = 0.5, ls='--', label='mean uncorrected = {:2.1f}'.format(np.mean(ar_uncor)))
    ax.plot(df_mod_obs['date'], np.cumsum(correct_values), 'g-', lw=0.5, label='corrected')
    #plt.axhline(np.mean(ar_corr), color='green', lw = 0.5, ls='--', label='mean corrected = {:2.1f}'.format(np.mean(ar_corr)))
    ax.plot(df_mod_obs['date'], np.cumsum(obs_data), color='skyblue', lw=0.5, label='observed')
    #plt.axhline(np.mean(ar_obs), color='skyblue', lw = 0.5, ls='--', label='mean observed = {:2.1f}'.format(np.mean(ar_obs)))
    #diff = ar_uncor-ar_corr
    #ax.plot(ar_yr, ar_uncor-ar_corr, 'g--', lw=0.5, label='uncorrected - corrected')
    #plt.axhline(0, lw=0.9, color = 'gray')
    #plt.axhline(np.mean(ar_uncor-ar_corr), lw=0.5, color='green', ls='--', label='mean adjust = {:2.1f} inches'.format(np.mean(ar_uncor-ar_corr)))
    ax.set_xlabel('date', fontsize=9)
    ax.set_title('{} annual {} bias correction'.format(modlist[modindex], paramlist[paramindex]))
    #ax.set_xlim(np.min(ar_yr), np.max(ar_yr))
    if paramlist[paramindex] == 'ppt':
        ax.set_ylabel('total annual precip in inches', fontsize=9)
        #ax.set_ylim(0,40)
    else:
        ax.set_ylabel('mean annual {} in deg C'.format(paramlist[paramindex]), fontsize=9)
        #if paramlist[paramindex]=='tmx':
        #    ax.set_ylim(22,30)
        #else:
            #ax.set_ylim(6,12)
    ax.legend()
    plt.savefig('{}_{}_corr_cum.png'.format(modlist[modindex], paramlist[paramindex]))
    plt.show()

#bc = BiasCorrection()
#fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
#ax.plot(df_mod_hist['date'].values, model_data, 'g-', lw=0.5, label='uncorrected')
#ax.set_title('{} historical precipitation'.format(modlist[modindex]))
#ax.plot(df_mod_hist['date'].values, correct_values-model_data, 'b-', lw=0.5, label='corrected - uncorrected')
#ax.legend()
#plt.savefig('{}_ppt_corr_adjust.png'.format(modlist[modindex]))
#plt.show()
print('done.')


