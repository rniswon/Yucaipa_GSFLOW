def getmonth(df, fld, param):
    df['mo'] = pd.DatetimeIndex(df['date']).month
    ar_mm = np.array([],dtype=float)
    startdate = df['date'].min()
    enddate = df['date'].max()
    nyr = enddate.year - startdate.year
    molist = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for m in molist:
        df_m = df.loc[df['mo']==m]
        if param=='ppt':
            ar_mm = np.append(ar_mm, df_m[fld].sum())
        else:
            ar_mm = np.append(ar_mm, df_m[fld].mean())
    if param=='ppt':
        ar_mm = ar_mm/nyr
    return ar_mm


def local_quantile_correction(obs_data, mod_data, sce_data, modified=True):
    mdata = mod_data[~np.isnan(mod_data)]
    #dmask = np.ma.masked_where(mod_data<0.05, mod_data)
    randrain = 0.05 * np.random.randn(len(mod_data))
    mdata = np.where(mdata<0.05, randrain, mdata)
    cdf = ECDF(mdata)
    #cdf = ECDF(mod_data) # not used because of nan values being used in correction
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    if modified:
        mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
        iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))
        f = np.true_divide(iqr_obs_data, iqr_mod_data)
        cor = g * mid + f * (cor - mid)
        return sce_data + cor
    else:
        return sce_data + cor


import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import pandas as pd
from datetime import datetime, timedelta
from bias_correction import BiasCorrection, XBiasCorrection
import xarray as xr
from matplotlib import pyplot as plt

modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
paramlist = ['tmx', 'tmn', 'ppt']
scenlist = ['rcp45', 'rcp85']

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
# read in the future data
df_sce = pd.read_csv('gcm_table.csv', header=0)
df_sce['date'] = pd.to_datetime(df_sce['date'])
#df_sce = df_sce.iloc[:4000]
df_sce_out = pd.DataFrame()
df_sce_out['date'] = df_sce['date']
startdatesce = df_sce['date'].min()
strt_sce = datetime.isoformat(startdatesce)[:10]
enddatesce = df_sce['date'].max()
enddt_sce = datetime.isoformat(enddatesce)[:10]

for modindex in range(4):
    for scenindex in range(2):
        for paramindex in range(3): # only run ppt
            # pull out the observed data
            obs_data = df_mod_obs[paramlist[paramindex]].values
            #obs_data_s = obs_data[:, np.newaxis, np.newaxis]
            obs_month = getmonth(df_mod_obs, paramlist[paramindex], paramlist[paramindex])
            # get the historical data for this model and parameter
            fld = '{}_hst_{}'.format(modlist[modindex], paramlist[paramindex])
            model_data = df_mod_obs[fld].values
            #model_data_s = model_data[:, np.newaxis, np.newaxis]
            model_month = getmonth(df_mod_obs, fld, paramlist[paramindex])

            # now get the model future data
            fld = '{}_{}_{}'.format(modlist[modindex], scenlist[scenindex], paramlist[paramindex])
            sce_data = df_sce[fld].values
            sce_month = getmonth(df_sce, fld, paramlist[paramindex])
            #sce_data_s = sce_data[:, np.newaxis, np.newaxis]
            #lat = range(1)
            #lon = range(1)
            #
            #strt = datetime.isoformat(startdate)[:10]
            #enddt = datetime.isoformat(enddate)[:10]
            # all data has been loaded, make xarrays for the bias-correct library
            #xobs_data = xr.DataArray(obs_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
            #xmodel_data = xr.DataArray(model_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
            #xsce_data = xr.DataArray(sce_data_s, dims=['time','lat','lon'], coords=[pd.date_range(strt_sce, enddt_sce, freq='D'), lat, lon])

            # build the BiasCorrection object with just the historical period and correct the GCM historical data
            #ds = xr.Dataset({'obs_data':xobs_data, 'model_data':xmodel_data, 'sce_data': xsce_data})
            correct_values = local_quantile_correction(obs_data, model_data, sce_data, False)
            correct_values = np.where(correct_values<0.05,0.,correct_values)
            #correct_values = quantile_correction(ds['obs_data'], ds['model_data'], ds['sce_data'], False)
            #xbc = XBiasCorrection(reference, model, data_to_be_corrected)
            #bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
            #if paramindex == 2:
            #    df2 = bc.correct(method='basic_quantile_ppt')
            #else:
            #df2 = bc.correct(method='basic_quantile')
            #correct_values = df2.values[0][0]
            #correct_values[correct_values<0]=0.
            #correct_values = correct_values[~np.isnan(correct_values)]
            df_sce['corrected'] = correct_values
            cor_month = getmonth(df_sce, 'corrected', paramlist[paramindex])
            # plot months
            molist = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            monames = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
            fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
            ax.plot(monames, obs_month, color='skyblue', lw=0.7, label='observed')
            ax.plot(monames, model_month, color='darkblue', lw=0.7, label='model')
            ax.plot(monames, sce_month, color='limegreen', lw=0.7, label='future uncor')
            ax.plot(monames, cor_month, color='forestgreen', lw=0.7, label='future corrected')
            ax.legend()
            if paramlist[paramindex]=='ppt':
                ax.set_title('monthly mean total precipitation {}_{}'.format(modlist[modindex],
                                                                                scenlist[scenindex]))
                ax.set_ylabel('precip in inches', fontsize=9)
            else:
                ax.set_title('monthly mean {0:}, {1:}_{2:}'.format(paramlist[paramindex],
                                                                        modlist[modindex], scenlist[scenindex]))
                ax.set_ylabel('mean monthly {} in deg C'.format(paramlist[paramindex]), fontsize=9)
            #plt.bar(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], cor_month)
            plt.savefig('{}_{}_{}_monthly.png'.format(modlist[modindex], scenlist[scenindex], paramlist[paramindex]))
            plt.show()

            if paramlist[paramindex]=='ppt':
                fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
                ax.set_title('{}_{} {} bias correction'.format(modlist[modindex], scenlist[scenindex], paramlist[paramindex]))
                ax.plot(df_sce['date'], np.cumsum(sce_data), 'b-', lw=0.5, label='uncorrected')
                ax.plot(df_sce['date'], np.cumsum(correct_values), 'g--', lw=0.5, label='corrected')
                #print(np.sum(correct_values - sce_data))
                ax.set_xlabel('date', fontsize=9)
                if paramlist[paramindex] == 'ppt':
                    ax.set_ylabel('cumulative precip in mm', fontsize=9)
                    ax.set_ylim(0,np.max(np.cumsum(correct_values))+1000)
                else:
                    ax.set_ylabel('mean annual {} in deg C'.format(paramlist[paramindex]), fontsize=9)
                    #if paramlist[paramindex]=='tmx':
                    #ax.set_ylim(np.min(correct_values),np.max(correct_values))
                    #else:
                    #    ax.set_ylim(6,12)
                ax.legend()
                plt.savefig('{}_{}_{}_corr_cum_future.png'.format(modlist[modindex], scenlist[scenindex], paramlist[paramindex]))
                plt.show()

print('done.')


