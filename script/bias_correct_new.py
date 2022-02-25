def quantile_correction(obs_data, mod_data, ar_sce, par, cutoff):
    #mdata = mod_data[~np.isnan(mod_data)]
    # This routine sets precip values less than cutoff to a random value
    if cutoff > 0. and par=='ppt':
        randrain = cutoff * np.random.randn(len(mod_data))
        randrain = randrain + abs(np.min(randrain))
        mdata = np.where(mod_data<cutoff, randrain, mod_data)
        cdf = ECDF(mdata)
        p = cdf(ar_sce) * 100
        cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mdata]])
        correct = ar_sce + cor

    else:
        cdf = ECDF(mod_data)
        #cdf = ECDF(mod_data) # not used because of nan values being used in correction
        p = cdf(ar_sce) * 100
        cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
        correct = ar_sce + cor
    correct = np.where(correct<0., 0., correct)
    return correct

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

##################################################################################
# this script performs bias correction on 8 GCM data sets of precipitation
# and plots the CDF and comparisons of cumulative precipitation
#
#  author: Derek Ryter
#
##################################################################################
modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
wet_cutoff_list = [0.02, 0.03, 0.03, 0.02, 0.01]
par = 'ppt'
#paramlist = ['tmx', 'tmn', 'ppt']
paramlist = ['ppt']
scenlist = ['rcp45', 'rcp85']
######################################################################

## load the data
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
ar_sce_date = df_sce['date'].values
#df_sce = df_sce.iloc[:4000]
df_cor_sce = pd.DataFrame()
df_cor_sce['date'] = df_sce['date']
startdatesce = df_sce['date'].min()
strt_sce = datetime.isoformat(startdatesce)[:10]
enddatesce = df_sce['date'].max()
enddt_sce = datetime.isoformat(enddatesce)[:10]
# historic period arrays that won't change
ar_obs = df_mod_obs[par]
cdf_obs = ECDF(ar_obs)
# load a DataFrame with the corrected results to save later
df_cor_hist = pd.DataFrame()
ar_h_date = df_mod_hist['date']
df_cor_hist['date'] = df_mod_hist['date']
########################################################
# perform the bias correction on the historical period and plot the cdf for each model
# each model has 2 GHG scenarios: RCP45 and RCP85
fig_h, ax_h = plt.subplots(figsize=(10, 7), tight_layout=True)
ax_h.plot(ar_h_date, np.cumsum(ar_obs), lw=1.0, label='observed ppt')
fig_s, ax_s = plt.subplots(figsize=(10, 7), tight_layout=True)
for mod in modlist:
    for scen in scenlist:
        fld_hst = '{}_hst_{}'.format(mod, par)
        ar_model = df_mod_obs[fld_hst].values
        # perform bias correction on the model data and add the result to the DataFrame
        cor_hist = quantile_correction(obs_data=ar_obs, mod_data=ar_model, ar_sce=ar_model, par=par, cutoff=0.)
        df_cor_hist[fld_hst] = cor_hist
        # plot results
        ax_h.plot(ar_h_date, np.cumsum(cor_hist), label=fld_hst)

        # get the future data
        fld = '{}_{}_{}'.format(mod, scen, par)
        ar_sce = df_sce[fld].values
        # bias correct the future model and add it to the DataFrame
        cor_sce = quantile_correction(obs_data=ar_obs, mod_data=ar_model, ar_sce=ar_sce, par=par, cutoff=0.)

        # plot the results
        ax_s.plot(ar_sce_date, np.cumsum(cor_sce), label=fld)

df_cor_sce.to_csv('gcm_table_corrected.csv', index=False)
df_cor_hist.to_csv('gcm_hist_table_cor.csv', index=False)
ax_h.set_title('Cumulative precipitation, historical model period')
ax_h.set_ylim(0,800)
ax_h.set_ylabel('cumulative precipitation in inches', fontsize=9)
ax_h.legend()
ax_s.set_title('Cumulative precipitation, future model period')
ax_s.set_ylabel('cumulative precipitation in inches', fontsize=9)
ax_h.set_ylim(0,1600)
ax_s.legend()
plt.show()

for mod in modlist:
    for scen in scenlist:
        fld = '{}_{}_{}'.format(mod, scen, par)
        fld_hst = '{}_hst_{}'.format(mod, par)
        fig_c, ax_c = plt.subplots(figsize=(6, 6), tight_layout=True)
        # plot cdf for obs data
        ax_c.plot(cdf_obs.x, cdf_obs.y, color='blue', lw=0.9, label='observed')
        # plot cdfs for historical model
        cdf_mod_hist_cor = ECDF(df_cor_hist[fld_hst])
        ax_c.plot(cdf_mod_hist_cor.x, cdf_mod_hist_cor.y, color='orange', lw=0.7, label='hist model corrected')
        cdf_mod_hist_unc = ECDF(df_mod_hist[fld_hst])
        ax_c.plot(cdf_mod_hist_unc.x, cdf_mod_hist_unc.y, color='red', lw=0.7, label='hist model uncorrected')
        # get the uncorrected cdf
        cdf_mod_sce_unc = ECDF(df_sce[fld])
        ax_c.plot(cdf_mod_sce_unc.x, cdf_mod_sce_unc.y, color='lime', lw=0.7, label='GCM model uncorrected')
        # get the corrected cdf
        cdf_mod_sce_cor = ECDF(df_cor_sce[fld])
        ax_c.plot(cdf_mod_sce_cor.x, cdf_mod_sce_cor.y, color='green', lw=0.7, label='GCM model corrected')
        plt.show()