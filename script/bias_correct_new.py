def quantile_correction(obs_data, mod_data, ar_sce, par, cutoff):
    mod_data = mod_data[~np.isnan(mod_data)]
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
        cdf = ECDF(mod_data[~np.isnan(mod_data)])
        #cdf = ECDF(mod_data) # not used because of nan values being used in correction
        p = cdf(ar_sce) * 100
        cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
        correct = ar_sce + cor
        if par=='ppt':
            correct = np.where(correct<0., 0., correct)
    return correct

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib import pyplot as plt

##################################################################################
# this script performs bias correction on 8 GCM data sets of precipitation
# and plots the CDF and comparisons of cumulative precipitation
#
#  author: Derek Ryter
#
##################################################################################
saveplots = True
modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
par = 'ppt'
paramlist = ['ppt', 'tmn', 'tmx']
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
# load a DataFrame with the corrected results to save later
df_cor_hist = pd.DataFrame()
ar_h_date = df_mod_hist['date']
df_cor_hist['date'] = df_mod_hist['date']
#########################################################################################
# perform the bias correction on the historical period and plot the cdf for each model
# each model has 2 GHG scenarios: RCP45 and RCP85
#########################################################################################
# historical
i = 0
#fig_h, ax_h = plt.subplots(figsize=(10, 7), tight_layout=True)
#ax_h.plot(ar_h_date, np.cumsum(ar_obs), color='blue', lw=1.0, label='observed ppt')
#for mod in modlist:
#    fld_hst = '{}_hst_{}'.format(mod, par)
#    ar_model = df_mod_obs[fld_hst].values
    # perform bias correction on the model data and add the result to the DataFrame
#    cor_hist = quantile_correction(obs_data=ar_obs, mod_data=ar_model, ar_sce=ar_model, par=par, cutoff=0.)
#    df_cor_hist[fld_hst] = cor_hist
    # plot results
    #ax_h.plot(ar_h_date, np.cumsum(cor_hist), lw=0.6, label=mod)
# save corrected precip data
#df_cor_hist.to_csv('gcm_hist_table_cor.csv', index=False)
# plot historical cumulative precip
#ax_h.set_title('Bias-corrected and observed cumulative precipitation, historical model period')
#ax_h.set_xlim(df_mod_obs['date'].min(), df_mod_obs['date'].max())
#ax_h.set_ylim(0,800)
#ax_h.yaxis.set_minor_locator(MultipleLocator(25))
#ax_h.set_ylabel('cumulative precipitation in inches', fontsize=12)
#ax_h.set_xlabel('year', fontsize=12)
#ax_h.xaxis.set_tick_params(which='both', direction='in', labelsize=10)
#ax_h.yaxis.set_tick_params(which='both', direction='in', labelsize=10)
#ax_h.legend(fancybox=False, edgecolor='black', framealpha=1.)
#if saveplots:
#    plt.savefig('./plots/cumulative_ppt_hist.png')
#plt.show()

########################################################
# future
colorlist = ['forestgreen', 'seagreen', 'limegreen', 'chartreuse', 'firebrick', 'red', 'chocolate', 'orange']
#fig_s, ax_s = plt.subplots(figsize=(10, 7), tight_layout=True)
for scen in scenlist:
    for mod in modlist:
        for par in paramlist:
            # historic period arrays that won't change
            ar_obs = df_mod_obs[par]
            # get the future data
            fld = '{}_{}_{}'.format(mod, scen, par)
            ar_sce = df_sce[fld].values
            fld_hst = '{}_hst_{}'.format(mod, par)
            ar_model = df_mod_obs[fld_hst].values
            # bias correct the future model and add it to the DataFrame
            cor_sce = quantile_correction(obs_data=ar_obs, mod_data=ar_model, ar_sce=ar_sce, par=par, cutoff=0.)
            df_cor_sce[fld] = cor_sce
            # plot the results
            #ax_s.plot(ar_sce_date, np.cumsum(cor_sce), color=colorlist[i], lw=0.6, label='{} {}'.format(mod, scen))
            #i += 1

df_cor_sce.to_csv('gcm_table_corrected.csv', index=False)
# plot future cumulative precip
#ax_s.set_title('Bias-corrected cumulative precipitation, future model period')
#ax_s.set_ylabel('cumulative precipitation in inches', fontsize=12)
#ax_s.set_xlabel('year', fontsize=12)
#ax_s.set_xlim(df_sce['date'].min(), df_sce['date'].max())
#ax_s.xaxis.set_tick_params(which='both', direction='in', labelsize=10)
#ax_s.yaxis.set_tick_params(which='both', direction='in', labelsize=10)
#ax_s.set_ylim(0,1600)
#ax_s.yaxis.set_minor_locator(MultipleLocator(50))
#ax_s.legend(fancybox=False, edgecolor='black', framealpha=1.)
if saveplots:
    plt.savefig('./plots/cumulative_ppt_future.png')
plt.show()

############################################## plot the cdfs
cdf_obs = ECDF(ar_obs[ar_obs>0])
linewt = 0.9
for mod in modlist:
    for scen in scenlist:
        fld = '{}_{}_{}'.format(mod, scen, par)
        fld_hst = '{}_hst_{}'.format(mod, par)
        fig_c, ax_c = plt.subplots(figsize=(7, 6), tight_layout=True)
        # plot cdf for obs data
        ax_c.plot(cdf_obs.x, cdf_obs.y, color='blue', lw=0.9, label='observed')
        # plot cdfs for historical model
        cdf_mod_hist_unc = ECDF(df_mod_hist[fld_hst][df_mod_hist[fld_hst]>0])
        ax_c.plot(cdf_mod_hist_unc.x, cdf_mod_hist_unc.y, color='orange', lw=linewt, label='historical model uncorrected')
        cdf_mod_sce_unc = ECDF(df_sce[fld][df_sce[fld]>0])
        ax_c.plot(cdf_mod_sce_unc.x, cdf_mod_sce_unc.y, color='limegreen', lw=linewt, label='GCM model uncorrected')
        cdf_mod_hist_cor = ECDF(df_cor_hist[fld_hst][df_cor_hist[fld_hst]>0])
        ax_c.plot(cdf_mod_hist_cor.x, cdf_mod_hist_cor.y, color='red', lw=linewt, label='historical model corrected')
        # get the uncorrected cdf
        # get the corrected cdf
        cdf_mod_sce_cor = ECDF(df_cor_sce[fld][df_cor_sce[fld]>0])
        ax_c.plot(cdf_mod_sce_cor.x, cdf_mod_sce_cor.y, color='green', lw=linewt, label='GCM model corrected')
        ax_c.set_title('Empirical CDFs for model {} scenario {}'.format(mod, scen))
        ax_c.set_xlim(0,3.0)
        ax_c.xaxis.set_tick_params(which='both', direction='in', labelsize=10)
        ax_c.yaxis.set_tick_params(which='both', direction='in', labelsize=10)
        ax_c.set_ylim(0,1.1)
        ax_c.legend(loc='lower right', fancybox=False, edgecolor='black', framealpha=1.)
        if saveplots:
            plt.savefig('./plots/cdf_{}_{}.png'.format(mod, scen))
        plt.show()