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


def local_quantile_correction(obs_data, mod_data, sce_data, par, cutoff):
    #mdata = mod_data[~np.isnan(mod_data)]
    # This routine sets precip values less than cutoff to a random value
    if cutoff > 0. and par=='ppt':
        randrain = cutoff * np.random.randn(len(mod_data))
        randrain = randrain + abs(np.min(randrain))
        mdata = np.where(mod_data<cutoff, randrain, mod_data)
        cdf = ECDF(mdata)
        p = cdf(sce_data) * 100
        cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mdata]])
        correct = sce_data + cor
        correct = np.where(correct<0., 0., correct)

    else:
        cdf = ECDF(mod_data)
        #cdf = ECDF(mod_data) # not used because of nan values being used in correction
        p = cdf(sce_data) * 100
        cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
        correct = sce_data + cor
    return correct

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from bias_correction import BiasCorrection, XBiasCorrection
import xarray as xr


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
#df_sce = df_sce.iloc[:4000]
df_sce_out = pd.DataFrame()
df_sce_out['date'] = df_sce['date']
startdatesce = df_sce['date'].min()
strt_sce = datetime.isoformat(startdatesce)[:10]
enddatesce = df_sce['date'].max()
enddt_sce = datetime.isoformat(enddatesce)[:10]

################################################################
use_library = True
plot_cum = True
#plot
use_window = False
use_season = True
cutoff = 0.0
modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
wet_cutoff_list = [0.02, 0.03, 0.03, 0.02, 0.01]
#paramlist = ['tmx', 'tmn', 'ppt']
paramlist = ['ppt']
scenlist = ['rcp45', 'rcp85']
################################################################
i = 0
for mod in modlist:
    # get the wet season cutoff for this model
    wet_cutoff = wet_cutoff_list[i]
    i += 1
    for scen in scenlist:
        for par in paramlist:
            # pull out the observed data
            obs_data = df_mod_obs[par].values
            #obs_data_s = obs_data[:, np.newaxis, np.newaxis]
            obs_month = getmonth(df_mod_obs, par, par)
            # get the historical data for this model and parameter
            fld_hst = '{}_hst_{}'.format(mod, par)
            model_data = df_mod_obs[fld_hst].values
            #model_data_s = model_data[:, np.newaxis, np.newaxis]
            model_month = getmonth(df_mod_obs, fld_hst, par)
            # now get the model future data
            fld = '{}_{}_{}'.format(mod, scen, par)
            sce_data = df_sce[fld].values
            sce_month = getmonth(df_sce, fld, par)

            if use_library:
                lat = range(1)
                lon = range(1)
                tagline = 'base'
                titleline = 'original cor method'
                sce_data_s = sce_data[:, np.newaxis, np.newaxis]
                # pull out precipitation data
                strt = datetime.isoformat(startdate)[:10]
                enddt = datetime.isoformat(enddate)[:10]

                xobs_data = xr.DataArray(obs_data[:, np.newaxis, np.newaxis], dims=['time','lat','lon'],
                                         coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
                xmodel_data = xr.DataArray(model_data[:, np.newaxis, np.newaxis], dims=['time','lat','lon'],
                                           coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
                xmod_fix = xr.DataArray(model_data[:, np.newaxis, np.newaxis], dims=['time','lat','lon'], 
                                        coords=[pd.date_range(strt, enddt, freq='D'), lat, lon])
                xsce_data = xr.DataArray(sce_data[:, np.newaxis, np.newaxis], dims=['time','lat','lon'],
                                         coords=[pd.date_range(strt_sce, enddt_sce, freq='D'), lat, lon])
                # correct the historical data using
                ds = xr.Dataset({'obs_data':xobs_data, 'model_data':xmodel_data, 'sce_data': xmod_fix})
                bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
                df2 = bc.correct(method='basic_quantile')
                cor_hist = df2.values[0][0]
                cor_hist = cor_hist[~np.isnan(cor_hist)]

                # correct the future data (sce_data)
                ds = xr.Dataset({'obs_data':xobs_data, 'model_data':xmodel_data, 'sce_data': xsce_data})
                bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
                df2 = bc.correct(method='basic_quantile')

                correct_values = df2.values[0][0]
                correct_values = correct_values[~np.isnan(correct_values)]
                cdfdata = ECDF(model_data[model_data>0])
                cdfobs = ECDF(obs_data[obs_data>0])
                cdfcor = ECDF(correct_values[correct_values>0])
                cdfsce = ECDF(sce_data[sce_data>0])
                cdfhistcor = ECDF(cor_hist[cor_hist>0])
                fig, ax = plt.subplots()
                ax.plot(cdfobs.x, cdfobs.y, label='obs')
                ax.plot(cdfcor.x, cdfcor.y, label='corrected')
                ax.plot(cdfdata.x, cdfdata.y, label='model hist uncor')
                ax.plot(cdfhistcor.x, cdfhistcor.y, label='historical corrected')
                ax.plot(cdfsce.x, cdfsce.y, label='future uncorrected')
                ax.legend()
                ax.set_title(fld)
                plt.show()

            else:
                if use_window:
                    print('processing data using month windows')
                    cor_hist = local_quantile_correction(obs_data, model_data, model_data, par, cutoff)
                    dry_cutoff = 0.01
                    if use_season:
                        titleline = 'cutoff = {:4.2f} wet, {:4.2f} dry season'.format(wet_cutoff, dry_cutoff)
                        tagline = 'season'
                    else:
                        titleline = 'cutoff = {:4.2f} window'.format(cutoff)
                        tagline = 'cutoff_{:4.2f}_window'.format(cutoff)
                    correct_values = np.array([], dtype=float)
                    cor_hist = np.array([], dtype=float)
                    # loop through the years in the sce_data
                    # process the historical data and get the corrected
                    for yr in range(startdatesce.year, enddatesce.year+1):
                        # for each year, loop through the months and extract the data to correct
                        # also get the historical data to use. the historical data is the month to be
                        # corrected and the adjacent months
                        # select a month, correct it, and append the corrected array to correct_values
                        for mo in range(1, 13):
                            # get the sce data for this month
                            df_sc = df_sce.loc[(df_sce['month']==mo) & (df_sce['year']==yr)]
                            sce_mo = df_sc[fld]
                            model_m = df_mod_obs.loc[(df_mod_obs['month']==mo) & (df_mod_obs['year']=='year')]
                            # get the window data for model and observed
                            # this data will be in df_mod_obs
                            if use_season:
                                # use two periods: the dry season from April - December, and the rest the wet season
                                if mo <=3:
                                    df_pdf = df_mod_obs.loc[(df_mod_obs['month']<=3)]
                                    coff = wet_cutoff
                                else:
                                    df_pdf = df_mod_obs.loc[(df_mod_obs['month']>=4) & (df_mod_obs['month']<=11)]
                                    coff = dry_cutoff
                            else:
                                if mo == 1:
                                    df_pdf = df_mod_obs.loc[(df_mod_obs['month']==12) | (df_mod_obs['month']<=2)]
                                    #cutoff = 0.05
                                elif mo == 12:
                                    df_pdf = df_mod_obs.loc[(df_mod_obs['month']>=11) | (df_mod_obs['month']==1)]
                                else:
                                    df_pdf = df_mod_obs.loc[(df_mod_obs['month']>=mo-1) & (df_mod_obs['month']<=mo+1)]
                            model_d = df_pdf[fld_hst].values
                            obs_d = df_pdf[par].values
                            cor_h = local_quantile_correction(obs_d, model_d, model_m, par, coff)
                            cor_hist = np.append(cor_hist, cor_h)
                            correct_m = local_quantile_correction(obs_d, model_d, sce_mo, par, coff)
                            correct_values = np.append(correct_values, correct_m)
                else:
                    titleline = 'cutoff = {:4.3f}'.format(cutoff)
                    tagline = 'cutoff_{:4.3f}'.format(cutoff)
                    correct_values = local_quantile_correction(obs_data, model_data, sce_data, par, cutoff)
                    cor_hist = local_quantile_correction(obs_data, model_data, model_data, par, cutoff)

            df_mod_hist['cor_hist'] = cor_hist
            cor_hist_mo = getmonth(df_mod_hist,'cor_hist', par)

            df_sce['corrected'] = correct_values
            cor_month = getmonth(df_sce, 'corrected', par)
            df_sce_out[fld]=correct_values
            ################################################################################# plot months
            molist = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            monames = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
            fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
            ax.plot(monames, obs_month, color='blue', lw=0.8, label='observed')
            ax.plot(monames, model_month, color='orange', lw=0.8, label='historical')
            ax.plot(monames, cor_hist_mo, color='red', lw=0.9, label='corrected historical cutoff = {:4.3f}'.format(cutoff))
            ax.plot(monames, sce_month, color='limegreen', lw=0.8, label='future uncorrected')
            ax.plot(monames, cor_month, color='forestgreen', lw=0.8, label='future corrected')
            ax.legend()
            if par=='ppt':
                ax.set_title('Monthly mean total precipitation {}_{} {}'.format(mod,
                                                                                scen, titleline))
                ax.set_ylabel('precip in inches', fontsize=9)
                ax.set_ylim(0.,5.5)
            else:
                ax.set_title('Monthly mean {}, {}_{} {}'.format(par, mod,
                                                                scen, titleline))
                ax.set_ylabel('mean monthly {} in deg C'.format(par), fontsize=9)
            #plt.bar(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], cor_month)
            plt.savefig('./plots/{}_{}_{}_monthly_{}.png'.format(mod, scen, par,
                                                         tagline))
            plt.show()
            if plot_cum:
                if par=='ppt':
                    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
                    ax.set_title('{}_{} {} bias correction {}'.format(mod, scen,
                                                                      par, tagline))
                    ax.plot(df_sce['date'], np.cumsum(sce_data), 'b-', lw=0.5, label='uncorrected')
                    ax.plot(df_sce['date'], np.cumsum(correct_values), 'g--', lw=0.5, label='corrected')
                    #print(np.sum(correct_values - sce_data))
                    ax.set_xlabel('date', fontsize=9)
                    if par == 'ppt':
                        ax.set_ylabel('cumulative precip in mm', fontsize=9)
                        ax.set_ylim(0,np.max(np.cumsum(sce_data))+100)
                    else:
                        ax.set_ylabel('mean annual {} in deg C'.format(par), fontsize=9)
                        #if par=='tmx':
                        #ax.set_ylim(np.min(correct_values),np.max(correct_values))
                        #else:
                        #    ax.set_ylim(6,12)
                    ax.legend()
                    plt.savefig('./plots/{}_{}_{}_corr_cum_{}.png'.format(mod, scen,
                                                                  par, tagline))
                    plt.show()
df_sce_out.to_csv('gcm_table_correct.csv', index=False)
print('done.')


