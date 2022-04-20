def plotgsflow(btype):
    j = 0
    ar_yr = np.arange(2015, 2100)
    for cat in catlist:   # loop through the categories and plot each model for them
        print('plotting gsflow budget, category {}, {}'.format(cat, btype))
        meanflow = df_gs_mean[cat].mean()   # this is the mean annual flow from the original model
        c = 0
        fig, ax = plt.subplots(figsize=(9,5), tight_layout=True)
        for scen in scenlist:
            if scen=='rcp85':
                lstyle = '--'
            else:
                lstyle = '-'
            for mod in modlist:
                fname = '../data_files/gsflow_{}_{}.csv'.format(mod, scen)
                if os.path.isfile(fname):   # not all the budget files might be done, so plot it if it is
                    df_gsflow = pd.read_csv(fname)
                    df_gsflow['Date'] = pd.to_datetime(df_gsflow['Date'])
                    if btype=='cum':
                        ar_date = df_gsflow['Date'].values
                        ar_data = df_gsflow[cat].values
                        ax.plot(ar_date, np.cumsum(ar_data), color=colorlist[c], linestyle = lstyle, lw = 0.7,
                                label='{} {}'.format(mod, scen))
                        ax.set_xlim(np.min(ar_date), np.max(ar_date))
                    else:
                        ar_data = np.array([], dtype=float)
                        for yr in ar_yr:
                            sdate = datetime(yr, 1, 1)
                            edate = datetime(yr+1, 1, 1)
                            df = df_gsflow.loc[(df_gsflow['Date']>=sdate) & (df_gsflow['Date']<edate)]
                            ar_data = np.append(ar_data, df[cat].mean())
                        ar_data = (ar_data - meanflow) * cft2aft
                        n2 = int(n/2)
                        # now run the moving average filter
                        ar_mean_yr_filter = np.array([], dtype=float)
                        ar_yr_flt = np.array([], dtype=int)
                        for i in range(n2, len(ar_data)-n2):
                            ar_yr_flt = np.append(ar_yr_flt, 2014+i)
                            s = i - n2
                            e = i + n2
                            ar_mean_yr_filter = np.append(ar_mean_yr_filter, np.mean(ar_data[s:e]))
                            #ar_yr_sub = np.append(ar_yr_sub, ar_yrs[i])
                        ax.plot(ar_yr_flt, ar_mean_yr_filter, color=colorlist[c], linestyle = lstyle, lw = 0.7,
                                label='{} {} ({:4.3f})'.format(mod, scen, np.mean(ar_mean_yr_filter)))
                    c+=1
                    del df_gsflow

        if btype=='cum':
            ax.set_title('Cumulative {} for the Yucaipa GSFLOW model'.format(cat))
            ax.set_ylabel('cumulative flow in acre feet', fontsize=9)
            outname = '../report_figs/yuc_{}_{}.png'.format(cat.replace(' ', '_'), btype)
            ax.legend(loc='best', title='Model used', title_fontsize=8, fontsize=7, ncol=2, fancybox=False, edgecolor='black')
        else:
            ax.set_xlim(np.min(ar_yr_flt), np.max(ar_yr_flt))
            plt.axhline(0, lw=0.6)
            outname = '../report_figs/yuc_{}_depart_{}.png'.format(cat.replace(' ', '_'), btype)
            ax.set_ylabel('flow in acre feet', fontsize=9)
            ax.set_title('{} departure from mean for the Yucaipa GW Basin full model'.format(cat))
            ax.legend(loc='best', title='Model used (mean deviation)', title_fontsize=8, fontsize=7, ncol=2, fancybox=False, edgecolor='black')
        ax.set_xlabel('year')
        ax.xaxis.set_tick_params(direction='in', labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(direction='in', labelsize=7)
        plt.savefig(outname)
        plt.show()
        j += 1


import os
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt

# read data files and look at the mean annual precipitation and temperature

#########################
# gsflow budget categories to plot
cats = ['HORT+DUNN_IN', 'net_stream', 'net_storage', 'recharge', 'UZFET_OUT', 'GWET_OUT', 'pumpage']
lablist = ['HORT+DUNN SR to streams', 'stream leakage', 'change in storage', 'recharge', 'UZF ET', 'GW ET', 'pumpage']
catlist = ['HortSroff2Stream_Q','DunnSroff2Stream_Q','Interflow2Stream_Q','Precip_Q',
           'UnsatET_Q','SatET_Q','SoilDrainage2Unsat_Q','Sat2Grav_Q','RechargeUnsat2Sat_Q']
#########################
# models to plot
modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
#modlist = ['CanESM2', 'CNRMCM5', 'MIROC5']
scenlist = ['rcp45', 'rcp85']
#scenlist = ['rcp45', 'rcp85']
btype = 'inc'       # 'cum' for cumulative budget; any other string for incremental
nsp = 816
n = 30
printStor = False
cft2aft = 0.00002295684
savefig = True
################################################################### simulation
# load model
newlin = '\n'


