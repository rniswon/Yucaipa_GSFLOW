def movingavg(ar_yrs, df_bud, cat):
    # get the moving average smoothed data
    ar_mean_yr = np.array([], dtype=float)
    ar_yr_sub = np.array([], dtype=float)
    basemean = df_base[cat].mean()
    for yr in ar_yrs:   # select each year and get the mean, then assign it to an array
        sdate = datetime(yr, 1, 1)
        edate = datetime(yr+1, 1, 1)
        df = df_bud.loc[(df_bud['date']>=sdate) & (df_bud['date']<edate)]
        ar_vals = df[cat].values - basemean   # convert to deviation from mean of base model run
        ar_mean_yr = np.append(ar_mean_yr, np.mean(ar_vals) * cft2aft)
        # the mean flow in acre-feet is in the array. Now process it with a moving average.
    n2 = int(n/2)
    # now run the moving average filter
    ar_mean_yr_filter = np.array([], dtype=float)
    for i in range(n2, len(ar_mean_yr)-n2):
        s = i - n2
        e = i + n2
        ar_mean_yr_filter = np.append(ar_mean_yr_filter, np.mean(ar_mean_yr[s:e]))
        ar_yr_sub = np.append(ar_yr_sub, ar_yrs[i])
    return ar_yr_sub, ar_mean_yr_filter

def getdf_bud(fname, startingsp):
    df_bud = pd.read_csv(fname, names=cols, skiprows=1, sep='[\s,]{2,20}')    # use this delimiter because of spaces in the header line
    df_bud = df_bud[(df_bud['ZONE']==2) & (df_bud['PERIOD']>startingsp)]
    df_bud['STREAMLEAKAGE_IN'] = pd.to_numeric(df_bud['STREAMLEAKAGE_IN'])
    df_bud['net_stream']=df_bud['STREAMLEAKAGE_IN']-df_bud['STREAMLEAKAGE_OUT']
    df_bud['pumpage']=df_bud['MNW2_IN']-df_bud['MNW2_OUT']
    df_bud['net_storage']=df_bud['STORAGE_OUT']-df_bud['STORAGE_IN']
    ar_date = np.array([],dtype='datetime64')
    for index, row in df_bud.iterrows():
        thisdate = zerodate + timedelta(days=row['TOTIM'])
        ar_date = np.append(ar_date, thisdate)
    df_bud['date'] = ar_date
    return df_bud

def plotcats(btype):
    print('plotting budget...')
    # plot all models on the same plot
    ar_yrs = np.arange(2011, 2100)
    startingsp = 769
    j = 0
    for cat in cats:   # loop through the categories and plot each model for them
        print('working on {}'.format(cat))
        c = 0
        fig, ax = plt.subplots(figsize=(9,5), tight_layout=True)
        ymn = 1000
        ymx = -1000
        for scen in scenlist:
            if scen=='rcp85':
                lstyle = '--'
            else:
                lstyle = '-'
            df_mean = pd.DataFrame()
            for mod in modlist:
                fname = '../data_files/yuczone_yucaipa_{}_{}.csv.2.csv'.format(mod, scen)
                if os.path.isfile(fname):   # not all the budget files might be done, so plot it if it is
                    df_bud=getdf_bud(fname, startingsp)
                    if btype == 'cum':
                        ar_data = df_bud[cat]
                        if btype=='cum':
                            df_mean[mod] = np.cumsum(ar_data)
                        else:
                            df_mean[mod] = ar_data
                        ax.plot(df_bud['date'].values, np.cumsum(ar_data), color=colorlist[c], linestyle = lstyle, lw = 0.7,
                                label='{} {}'.format(mod, scen))
                        ax.set_ylabel('cumulative flow in acre feet', fontsize=9)
                        ax.set_title('Cumulative {} for the Yucaipa GW Basin'.format(lablist[j].title()))
                        ax.set_xlim(np.min(df_bud['date'].values), np.max(df_bud['date'].values))
                        outname = '../report_figs/yuc_gsflow_{}_{}.png'.format(cat.replace(' ', '_'), btype)
                        titlestr = 'Cumulative {} for the Yucaipa GW Basin'.format(lablist[j].title())
                    else:
                        n2 = n/2
                        ar_yr_sub, ar_mean_yr_filter = movingavg(ar_yrs, df_bud, cat)
                        ax.plot(ar_yr_sub, ar_mean_yr_filter, color=colorlist[c], lw = 0.8, linestyle = lstyle,
                                label='{} {}'.format(mod, scen))
                        if ymn > np.min(ar_mean_yr_filter):
                            ymn = np.min(ar_mean_yr_filter)
                        if ymx < np.max(ar_mean_yr_filter):
                            ymx = np.max(ar_mean_yr_filter)
                        ax.set_xlim(2015+n2, 2099-n2)
                        ymn -= abs(ymn)*0.1
                        ymx += ymx*0.1
                        #ax.set_ylim(ymn, ymx)
                        ax.set_ylabel('flow in acre feet {} year moving average'.format(n), fontsize=9)
                        plt.axhline(0, lw=0.6)
                        outname = '../report_figs/yuc_gsflow_depart_{}_{}.png'.format(cat.replace(' ', '_'), btype)
                        titlestr = '{} deviation from mean for the Yucaipa GW Basin'.format(lablist[j].title())
                    c+=1
                    del df_bud
        ax.set_title(titlestr)
        ax.set_xlabel('year')
        ax.xaxis.set_tick_params(direction='in', labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(direction='in', labelsize=7)
        ax.legend(ncol=2, fontsize=7)
        if savefig:
            plt.savefig(outname)
        else:
            plt.show()
        j += 1

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

################################################################### folders
# This script extracts budgets from list files of GCM future models and
# saves the budget tables in csv files. If they are already in csv files
# then it can read the budget tables.
# The list files are 2-3 GB so it's good to create csv files and read them.
# This script then plots the budget time series.
################################################################### folders
# set up folders
thisFolder = os.getcwd()
OPJ = os.path.join
root = os.path.sep.join(thisFolder.split(os.path.sep)[:-2])
#modelFolder = os.path.join(root, 'GCM', 'models')
plotFolder = OPJ(root, 'yucaipa_git', 'Yucaipa_GSFLOW', 'report_figs')
################################################################### switches
cols = ['TOTIM','PERIOD','STEP','ZONE','STORAGE_IN','CONSTANTHEAD_IN','WELLS_IN','HEADDEPBOUNDS_IN','STREAMLEAKAGE_IN',
        'UZFET_IN','GWET_IN','UZFINFILTR_IN','recharge','SURFACELEAKAGE_IN','HORT+DUNN_IN','STORAGECHANGE_IN','MNW2_IN',
        'FromOtherZones_IN','TotalIN','STORAGE_OUT','CONSTANTHEAD_OUT','WELLS_OUT','HEADDEPBOUNDS_OUT',
        'STREAMLEAKAGE_OUT','UZFET_OUT','GWET_OUT','UZFINFILTR_OUT','UZFRECHARGE_OUT','SURFACELEAKAGE_OUT',
        'HORT+DUNN_OUT','STORAGECHANGE_OUT','MNW2_OUT','ToOtherZones_OUT','TotalOut','IN-OUT','PercentError',
        'FROMZONE1','FROMZONE2','TOZONE1','TOZONE2']
#########################
# budget categories to plot
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
#cats = ['UZF_RECHARGE_IN']
#colorlist = ['blue', 'green', 'orange', 'red', 'limegreen', 'purple', 'chocolate', 'goldenrod']
colorlist = ['blue', 'red', 'green', 'chocolate', 'blue', 'red', 'green', 'chocolate']
lslist = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
zerodate = datetime(1947, 1, 1)
# get base model budget
df_base = getdf_bud('../data_files/yuczone_base_model.2.csv', 1)
df_gs_base = pd.read_csv('../data_files/gsflow_base.csv', header=0)   # historical period model for stats
df_gs_base['Date'] = pd.to_datetime(df_gs_base['Date'])
ar_yrs = np.arange(1947, 2015)
df_gs_mean = pd.DataFrame(columns=['year'] + catlist)
for yr in ar_yrs:
    sdate = datetime(yr, 1, 1)
    edate = datetime(yr+1, 1, 1)
    df = df_gs_base.loc[(df_gs_base['Date']>=sdate) & (df_gs_base['Date']<edate)]
    dict = {'year':[yr]}
    for cat in catlist:
        dict[cat] = df[cat].mean()
    df_means = pd.DataFrame(dict)
    df_gs_mean = df_gs_mean.append(df_means)

plotcats('cum')
plotcats('inc')
plotgsflow('inc')
#plotgsflow('cum')

