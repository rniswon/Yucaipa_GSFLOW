import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import os
import seaborn as sns
from tqdm import tqdm

def getdf_bud(fname, startingsp):
    df_bud = pd.read_csv(fname, names=cols, skiprows=1, sep='[\s,]{2,20}')    # use this delimiter because of spaces in the header line
    df_bud = df_bud[(df_bud['ZONE']==2) & (df_bud['PERIOD']>startingsp)]
    df_bud['STREAMLEAKAGE_IN'] = pd.to_numeric(df_bud['STREAMLEAKAGE_IN'])
    df_bud['net_stream']=df_bud['STREAMLEAKAGE_IN']-df_bud['STREAMLEAKAGE_OUT']
    df_bud['pumpage']=df_bud['MNW2_IN']-df_bud['MNW2_OUT']
    df_bud['net_storage']=df_bud['STORAGE_OUT']-df_bud['STORAGE_IN']
    # ar_date = np.array([],dtype='datetime64')
    # for index, row in df_bud.iterrows():
    #     thisdate = zerodate + timedelta(days=row['TOTIM'])
    #     ar_date = np.append(ar_date, thisdate)
    # df_bud['date'] = ar_date

    df_bud['date'] = pd.to_datetime(zerodate) + pd.to_timedelta(df_bud['TOTIM']-1, unit='D')

    return df_bud

def plot_contour(xvalue, yvalue, zvalue, stat = 'mean', bins = 150):
    ret = stats.binned_statistic_2d(xvalue, yvalue, zvalue, stat, bins = bins)
    x = ret.x_edge[1:]
    y = ret.y_edge[1:]

    X, Y = np.meshgrid(x, y)
    z = ret.statistic
    return x, y, z



OPJ = os.path.join
thisFolder = os.getcwd()
root = os.path.sep.join(thisFolder.split(os.path.sep)[:-3])
data_folder = os.path.abspath("..\data_files")
saveplots = True

###############################################################################################

# models to plot
modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
scenlist = ['rcp45', 'rcp85']
catlist = ['HortSroff2Stream_Q','DunnSroff2Stream_Q','Interflow2Stream_Q','Precip_Q',
           'UnsatET_Q','SatET_Q','SoilDrainage2Unsat_Q','Sat2Grav_Q','RechargeUnsat2Sat_Q']
skipcount = 6
cols = ['TOTIM','PERIOD','STEP','ZONE','STORAGE_IN','CONSTANTHEAD_IN','WELLS_IN','HEADDEPBOUNDS_IN','STREAMLEAKAGE_IN',
        'UZFET_IN','GWET_IN','UZFINFILTR_IN','recharge','SURFACELEAKAGE_IN','HORT+DUNN_IN','STORAGECHANGE_IN','MNW2_IN',
        'FromOtherZones_IN','TotalIN','STORAGE_OUT','CONSTANTHEAD_OUT','WELLS_OUT','HEADDEPBOUNDS_OUT',
        'STREAMLEAKAGE_OUT','UZFET_OUT','GWET_OUT','UZFINFILTR_OUT','UZFRECHARGE_OUT','SURFACELEAKAGE_OUT',
        'HORT+DUNN_OUT','STORAGECHANGE_OUT','MNW2_OUT','ToOtherZones_OUT','TotalOut','IN-OUT','PercentError',
        'FROMZONE1','FROMZONE2','TOZONE1','TOZONE2']
data_cols = ['year', 'month', 'day', 'd1', 'd2', 'd3', 'tmx1', 'tmx2', 'tmx3', 'tmx4',
             'tmn1', 'tmn2', 'tmn3', 'tmn4', 'ppt', 'ro1', 'ro2', 'ro3', 'ro4']
zerodate = datetime(1947, 1, 1)
date_columns = ['year', 'month', 'day']
colorlist = ['blue', 'red', 'green', 'chocolate', 'blue', 'red', 'green', 'chocolate']
cats = ['recharge', 'net_stream', 'net_storage', 'GWET_OUT']

for cat in cats:
    all_scenarios = []
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    for mod in tqdm(modlist):
        i = 0
        for scen in scenlist:
            ## read the MDOFLOW budget extracted by zonebud
            df_bud = getdf_bud('../data_files/yuczone_yucaipa_{}_{}.csv.2.csv'.format(mod, scen), 716)
            df_bud['year'] = df_bud['date'].dt.year
            df_bud['month'] = df_bud['date'].dt.month
            annual_bud = df_bud.groupby(by=['year']).mean().reset_index()

            ## read the input data file
            datafilename = os.path.join(data_folder, 'Yucaipa_{0:}_{1:}.data'.format(mod, scen) )
            df_data = pd.read_csv(datafilename, skiprows=skipcount, names=data_cols, delim_whitespace=True)
            # the loop slows down the code.. this is cleaner
            df_data['date'] = pd.to_datetime(df_data[['year', 'month', 'day']])
            df_data['tmean'] = (df_data['tmn1'] + df_data['tmx1'])*0.5

            datafilename = os.path.join(data_folder, 'gsflow_{0:}_{1:}.csv'.format(mod, scen))
            gsflow_bdg = pd.read_csv(datafilename)

            annual_climate_data = df_data.groupby('year').mean().reset_index()
            historical_climate = annual_climate_data[annual_climate_data['year']<=2014]



            df_plot_data = annual_bud.merge(annual_climate_data, how = 'left', on = 'year')
            df_plot_data['model'] = mod
            df_plot_data['scenario'] = scen
            all_scenarios.append(df_plot_data.copy())



           # plt.scatter(ppt_.mean(), tmean_.mean(), c = df_plot_data[cat].mean() ) # , c=df_plot_data[cat]
            # x, y, z = plot_contour(ppt_,tmean_ , df_plot_data[cat].values)
            #
            # cs = ax.contourf(x, y, z, cmap=cm.PuBu_r)
            #cbar = fig.colorbar(cs)
            # i += 1
            # ax.set_ylabel('temperature')
            # ax.set_xlabel('precipitation')
            # ax.set_title('Yucaipa GW Basin: {} for {}_{} model'.format(cat, mod, scen))
            # if saveplots:
            #     plt.savefig(OPJ(root, 'yucaipa_git', 'Yucaipa_GSFLOW', 'report_figs',
            #                     'plot_{}_{}_{}.png'.format(mod, cat, scen)))
            # plt.show()
    all_scenarios = pd.concat(all_scenarios)
    sns.violinplot(data=all_scenarios, x='model', y=cat, hue='scenario', split=True)
    ppt_ = df_plot_data['ppt'].values - historical_climate['ppt'].mean()
    tmean_ = df_plot_data['tmean'].values - historical_climate['tmean'].mean()
