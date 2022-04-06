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

def plot_contour(xvalue, yvalue, zvalue, stat = 'mean', bins = 150):
    ret = stats.binned_statistic_2d(xvalue, yvalue, zvalue, stat, bins = bins)
    x = ret.x_edge[1:]
    y = ret.y_edge[1:]

    X, Y = np.meshgrid(x, y)
    z = ret.statistic
    return x, y, z


import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import os

OPJ = os.path.join
thisFolder = os.getcwd()
root = os.path.sep.join(thisFolder.split(os.path.sep)[:-3])
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
    for mod in modlist:
        i = 0
        for scen in scenlist:
            fig, ax = plt.subplots(figsize=(8,6),tight_layout=True)
            ## read the MDOFLOW budget extracted by zonebud
            df_bud = getdf_bud('yuczone_yucaipa_{}_{}.csv.2.csv'.format(mod, scen), 716)
            ## read the input data file
            datafilename = OPJ(root, '{0:}_{1:}'.format(mod, scen),'input','Yucaipa_{0:}_{1:}.data'.format(mod, scen))
            df_data = pd.read_csv(datafilename, skiprows=skipcount, names=data_cols, delim_whitespace=True)
            ar_date = np.array([], dtype='datetime64')
            for index, row in df_data.iterrows():
                ar_date = np.append(ar_date, datetime(int(row['year']), int(row['month']), int(row['day'])))
            df_data['date'] = ar_date
            df_plot_data = df_data.merge(df_bud, left_on='date', right_on='date')
            df_plot_data['tmean'] = (df_plot_data['tmn1'] + df_plot_data['tmx1']) / 2
            x, y, z = plot_contour(df_plot_data['ppt'].values, df_plot_data['tmean'].values, df_plot_data[cat].values)
            #
            cs = ax.contourf(x, y, z, cmap=cm.PuBu_r)
            cbar = fig.colorbar(cs)
            i += 1
            ax.set_ylabel('temperature')
            ax.set_xlabel('precipitation')
            ax.set_title('Yucaipa GW Basin: {} for {}_{} model'.format(cat, mod, scen))
            if saveplots:
                plt.savefig(OPJ(root, 'yucaipa_git', 'Yucaipa_GSFLOW', 'report_figs',
                                'plot_{}_{}_{}.png'.format(mod, cat, scen)))
            plt.show()
