import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import os
import seaborn as sns
from tqdm import tqdm

gsflow_columns = "Date	StreamOut_Q	HortSroff2Stream_Q	DunnSroff2Stream_Q	Interflow2Stream_Q	Stream2Unsat_Q	StreamExchng2Sat_Q" \
"Canopy_S	SnowPweqv_S	Imperv_S	Dprst_S	Cap_S	Grav_S	Unsat_S	Sat_S	UnsatStream_S	Lake_S	Stream_S" \
"	Precip_Q	NetBoundaryFlow2Sat_Q	NetWellFlow_Q	BoundaryStreamFlow_Q	CanopyEvap_Q	SnowEvap_Q	ImpervEvap_Q" \
"	DprstEvap_Q	CapET_Q	SwaleEvap_Q	UnsatET_Q	SatET_Q	LakeEvap_Q	DunnInterflow2Lake_Q	HortSroff2Lake_Q	" \
"Lake2Unsat_Q	LakeExchng2Sat_Q	SoilDrainage2Unsat_Q	Sat2Grav_Q	RechargeUnsat2Sat_Q	Infil2Soil_Q	KKITER"

intg_inflow = ['Precip_Q', 'NetBoundaryFlow2Sat_Q', 'NetWellFlow_Q']
intg_ouflow = ['CanopyEvap_Q', 'ImpervEvap_Q', 'DprstEvap_Q', 'CapET_Q', 'SwaleEvap_Q', 'UnsatET_Q', 'SatET_Q']


def getdf_bud(fname, startingsp):
    df_bud = pd.read_csv(fname, names=cols, skiprows=1,
                         sep='[\s,]{2,20}')  # use this delimiter because of spaces in the header line
    df_bud = df_bud[(df_bud['ZONE'] == 2) & (df_bud['PERIOD'] > startingsp)]
    df_bud['STREAMLEAKAGE_IN'] = pd.to_numeric(df_bud['STREAMLEAKAGE_IN'])
    df_bud['net_stream'] = df_bud['STREAMLEAKAGE_IN'] - df_bud['STREAMLEAKAGE_OUT']
    df_bud['pumpage'] = df_bud['MNW2_IN'] - df_bud['MNW2_OUT']
    df_bud['net_storage'] = df_bud['STORAGE_OUT'] - df_bud['STORAGE_IN']

    df_bud['date'] = pd.to_datetime(zerodate) + pd.to_timedelta(df_bud['TOTIM'] - 1, unit='D')

    return df_bud


def plot_contour(xvalue, yvalue, zvalue, stat='mean', bins=150):
    ret = stats.binned_statistic_2d(xvalue, yvalue, zvalue, stat, bins=bins)
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
catlist = ['HortSroff2Stream_Q', 'DunnSroff2Stream_Q', 'Interflow2Stream_Q', 'Precip_Q',
           'UnsatET_Q', 'SatET_Q', 'SoilDrainage2Unsat_Q', 'Sat2Grav_Q', 'RechargeUnsat2Sat_Q']
skipcount = 6
cols = ['TOTIM', 'PERIOD', 'STEP', 'ZONE', 'STORAGE_IN', 'CONSTANTHEAD_IN', 'WELLS_IN', 'HEADDEPBOUNDS_IN',
        'STREAMLEAKAGE_IN',
        'UZFET_IN', 'GWET_IN', 'UZFINFILTR_IN', 'recharge', 'SURFACELEAKAGE_IN', 'HORT+DUNN_IN', 'STORAGECHANGE_IN',
        'MNW2_IN',
        'FromOtherZones_IN', 'TotalIN', 'STORAGE_OUT', 'CONSTANTHEAD_OUT', 'WELLS_OUT', 'HEADDEPBOUNDS_OUT',
        'STREAMLEAKAGE_OUT', 'UZFET_OUT', 'GWET_OUT', 'UZFINFILTR_OUT', 'UZFRECHARGE_OUT', 'SURFACELEAKAGE_OUT',
        'HORT+DUNN_OUT', 'STORAGECHANGE_OUT', 'MNW2_OUT', 'ToOtherZones_OUT', 'TotalOut', 'IN-OUT', 'PercentError',
        'FROMZONE1', 'FROMZONE2', 'TOZONE1', 'TOZONE2']
data_cols = ['year', 'month', 'day', 'd1', 'd2', 'd3', 'tmx1', 'tmx2', 'tmx3', 'tmx4',
             'tmn1', 'tmn2', 'tmn3', 'tmn4', 'ppt', 'ro1', 'ro2', 'ro3', 'ro4']
zerodate = datetime(1947, 1, 1)
date_columns = ['year', 'month', 'day']
colorlist = ['blue', 'red', 'green', 'chocolate', 'blue', 'red', 'green', 'chocolate']
cats = ['recharge', 'net_stream', 'net_storage', 'GWET_OUT']

gsflow_historical = pd.read_csv(os.path.join(data_folder, 'gsflow_historical.csv'))
gsflow_historical['Date'] = pd.to_datetime(gsflow_historical['Date'])
gsflow_historical['year'] = gsflow_historical['Date'].dt.year
ann_hist_gsflow = gsflow_historical.groupby(by=['year']).mean()

all_gsflows = []

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
        datafilename = os.path.join(data_folder, 'Yucaipa_{0:}_{1:}.data'.format(mod, scen))
        df_data = pd.read_csv(datafilename, skiprows=skipcount, names=data_cols, delim_whitespace=True)
        df_data['date'] = pd.to_datetime(df_data[['year', 'month', 'day']])
        df_data['tmean'] = (df_data['tmn1'] + df_data['tmx1']) * 0.5

        #read GSFLOW
        datafilename = os.path.join(data_folder, 'gsflow_{0:}_{1:}.csv'.format(mod, scen))
        gsflow_bdg = pd.read_csv(datafilename)
        gsflow_bdg['date'] = pd.to_datetime(gsflow_bdg['Date'])
        del(gsflow_bdg['Date'])
        gsflow_bdg['year'] = gsflow_bdg['date'].dt.year
        gsflow_bdg['month'] = gsflow_bdg['date'].dt.month
        annual_gsflow = gsflow_bdg.groupby(by=['year']).mean().reset_index()

        annual_climate_data = df_data.groupby('year').mean().reset_index()
        historical_climate = annual_climate_data[annual_climate_data['year'] <= 2014]

        df_plot_data = annual_bud.merge(annual_climate_data, how='left', on='year')
        df_plot_data = df_plot_data.merge(annual_gsflow, how='left', on='year')
        all_dfs = gsflow_bdg.merge(df_data[['date', 'tmx1', 'tmn1', 'tmean']], how='left', on='date')
        del(all_dfs['month'])
        del(all_dfs['year'])
        all_dfs = all_dfs.merge(df_bud, how='left', on='date')

        all_dfs['model'] = mod
        all_dfs['scenario'] = scen
        all_scenarios.append(all_dfs.copy())

# This data_frame "all_scenarios" has all daily data for all models and scenarios
# you groupby it month or year
all_scenarios = pd.concat(all_scenarios)
all_scenarios.reset_index(inplace=True)
all_scenarios['Climate Scenario'] = all_scenarios['model'] + all_scenarios['scenario']

# this is simple example how to plot data
cat = 'recharge'
sns.violinplot(data=all_scenarios, x='model', y=cat, hue='scenario', split=True)
