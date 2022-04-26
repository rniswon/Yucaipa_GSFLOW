def assign_wy(row):
    if row.Date.month>=10:
        return(pd.datetime(row.Date.year+1,1,1).year)
    else:
        return(pd.datetime(row.Date.year,1,1).year)

def getdf(sv_name, gs_name, startyear, endyear):
    ##################
    # get the statvar data, which is in inches from the PRMS model run without GSFLOW
    df_statvar = pd.read_csv(sv_name, skiprows=7, names=cols, delim_whitespace=True,
                             index_col='id')
    df_statvar['Date'] = pd.to_datetime(df_statvar[['year', 'month', 'day']])
    df_statvar['PET'] = df_statvar['basin_potet']
    ##################
    # get the GSFLOW budget this is in cubic feet per day
    # to convert to inches
    df_gsflow = pd.read_csv(gs_name, header=0)
    df_gsflow['Date'] = pd.to_datetime(df_gsflow['Date'])
    # there are 14,012 active cells. Each is 150m square  4.88668E+11
    df_gsflow['AET'] = (df_gsflow['SatET_Q']+df_gsflow['UnsatET_Q']+df_gsflow['CanopyEvap_Q']+
                        df_gsflow['SnowEvap_Q']+df_gsflow['ImpervEvap_Q']+df_gsflow['DprstEvap_Q']+
                        df_gsflow['SwaleEvap_Q']+df_gsflow['UnsatET_Q']+df_gsflow['LakeEvap_Q'])

    #### now merge the dataframes to get the values for matching days
    df_all = df_statvar.merge(df_gsflow, left_on='Date', right_on='Date')
    # assign the water year
    df_all['WY'] = df_all.apply(lambda x: assign_wy(x), axis=1)
    # slice the future period 2015 - 2099
    df_all = df_all.loc[(df_all['WY']>=startyear) & (df_all['WY']<=endyear)]
    # get the mean daily values for AET/PPT and PET/PPT
    df_statvar_year = df_all.groupby(['WY'])['AET', 'PET', 'basin_ppt', 'Precip_Q'].sum()
    df_statvar_year['AETP'] = df_statvar_year['AET']/df_statvar_year['Precip_Q']
    df_statvar_year['PETP'] = df_statvar_year['PET']/df_statvar_year['basin_ppt']
    # now get the 5-year moving average
    df_statvar_year['AETP_MA'] = df_statvar_year['AETP'].rolling(5).mean()
    df_statvar_year['PETP_MA'] = df_statvar_year['PETP'].rolling(5).mean()
    df_statvar_year = df_statvar_year.sort_values(by='PETP_MA')
    df_statvar_year = df_statvar_year.loc[df_statvar_year['PETP_MA']>0]
    #ar_aep = moving_average(df_statvar_year['AEP'].values, 365)
    #ar_pep = moving_average(df_statvar_year['PEP'].values, 365)
    ar_aep = df_statvar_year['AETP_MA'].values
    ar_pep = df_statvar_year['PETP_MA'].values
    return ar_aep, ar_pep

def func1(x, a, b, c):
    return a*x**2+b*x+c

def func2(x, a, b, c):
    return a*x**3+b*x+c

def func3(x, a, b, c):
    return a*x**3+b*x**2+c

def func4(x, a, b, c):
    return a*exp(b*x)+c

def linear(x, a, b):
    return a*x+b

def fitcurve1(x, y):
    popt, pcov = curve_fit(func1, x, y)
    a, b, c = popt[0], popt[1], popt[2]
    yfit1 = a*x**2+b*x+c
    return yfit1

def fitcurve2(x, y):
    popt, pcov = curve_fit(func2, x, y)
    a, b, c = popt[0], popt[1], popt[2]
    yfit = a*x**3+b*x+c
    return yfit

def fitcurve3(x, y):
    popt, pcov = curve_fit(func3, x, y)
    a, b, c = popt[0], popt[1], popt[2]
    yfit = a*x**2+b*x+c
    return yfit

def fitcurve4(x, y):
    popt, pcov = curve_fit(func4, x, y)
    a, b, c = popt[0], popt[1], popt[2]
    yfit = a*exp(b*x)+c
    return yfit

#######################################################################
# this script is to read PRMS statvar and GSFLOW budget files,
# calculate the total AET and PET, and the ratios AET/PPT and PET/PPT.
# these ratios are calculated on annual sums and a 5-year moving
# average applied. They are then plotted as points with the PET/PPT
# on the x axis for a Budyko curve
#

import numpy as np
from numpy import exp
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OPJ = os.path.join
cols = ['id', 'year', 'month', 'day', 'f1', 'f2', 'f3', 'basin_ppt', 'basin_ssflow_cfs', 'basin_sroff',
        'basin_recharge', 'basin_potet', 'basin_actet']
modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green'), ('HadGEM2ES', 'red'), ('MIROC5', 'magenta')]
#modlist = ['CanESM2', 'CNRMCM5']
scenlist = [('rcp45', 's', '-'), ('rcp85', '+', '--')]
# get historical data
# getdf extracts the PET from the PRMS statvar file and the AET from the gsflow.csv for the period specified
# it also assigns the water year and sums by water year
ar_aet_h, ar_pet_h = getdf('../data_files/CanESM2_rcp45_statvar_prms.dat', '../data_files/gsflow_base.csv', 1947, 2014)
yfit_h = fitcurve1(ar_pet_h, ar_aet_h)

for mod in modlist:
    for scen in scenlist:
        fig, ax = plt.subplots()
        ax.plot(ar_pet_h, ar_aet_h, lw=0, marker='o', ms=2, mfc='black', mec='black')
        ax.plot(ar_pet_h, yfit_h, color='black', ls='--', lw=0.8, label='Historical')
        sv_name = '../data_files/{}_{}_statvar_prms.dat'.format(mod[0], scen[0])
        ar_aet, ar_pet = getdf(sv_name, '../data_files/gsflow_{}_{}.csv'.format(mod[0], scen[0]), 2015, 2099)
        ax.plot(ar_pet, ar_aet, lw=0, marker=scen[1], ms=3, mfc=mod[1], mec=mod[1])
        yfit1 = fitcurve1(ar_pet, ar_aet)
        ax.plot(ar_pet, yfit1, color=mod[1], ls=scen[2], lw=0.6, label='{} {}'.format(mod[0], scen[0]))
        #plt.axhline(1.0)
        #plt.axvline(1.0, ls='--', lw=0.5)
        ax.set_ylabel('AE/PPT')
        ax.set_xlabel('PE/PPT')
        #ax.set_ylim(0, 0.10)
        #ax.set_xlim(0, 50)
        ax.legend(loc='lower right')
        plt.savefig(OPJ('C:\\', 'Users', 'dryter', 'OneDrive - DOI', 'Yucaipa', '{}_{}_budyko.png'.format(mod[0], scen[0])))
        plt.savefig('./plots/{}_{}_budyko.png'.format(mod[0], scen[0]))
        plt.show()

print('done.')