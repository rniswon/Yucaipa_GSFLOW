def getmean(fname, start, end):
    df = pd.read_csv(fname, names=data_col, skiprows=6, delim_whitespace=True)   # historical period model for stats
    df = df.loc[(df['year']>=start) & (df['year']<=end)]
    df['mt'] = (df['tmx'] + df['tmn'])/2
    df1 = df.groupby(['year'])['ppt'].sum().reset_index()
    ppt = df1['ppt'].mean()
    ppt_min = df1['ppt'].min()
    ppt_max = df1['ppt'].max()
    df2 = df.groupby(['year'])['mt', 'tmn', 'tmx'].mean().reset_index()
    tmp = df2['mt'].mean()
    up_err = df2['tmx'].mean()-df2['mt'].mean()
    lo_err = df2['mt'].mean()-df2['tmn'].mean()
    return ppt, ppt_min, ppt_max, tmp, lo_err, up_err

# this script looks at climate data from the historical period and future models to look at how they are
# different with respect to mean annual temperature and precitpitation

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

data_col = ['year', 'month', 'day', 'f1', 'f2', 'f3', 'tmx', 'tm1', 'tm2', 'tm3', 'tmn', 'tmn1', 'tmn2', 'tmn3', 'ppt',
            'sr1', 'sr2', 'sr3', 'sr4']
colorlist = ['blue', 'red', 'green', 'chocolate', 'blue', 'red', 'green', 'chocolate']

# get the historical period
hist_ppt, hist_ppt_min, hist_ppt_max, hist_temp, hist_tmn, hist_tmx = getmean('../data_files/Yucaipa_CanESM2_rcp45.data', 1947, 2014)
fig, ax = plt.subplots(figsize = (7,6))
plt.axhline(hist_temp, color='gray', lw=0.5, ls='--')
plt.axvline(hist_ppt, color='gray', lw=0.5, ls='--')
ax.plot(hist_ppt, hist_temp, marker='s', ms=6,
            mfc='black', label='historical (1947 - 2014)', lw=0)
#########################
# models to plot
modlist = [('CanESM2', 'blue'), ('CNRMCM5', 'green'), ('HadGEM2ES', 'red'), ('MIROC5', 'magenta')]
scenlist = [('rcp45', 'o'), ('rcp85', '^')]
for mod in modlist:
    for scen in scenlist:
        ppt, ppt_min, ppt_max,  tmp, tmn, tmx = getmean('../data_files/Yucaipa_{}_{}.data'.format(mod[0],
                                                                                                  scen[0]), 2015, 2099)
        ax.errorbar(ppt, tmp, lw=0, marker=scen[1], ms=5, mfc=mod[1], mec=mod[1],
                    label='{} {}'.format(mod[0], scen[0]))

ax.set_ylabel('mean annual temperature in degrees Celcius')
ax.set_xlabel('mean annual precipitation in inches')
ax.set_ylim(15, 25)
ax.xaxis.set_tick_params(direction='in', labelsize=9)
ax.yaxis.set_tick_params(direction='in', labelsize=9)
ax.legend(fontsize=7)
plt.savefig('../report_figs/compare_hist_future_climate.png')
plt.show()