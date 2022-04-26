def getarray(line):
    line = line.replace(' ','')
    line = line[:len(line)-2]
    a = line.strip().split(',')
    d = timedelta(days=int(float(a[0])))
    thisdate = startdate + d
    firstcols = [thisdate.strftime('%m/%d/%Y')] + a[:4]
    ar = np.array([], dtype=float)
    for cat in a[4:]:
        ar = np.append(ar, float(cat))
    return firstcols, ar

from datetime import datetime, timedelta
import numpy as np

modlist = ['CanESM2', 'CNRMCM5', 'HadGEM2ES', 'MIROC5']
scenlist = ['rcp45', 'rcp85']
newlin = '\n'
startdate = datetime(1947, 1, 1)
for mod in modlist:
    for scen in scenlist:
        zb_name = '../data_files/yuczone_yucaipa_{}_{}.csv.2.csv'.format(mod, scen)
        bud_name = '../data_files/{}_{}_combine_zone.csv'.format(mod, scen)
        inf = open(zb_name, 'r')
        outf = open(bud_name, 'w')
        line = inf.readline()
        line = line.replace(' ','')
        line = line[:len(line)-2]
        a = line.strip().split(',')
        headlist = ['date'] + a
        outf.write(','.join(headlist)+newlin)
        while True:
            line = inf.readline()
            if line == '':
                break
            fcol1, ar1 = getarray(line)
            line = inf.readline()
            fcol2, ar2 = getarray(line)
            ar_line = ar1 + ar2
            linelist = []
            for cat in ar_line:
                linelist.append(str(cat))
            #ar_str = np.array2string(ar_line, precision=0, separator=',',max_line_width=500, floatmode='fixed')
            #ar_str = ar_str.replace('[','').replace(']','')
            lineout = ','.join(fcol1) + ',' + ','.join(linelist) + newlin
            outf.write(lineout)

        outf.close()
        inf.close()
        print('done.')
#yuczone_yucaipa_CanESM2_rcp45.csv.2.csv