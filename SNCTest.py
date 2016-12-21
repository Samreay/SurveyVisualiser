import sncosmo as snc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table as table

'''
#SNCTest.py

A package test
'''

N=1000
tpeak=0


gain=[1]*N
skynoise=[0]*N
zp=[0]*N
zpsys=['ab']*N

model = snc.Model(source='salt2')

for z in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    print(z)
    plt.figure()
    for filter,c in zip(['besselli','bessellr','bessellv','bessellb','bessellux'],['k','r','g','b','purple']):
        band = [filter] * N
        print("Plotting for ",filter,"c=",c)

        span=200
        tmin = tpeak - span * 1 / 4
        tmax = tpeak + span * 3 / 4
        t = np.linspace(tmin, tmax, N)

        obs = table({'time': t,
                     'band': band,
                     'gain': gain,
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': zpsys})

        x0=4e-6
        params = {'z': z, 'x0': x0, 't0': 0, 'x1': 0.1, 'c': -0.1}

        try:
            lcs = snc.realize_lcs(obs, model, [params],scatter=False)
            x,y=lcs[0]["time"],lcs[0]["flux"]/x0/1e-4
            plt.plot(x, y, c=c)
        except:
            pass

    plt.legend(['besselli', 'bessellr', 'bessellv', 'bessellb', 'bessellux'],loc='best')
    plt.xlim(xmin=-50,xmax=150)
    plt.ylim(ymin=0,ymax=1)

    plt.axvline(-25)
    plt.axvline(100)

    title=str(z)
    plt.title(title)
    plt.savefig("Redshift-"+title+".png")

