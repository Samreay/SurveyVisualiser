import sncosmo as snc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table as table

#SNCTest.py
'''
A package test
'''

N=1000
tpeak=0


gain=[1]*N
skynoise=[0]*N
zp=[0]*N
zpsys=['ab']*N

model = snc.Model(source='salt2')

for x1 in [1]:
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
        z=0.4
        params = {'z': 0.4, 'x0': x0, 't0': 0, 'x1': 0.1, 'c': -0.1}

        lcs = snc.realize_lcs(obs, model, [params],scatter=False)
        x,y=lcs[0]["time"],lcs[0]["flux"]
        plt.plot(x,y,c=c)
        plt.xlim(xmin=-50,xmax=150)
        title=str(z)+","+str(x0)+","+str(x1)+","+str(filter)
        plt.ylim(ymin=0)
        plt.title(title)



plt.show()