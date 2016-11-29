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


band=['bessellb']*N
gain=[1]*N
skynoise=[0]*N
zp=[0]*N
zpsys=['ab']*N

model = snc.Model(source='salt2')

for x1 in np.linspace(-4,4,1):
    for c in np.linspace(-.2, .6, 1):

        print("Plotting for x1=",x1,"c=",c)

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

        params = {'z': 0.4, 'x0': 1e-5, 't0': 0, 'x1': 0.1, 'c': -0.1}

        lcs = snc.realize_lcs(obs, model, [params],scatter=False)
        x,y=lcs[0]["time"],lcs[0]["flux"]
        plt.plot(x,y)



plt.show()