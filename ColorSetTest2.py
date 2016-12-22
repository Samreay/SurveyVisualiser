from surveyvis.surveys import OzDES, SupernovaSurvey
from surveyvis.visualiser import Visualisation
#import matplotlib.pyplot as plt
#import numpy as np
#from joblib import Parallel, delayed
#import os


#s.t_line=np.linspace(56500,    58000,  32, endpoint=False)
#s.set_all_colors()

t=57412

o = OzDES()
s=SupernovaSurvey()


for redshift in [False,True]:

    vis = Visualisation()x
    vis.add_survey(o)
    vis.add_survey(s)

    for falsecolor in ['redshiftonly']:
        for contrast in [1,2,4]:
            name=str(redshift)+"_"+str(falsecolor)+"_"+str(contrast).replace('.','-')
            print(name)
            vis.render3d(t=t, filename=name, azim=300, elev=20, falsecolor=falsecolor, contrast=contrast, redshift=redshift)
