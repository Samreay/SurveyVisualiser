from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, Taipan, SupernovaSurvey
from surveyvis.visualiser import Visualisation
import numpy as np
from joblib import Parallel, delayed
import os

s=SupernovaSurvey()
s.t_line=np.linspace(56548.121,    57412.457,  32, endpoint=False)
s.set_all_colors()

print(s.fluxes)

t=s.t_line[14]

r,g,b=s.get_color(t,'r'),s.get_color(t,'g'),s.get_color(t,'b')

print(r,g,b)