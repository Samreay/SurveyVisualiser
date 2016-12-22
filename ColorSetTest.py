from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, Taipan, SupernovaSurvey
from surveyvis.visualiser import Visualisation
import numpy as np
import os

s=SupernovaSurvey()
s.t_line=np.linspace(56548.121,    57412.457,  32, endpoint=False)
s.set_all_colors()
s.set_color('r')

t=s.t_line[14]


print(r+g+b)