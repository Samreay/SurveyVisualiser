from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, Taipan, SupernovaSurvey
from surveyvis.visualiser import Visualisation
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import os

s=SupernovaSurvey()
#s.t_line=np.linspace(56500,    58000,  32, endpoint=False)
#s.set_all_colors()

t=57300

r,g,b=s.get_color(t,'r'),s.get_color(t,'g'),s.get_color(t,'b')


vis=Visualisation()
o=OzDES()
vis.add_survey(o)
vis.add_survey(s)

vis.render3d(t=t,filename="Test")
