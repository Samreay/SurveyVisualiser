from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens
from surveyvis.visualiser import Visualisation
import numpy as np
from joblib import Parallel, delayed
import os


def make3d(name, vis, i, maxr, minr):
    name = "output/%s" % name
    rad = i * np.pi / 180
    elev = 30 + 30 * np.cos(rad)
    d = min(np.abs(200 - i), np.abs(360 + i - 200))
    r = maxr - (maxr - minr) * (1 - np.exp(-(d / 140) ** 2))
    if not os.path.exists(name):
        os.makedirs(name)
    vis.render3d("%s/3d_%d.png" % (name, i), rmax=r, elev=elev, azim=i)


def make_video(name, data):
    vis = Visualisation()
    if isinstance(data, list):
        for d in data:
            vis.add_survey(d)
    else:
        vis.add_survey(data)
    rs = [s.zmax for s in vis.surveys]
    if len(rs) == 1:
        maxr = 0.7 * max(rs)
        minr = maxr
    else:
        maxr = 0.7 * max(rs)
        minr = 0.7 * min(rs)
    Parallel(n_jobs=4)(delayed(make3d)(name, vis, int(i), minr, maxr) for i in np.linspace(0, 360, 360, endpoint=False))


def make(name, data):
    output = "output"
    name = "%s/%s" % (output, name)
    if not os.path.exists(output):
        os.makedirs(output)
    vis = Visualisation()
    if isinstance(data, list):
        for d in data:
            vis.add_survey(d)
    else:
        vis.add_survey(data)
    vis.render_latex(name.replace(".png", "_latex.png"))
    vis.render2d(name)
    print("Made figure for %s" % name)


def get_permutations():
    w = WiggleZ()
    t = TwoDegreeField()
    s = SDSS()
    g = Gama()
    x = SixDegreefField()
    o = OzDES()
    l = Tdflens()
    groups = [[w, t, s, g, x, o], [w, t, s, g, x], w, t, g, s, x, o, l, [l, o, t]]
    names = ["all", "all_nooz", "wigglez", "2df", "gama", "sdss", "6df", "ozdes", "2dflens", "sub"]
    return groups, names


def make_figures(name=None):
    groups, names = get_permutations()
    Parallel(n_jobs=4)(delayed(make)(n + ".png", g) for n, g in zip(names, groups) if name is None or name == n)


def make_all_video(name=None):
    groups, names = get_permutations()
    for n, g in zip(names, groups):
        if name is None or name == n:
            make_video(n, g)


if __name__ == "__main__":
    make_figures()
    make_all_video()
    # make_figures("2df")
    # make_figures("all")
    # make_all_video("all")
    # make_figures("ozdes")
    # make_figures("sdss")
    # vis = Visualisation()
    # s = TwoDegreeField()
    # vis.add_survey(s)
    # make3d("2df", vis, 0, 0.7 * s.zmax, 0.7 * s.zmax)

