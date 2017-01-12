from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, Taipan
from surveyvis.visualiser import Visualisation
import numpy as np
from joblib import Parallel, delayed
import os


def make3d(name, vis, i, maxr, minr):
    name = "output/%s" % name
    rad = i * np.pi / 180
    elev = -(30 + 30 * np.cos(rad))
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
    vis.render_latex(name.replace(".png", "_latex.png"), colours=["#0000FF", "#FF0000"]) #["#1E3B9C", "#941313"]
    vis.render_latex(name.replace(".png", "_latex2.png"), colours=["#0000FF", "#FF0000"][::-1])
    # vis.render2d(name)
    print("Made figure for %s" % name)


def get_permutations():
    w = WiggleZ()
    w125 = WiggleZ()
    w125.zmax = 1.25
    w150 = WiggleZ()
    w150.zmax = 1.50
    w200 = WiggleZ()
    w200.zmax = 2.0
    t = TwoDegreeField()
    s = SDSS()
    g = Gama()
    x = SixDegreefField()
    o = OzDES()
    o2 = OzDES()
    o3 = OzDES()
    w.zmax = 1.5
    o2.zmax = 1.0
    o3.zmax = 4.0
    l = Tdflens()
    p = Taipan()
    groups = [[w, t, s, g, x, o], [w, t, s, g, x], w, t, g, s, x, o, l, [l, o2, t], p, [l, t, o3], [w, t], [w125, t], [w150, t], [w200, t]]
    names = ["all", "all_nooz", "wigglez", "2df", "gama", "sdss", "6df", "ozdes", "2dflens", "sub", "taipan", "ozdes_deep", "michael100","michael125","michael150","michael200"]
    return groups, names


def make_figures(name=None):
    groups, names = get_permutations()
    Parallel(n_jobs=4)(delayed(make)(n + ".pdf", g) for n, g in zip(names, groups) if name is None or name == n)
    Parallel(n_jobs=4)(delayed(make)(n + ".eps", g) for n, g in zip(names, groups) if name is None or name == n)


def make_all_video(name=None):
    groups, names = get_permutations()
    for n, g in zip(names, groups):
        if name is None or name == n:
            make_video(n, g)


if __name__ == "__main__":
    # make_figures()
    # make_all_video()
    make_figures("michael100")
    make_figures("michael125")
    make_figures("michael150")
    make_figures("michael200")
    # make_figures("ozdes_deep")
    # make_figures("ozdes")
    # make_figures("sub")
    # make_figures("all")
    # make_all_video("6df")
    # make_figures("ozdes")
    # make_figures("sdss")
    # vis = Visualisation()
    # s = SixDegreefField()
    # vis.add_survey(s)
    # make3d("6df", vis, 0, 0.7 * s.zmax, 0.7 * s.zmax)
    # make3d("6df", vis, 180, 0.7 * s.zmax, 0.7 * s.zmax)

