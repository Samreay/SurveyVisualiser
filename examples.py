from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, Taipan, SupernovaSurvey
from surveyvis.visualiser import Visualisation
import numpy as np
from joblib import Parallel, delayed
import os


def make3d(name, vis, i, maxr, minr, low_quality=False, t=0,  plotsupernovae=False, blur=True):
    """
     Outputs a PNG image of the surveys visualised at a
     calculated camera position

     Parameters
     ----------
     name : str
        Name of the folder in which to put the file, and the start of the filename
    vis : Visualisation
        A Visualisation object
    i : int
        The index of the image. Aka the degrees around the camera has moved.
    maxr : float
        The maximum distance to put the camera
    minr : float
        The minimum distance to put the camera
    """
    name = "output/%s" % name



    if plotsupernovae:
        SN_in_vis = False
        for s in vis.surveys:
            if isinstance(s,SupernovaSurvey):
                SN_in_vis = True

        if SN_in_vis == False:
            print("Adding Supernovae from make3d()")
            s=SupernovaSurvey()
            s.t_line=np.array([t])
            s.set_all_colors()
            vis.add_survey(s)

    rad = i * np.pi / 180
    elev = -(30 + 30 * np.cos(rad))
    d = min(np.abs(200 - i), np.abs(360 + i - 200))
    r = maxr - (maxr - minr) * (1 - np.exp(-(d / 140) ** 2))
    if not os.path.exists(name):
        os.makedirs(name)
    vis.render3d("%s/3d_%d_%f.png" % (name, i, t), rmax=r, elev=elev, azim=i, low_quality=low_quality, t=t, blur=blur)


def make_video(name, data, low_quality=False,   no_frames=360, plotsupernovae=False, blur=True, tlist=np.array([])):
    """
    Render out all the still frames needed to make a video for the given data

    After rendering out the video series, you turn this into an mpg video
    by running the `make.bat`. For example, if I have generated all the
    images for 2df, I can call `make.bat 2df` to turn those images
    into a video.

    Parameters
    ----------
    name : str
        The name of the folder to put results in
    data : list[Survey] | Survey
        A list of Surveys add to the Visualisation, or a single
        Survey if you only want one.
    """

    # Create an empty visualisation
    vis = Visualisation()

    # Load the data into it
    if isinstance(data, list):
        for d in data:
            vis.add_survey(d)
    else:
        vis.add_survey(data)

    if plotsupernovae:
        print("Adding Supernovae To Visualiser from make_video()")
        supersurvey=SupernovaSurvey()
        supersurvey.t_line=tlist
        supersurvey.set_all_colors()

        vis.add_survey(supersurvey)

    # Get the redshift limits for each survey
    rs = [s.zmax for s in vis.surveys]

    # Calculate the camera radius (min and max) from these values
    if len(rs) == 1:
        maxr = 0.7 * max(rs)
        minr = maxr
    else:
        maxr = 0.7 * max(rs)
        minr = 0.7 * min(rs)

    # Using 4 cores, call make3d for each degree from 0 to 360
    ilist=np.linspace(0,            0,          no_frames, endpoint=False)

    Parallel(n_jobs=1)(delayed(make3d)(name, vis, int(i), minr, maxr, low_quality, t, plotsupernovae, blur) for i,t in zip( ilist, tlist ))


def make(name, data):
    """
    Render out the 2D images for the given data. Once for a latex image, and one for a colour image.

    Parameters
    ----------
    name : str
        The name of the output files
    data : list[Survey] | Survey
        A list of Surveys add to the Visualisation, or a single
        Survey if you only want one.
    """

    # Set up filenames, make the folders if they dont exist
    output = "output"
    name = "%s/%s" % (output, name)
    if not os.path.exists(output):
        os.makedirs(output)

    # Create Visualisation
    vis = Visualisation()

    # Load data into Visualisation
    if isinstance(data, list):
        for d in data:
            vis.add_survey(d)
    else:
        vis.add_survey(data)

    # Render the latex plot out
    vis.render_latex(name.replace(".png", "_latex.png"))

    # Render the colour plot
    vis.render2d(name)

    print("Made figure for %s" % name)


def get_permutations():
    """
    A helper function I call, that gives me a list of all the different plots I want
    to create with their data.
    """
    w = WiggleZ()
    t = TwoDegreeField()
    s = SDSS()
    g = Gama()
    x = SixDegreefField()
    o = OzDES()
    o2 = OzDES()
    o3 = OzDES()
    o2.zmax = 1.0
    o3.zmax = 4.0
    l = Tdflens()
    p = Taipan()
    groups = [[w, t, s, g, x, o], [w, t, s, g, x], w, t, g, s, x, o, l, [l, o2, t], p, [l, t, o3]]
    names = ["all", "all_nooz", "wigglez", "2df", "gama", "sdss", "6df", "ozdes", "2dflens", "sub", "taipan", "ozdes_deep"]
    return groups, names


def make_figures(name=None, blur=True):
    """
    Makes all 2D figures for all permutations of data that I want

    Parameters
    ----------
    name : str [optional]
        If None, makes all permutations. If a string,
        will only make the permutation matching with the same name.
        See output of `get_permutations` for a list of names
    """
    groups, names = get_permutations()
    # Using 4 cores, make all the images we want
    Parallel(n_jobs=1)(delayed(make)(n + ".png", g) for n, g in zip(names, groups) if name is None or name == n)


def make_all_video(name=None, low_quality=False, no_frames=360, plotsupernovae=False, blur=True, tlist=np.array([])):
    """
    Makes all video series for all permutations of data that I want

    Parameters
    ----------
    name : str [optional]
        If None, makes all permutations. If a string,
        will only make the permutation matching with the same name.
        See output of `get_permutations` for a list of names
    """

    # Note we dont use Parallel processing here, because make_video already uses it
    groups, names = get_permutations()
    for n, g in zip(names, groups):
        if name is None or name == n:
            make_video(n, g, low_quality=low_quality, no_frames=no_frames, plotsupernovae=plotsupernovae, blur=blur, tlist=tlist)


if __name__ == "__main__":
    # Uncomment the below two lines to do everything
    # make_figures()
    # make_all_video()

    # As an example, make the 6df figures and video
    #make_figures("6df")
    noframes=32
    tlist=np.linspace(56548.121,    56548.121-2*(56548.121-57412.457),  noframes, endpoint=False)
    make_all_video("ozdes", low_quality=False, no_frames=noframes, plotsupernovae=True, blur=True, tlist=tlist)


    # Uncomment one of the below lines (and comment out the above two)
    # to make only the plot declared

    # make_figures("ozdes_deep")
    # make_figures("ozdes")
    # make_figures("sub")
    # make_figures("all")
    # make_all_video("6df")
    # make_figures("ozdes")
