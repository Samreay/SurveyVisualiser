from surveyvis.camera import OrbitZoomCamera
from surveyvis.surveys import WiggleZ, TwoDegreeField, Gama, SDSS, SixDegreefField, Dummy, Dummy2, OzDES, Tdflens, \
    Taipan, RandomSupernovae, OzDESSupernovae, SupernovaeSurvey, OzDESSupernovaeAll
from surveyvis.visualiser import Visualisation
import numpy as np
from joblib import Parallel, delayed
import os


def make3d(name, vis, i, i_max, low_quality=False):
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

    if not os.path.exists(name):
        os.makedirs(name)
    vis.render3d("%s/3d_%d.png" % (name, i), 1.0 * i / i_max, low_quality=low_quality)


def make_video(name, data, low_quality=False, num_frames=360):
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
    print("Making video for %s" % name)
    # Create an empty visualisation
    vis = Visualisation()

    # Load the data into it
    if isinstance(data, list):
        for d in data:
            vis.add_survey(d)
    else:
        vis.add_survey(data)

    has_supernova = len([s for s in vis.surveys if isinstance(s, SupernovaeSurvey)]) > 0
    num_turns = 1 if not has_supernova else 2

    num_frames *= num_turns  # Atm extending running time if more turns. May remove

    # Get the redshift limits for each survey
    rs = [s.zmax for s in vis.surveys]

    # Calculate the camera radius (min and max) from these values
    if len(rs) == 1:
        maxr = 0.7 * max(rs)
        minr = maxr
    else:
        maxr = 0.7 * max(rs)
        minr = 0.7 * min(rs)

    vis.set_camera(OrbitZoomCamera(minr, maxr, num_turns=num_turns))

    # Using 4 cores, call make3d for each degree from 0 to 360
    Parallel(n_jobs=3)(delayed(make3d)(name, vis, i, num_frames, low_quality) for i in range(226, num_frames))


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


def get_permutations(full_data=False):
    """
    A helper function I call, that gives me a list of all the different plots I want
    to create with their data.
    """
    if full_data:
        w = WiggleZ()
        t = TwoDegreeField()
        s = SDSS()
        g = Gama()
        x = SixDegreefField()
        o = OzDES()
        o2 = OzDES()
        o3 = OzDES()
        o2.zmax = 0.7
        o3.zmax = 4.0
        l = Tdflens()
        p = Taipan()
        rs = RandomSupernovae()
        ozs = OzDESSupernovae()
        ozsa = OzDESSupernovaeAll()
        groups = [[w, t, s, g, x, o], [w, t, s, g, x], [w, t, s, g, x, o, ozs], w, t, g, s, x, o, l, [l, o2, t], p, [l, t, o3], [o2, ozs], [o2, ozsa]]
        names = ["all", "all_nooz", "all_supernova", "wigglez", "2df", "gama", "sdss", "6df", "ozdes", "2dflens", "sub", "taipan",
                 "ozdes_deep", "ozdes_nova", "ozdes_allnova"]
    else:
        t = TwoDegreeField()
        s = SDSS()
        x = SixDegreefField()
        groups = [[t, s, x], t, s, x]
        names = ["all_small", "2df", "sdss", "6df"]
    return groups, names


def make_figures(name=None):
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
    Parallel(n_jobs=4)(delayed(make)(n + ".png", g) for n, g in zip(names, groups) if name is None or name == n)


def make_all_video(name=None, low_quality=False, num_frames=360):
    """
    Makes all video series for all permutations of data that I want

    Parameters
    ----------
    name : str [optional]
        If None, makes all permutations. If a string,
        will only make the permutation matching with the same name.
        See output of `get_permutations` for a list of names
    """
    print("Making all videos")

    # Some data I cant release to github unfortunately
    full_data = os.path.exists("surveyvis/data/supernovae.npy")
    if full_data:
        print("Found full data")
    else:
        print("Found github data")

    # Note we dont use Parallel processing here, because make_video already uses it
    groups, names = get_permutations(full_data=full_data)
    for n, g in zip(names, groups):
        if name is None or name == n:
            make_video(n, g, low_quality=low_quality, num_frames=num_frames)


if __name__ == "__main__":
    # Uncomment the below two lines to do everything
    # make_figures()
    make_all_video()

    # As an example, make the 6df figures and video
    # make_figures("6df")

    # Uncomment one of the below lines (and comment out the above two)
    # to make only the plot declared

    # make_figures("ozdes_deep")
    # make_figures("ozdes")
    # make_figures("sub")
    # make_figures("all")
    # make_all_video("6df")
    # make_figures("ozdes")
    # make_all_video("ozdes_nova")
    # make_all_video("ozdes_allnova")
    # make_all_video("all_supernova")
