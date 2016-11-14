import numpy as np


class Survey(object):
    def __init__(self, ra, dec, z, zmax=1.0):
        """
        An object to hold coordinates and plotting information

        Parameters
        ----------
        ra : np.ndarray
            A 1D numpy array of all RA positions on the sky
        dec : np.ndarray
            A 1D numpy array of all the Declination positions on the sky
        z : np.ndarray
            A 1D numpy array of all the redshifts
        zmax : float [optional]
            The redshift at which the camera should orbit
        """

        self.ra = ra
        self.dec = dec
        self.z = z
        self.zmax = zmax

        # Conver to Cartesian
        dec = np.pi / 2 - dec
        self.ys = np.cos(ra) * np.sin(dec) * z
        self.xs = np.sin(ra) * np.sin(dec) * z
        self.zs = np.cos(dec) * z

        # Some plotting values you can override. Scatter size, alpha and colour.
        self.size = 1.0
        self.alpha = 0.5
        self.color = "#1E3B9C"


class Dummy(Survey):
    def __init__(self):
        """
        A dummy survey that has points uniformly distributed on the southern hemisphere.

        Useful for debugging plots and aspect ratios.
        """
        rag, decg = np.meshgrid(np.linspace(0, 360, 100), np.linspace(0, -90, 100))
        ra = rag.flatten()
        dec = decg.flatten()
        z = np.ones(ra.shape)
        super().__init__(ra * np.pi / 180, dec * np.pi / 180, z, zmax=1.4)
        self.size = 5.0


class Dummy2(Survey):
    def __init__(self):
        """
        A dummy survey that has points uniformly distributed on the southern hemisphere, but at
        a different redshift than the `Dummy` class.

        Useful for debugging plots and aspect ratios.
        """
        rag, decg = np.meshgrid(np.linspace(0, 360, 100), np.linspace(0, -90, 100))
        ra = rag.flatten()
        dec = decg.flatten()
        z = 0.1 * np.ones(ra.shape)
        super().__init__(ra * np.pi / 180, dec * np.pi / 180, z, zmax=1.4)
        self.size = 3.0


class WiggleZ(Survey):
    def __init__(self):
        """
        The WiggleZ survey. Data is stored in an (n x 3) matrix as a file, with the columns being
        RA, DEC, and z, with RA and DEC in degrees. We open the file, convert to radians, and
        initialise the superclass with the right variables.
        """
        data = np.load("surveyvis/data/wigglez.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.0)


class TwoDegreeField(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/2df.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.3)
        self.color = "#941313"
        self.alpha = 0.7
        self.size = 1.0


class Gama(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/gama.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.5)
        self.color = "#28AD2C"
        self.size = 0.5
        self.alpha = 0.2


class SDSS(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/sdss.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.2)
        self.color = "#BF40CF"
        self.size = 0.8
        self.alpha = 0.2


class SixDegreefField(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/6df.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.15)
        self.color = "#e2a329"


class OzDES(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/ozdes.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.6)
        self.size = 1.1
        self.alpha = 1.0
        self.color = "#da6016"


class Tdflens(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/tdflens.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.0)


class Taipan(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/taipan.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.2)
        self.size = 1.0
        self.alpha = 0.4
