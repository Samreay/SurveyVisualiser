import numpy as np


class Survey(object):
    def __init__(self, ra, dec, z, zmax=None):
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
        zmax = zmax or np.max(z)
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


class StaticSurvey(Survey):
    def __init__(self, ra, dec, z, zmax=1.0):
        super().__init__(ra, dec, z, zmax=zmax)


class SupernovaeSurvey(Survey):
    """
    The class that contains supernovae
    """

    def __init__(self, ra, dec, z, ts, mb, x1, c, zmax=None, redshift=False):

        super().__init__(ra, dec, z, zmax=zmax)

        self.ts = ts
        self.x1s = x1
        self.mbs = mb
        self.cs = c
        self.redshift = redshift

        x0 = np.exp(-0.9209 * mb + 9.7921)  # Note that we dont show varying mB this is mostly just for sncosmo
        self.x0s = x0

    def get_time_range(self):
        return np.min(self.ts) - 50, np.max(self.ts) + 100

    def get_colors(self, time, style="ivu3", layers=1):
        import sncosmo
        model = sncosmo.Model(source='salt2')

        # Get band filter name to work with sncosmo
        bands = ["bessellr", "bessellv", "bessellb", "besselli", "bessellux"]

        colours = []
        for x0, x1, c, t, z in zip(self.x0s, self.x1s, self.cs, self.ts, self.z):
            if not self.redshift:
                z = 0.2
            # z = 0.05
            model.set(z=z, t0=t, x0=x0, x1=x1, c=c)
            fluxes = []
            for b in bands:
                try:
                    flux = model.bandflux(b, time)
                except ValueError:
                    flux = 0
                fluxes.append(flux)
            colour = self.get_color_from_bands(fluxes, style)
            colours.append(colour)
        colours = np.array(colours)
        # Now get alphas
        length_peak = 1
        min_t = -15
        diff = time - self.ts
        alphas = []
        for d in diff:
            if d < min_t:
                alphas.append(0)
            elif d < length_peak:
                alphas.append(1)
            else:
                alphas.append(np.exp(-(d - length_peak) / 2))

        alphas = np.array(alphas) / np.sqrt(layers)
        rgba = np.hstack((colours, alphas[:, None]))
        return rgba

    def get_color_from_bands(self, fluxes, style):
        vr, vg, vb, ir, uv = tuple(fluxes)

        if style == 'rgb':
            r = vr
            g = vg
            b = vb
        elif style == 'ivu1':
            r = ir
            g = vg
            b = uv
        elif style == 'ivu2':
            r = ir
            g = (vb + vg + vr) / 3
            b = uv / 2
        elif style == 'ivu3':
            r = vg / 2 + vr + ir
            g = vb / 2 + vg + 2 / 3 * vr
            b = ir + vb + 1 / 2 * uv
        elif style == 'rbslide':
            r = ir + (vr + vg + vb) / 2
            g = (vr + vg + vb) / 2
            b = (vr + vg + vb) / 2 + uv
        elif style == 'rbslide2':
            r = ir + vr + (vr + vg + vb) / 2
            g = (vr + vg + vb) / 2
            b = (vr + vg + vb) / 2 + vb + uv

        rgb = np.array([r, g, b])  # Flux Array
        rgb += 0.2 * np.max(rgb)
        rgb *= rgb > 0
        if not np.max(rgb) == 0:
            rgb /= np.max(rgb)
        return rgb

    def get_size(self, time):
        diff = time - self.ts
        t0_size = 15
        size = (diff < 0) * np.exp(diff / 5)
        size += (diff > 0) * (1 + diff * 0.03)
        return t0_size * size


class RandomSupernovae(SupernovaeSurvey):
    def __init__(self, n=500):

        z = np.random.uniform(0.1, 1.0, n)
        ra = np.random.uniform(0, 2 * np.pi, n)
        dec = np.random.uniform(-np.pi / 2, np.pi / 2, n)
        ts = np.random.uniform(5000, 5500, n)

        mb = np.random.normal(20, 1, n)  # This universe has some odd cosmology
        x1 = np.random.normal(0, 0.8, n)
        c = np.random.normal(0, 0.1, n)

        super().__init__(ra, dec, z, ts, mb, x1, c)


class OzDESSupernovae(SupernovaeSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/supernovae.npy")

        ra = data[:, 0] * np.pi / 180
        dec = data[:, 1] * np.pi / 180
        z = data[:, 2]

        ts = data[:, 3]
        mb = data[:, 4]
        x1 = data[:, 5]
        c = data[:, 6]

        super().__init__(ra, dec, z, ts, mb, x1, c)


class OzDESSupernovaeAll(SupernovaeSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/supernovae.npy")

        ra = data[:, 0] * np.pi / 180
        dec = data[:, 1] * np.pi / 180
        z = data[:, 2]

        ts = data[:, 3]
        mb = data[:, 4]
        x1 = data[:, 5]
        c = data[:, 6]

        super().__init__(ra, dec, z, ts, mb, x1, c)


class Dummy(StaticSurvey):
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


class Dummy2(StaticSurvey):
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


class WiggleZ(StaticSurvey):
    def __init__(self):
        """
        The WiggleZ survey. Data is stored in an (n x 3) matrix as a file, with the columns being
        RA, DEC, and z, with RA and DEC in degrees. We open the file, convert to radians, and
        initialise the superclass with the right variables.
        """
        data = np.load("surveyvis/data/wigglez.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.0)


class TwoDegreeField(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/2df.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.3)
        self.color = "#941313"
        self.alpha = 0.7
        self.size = 2.0


class Gama(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/gama.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.5)
        self.color = "#28AD2C"
        self.size = 1
        self.alpha = 0.2


class SDSS(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/sdss.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.2)
        self.color = "#BF40CF"
        self.size = 0.8
        self.alpha = 0.2


class SixDegreefField(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/6df.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.15)
        self.color = "#e2a329"
        self.size = 2

class OzDES(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/ozdes.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.6)
        self.size = 1.0
        self.alpha = 1.0
        self.color = "#da6016"


class Tdflens(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/tdflens.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=1.0)


class Taipan(StaticSurvey):
    def __init__(self):
        data = np.load("surveyvis/data/taipan.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.2)
        self.size = 1.0
        self.alpha = 0.4
