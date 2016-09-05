import numpy as np


class Survey(object):
    def __init__(self, ra, dec, z, zmax=1.0):
        self.ra = ra
        self.dec = dec
        self.z = z
        self.zmax = zmax

        dra = 0
        ddec = -1.0
        self.zs = np.sin(ra + dra) * np.cos(dec + ddec) * z
        self.xs = np.sin(ra + dra) * np.sin(dec + ddec) * z
        self.ys = np.cos(ra + dra) * z
        self.size = 1.0
        self.alpha = 0.5
        self.color = "#1E3B9C"


class Dummy(Survey):
    def __init__(self):
        rag, decg = np.meshgrid(np.linspace(0, 360, 100), np.linspace(0, -90, 100))
        ra = rag.flatten()
        dec = decg.flatten()
        z = np.ones(ra.shape)
        super().__init__(ra * np.pi / 180, dec * np.pi / 180, z, zmax=1.4)
        self.size = 5.0


class Dummy2(Survey):
    def __init__(self):
        rag, decg = np.meshgrid(np.linspace(0, 360, 100), np.linspace(0, -90, 100))
        ra = rag.flatten()
        dec = decg.flatten()
        z = 0.1 * np.ones(ra.shape)
        super().__init__(ra * np.pi / 180, dec * np.pi / 180, z, zmax=1.4)
        self.size = 3.0


class WiggleZ(Survey):
    def __init__(self):
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
        # self.size = 1.1
        # self.alpha = 1.0


class Taipan(Survey):
    def __init__(self):
        data = np.load("surveyvis/data/taipan.npy")
        super().__init__(data[:, 0] * np.pi / 180, data[:, 1] * np.pi / 180, data[:, 2], zmax=0.2)
        self.size = 1.0
        self.alpha = 0.4
