import numpy as np
import sncosmo as snc
from astropy.table import Table as table


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


class SupernovaSurvey(Survey):
    """
    The class that contains supernovae
    """

    def __init__(self):

        #Load the data from the file
        data = np.load("surveyvis/data/supernovae.npy")
        ra=data[:, 0] * np.pi / 180
        dec=data[:, 1] * np.pi / 180
        z=data[:, 2]

        ts=data[:, 3]
        mb=data[:, 4]
        x0=np.exp(-0.9209*mb+9.7921)
        stretch=data[:, 5]
        col=data[:, 6]

        #Generate class
        super().__init__(ra, dec, z, zmax=0.85)
        self.ts = ts #Peak Times

        self.stretch=stretch
        self.mb=mb
        self.color=col
        self.x0=x0

        #Create empty flux arrays
        self.colnames=['r', 'g', 'b', 'i', 'u']

        self.flux_r = np.array([[]])
        self.flux_g = np.array([[]])
        self.flux_b = np.array([[]])
        self.flux_i = np.array([[]])
        self.flux_u = np.array([[]])

        self.fluxes=[self.flux_r, self.flux_g, self.flux_b, self.flux_i, self.flux_u]

        self.t_line = np.array([]) #Time line used to generate colors. Needs to be set to do generate color arrays

    def get_color(self, t, colname, redshift=True):
        #Returns a vector of fluxes in a particular band for all supernovae at a specific time

        #For single frame renders, check if time has not been set
        if len(self.t_line)==0:
            print("Time Line Not set. Rending for t=",t)
            self.t_line=np.array([t])
            self.set_all_colors(redshift=redshift)

        #Aquire the flux array
        for name,arr in zip(self.colnames,self.fluxes):
            if name==colname:
                fluxarray=arr
                break


        #Reading time to get fluxes
        if t in self.t_line:
            #Read Direct
            for tl,fluxrow in zip(self.t_line, fluxarray):
                if tl==t:
                    return(fluxrow)

        elif t>min(self.t_line) and t<max(self.t_line):
            #Linearly Interpolate
            i2=np.argmax(self.t_line>t)
            i1=i2-1

            return(fluxarray[:,i1]+(t-self.t_line[i1])/(self.t_line[i2]-self.t_line[i1])*(fluxarray[:,i2]-fluxarray[:,i1]))

        else:
            print("Possible time-domain error on flux get. Returning Zero Flux")
            return( np.zeros([len( self.ra )]) )

    def set_color(self, colname, redshift=True):
        #Use self.t_line to generate color array

        #Get the array you're working with:
        for name, index in zip(self.colnames, np.arange( len(self.colnames) )):
            if name==colname:
                fluxarray = self.fluxes[index]
                print("Setting Supernovae Colours for ",name)
                break

        #Make array empty and correct shape
        self.fluxes[index]=np.zeros([len(self.t_line),len(self.ra)])

        model = snc.Model(source='salt2')

        #Get band filter name to work with sncosmo
        bands = ["bessellr", "bessellv", "bessellb", "besselli", "bessellux"]
        for bn,cn in zip(bands,self.colnames):
            if colname==cn:
                bandname=bn
                break

        N=len(self.t_line)

        band = [bandname] * N
        gain = [1] * N
        skynoise = [0] * N
        zp = [0] * N
        zpsys = ['ab'] * N

        obs = table({'time': self.t_line,
                     'band': band,
                     'gain': gain,
                     'skynoise': skynoise,
                     'zp': zp,
                     'zpsys': zpsys})

        #Itterate over each supernovae and get the light curve

        for i,zi,tsi,x0i,x1i,ci, in zip(np.arange(len(self.z)), self.z, self.ts, self.x0, self.xs, self.color):

            if redshift==False:
                zi=0

            params = {'z': zi, 't0': tsi, 'x0': x0i, 'x1': x1i, 'c': ci}
            try:
                lcs = snc.realize_lcs(obs, model, [params], scatter=False)
                fcol=np.array(lcs[0]["flux"])
            except:
                fcol=np.zeros([len(self.t_line)])

            self.fluxes[index][:,i]=fcol

    def set_all_colors(self, redshift=True):
        for name in self.colnames:
            self.set_color(name, redshift=redshift)




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
