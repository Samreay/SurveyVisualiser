import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
from surveyvis.surveys import SupernovaeSurvey
from surveyvis.camera import Camera
import gc

class Visualisation(object):
    def __init__(self):
        """
        The workhorse class, which is responsible for accepting data and
        exposing a variety of methods which can be used to plot the data
        """

        self.surveys = []  # A list of all surveys to plot
        self.camera = None

        # Various background colours for our figures and axes
        self.plot_background_color = (0, 0, 0, 0)
        self.axis_background_color = (0, 0, 0, 1)
        self.plot_second_background_color = (0, 0, 0, 1)

        # Theta grid for plotting the hours (1hr = 15deg)
        self.theta_grid = np.linspace(0, 2, 24, endpoint=False) * 180
        self.theta_labels = [r"$%d^{\rm hr}$" % i for i in range(24)]

        # z grid for when we plot redshift lines on top of everything
        self.z_grid = [0.25, 0.5, 0.75, 1.0]
        self.z_labels = ["$z=0.25$", "$z=0.50$", "$z=0.75$", "$z=1.00$"]

    def add_survey(self, survey):
        """
        Adds the survey to the list of surveys to get plotted

        Parameters
        ----------
        survey : Survey
            The survey to add
        """
        self.surveys.append(survey)

    def set_camera(self, camera):
        assert isinstance(camera, Camera), "This is not a camera!"
        self.camera = camera

    def render3d(self, filename, ratio, low_quality=False):

        """
        Render a 3D still to file

        Parameters
        ----------
        filename : str
            The filename to save the plot to
        """
        image_ratio = 1920.0 / 1080.0

        azim, elev, rmax = self.camera.get_azim_elevation_radius(ratio)

        time_bounds = np.array([s.get_time_range() for s in self.surveys if isinstance(s, SupernovaeSurvey)])
        if len(time_bounds) > 0:
            t_min, t_max = np.min(time_bounds), np.max(time_bounds)
        else:
            t_min = 0
            t_max = 1
        time = t_min + (t_max - t_min) * ratio

        if low_quality:
            print("Making low quality")
            s_size = 4
            layers = 2
        else:
            print("Making high quality")
            s_size = 10
            layers = 20

        size = (s_size, s_size)
        pixel_height = 192 * s_size / image_ratio
        finsize = (s_size, s_size / image_ratio)

        # Create a figure
        ax, fig = self._get_3d_fig(azim, elev, rmax, size)

        # Draw the background
        fig.canvas.draw()

        # Steal the RGB canvas from matplotlib, and turn it to int16 so we dont overflow
        w, h = fig.canvas.get_width_height()
        first = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy()
        first[first[:, :, -1] == 0] = 0  # If alpha = 0, throw out colour data
        stacked = np.zeros(first.shape)

        # Get a series of images, one per layer, so that we can stack them
        imgs = []
        for i in range(layers):
            # Reset the axis (clear previous layer)
            self._reset_axis(ax, rmax)

            # Plot each survey
            for s in self.surveys:
                if isinstance(s, SupernovaeSurvey):
                    ratio_i = max(0.01, ((i + 1) / layers) ** 3)
                    colors = s.get_colors(time, layers=layers)
                    size = s.get_size(time)
                    ax.scatter(s.xs, s.ys, s.zs, lw=0, s=size * ratio_i, c=colors)

                else:
                    ax.scatter(s.xs[i::layers], s.ys[i::layers], s.zs[i::layers], lw=0, alpha=0.9 * s.alpha,
                               s=0.3 * s.size * s.zmax / rmax, c=s.color)

            # Render the plot and then steal the RGB buffer again
            fig.canvas.draw()
            imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy())
        fig.clf()
        plt.close()
        gc.collect()

        fig.clf()
        plt.close()
        gc.collect()

        # Stack all the images, so that we have two stacked layers, "first" and "stacked"
        for img in imgs:
            img[img[:, :, -1] == 0] = 0
            first += img
            stacked += img

        if not low_quality:
            first = self._blur(first, stacked)

        # And reclip
        first = np.clip(first, 0, 255)
        # Turn it back to 8bits
        first = first.astype(np.uint8)

        # Crop out the top and bottom so we get the right aspect ratio for our video
        i1 = int(w // 2 - pixel_height // 2)
        i2 = int(w // 2 + pixel_height // 2)
        first = first[i1:i2, :, :]

        # Create a new plot, where we will plot our final image, which is called "first"
        self._render_final_layer(filename, finsize, first)

    def _reset_axis(self, ax, rmax):
        ax.clear()
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)

    def _render_final_layer(self, filename, finsize, first):
        fig, ax = plt.subplots(figsize=finsize, dpi=192, frameon=False)
        plt.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(first, aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Save this out to file.
        fig.savefig(filename, dpi=192, bbox_inches=extent, pad_inches=0, transparent=True)
        fig.clf()
        plt.close()
        gc.collect()
        print("Saved to %s" % filename)

    def _get_3d_fig(self, azim, elev, rmax, size):
        fig = plt.figure(figsize=size, dpi=192, facecolor=self.plot_background_color, frameon=False)
        ax = fig.add_subplot(111, projection='3d', axisbg=self.axis_background_color)
        # Set background colours and screw around with things
        plt.gca().patch.set_facecolor((0, 0, 0, 0))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        ax.w_xaxis.set_pane_color(self.axis_background_color)
        ax.w_yaxis.set_pane_color(self.axis_background_color)
        ax.w_zaxis.set_pane_color(self.axis_background_color)
        # Set viewing limits and camera
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)
        ax.set_aspect("equal")
        return ax, fig

    def _blur(self, first, stacked):
        # Clip the values between 0 and 255 so we can turn back to 8bits
        first = np.clip(first, 0, 255)
        stacked = np.clip(stacked, 0, 255)

        # Resize stacked so it is 25% of its original size, because this step is *slow*
        stacked = imresize(stacked, 25)
        # Run a gaussian filter of the layer to blur it, to simulate glow of some sort. Blue R, G, B, alpha individually
        smoothed = np.dstack(
            [gaussian_filter(stacked[:, :, i], sigma=4, truncate=3) for i in range(stacked.shape[2])])
        # Now blur the colours togeter
        s2 = gaussian_filter(stacked, sigma=10, truncate=3)
        # Modify blur ratios (non-colour blur to colour blur)
        add = np.floor(0.5 * smoothed + s2)
        # Scale it back up to size
        add = imresize(add, 400)
        # Reclip it and decrease intensity to 40%
        add = np.clip(add * 0.4, 0, 255)
        # Turn it back to int16 so we can add it (gaussian_filter makes it all doubles)
        add = add.astype(np.int16)
        # Mask out 0 alpha values
        add[add[:, :, -1] == 0] = 0

        # Finally, add in our glow
        first += add
        return first

    def render_latex(self, filename, grid=True, grid_color="#333333", theta_color="#111111", outline=True, dpi=600):
        """
        Render out a latex picture to file

        Parameters
        ----------
        filename : str
            The filename to save the plot to
        grid : boolean [optional]
            Whether or not to plot a theta and z grid on top
        grid_color : str (hex colour), [optional]
            The colour to have the tick labels
        theta_color : str (hex colour), [optional]
            The colour to have the theta grid
        outline : boolean, [optional]
            Whether or not to draw a circular outline on the plot
        dpi : int, [optional]
            The dpi to render out as.

        """

        # Decalre a function formatter to print redshifts correctly
        def formatter(x, p):
            return "$z=%.2f$" % x

        # A lot of this code is duplicated from the 3D steps, so I will be briefer in my explanations here.

        # rmax values for plot limits
        rmax = max([s.zmax for s in self.surveys])

        # Get figure, axes and set them up
        fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor=self.plot_background_color, frameon=False)
        ax = fig.add_subplot(111, projection='polar', axisbg=self.axis_background_color)
        ax.clear()
        ax.set_rlim(0, rmax)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="lower"))
        ax.grid(grid)
        ax.set_thetagrids(self.theta_grid, frac=1.06)
        ax.set_rlabel_position(90)
        ax.xaxis.label.set_color(grid_color)
        ax.yaxis.label.set_color(grid_color)
        ax.tick_params(axis='x', colors=theta_color)
        ax.tick_params(axis='y', colors=grid_color)
        ax.grid(color=grid_color)
        ax.spines['polar'].set_color(grid_color)
        ax.spines['polar'].set_visible(outline)
        ax.set_xticklabels(self.theta_labels, fontsize=14)
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter))
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('center')
            tick.label1.set_verticalalignment('top')

        # For each survey, plot the points *in black*
        for s in self.surveys:
            ax.scatter(s.ra, s.z, lw=0, alpha=0.7 * s.alpha, s=s.size * np.power(s.zmax / rmax, 1.7), c='k')

        # Save out the figure with minimal borders
        plt.tight_layout()
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1, transparent=True)

    def render2d(self, filename, grid=True, grid_color="#AAAAAA", theta_color="#111111", outline=True, backplot=True,
                 layers=20, dpi=600):
        """
        Render out a colour 2D picture to file

        Parameters
        ----------
        filename : str
            The filename to save the plot to
        grid : boolean [optional]
            Whether or not to plot a theta and z grid on top
        grid_color : str (hex colour), [optional]
            The colour to have the tick labels
        theta_color : str (hex colour), [optional]
            The colour to have the theta grid
        outline : boolean, [optional]
            Whether or not to draw a circular outline on the plot
        backplot : boolean, [optional]
            Whether to make each dot opaque or all them to be translucent
        layers : int, [optional]
            How many layers to use for additive blending
        dpi : int, [optional]
            The dpi to render out as.

        """

        # Custom label formatter
        def formatter(x, p):
            return "$z=%.2f$" % x

        # Plot limits
        rmax = max([s.zmax for s in self.surveys])

        # Get new plot and set it up
        fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor=self.plot_background_color, frameon=False)
        ax = fig.add_subplot(111, projection='polar', axisbg=self.axis_background_color)
        ax.clear()
        ax.set_rlim(0, rmax)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="lower"))
        ax.grid(grid)
        ax.set_thetagrids(self.theta_grid, frac=1.06)
        ax.set_rlabel_position(90)
        ax.xaxis.label.set_color(grid_color)
        ax.yaxis.label.set_color(grid_color)
        ax.tick_params(axis='x', colors=theta_color)
        ax.tick_params(axis='y', colors=grid_color)
        ax.grid(color=grid_color)
        ax.spines['polar'].set_color(grid_color)
        ax.spines['polar'].set_visible(outline)
        ax.set_xticklabels(self.theta_labels, fontsize=14)
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter))
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('center')
            tick.label1.set_verticalalignment('top')
        plt.tight_layout()

        # If backplot, render a black background behind each dot
        if backplot:
            for s in self.surveys:
                ax.scatter(s.ra, s.z, lw=0, alpha=1, s=s.size * s.zmax / rmax, c="black")

        # Draw the canvas, and steal the RBGA image buffer
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        fig.canvas.draw()
        first = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy()
        first[first[:, :, -1] == 0] = 0
        stacked = np.zeros(first.shape)

        # Render out each layer
        imgs = []
        for i in range(layers):
            # Reset axis
            ax.clear()
            ax.patch.set_facecolor(self.plot_second_background_color)
            ax.set_thetagrids(self.theta_grid, frac=1.06)
            # ax.set_rgrids(self.z_grid, angle=90)
            ax.set_rlabel_position(90)
            ax.tick_params(axis='x', colors=(0, 0, 0, 0))
            ax.tick_params(axis='y', colors=(0, 0, 0, 0))
            ax.grid(color=grid_color)
            ax.spines['polar'].set_color(grid_color)
            ax.spines['polar'].set_visible(outline)
            ax.set_xticklabels(self.theta_labels, fontsize=14)
            # Currently have commented out code to render the redshift labels because it was clutter
            # ax.set_yticklabels(self.z_labels, fontsize=12)
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter))
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_horizontalalignment('center')
                tick.label1.set_verticalalignment('top')
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_rlim(0, rmax)

            # Now render out each survey
            for s in self.surveys:
                ax.scatter(s.ra[i::layers], s.z[i::layers], lw=0, alpha=s.alpha,
                           s=s.size * np.power(s.zmax / rmax, 1.7), c=s.color)

            # Update plot layout and colours, and then render it again
            plt.tight_layout()
            ax.tick_params(axis='x', colors=(0, 1, 0, 0))
            ax.tick_params(axis='y', colors=(0, 0, 0, 0))
            fig.canvas.draw()

            # Steal the image buffer
            imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy())

        # Stack the layers
        for img in imgs:
            img[img[:, :, -1] == 0] = 0
            stacked += img
            first += img

        # Clip them to 8bit
        stacked = np.clip(stacked, 0, 255)
        first = np.clip(first, 0, 255)

        # As detailed in the 3D algorithm, we fake some blur to get something like glow
        smoothed = np.dstack([gaussian_filter(stacked[:, :, i], sigma=10, truncate=3) for i in range(stacked.shape[2])])
        s2 = gaussian_filter(stacked, sigma=10, truncate=3) * 0.2
        m = np.mean(smoothed[:, :, :3], axis=2)
        smoothed = np.dstack((m, m, m, smoothed[:, :, -1]))
        add = (0.5 * np.floor(smoothed.astype(np.float32) + s2)).astype(np.int16)
        add[add[:, :, -1] == 0] = 0

        # Add the blur, clip and convert to 8bits
        first += add
        first = np.clip(first, 0, 255)
        first = first.astype(np.uint8)

        # Create an empty figure for us to put our stacked layer
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi, frameon=False)
        plt.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(first, aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # And save it off
        fig.savefig(filename, dpi=dpi, bbox_inches=extent, pad_inches=0, transparent=True)
        # If we're saving a PNG, also save a lower resolution image to give options.
        if ".png" in filename:
            fig.savefig(filename.replace(".png", "_small.png"), dpi=dpi // 2, bbox_inches=extent, pad_inches=0,
                        transparent=True)
