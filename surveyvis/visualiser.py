import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize


class Visualisation(object):
    def __init__(self):
        self.surveys = []
        self.plot_background_color = (0, 0, 0, 0)
        self.axis_background_color = (0, 0, 0, 1)
        self.plot_second_background_color = (0, 0, 0, 1)

        self.theta_grid = np.linspace(0, 2, 24, endpoint=False) * 180
        self.theta_labels = [r"$%d^{\rm hr}$" % i for i in range(24)]

        self.z_grid = [0.25, 0.5, 0.75, 1.0]
        self.z_labels = ["$z=0.25$", "$z=0.50$", "$z=0.75$", "$z=1.00$"]

    def add_survey(self, survey):
        self.surveys.append(survey)

    def render3d(self, filename, rmax=None, elev=60, azim=70, layers=20):
        if rmax is None:
            rmax = 0.4 * max([s.zmax for s in self.surveys])
        fig = plt.figure(figsize=(10,10), dpi=192, facecolor=self.plot_background_color, frameon=False)
        # fig = plt.figure(figsize=(10, 5.625), dpi=192, facecolor=self.plot_background_color, frameon=False)

        ax = fig.add_subplot(111, projection='3d', axisbg=self.axis_background_color)
        plt.gca().patch.set_facecolor((0, 0, 0, 0))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        ax.w_xaxis.set_pane_color(self.axis_background_color)
        ax.w_yaxis.set_pane_color(self.axis_background_color)
        ax.w_zaxis.set_pane_color(self.axis_background_color)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)
        ax.set_aspect("equal")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        fig.canvas.draw()
        first = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy()
        first[first[:, :, -1] == 0] = 0
        stacked = np.zeros(first.shape)

        imgs = []
        for i in range(layers):
            ax.clear()
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_xlim(-rmax, rmax)
            ax.set_ylim(-rmax, rmax)
            ax.set_zlim(-rmax, rmax)
            for s in self.surveys:
                ax.scatter(s.xs[i::layers], s.ys[i::layers], s.zs[i::layers], lw=0, alpha=0.9*s.alpha, s=0.3 * s.size * s.zmax / rmax, c=s.color)

            fig.canvas.draw()
            imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy())

        for img in imgs:
            img[img[:, :, -1] == 0] = 0
            first += img
            stacked += img
        first = np.clip(first, 0, 255)
        stacked = np.clip(stacked, 0, 255)
        stacked = imresize(stacked, 25)
        smoothed = np.dstack([gaussian_filter(stacked[:, :, i], sigma=4, truncate=3) for i in range(stacked.shape[2])])
        s2 = gaussian_filter(stacked, sigma=10, truncate=3)
        add = np.floor(0.5 * smoothed + s2)
        add = imresize(add, 400)
        add = np.clip(add * 0.4, 0, 255)
        add = add.astype(np.int16)
        add[add[:, :, -1] == 0] = 0
        first += add
        first = np.clip(first, 0, 255)

        first = first.astype(np.uint8)
        i1 = w//2 - 1080//2
        i2 = w//2 + 1080//2
        first = first[i1:i2, :, :]

        fig, ax = plt.subplots(figsize=(10, 5.625), dpi=192, frameon=False)
        plt.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(first, aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=192, bbox_inches=extent, pad_inches=0, transparent=True)
        print("Saved to %s" % filename)

    def render_latex(self, *filenames, grid=True, grid_color="#333333", theta_color="#111111", outline=True, dpi=600):
        def formatter(x, p):
            return "$z=%.2f$" % x
        rmax = max([s.zmax for s in self.surveys])
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

        for s in self.surveys:
            ax.scatter(s.ra, s.z, lw=0, alpha=0.7 * s.alpha, s=s.size * np.power(s.zmax / rmax, 1.7), c='k')
        plt.tight_layout()
        for filename in filenames:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1, transparent=True)

    def render2d(self, *filenames, grid=True, grid_color="#AAAAAA", theta_color="#111111", outline=True, backplot=True, layers=20, dpi=600):
        def formatter(x, p):
            return "$z=%.2f$" % x
        rmax = max([s.zmax for s in self.surveys])
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

        if backplot:
            for s in self.surveys:
                ax.scatter(s.ra, s.z, lw=0, alpha=1, s=s.size*s.zmax/rmax, c="black")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        fig.canvas.draw()
        first = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy()

        first[first[:, :, -1] == 0] = 0
        stacked = np.zeros(first.shape)
        imgs = []
        for i in range(layers):
            ax.clear()
            ax.patch.set_facecolor(self.plot_second_background_color)
            ax.set_thetagrids(self.theta_grid, frac=1.06)
            #ax.set_rgrids(self.z_grid, angle=90)
            ax.set_rlabel_position(90)
            ax.tick_params(axis='x', colors=(0, 0, 0, 0))
            ax.tick_params(axis='y', colors=(0, 0, 0, 0))
            ax.grid(color=grid_color)

            ax.spines['polar'].set_color(grid_color)
            ax.spines['polar'].set_visible(outline)

            ax.set_xticklabels(self.theta_labels, fontsize=14)
            #ax.set_yticklabels(self.z_labels, fontsize=12)
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter))
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_horizontalalignment('center')
                tick.label1.set_verticalalignment('top')

            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_rlim(0, rmax)
            for s in self.surveys:
                ax.scatter(s.ra[i::layers], s.z[i::layers], lw=0, alpha=s.alpha, s=s.size*np.power(s.zmax / rmax, 1.7), c=s.color)

            plt.tight_layout()
            ax.tick_params(axis='x', colors=(0, 1, 0, 0))
            ax.tick_params(axis='y', colors=(0, 0, 0, 0))
            fig.canvas.draw()
            # fig.savefig(filenames[0].replace(".png", "2.png"))
            imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1).copy())

        for img in imgs:
            img[img[:, :, -1] == 0] = 0
            stacked += img
            first += img
            # first[:, :, -1] += 2 * img[:, :, -1]
        stacked = np.clip(stacked, 0, 255)
        first = np.clip(first, 0, 255)

        smoothed = np.dstack([gaussian_filter(stacked[:, :, i], sigma=10, truncate=3) for i in range(stacked.shape[2])])
        s2 = gaussian_filter(stacked, sigma=10, truncate=3) * 0.2
        m = np.mean(smoothed[:, :, :3], axis=2)
        smoothed = np.dstack((m, m, m, smoothed[:, :, -1]))
        add = (0.5 * np.floor(smoothed.astype(np.float32) + s2)).astype(np.int16)
        add[add[:, :, -1] == 0] = 0
        first += add
        first = np.clip(first, 0, 255)

        first = first.astype(np.uint8)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi, frameon=False)
        plt.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(first, aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        for filename in filenames:
            fig.savefig(filename, dpi=dpi, bbox_inches=extent, pad_inches=0, transparent=True)
