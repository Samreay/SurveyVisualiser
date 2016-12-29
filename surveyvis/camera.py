from abc import ABCMeta, abstractmethod
import numpy as np


class Camera(metaclass=ABCMeta):

    @abstractmethod
    def get_azim_elevation_radius(self, ratio):
        pass


class OrbitZoomCamera(Camera):
    def __init__(self, min_radius, max_radius, num_turns=1, zoom_loc=200, zoom_width=140):
        self.num_turns = num_turns
        self.zoom_loc = zoom_loc
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.zoom_width = zoom_width

    def get_azim_elevation_radius(self, ratio):
        rad = ratio * 2 * np.pi * self.num_turns

        azim = (180 / np.pi) * np.remainder(rad, 2 * np.pi)
        elev = -(30 + 30 * np.cos(rad))
        d = min(np.abs(self.zoom_loc - azim), np.abs(2 * np.pi + azim - self.zoom_loc))
        radius = self.max_radius - (self.max_radius - self.min_radius) * (1 - np.exp(-(d / self.zoom_width) ** 2))
        print(azim, elev)
        return azim, elev, radius
