# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
import cv2

WINDOW_FLAGS = cv2.WINDOW_NORMAL  # cv2.WINDOW_AUTOSIZE


class NFOV():
    def __init__(self, fov_deg, fov_lens_horiz_deg, height=1080, width=1920):
        self.fov_rad = np.deg2rad(fov_deg)
        self.lens_fov_horiz_rad = np.deg2rad(fov_lens_horiz_deg)
        self.lens_fov_vert_rad = self.lens_fov_horiz_rad / 16 * 9
        self.height = height
        self.width = width
        self.tilt = 50

    @property
    def FOV(self):
        return np.array([self.fov_rad, self.fov_rad / 16 * 9])

    # @property
    # def lens_fov_vert_rad(self):
    #     return self.lens_fov_horiz_rad / 16 * 9

    @property
    def limits(self):
        return np.array([self.lens_fov_horiz_rad, self.lens_fov_vert_rad]) / 2

    def _screen2spherical(self, coord_screen):
        """ In range: [0, 1], out range: [-FoV_lens/2, FoV_lens/2] """
        return (coord_screen * 2 - 1) * self.limits

    def _spherical2screen(self, coord_spherical):
        """ In range: [-FoV_lens/2, FoV_lens/2], out range: [0, 1] """
        x, y = coord_spherical.T
        horiz_limit, vert_limit = self.limits
        x = (x / horiz_limit + 1.) * 0.5
        y = (y / vert_limit + 1.) * 0.5
        return np.array([x, y]).T

    def _get_frame_spherical_fov(self):
        frame_screen = self._get_coords_screen_frame()
        frame_spherical = self._screen2spherical(frame_screen)
        frame_spherical_fov = frame_spherical * self.FOV / 2
        return frame_spherical_fov

    def _get_coords_screen_frame(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width),
                             np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _gnomonic_forward(self, coord_spherical, center=None):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        lambda_rad = coord_spherical.T[0]
        phi_rad = coord_spherical.T[1]
        if center is None:
            center_pan_rad = -self.cp[0]
            center_tilt_rad = -self.cp[1]
        else:
            center_pan_rad, center_tilt_rad = center

        cos_c = np.sin(center_tilt_rad) * np.sin(phi_rad) + np.cos(center_tilt_rad) * \
            np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)
        x = (np.cos(phi_rad) *
             np.sin(lambda_rad - center_pan_rad)) / cos_c
        y = (np.cos(center_tilt_rad) * np.sin(phi_rad) - np.sin(center_tilt_rad)
             * np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)) / cos_c

        return np.array([x, y]).T

    def _gnomonic_inverse(self, coord_spherical, center=None):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        x = coord_spherical.T[0]
        y = coord_spherical.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        cp0, cp1 = self.cp
        if center is not None:
            cp0, cp1 = center

        lat = np.arcsin(
            cos_c * np.sin(cp1) + (y * sin_c * np.cos(cp1)) / rou)
        lon = cp0 + np.arctan2(x * sin_c, rou * np.cos(cp1)
                               * cos_c - y * np.sin(cp1) * sin_c)

        return np.array([lon, lat]).T

    def _remap(self, coords):
        """ In range: [0, 1] """

        map_x = (coords[:, 0] * self.frame_width).astype(np.float32)
        map_x = np.reshape(map_x, [self.height, self.width])

        map_y = (coords[:, 1] *
                 self.frame_height).astype(np.float32)
        map_y = np.reshape(map_y, [self.height, self.width])

        return cv2.remap(frame_orig, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def draw_coords_(self, frame_orig, coords):
        xs = np.floor(coords.T[0] * self.frame_width).astype(np.int32)
        ys = np.floor(coords.T[1] * self.frame_height).astype(np.int32)

        skip = 150

        for x, y in zip(xs[::skip], ys[::skip]):
            cv2.circle(frame_orig, [x, y], radius=5,
                       color=[255, 0, 255], thickness=-1)

        return frame_orig

    def toNFOV(self, frame_orig, center_point):
        self.frame_orig = frame_orig
        self.frame_height = frame_orig.shape[0]
        self.frame_width = frame_orig.shape[1]
        self.frame_channel = frame_orig.shape[2]

        # center point
        self.cp = self._screen2spherical(center_point)

        # frame coords
        coords_screen_orig = self._get_coords_screen_frame()
        coords_spherical = self._screen2spherical(coords_screen_orig)
        coords_spherical_fov = coords_spherical * (self.FOV / 2 / self.limits)

        # map onto a sphere
        coords_spherical_fov = self._gnomonic_inverse(
            coords_spherical_fov)

        # map onto a plane
        # coords_spherical_fov = self._gnomonic_forward(
        #     coords_spherical_fov)
        # coords_spherical_fov = self._gnomonic_inverse(
        #     coords_spherical_fov, center=(0, self.tilt))
        coords_screen_fov = self._spherical2screen(coords_spherical_fov)

        frame_painted = self.draw_coords_(frame_orig.copy(), coords_screen_fov)

        return self._remap(coords_screen_fov), frame_painted

    def get_stats(self):
        return {
            "fov_lens_horiz": np.rad2deg(self.lens_fov_horiz_rad),
            "fov_lens_vert": np.rad2deg(self.lens_fov_vert_rad),
            "fov": np.rad2deg(self.FOV),
            "tilt": np.rad2deg(self.tilt)
        }


# test the class
if __name__ == '__main__':
    import imageio as im
    # frame_orig = im.imread('images/curvilinear.png')
    # frame_orig = im.imread('images/pitch.png')
    frame_orig = im.imread('images/pitch_grid.png')
    # frame_orig = im.imread('images/360.jpg')
    # frame_orig = im.imread('images/pitch_fisheye.png')
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)

    nfov = NFOV(fov_deg=50, fov_lens_horiz_deg=115)
    # nfov = NFOV(fov_deg=90, fov_lens_horiz_deg=360)
    # nfov = NFOV(fov_deg=120, fov_lens_horiz_deg=150)

    cv2.namedWindow("frame", WINDOW_FLAGS)
    cv2.namedWindow("frame_orig", WINDOW_FLAGS)

    # camera center point (valid range [0,1])
    center_point = np.array([0.5, 0.5])

    dx = 0.05
    dz = .25
    dfov = .025
    dtilt = np.deg2rad(5)
    while True:
        try:
            img, frame_orig_painted = nfov.toNFOV(frame_orig, center_point)
        except Exception as e:
            print(e)

        cv2.imshow('frame', img)
        cv2.imshow("frame_orig", frame_orig_painted)
        print(nfov.get_stats())
        key = cv2.waitKey(0)
        if key == ord('d'):
            center_point += np.array([dx, 0])
        elif key == ord('a'):
            center_point -= np.array([dx, 0])
        elif key == ord('w'):
            center_point -= np.array([0, dx])
        elif key == ord('s'):
            center_point += np.array([0, dx])
        elif key == ord('+'):
            nfov.fov_rad -= dz
        elif key == ord('-'):
            nfov.fov_rad += dz
        elif key == ord('8'):
            nfov.lens_fov_vert_rad += dfov
        elif key == ord('2'):
            nfov.lens_fov_vert_rad -= dfov
        elif key == ord('6'):
            nfov.lens_fov_horiz_rad += dfov
        elif key == ord('4'):
            nfov.lens_fov_horiz_rad -= dfov
        elif key == ord('p'):
            nfov.tilt += dtilt
        elif key == ord('m'):
            nfov.tilt -= dtilt
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
