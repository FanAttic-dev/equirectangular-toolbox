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
    def __init__(self, fov_deg, fov_lens_horiz_deg, fov_lens_vert_deg, height=540, width=960):
        self.fov_rad = np.deg2rad(fov_deg)
        self.fov_lens_horiz_rad = np.deg2rad(fov_lens_horiz_deg)
        self.fov_lens_vert_rad = np.deg2rad(fov_lens_vert_deg)
        self.height = height
        self.width = width

    @property
    def FOV(self):
        return np.array([self.fov_rad, self.fov_rad])

    @property
    def limits(self):
        return np.array([self.fov_lens_horiz_rad, self.fov_lens_vert_rad]) / 2

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
        frame_screen = self._get_screen_frame()
        frame_spherical = self._screen2spherical(frame_screen)
        frame_spherical_fov = frame_spherical * self.FOV / 2
        return frame_spherical_fov

    def _get_screen_frame(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width),
                             np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _gnomonic_forward(self, coord_spherical):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        lambda_rad = coord_spherical.T[0]
        phi_rad = coord_spherical.T[1]
        center_pan_rad = -self.cp[0]
        center_tilt_rad = -self.cp[1]

        cos_c = np.sin(center_tilt_rad) * np.sin(phi_rad) + np.cos(center_tilt_rad) * \
            np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)
        x = (np.cos(phi_rad) *
             np.sin(lambda_rad - center_pan_rad)) / cos_c
        y = (np.cos(center_tilt_rad) * np.sin(phi_rad) - np.sin(center_tilt_rad)
             * np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)) / cos_c

        return np.array([x, y]).T

    def _gnomonic_inverse(self, coord_spherical):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        x = coord_spherical.T[0]
        y = coord_spherical.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1])
                                      * cos_c - y * np.sin(self.cp[1]) * sin_c)

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        # [0, frame_size]
        uf = np.mod(screen_coord.T[0], 1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1], 1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        # coords of pixel to top right
        x2 = np.add(x0, np.ones(uf.shape).astype(int))
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame_orig, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(
            np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        # import matplotlib.pyplot as plt
        # plt.imshow(nfov)
        # plt.show()
        return nfov

    def _remap(self, gnomonicCoord):
        ...

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

        self.cp = self._screen2spherical(center_point)
        coord_spherical = self._get_frame_spherical_fov()
        # coord_gnomonic = self._gnomonic_forward(coord_spherical)
        coord_gnomonic = self._gnomonic_inverse(coord_spherical)
        coord_gnomonic = self._spherical2screen(coord_gnomonic)

        frame_painted = self.draw_coords_(frame_orig.copy(), coord_gnomonic)
        return self._bilinear_interpolation(coord_gnomonic), frame_painted
        # return self._remap(gnomonicCoord), frame_painted

    def get_stats(self):
        return {
            "fov_lens_horiz": np.rad2deg(self.fov_lens_horiz_rad),
            "fov_lens_vert": np.rad2deg(self.fov_lens_vert_rad),
            "fov": np.rad2deg(self.FOV),
        }


# test the class
if __name__ == '__main__':
    import imageio as im
    # frame_orig = im.imread('images/pitch.png')
    frame_orig = im.imread('images/360.jpg')
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)

    nfov = NFOV(fov_deg=50, fov_lens_horiz_deg=115, fov_lens_vert_deg=90)

    cv2.namedWindow("frame", WINDOW_FLAGS)
    cv2.namedWindow("frame_orig", WINDOW_FLAGS)

    # camera center point (valid range [0,1])
    center_point = np.array([0.5, 0.5])

    dx = 0.05
    dz = .01
    dfov = .1
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
            nfov.fov_lens_vert_rad += dfov
        elif key == ord('2'):
            nfov.fov_lens_vert_rad -= dfov
        elif key == ord('6'):
            nfov.fov_lens_horiz_rad += dfov
        elif key == ord('4'):
            nfov.fov_lens_horiz_rad -= dfov
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
