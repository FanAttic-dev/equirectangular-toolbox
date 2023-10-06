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


class NFOV():
    def __init__(self, height=540, width=960):
        self.fov = np.deg2rad(50)
        # self.limit_horiz = pi
        self.limit_horiz = np.deg2rad(115/2)
        # self.limit_vert = pi * 0.5
        self.limit_vert = np.deg2rad(66/2)
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    @property
    def FOV(self):
        return np.array([self.fov, self.fov / 16 * 9])

    @property
    def limits(self):
        return np.array([self.limit_horiz, self.limit_vert])

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * self.limits \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * self.limits * \
            (self.FOV / (self.limits * 2) * np.ones(self.screen_points.shape))

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width),
                             np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        # lon = x + self.cp[0]
        # lat = y + self.cp[1]

        center_pan_rad = -self.cp[0]
        center_tilt_rad = -self.cp[1]
        lambda_rad = x
        phi_rad = y
        cos_c = np.sin(center_tilt_rad) * np.sin(phi_rad) + np.cos(center_tilt_rad) * \
            np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)
        lon = (np.cos(phi_rad) *
               np.sin(lambda_rad - center_pan_rad)) / cos_c
        lat = (np.cos(center_tilt_rad) * np.sin(phi_rad) - np.sin(center_tilt_rad)
               * np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)) / cos_c

        # rou = np.sqrt(x ** 2 + y ** 2)
        # c = np.arctan(rou)
        # # c = np.arcsin(rou / self.r)
        # sin_c = np.sin(c)
        # cos_c = np.cos(c)

        # lat = np.arcsin(
        #     cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        # lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1])
        #                               * cos_c - y * np.sin(self.cp[1]) * sin_c)

        # [-pi, pi] -> [0, 1]
        lon = (lon / self.limit_horiz + 1.) * 0.5
        lat = (lat / self.limit_vert + 1.) * 0.5

        return np.array([lon, lat]).T

    def _calcSphericaltoGnomonic_inverse(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        # lon = x + self.cp[0]
        # lat = y + self.cp[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        # c = np.arcsin(rou / self.r)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1])
                                      * cos_c - y * np.sin(self.cp[1]) * sin_c)

        # [-pi, pi] -> [0, 1]
        lat = (lat / self.limit_vert + 1.) * 0.5
        lon = (lon / self.limit_horiz + 1.) * 0.5

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

    def draw_coords_(self, frame_orig, sphericalCoord):
        xs = np.floor(sphericalCoord.T[0] * self.frame_width).astype(np.int32)
        ys = np.floor(sphericalCoord.T[1] * self.frame_height).astype(np.int32)

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

        self.cp = self._get_coord_rad(
            center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        sphericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        frame_painted = self.draw_coords_(frame_orig.copy(), sphericalCoord)
        return self._bilinear_interpolation(sphericalCoord), frame_painted

    def get_stats(self):
        return {
            "limit_horiz": np.rad2deg(self.limit_horiz*2),
            "limit_vert": np.rad2deg(self.limit_vert*2),
            "fov": np.rad2deg(self.FOV),
        }


# test the class
if __name__ == '__main__':
    import imageio as im
    frame_orig = im.imread('images/pitch.png')
    frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)
    nfov = NFOV()
    WINDOW_FLAGS = cv2.WINDOW_NORMAL  # cv2.WINDOW_AUTOSIZE
    cv2.namedWindow("frame", WINDOW_FLAGS)
    cv2.namedWindow("frame_orig", WINDOW_FLAGS)
    # camera center point (valid range [0,1])
    center_point = np.array([0.5, 0.5])
    dx = 0.01
    dz = .01
    dfov = .01
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
            nfov.fov -= dz
        elif key == ord('-'):
            nfov.fov += dz
        elif key == ord('8'):
            nfov.limit_vert += dfov
        elif key == ord('2'):
            nfov.limit_vert -= dfov
        elif key == ord('6'):
            nfov.limit_horiz += dfov
        elif key == ord('4'):
            nfov.limit_horiz -= dfov
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
