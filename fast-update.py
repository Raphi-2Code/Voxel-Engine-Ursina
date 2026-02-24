from ursina import *
from perlin_noise import *
from numba import njit
import numpy as np
import math

xpos = 0
zpos = 0
chunk_size = 16
cube_faces = [
    (0, 1, 0, 180, 0, 0),
    (0, 2, 0, 0, 0, 0),
    (0, 1.5, 0.5, 90, 0, 0),
    (0, 1.5, -0.5, -90, 0, 0),
    (0.5, 1.5, 0, 0, 0, 90),
    (-0.5, 1.5, 0, 0, 0, -90),
]

seed = sum(ord(c) for c in 'terrain')
octaves = 0.5
frequency = 8
amplitude = 1


class Perlin:
    def __init__(self):
        self.seed = seed
        self.octaves = max(1, int(octaves))
        self.freq = frequency
        self.amplitude = amplitude
        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        if not isinstance(x, (int, float)) or not isinstance(z, (int, float)):
            return 0
        return self.pNoise([x / self.freq, z / self.freq]) * self.amplitude


def compute_height(noise, x, z):
    return max(0, math.floor(noise.get_height(round(x / 2), round(z / 2)) * 7.5))


cube_faces_arr = np.array(
    [[f[0], f[1], f[2]] for f in cube_faces], dtype=np.float64
)


@njit
def process_chunk(heights, cf, x0, z0, cs, ho_x, ho_z):
    # Kapazität erhöht, da bei steilem Gelände deutlich mehr
    # Flächen pro (x,z) Koordinate entstehen können.
    cap = cs * cs * 30
    px = np.empty(cap, np.float64)
    py = np.empty(cap, np.float64)
    pz = np.empty(cap, np.float64)
    fi = np.empty(cap, np.int32)
    n = 0

    ndx = np.array([-1, 1, 0, 0], np.int32)
    ndz = np.array([0, 0, -1, 1], np.int32)
    nfi = np.array([5, 4, 3, 2], np.int32)

    for x in range(x0, x0 + cs):
        for z in range(z0, z0 + cs):
            hx = x - ho_x
            hz = z - ho_z
            y = heights[hx, hz]

            # Obere Fläche zeichnen
            px[n] = cf[1, 0] + x
            py[n] = cf[1, 1] + y
            pz[n] = cf[1, 2] + z
            fi[n] = 1
            n += 1

            # Seitenflächen zeichnen
            for i in range(4):
                neighbor_y = heights[hx + ndx[i], hz + ndz[i]]

                # Wenn der Nachbar tiefer ist, fülle alle Blöcke dazwischen auf
                if neighbor_y < y:
                    f = nfi[i]
                    for current_y in range(neighbor_y + 1, y + 1):
                        px[n] = cf[f, 0] + x
                        py[n] = cf[f, 1] + current_y
                        pz[n] = cf[f, 2] + z
                        fi[n] = f
                        n += 1

    return px[:n], py[:n], pz[:n], fi[:n]


noise = Perlin()
total = 32 * chunk_size
hw = total + 2
ho_x = xpos - 1
ho_z = zpos - 1

heights = np.empty((hw, hw), dtype=np.int32)
for xi in range(hw):
    for zi in range(hw):
        heights[xi, zi] = compute_height(noise, ho_x + xi, ho_z + zi)

# Initialer Numba-Kompilierungsaufruf
process_chunk(heights, cube_faces_arr, xpos, zpos, chunk_size, ho_x, ho_z)

with open('chunks.txt', 'w') as yo:
    for xc in range(32):
        for zc in range(32):
            xs = xpos + xc * chunk_size
            zs = zpos + zc * chunk_size

            rpx, rpy, rpz, rfi = process_chunk(
                heights, cube_faces_arr,
                xs, zs, chunk_size,
                ho_x, ho_z,
            )

            faces2 = [Vec3(rpx[i], rpy[i], rpz[i]) for i in range(len(rpx))]
            faces3 = rfi.tolist()

            if faces2 and len(faces2) == len(faces3):
                yo.write(str([faces2, faces3]) + ",")
