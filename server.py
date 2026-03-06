from ursina import *
from perlin_noise import *
from numba import njit
import numpy as np
import math

xpos = 0
zpos = 0
chunk_size = 16

cube_faces = [
    (0, 1, 0, 180, 0, 0),      # 0: Bottom
    (0, 2, 0, 0, 0, 0),        # 1: Top
    (0, 1.5, 0.5, 90, 0, 0),   # 2: Front (+Z)
    (0, 1.5, -0.5, -90, 0, 0), # 3: Back (-Z)
    (0.5, 1.5, 0, 0, 0, 90),   # 4: Right (+X)
    (-0.5, 1.5, 0, 0, 0, -90), # 5: Left (-X)
]

terrain_seed = 67676767
cave_seed = 12345678

octaves = 0.5
frequency = 8
amplitude = 1


class Perlin:
    def __init__(self):
        self.seed = terrain_seed
        self.octaves = max(1, int(octaves))
        self.freq = frequency
        self.amplitude = amplitude
        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        if not isinstance(x, (int, float)) or not isinstance(z, (int, float)):
            return 0
        return self.pNoise([x / self.freq, z / self.freq]) * self.amplitude


def compute_height(noise, x, z):
    # Die Welt wird tiefer gemacht (Basis 15 + Variation), damit Platz für Höhlen ist!
    val = noise.get_height(round(x / 2), round(z / 2))
    return max(5, math.floor(15 + val * 15))


cube_faces_arr = np.array([[f[0], f[1], f[2]] for f in cube_faces], dtype=np.float64)


# ==========================================
# MAGISCHE 3D NUMBA NOISE FÜR HÖHLEN
# ==========================================
@njit
def hash_3d(x, y, z, seed):
    # Ein schneller Hash-Algorithmus mit Seed für Höhlen
    n = (x * 1619 + y * 31337 + z * 6971 + seed * 1013) % 2147483647
    n = (n ^ (n >> 8)) * 19081 % 2147483647
    n = (n ^ (n >> 9)) % 2147483647
    return n / 2147483647.0


@njit
def noise_3d(x, y, z, seed):
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    zi = int(math.floor(z))

    xf = x - xi
    yf = y - yi
    zf = z - zi

    # Smoothstep Interpolation
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    w = zf * zf * (3.0 - 2.0 * zf)

    c000 = hash_3d(xi, yi, zi, seed)
    c100 = hash_3d(xi + 1, yi, zi, seed)
    c010 = hash_3d(xi, yi + 1, zi, seed)
    c110 = hash_3d(xi + 1, yi + 1, zi, seed)
    c001 = hash_3d(xi, yi, zi + 1, seed)
    c101 = hash_3d(xi + 1, yi, zi + 1, seed)
    c011 = hash_3d(xi, yi + 1, zi + 1, seed)
    c111 = hash_3d(xi + 1, yi + 1, zi + 1, seed)

    x00 = c000 + u * (c100 - c000)
    x10 = c010 + u * (c110 - c010)
    x01 = c001 + u * (c101 - c001)
    x11 = c011 + u * (c111 - c011)

    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)

    return y0 + w * (y1 - y0)


@njit
def fbm_3d(x, y, z, seed):
    # Fractal Brownian Motion für organischer aussehende Höhlen
    return (
        noise_3d(x, y, z, seed) * 0.5
        + noise_3d(x * 2.0, y * 2.0, z * 2.0, seed) * 0.25
    )


@njit
def is_solid(x, y, z, heights, ho_x, ho_z, cave_seed):
    if y < 0:
        return True  # Grundgestein (Bedrock)

    hx = x - ho_x
    hz = z - ho_z

    if hx < 0 or hx >= heights.shape[0] or hz < 0 or hz >= heights.shape[1]:
        return False

    surface_y = heights[hx, hz]
    if y > surface_y:
        return False  # Luft über der Welt

    # Schützt die oberen 3 Blöcke, damit keine Höhlen das Gras durchlöchern
    depth = surface_y - y
    if depth < 3:
        return True

    # Separater Höhlen-Seed wird hier benutzt
    cave_val = fbm_3d(x * 0.08, y * 0.08, z * 0.08, cave_seed)
    if cave_val > 0.45:  # kleiner = mehr Höhlen, größer = weniger Höhlen
        return False

    return True


# ==========================================
# 3D VOXEL MESHING
# ==========================================
@njit
def process_chunk(heights, cf, x0, z0, cs, ho_x, ho_z, cave_seed):
    # Große Kapazität, da ein Chunk viele sichtbare Höhlen-Flächen haben kann
    cap = cs * cs * 150
    px = np.empty(cap, np.float64)
    py = np.empty(cap, np.float64)
    pz = np.empty(cap, np.float64)
    fi = np.empty(cap, np.int32)
    n = 0

    for x in range(x0, x0 + cs):
        for z in range(z0, z0 + cs):
            hx = x - ho_x
            hz = z - ho_z
            surface_y = heights[hx, hz]

            # Scanne die Welt vertikal vom Boden bis zur Oberfläche
            for y in range(0, surface_y + 1):
                if is_solid(x, y, z, heights, ho_x, ho_z, cave_seed):

                    # Bottom Face
                    if not is_solid(x, y - 1, z, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[0, 0] + x
                        py[n] = cf[0, 1] + y
                        pz[n] = cf[0, 2] + z
                        fi[n] = 0
                        n += 1

                    # Top Face
                    if not is_solid(x, y + 1, z, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[1, 0] + x
                        py[n] = cf[1, 1] + y
                        pz[n] = cf[1, 2] + z
                        fi[n] = 1
                        n += 1

                    # Front Face (+Z)
                    if not is_solid(x, y, z + 1, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[2, 0] + x
                        py[n] = cf[2, 1] + y
                        pz[n] = cf[2, 2] + z
                        fi[n] = 2
                        n += 1

                    # Back Face (-Z)
                    if not is_solid(x, y, z - 1, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[3, 0] + x
                        py[n] = cf[3, 1] + y
                        pz[n] = cf[3, 2] + z
                        fi[n] = 3
                        n += 1

                    # Right Face (+X)
                    if not is_solid(x + 1, y, z, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[4, 0] + x
                        py[n] = cf[4, 1] + y
                        pz[n] = cf[4, 2] + z
                        fi[n] = 4
                        n += 1

                    # Left Face (-X)
                    if not is_solid(x - 1, y, z, heights, ho_x, ho_z, cave_seed):
                        px[n] = cf[5, 0] + x
                        py[n] = cf[5, 1] + y
                        pz[n] = cf[5, 2] + z
                        fi[n] = 5
                        n += 1

    return px[:n], py[:n], pz[:n], fi[:n]


# ==========================================
# HAUPTPROGRAMM
# ==========================================
noise = Perlin()
total = 32 * chunk_size
hw = total + 2
ho_x = xpos - 1
ho_z = zpos - 1

print("Generiere 2D Heightmap...")
heights = np.empty((hw, hw), dtype=np.int32)
for xi in range(hw):
    for zi in range(hw):
        heights[xi, zi] = compute_height(noise, ho_x + xi, ho_z + zi)

print("Kompiliere Numba & generiere Chunks (mit Höhlen)...")
# Initialer Aufruf (kompiliert das Skript)
process_chunk(heights, cube_faces_arr, xpos, zpos, chunk_size, ho_x, ho_z, cave_seed)

with open("chunks.txt", "w") as yo:
    for xc in range(8):
        for zc in range(8):
            xs = xpos + xc * chunk_size
            zs = zpos + zc * chunk_size

            rpx, rpy, rpz, rfi = process_chunk(
                heights,
                cube_faces_arr,
                xs,
                zs,
                chunk_size,
                ho_x,
                ho_z,
                cave_seed,
            )

            faces2 = [Vec3(rpx[i], rpy[i], rpz[i]) for i in range(len(rpx))]
            faces3 = rfi.tolist()

            if faces2 and len(faces2) == len(faces3):
                yo.write(str([faces2, faces3]) + ",")

print("Fertig! Chunks inklusive Höhlen exportiert.")
