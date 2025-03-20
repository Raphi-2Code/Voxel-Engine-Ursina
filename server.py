from ursina import *
from perlin_noise import *
import math

xpos = 0
zpos = 0
chunk_size = 16
cube_faces = [
    (0,1,0,180,0,0),
    (0,2,0,0,0,0),
    (0,1.5,0.5,90,0,0),
    (0,1.5,-0.5,-90,0,0),
    (0.5,1.5,0,0,0,90),
    (-0.5,1.5,0,0,0,-90)
]
seed = ord('y') + ord('o')
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

with open('chunks.txt', 'w') as yo:
    noise = Perlin()
    for x_chunk in range(4):
        for z_chunk in range(4):
            chunk_faces2 = []
            chunk_faces3 = []
            for x in range(xpos + x_chunk * chunk_size, xpos + x_chunk * chunk_size + chunk_size):
                for z in range(zpos + z_chunk * chunk_size, zpos + z_chunk * chunk_size + chunk_size):
                    y = compute_height(noise, x, z)
                    if 1 < len(cube_faces):
                        elem = cube_faces[1]
                    else:
                        elem = cube_faces[0]
                    pos_top = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_top)
                    chunk_faces3.append(1)
                    for dx, dz, face_idx in [(-1,0,5), (1,0,4), (0,-1,3), (0,1,2)]:
                        x_adj = x + dx
                        z_adj = z + dz
                        y_adj = compute_height(noise, x_adj, z_adj)
                        if y_adj < y:
                            if face_idx < len(cube_faces):
                                elem = cube_faces[face_idx]
                            else:
                                elem = cube_faces[0]
                            pos_side = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                            chunk_faces2.append(pos_side)
                            chunk_faces3.append(face_idx)
            if chunk_faces2 and len(chunk_faces2) == len(chunk_faces3):
                yo.write(str([chunk_faces2, chunk_faces3]) + ",")
