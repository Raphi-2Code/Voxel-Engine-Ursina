from ursina import *
from perlin_noise import *
xpos=0
zpos=0
chunk_size=16
cube_faces = [(0, 1, 0, 180, 0, 0), (0, 2, 0, 0, 0, 0), (0, 1.5, 0.5, 90, 0, 0), (0, 1.5, -0.5, -90, 0, 0),
              (0.5, 1.5, 0, 0, 0, 90), (-0.5, 1.5, 0, 0, 0, -90)]
seed = ord('y') + ord('o')
octaves = 0.5
frequency = 8
amplitude = 1
class Perlin:
    def __init__(self):
        self.seed = seed
        self.octaves = octaves
        self.freq = frequency
        self.amplitude = amplitude

        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        return self.pNoise([x / self.freq, z / self.freq]) * self.amplitude
yo=open('chunks.txt','w')
q=[]
noise=Perlin()
chunk_faces2_=[]
chunk_faces3_=[]
for x_chunk in range(4):
    for z_chunk in range(4):
        chunk_faces2 = []
        chunk_faces3 = []
        for x in range(xpos + x_chunk*16, xpos + x_chunk*16 + chunk_size):
            for z in range(zpos + z_chunk*16, zpos + z_chunk*16 + chunk_size + 2):
                y = noise.get_height(round(round(x) / 2), round(round(z) / 2))
                y = math.floor(y * 7.5)
                elem = cube_faces[1]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                chunk_faces2.append(pos_i)
                chunk_faces3.append(1)
                chunk_faces2_.append(pos_i)
                chunk_faces3_.append(1)
                if pos_i + (0, 1, -1) in chunk_faces2_:
                    elem = cube_faces[2]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_i + (0,1,-1))
                    chunk_faces3.append(2)
                    chunk_faces2_.append(pos_i + (0,1,-1))
                    chunk_faces3_.append(2)
                if pos_i + (-1, -1, 0) in chunk_faces2_:
                    elem = cube_faces[5]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_i)
                    chunk_faces3.append(5)
                    chunk_faces2_.append(pos_i)
                    chunk_faces3_.append(5)
                if pos_i + (0, -1, -1) in chunk_faces2_:
                    elem = cube_faces[3]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_i)
                    chunk_faces3.append(3)
                    chunk_faces2_.append(pos_i)
                    chunk_faces3_.append(3)
                if pos_i + (-1, 1, 0) in chunk_faces2_:
                    elem = cube_faces[4]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_i + (-1, 1, 0))
                    chunk_faces3.append(4)
                    chunk_faces2_.append(pos_i + (-1, 1, 0))
                    chunk_faces3_.append(4)
                if pos_i + (0.5, -0.5, -1) in chunk_faces2_:
                    elem = cube_faces[3]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    chunk_faces2.append(pos_i)
                    chunk_faces3.append(3)
                    chunk_faces2_.append(pos_i)
                    chunk_faces3_.append(3)
                if pos_i + (-1, 1, 0) in chunk_faces2_:
                    elem = cube_faces[4]
                    pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                    if pos_i + (-1.5, 1.5, -1) in chunk_faces2_ and pos_i + (-1.5, 1.5, 1) in chunk_faces2_:
                        chunk_faces2.append(pos_i + (-1, 1 ,-1))
                        chunk_faces3.append(4)
                        chunk_faces2_.append(pos_i + (-1, 1, -1))
                        chunk_faces3_.append(4)
        yo.write(str([chunk_faces2,chunk_faces3])+str(","))
#yo.write(']')
#yo.write(str(',[')+str(q)+str(']'))