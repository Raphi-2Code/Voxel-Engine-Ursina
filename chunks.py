#An easy tool for generating chunks in ursina engine
#rules working for all rows except for the last row of chunk
#function to generate chunks
#you can change the chunk size
#made for perlin noises
from ursina import *
from perlin_noise import *
import math
cube_faces=[(0,1,0,180,0,0),(0,2,0,0,0,0),(0,1.5,0.5,90,0,0),(0,1.5,-0.5,-90,0,0),(0.5,1.5,0,0,0,90),(-0.5,1.5,0,0,0,-90)]
seed=ord('y')+ord('o')
octaves=0.5
frequency=8
amplitude=1
class Perlin:
    def __init__(self):
        self.seed = seed
        self.octaves = octaves
        self.freq = frequency
        self.amplitude = amplitude

        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        return self.pNoise([x/self.freq, z/self.freq]) * self.amplitude
def gen_chunk(texture,xpos,zpos,chunk_size):
    chunk_faces = []
    chunk_faces2 = []
    noise = Perlin()
    terrain = Entity()
    for x in range(xpos,xpos+chunk_size):
        for z in range(zpos,zpos+chunk_size):
            y = noise.get_height(round(round(x)/2), round(round(z)/2))
            y = math.floor(y * 7.5)
            elem = cube_faces[1]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append(face)
            chunk_faces2.append(face.position)
            if pos_i + (0, 1, -1) in chunk_faces2:
                elem = cube_faces[2]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face = Entity(model="plane", position=pos_i + (0, 1, -1), rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
            if pos_i + (-1, -1, 0) in chunk_faces2:
                elem = cube_faces[5]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
            if pos_i + (0, -1, -1) in chunk_faces2:
                elem = cube_faces[3]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
            if pos_i + (-1, 1, 0) in chunk_faces2:
                elem = cube_faces[4]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face = Entity(model="plane", position=pos_i + (-1, 1, 0), rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
            if pos_i + (0.5, -0.5, -1) in chunk_faces2:
                elem = cube_faces[3]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
                chunk_faces.append(face)
                chunk_faces2.append(face.position)
            if pos_i + (-1, 1, 0) in chunk_faces2:
                elem = cube_faces[4]
                pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
                rot_i = Vec3(elem[3], elem[4], elem[5])
                if pos_i + (-1.5, 1.5, -1) in chunk_faces2:
                    face = Entity(model="plane", position=pos_i + (-1, 1, -1), rotation=rot_i, parent=terrain)
                    chunk_faces.append(face)
                    chunk_faces2.append(face.position)

    terrain.combine()
    terrain.texture = texture
def title_window(title_,color_):
    Text(title_, y=.5, x=-.886, color=color_)
if __name__=="__main__":
    from ursina.prefabs.first_person_controller import *
    app = Ursina()
    gen_chunk("ursina-tutorials-main/assets/sandMinecraft.jfif", 0, 0, 60)
    player = FirstPersonController(gravity=0, speed=20)
    def input(key):
        if key == "m":
            player.y += 1
        if key == "y":
            player.y -= 1
    app.run()
