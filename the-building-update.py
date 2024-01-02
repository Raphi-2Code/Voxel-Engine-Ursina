from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np
app=Ursina()
#player=FirstPersonController(gravity=0)
player=FirstPersonController(gravity=0)
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
chunk_faces = []
chunk_faces2 = []
chunk_faces3 = []
noise = Perlin()
terrain = Entity()
texture="sand"
xpos=0
zpos=0
chunk_size=16
for x in range(xpos, xpos + chunk_size):
    for z in range(zpos, zpos + chunk_size):
        y = noise.get_height(round(round(x) / 2), round(round(z) / 2))
        y = math.floor(y * 7.5)
        elem = cube_faces[1]
        pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
        rot_i = Vec3(elem[3], elem[4], elem[5])
        face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
        chunk_faces.append([face.x,face.z])
        chunk_faces2.append(face.position)
        chunk_faces3.append(1)
        if pos_i + (0, 1, -1) in chunk_faces2:
            elem = cube_faces[2]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i + (0, 1, -1), rotation=rot_i, parent=terrain)
            chunk_faces.append([face.x, face.z])
            chunk_faces2.append(face.position)
            chunk_faces3.append(2)
        if pos_i + (-1, -1, 0) in chunk_faces2:
            elem = cube_faces[5]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append([face.x, face.z])
            chunk_faces2.append(face.position)
            chunk_faces3.append(5)
        if pos_i + (0, -1, -1) in chunk_faces2:
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append([face.x,face.z])
            chunk_faces2.append(face.position)
            chunk_faces3.append(3)
        if pos_i + (-1, 1, 0) in chunk_faces2:
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i + (-1, 1, 0), rotation=rot_i, parent=terrain)
            chunk_faces.append([face.x,face.z])
            chunk_faces2.append(face.position)
            chunk_faces3.append(4)
        if pos_i + (0.5, -0.5, -1) in chunk_faces2:
            elem = cube_faces[3]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain)
            chunk_faces.append([face.x,face.z])
            chunk_faces2.append(face.position)
            chunk_faces3.append(3)
        if pos_i + (-1, 1, 0) in chunk_faces2:
            elem = cube_faces[4]
            pos_i = Vec3(elem[0] + x, elem[1] + y, elem[2] + z)
            rot_i = Vec3(elem[3], elem[4], elem[5])
            if pos_i + (-1.5, 1.5, -1) in chunk_faces2  and pos_i+(-1.5,1.5,1) in chunk_faces2:
                face = Entity(model="plane", position=pos_i + (-1, 1, -1), rotation=rot_i, parent=terrain)
                chunk_faces.append([face.x,face.z])
                chunk_faces2.append(face.position)
                chunk_faces3.append(4)
#block = Entity(model="cube", texture="white_cube")
#blocky = block.combine()
#terrain.collider="mesh"
p = terrain.combine()
#terrain.collider='mesh'
#p.vertices.extend(blocky.vertices)
#p.project_uvs()
terrain.texture = texture
try:player.y = chunk_faces2[chunk_faces.index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]
except:pass
mode=1
save=0
q=(0,-9999,0)
c=Entity(model="cube",color=color.clear,collider="box")
c2=Entity(model="cube",texture="frame")
def update():
    global mode
    if mode==0:
        try:
            player.y = chunk_faces2[chunk_faces.index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]

        except:
            pass
    c2.position = floor(player.position + player.forward * 4)



count=0
def input(key):
    global p,mode,save,q,count
    if key=="g":
        if len(p.vertices)!=0:
            p_verts=p.vertices
            p_norms=p.normals
            p_uvs=p.uvs
            p.clear()
            [p_verts.pop(3+i) for i in range(6)]#verts pos distance to mouse pos < 0.5 -> remove verts
            [p_uvs.pop(3+i_) for i_ in range(6)]#index verts to remove -> update uvs
            p.vertices=p_verts
            p.normals=p_norms
            p.uvs=p_uvs
            p.generate()
    if key=="o":
        mode=1-mode
    if key=="m":
        player.y+=1
    if key=="y":
        player.y-=1
    if save==1:
        if mouse.hovered_entity == c:
            #try:
                    global chunk_faces,chunk_faces2,chunk_faces3
                    cube_=Entity()
                    aqc2=[]
                    aqc3=[]
                    pll=0
                    pos = Vec3(c.position) + (0, -1.5, 0)
                    for i__ in range(6):
                        elem = cube_faces[i__]#Vec3(vert)+Vec3(0.5,-1,0.5)
                        pos_i = Vec3(elem[0] + pos[0], elem[1] + pos[1], elem[2] + pos[2])
                        rot_i = Vec3(elem[3], elem[4], elem[5])
                        #face = Entity(model="plane", position=pos_i, rotation=rot_i, color=color.yellow, parent=cube_)
                        #chunk_faces.append([pos_i[0], pos_i[2]])
                        #chunk_faces2.append(pos_i)
                        #chunk_faces3.append(i__)
                        #aqc_1
                        aqc3.append(rot_i)
                        aqc2.append(pos_i)

                    p.clear()
                    destroy(terrain)
                    terrain2=Entity()
                    new_chunk_faces2=[]
                    new_chunk_faces=[]
                    new_chunk_faces3=[]
                    f_pos=[]
                    for element in chunk_faces2:
                        elem = cube_faces[chunk_faces3[pll]]
                        pos_i = Vec3(element[0], element[1], element[2])
                        rot_i = Vec3(elem[3], elem[4], elem[5])
                        if not pos_i in aqc2:
                            #print(aqc,pos_i)
                            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain2)
                            #if chunk_faces3[pll]!=2 and chunk_faces3[pll]!=3:
                            new_chunk_faces2.append((pos_i[0],pos_i[1],pos_i[2]))
                            new_chunk_faces.append([pos_i[0],pos_i[2]])
                            new_chunk_faces3.append(chunk_faces3[pll])
                        else:
                            f_pos.append(pos_i)
                        pll+=1#YESSSIRRR!
                    chunk_faces2=new_chunk_faces2
                    chunk_faces=new_chunk_faces
                    chunk_faces3=new_chunk_faces3
                    for anti_f_pos in aqc2:
                        if not anti_f_pos in f_pos:
                            face = Entity(model="plane", position=anti_f_pos, rotation=aqc3[aqc2.index(anti_f_pos)],parent=terrain2)
                            chunk_faces2.append(face.position)
                            chunk_faces.append([face.x,face.z])
                            if face.rotation==(180,0,0):chunk_faces3.append(0)
                            if face.rotation==(0,0,0):chunk_faces3.append(1)
                            if face.rotation==(90,0,0):chunk_faces3.append(2)
                            if face.rotation==(-90,0,0):chunk_faces3.append(3)
                            if face.rotation==(0,0,90):chunk_faces3.append(4)
                            if face.rotation==(0,0,-90):chunk_faces3.append(5)
                    p=terrain2.combine()
                    terrain2.texture=texture
                    c.y=-9999
            #except:print("SURFBREAD")
        else:c.y=-9999
        save=0
    if key=="right mouse down":
        l = []
        try:
            q = chunk_faces2[chunk_faces.index([round((player.forward[0]) * 4 + player.x),
                                                    round((player.forward[2]) * 4 + player.z)])] + (0, 0.5, 0)
        except:pass
        c.position=q
        save=1
    if key=="left mouse down":
        l = []
        try:
            q = chunk_faces2[chunk_faces.index([round((player.forward[0]) * 4 + player.x),
                                                round((player.forward[2]) * 4 + player.z)])] + (0, 0.5, 0)
        except:
            pass
        c.position = q
        save = 2
#        except:
#            pass
    if key=="up arrow":
        p_verts = p.vertices
        p_norms = p.normals
        p_uvs = p.uvs
        p.clear()
        try:
            [p_verts.pop(count*6+_i_) for _i_ in # mit _i_ mal nehmen und faces hervorheben
            range(6)]  # verts pos distance to mouse pos < 0.5 -> remove verts
            [p_uvs.pop(count*6+_i_) for _i_ in
            range(6)]  # index verts to remove -> update uvs
        except:pass
        p.vertices = p_verts
        p.normals = p_norms
        p.uvs = p_uvs
        p.generate()
        count+=1
#
#p.generate_normals(smooth=10)
app.run()
