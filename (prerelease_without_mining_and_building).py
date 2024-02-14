from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np
app=Ursina()#development_mode=False)
#player=FirstPersonController(gravity=0)
player=FirstPersonController(gravity=0)
cube_faces = [(0, 1, 0, 180, 0, 0), (0, 2, 0, 0, 0, 0), (0, 1.5, 0.5, 90, 0, 0), (0, 1.5, -0.5, -90, 0, 0),
              (0.5, 1.5, 0, 0, 0, 90), (-0.5, 1.5, 0, 0, 0, -90)]
cube_faces2 = [(0, 2, 0, 180, 0, 0), (0, 1, 0, 0, 0, 0), (0, 1.5, -0.5, 90, 0, 0), (0, 1.5, 0.5, -90, 0, 0),
              (-0.5, 1.5, 0, 0, 0, 90), (0.5, 1.5, 0, 0, 0, -90)]
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
chunk_rendering=10
all_chunks=[]
noise = Perlin()
texture = "sand"
xpos = 0
zpos = 0
chunk_size = 16
combined_terrains=[]
terrains=[]
def get_from_server_and_render():
    chunks_opened_=list(eval(open('chunks.txt','r').read()))
    for chunks_opened in chunks_opened_:
        terrain = Entity(texture="sand",color=color.red)
        print("yo")
        chunk_faces2=[]
        chunk_faces=[]
        chunk_faces3=[]
        for face_position in chunks_opened[0]:
            if not face_position[2]>chunks_opened[0][0][2]+15:#yo 15 it is
                chunk_faces2.append(face_position)
                chunk_faces.append([face_position[0],face_position[2]])
                chunk_faces3.append(chunks_opened[1][chunks_opened[0].index(face_position)])
                face = Entity(model="plane", position=face_position, rotation=cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]], parent=terrain)
        all_chunks.append([chunk_faces,chunk_faces2,chunk_faces3])
        chunk_faces2=[]
        chunk_faces=[]
        chunk_faces3=[]
        p = terrain.combine()
        combined_terrains.append(p)
        terrains.append(terrain)
    return all_chunks
all_chunks=get_from_server_and_render()
#player.position=chunk_faces2[0]
try:player.y = all_chunks[1][all_chunks[0].index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]
except:pass
mode=1
save=0
q=(0,-9999,0)
c=Entity(model="cube",color=color.clear,collider="box")
c2=Entity(model="cube",texture="frame")

def update():
    global mode
    if mode==0:
        x = player.x//chunk_size
        z = player.z//chunk_size
        try:
            chunk_faces2_=x*z+1
            chunk_faces_=x*z
            chunk_faces3_=x*z+2
            player.y = chunk_faces2_[chunk_faces_.index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]

        except:
            pass
    c2.position = floor(player.position + player.forward * 4)


chunk_net=["00","01","02","03","10","11","12","13","20","21","22","23","30","31","32","33"]
count=0
def build():
    global all_chunks, p
    cint=chunk_net.index(str(str(int(c.x // chunk_size)) + str(int(c.z // chunk_size))))
    chunk_faces,chunk_faces2,chunk_faces3=all_chunks[cint]
    # cube_=Entity()
    aqc2 = []
    aqc3 = []
    pll = 0
    pos = Vec3(c.position) + (0, -1.5, 0)
    for i__ in range(6):
        elem = cube_faces[i__]  # Vec3(vert)+Vec3(0.5,-1,0.5)
        pos_i = Vec3(elem[0] + pos[0], elem[1] + pos[1], elem[2] + pos[2])
        rot_i = Vec3(elem[3], elem[4], elem[5])
        # face = Entity(model="plane", position=pos_i, rotation=rot_i, color=color.yellow, parent=cube_)
        # chunk_faces.append([pos_i[0], pos_i[2]])
        # chunk_faces2.append(pos_i)
        # chunk_faces3.append(i__)
        # aqc_1
        aqc3.append(rot_i)
        aqc2.append(pos_i)

    combined_terrains[cint].clear()
    destroy(terrains[cint])
    terrain2 = Entity()
    new_chunk_faces2 = []
    new_chunk_faces = []
    new_chunk_faces3 = []
    f_pos = []
    for element in chunk_faces2:
        elem = cube_faces[chunk_faces3[pll]]
        pos_i = Vec3(element[0], element[1], element[2])
        rot_i = Vec3(elem[3], elem[4], elem[5])
        if not pos_i in aqc2:
            # print(aqc,pos_i)
            face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain2)
            # if chunk_faces3[pll]!=2 and chunk_faces3[pll]!=3:
            new_chunk_faces2.append((pos_i[0], pos_i[1], pos_i[2]))
            new_chunk_faces.append([pos_i[0], pos_i[2]])
            new_chunk_faces3.append(chunk_faces3[pll])
        else:
            f_pos.append(pos_i)
        pll += 1  # YESSSIRRR!
    chunk_faces2 = new_chunk_faces2
    chunk_faces = new_chunk_faces
    chunk_faces3 = new_chunk_faces3
    for anti_f_pos in aqc2:
        if not anti_f_pos in f_pos:
            face = Entity(model="plane", position=anti_f_pos, rotation=aqc3[aqc2.index(anti_f_pos)], parent=terrain2)
            chunk_faces2.append(face.position)
            chunk_faces.append([face.x, face.z])
            if face.rotation == (180, 0, 0): chunk_faces3.append(0)
            if face.rotation == (0, 0, 0): chunk_faces3.append(1)
            if face.rotation == (90, 0, 0): chunk_faces3.append(2)
            if face.rotation == (-90, 0, 0): chunk_faces3.append(3)
            if face.rotation == (0, 0, 90): chunk_faces3.append(4)
            if face.rotation == (0, 0, -90): chunk_faces3.append(5)
    all_chunks.append([chunk_faces,chunk_faces2,chunk_faces3])
    p = terrain2.combine()
    terrains[cint]=terrain2
    combined_terrains[cint] = p
    terrain2.texture = texture
    c.y = -9999
def mine():
    global all_chunks, p
    x=c.x//chunk_size
    z=c.z//chunk_size
    print(x,z)
    chunk_faces2 = all_chunks[x * z + 1]
    chunk_faces = all_chunks[x * z]
    chunk_faces3 = all_chunks[x * z + 2]
    p.clear()
    destroy(terrain)
    terrain2 = Entity()
    for cube_face in cube_faces2:
        pos___ = Vec3(cube_face[0], cube_face[1], cube_face[2]) + Vec3(c.position) + Vec3(0, -2.5, 0)
        if pos___ in chunk_faces2:
            cpos = chunk_faces2.index(pos___)
            chunk_faces3.pop(cpos)
            chunk_faces2.pop(cpos)
            chunk_faces.pop(cpos)
        else:
            chunk_faces2.append(pos___)
            chunk_faces.append([pos___[0], pos___[2]])
            chunk_faces3.append(cube_faces2.index(cube_face))
    new_chunk_faces = []
    new_chunk_faces2 = []
    new_chunk_faces3 = []
    pll = 0
    for element in chunk_faces2:
        elem = cube_faces[chunk_faces3[pll]]
        pos_i = Vec3(element[0], element[1], element[2])
        rot_i = Vec3(elem[3], elem[4], elem[5])
        face = Entity(model="plane", position=pos_i, rotation=rot_i, parent=terrain2)
        # if chunk_faces3[pll]!=2 and chunk_faces3[pll]!=3:
        new_chunk_faces2.append((pos_i[0], pos_i[1], pos_i[2]))
        new_chunk_faces.append([pos_i[0], pos_i[2]])
        new_chunk_faces3.append(chunk_faces3[pll])
        pll += 1
    print(pll, len(chunk_faces))
    chunk_faces2 = new_chunk_faces2
    chunk_faces3 = new_chunk_faces3
    chunk_faces = new_chunk_faces
    all_chunks.append([chunk_faces, chunk_faces2, chunk_faces3])
    p = terrain2.combine()
    terrain2.texture = "sand"

player.speed=20
print(len(all_chunks))
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
    if key=="l":
        player.y-=1
    if save==1:
        if mouse.hovered_entity == c:
            build()
        else:c.y=-9999
        save=0
    if save==2:
        if mouse.hovered_entity==c:
            mine()
        c.y=-9999
        save=0
    if key=="right mouse down":
        l = []
        print(str(str(int(c2.x // chunk_size)) + str(int(c2.z // chunk_size))))
        #print(cint)
        chunk_faces, chunk_faces2, chunk_faces3 = all_chunks[chunk_net.index(str(str(round(c2.x // chunk_size)) + str(round(c2.z // chunk_size))))]
        try:
            q = chunk_faces2[chunk_faces.index([round((player.forward[0]) * 4 + player.x),
                                                    round((player.forward[2]) * 4 + player.z)])] + (0, 0.5, 0)
        except:pass
        c.position=q
        save=1
    #if save==3 and mouse.hovered_entity==c:

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
    if key=="e":
        player.enabled=not player.enabled
        print(len(scene.entities))
    #print(all_chunks[round((c.x * chunk_size + c.z) // chunk_size)])
#p.generate_normals(smooth=10)
app.run()
