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
        terrain = Entity(texture="sand")
        print("yo")
        chunk_faces2=[]
        chunk_faces=[]
        chunk_faces3=[]
        for face_position in chunks_opened[0]:
##            if not face_position[2]>chunks_opened[0][0][2]+15:#yo 15 it is
                chunk_faces2.append(face_position)
                chunk_faces.append([face_position[0],face_position[2]])
                chunk_faces3.append(chunks_opened[1][chunks_opened[0].index(face_position)])
                face = Entity(model="plane", position=face_position, rotation=Vec3(cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][3],cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][4],cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][5]), parent=terrain)
                #print(cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]])
                #face.rotation = Vec3((face.rotation[0]), (face.rotation[1])-2, (face.rotation[2]))

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

# Utility helper to get the sign of a Vec3. Used to determine the face a
# player is looking at without relying on physics colliders.
def _vec_sign(v: Vec3) -> Vec3:
    """Return a Vec3 with the sign of each component of *v*."""
    return Vec3((1 if v.x > 0 else -1 if v.x < 0 else 0),
                (1 if v.y > 0 else -1 if v.y < 0 else 0),
                (1 if v.z > 0 else -1 if v.z < 0 else 0))


def get_target_face(max_distance: int = 12):
    """Return the face position and normal the player is currently looking at.

    The ray is traced from the camera for ``max_distance`` units in steps of
    0.5. The function checks faces inside the current chunk as well as across
    neighbouring chunks. This is a lightweight workaround for block selection
    without using colliders.
    """

    origin = camera.world_position
    direction = camera.forward

    for i in range(int(max_distance * 2)):
        step = i * 0.5
        point = origin + direction * step

        chunk_key = f"{int(point.x // chunk_size)}{int(point.z // chunk_size)}"
        if chunk_key not in chunk_net:
            continue

        chunk_idx = chunk_net.index(chunk_key)
        _, faces, _ = all_chunks[chunk_idx]

        closest_face = None
        min_dist = 0.6
        for fp in faces:
            d = distance(fp, point)
            if d < min_dist:
                min_dist = d
                closest_face = fp

        if closest_face is not None:
            normal = _vec_sign(direction)
            return closest_face, normal

    return None, None

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
    face_pos, _ = get_target_face()
    if face_pos:
        c2.position = face_pos
    else:
        c2.position = floor(player.position + player.forward * 4)


#chunk_net=["00","01","02","03","10","11","12","13","20","21","22","23","30","31","32","33"]

chunk_net = [f"{i}{j}" for i in range(4) for j in range(4)]

count=0


def build():
    global all_chunks, p, c, chunk_net, chunk_size, terrains, combined_terrains, texture

    cint = chunk_net.index(str(int(c.x // chunk_size)) + str(int(c.z // chunk_size)))
    chunk_faces, chunk_faces2, chunk_faces3 = all_chunks[cint]

    pos = Vec3(c.position) + (0, -1.5, 0)

    new_faces = []
    for i in range(6):
        elem = cube_faces[i]
        pos_i = Vec3(elem[0] + pos[0], elem[1] + pos[1], elem[2] + pos[2])
        rot_i = Vec3(elem[3], elem[4], elem[5])
        new_faces.append((pos_i, rot_i))

    affected_chunks = {}
    affected_chunks[cint] = {"faces": [], "to_remove": [], "to_add": []}

    cx = int(c.x // chunk_size)
    cz = int(c.z // chunk_size)
    for face_pos, face_rot in new_faces:
        face_chunk_x = int(face_pos[0] // chunk_size)
        face_chunk_z = int(face_pos[2] // chunk_size)
        face_chunk_idx = chunk_net.index(
            f"{face_chunk_x}{face_chunk_z}") if f"{face_chunk_x}{face_chunk_z}" in chunk_net else None

        if face_chunk_idx is not None:
            if face_chunk_idx not in affected_chunks:
                affected_chunks[face_chunk_idx] = {"faces": [], "to_remove": [], "to_add": []}
            affected_chunks[face_chunk_idx]["faces"].append((face_pos, face_rot))

    for chunk_idx, data in affected_chunks.items():
        chunk_faces, chunk_faces2, chunk_faces3 = all_chunks[chunk_idx]
        new_chunk_faces = []
        new_chunk_faces2 = []
        new_chunk_faces3 = []
        existing_faces = [Vec3(f[0], f[1], f[2]) for f in chunk_faces2]

        pll = 0
        for element in chunk_faces2:
            pos_i = Vec3(element[0], element[1], element[2])
            if pos_i not in [f[0] for f in data["faces"]]:
                new_chunk_faces2.append(element)
                new_chunk_faces.append([pos_i[0], pos_i[2]])
                new_chunk_faces3.append(chunk_faces3[pll])
            else:
                data["to_remove"].append(pos_i)
            pll += 1

        for face_pos, face_rot in data["faces"]:
            if face_pos not in existing_faces:
                data["to_add"].append((face_pos, face_rot))
                new_chunk_faces2.append((face_pos[0], face_pos[1], face_pos[2]))
                new_chunk_faces.append([face_pos[0], face_pos[2]])
                if face_rot == Vec3(180, 0, 0):
                    new_chunk_faces3.append(0)
                elif face_rot == Vec3(0, 0, 0):
                    new_chunk_faces3.append(1)
                elif face_rot == Vec3(90, 0, 0):
                    new_chunk_faces3.append(2)
                elif face_rot == Vec3(-90, 0, 0):
                    new_chunk_faces3.append(3)
                elif face_rot == Vec3(0, 0, 90):
                    new_chunk_faces3.append(4)
                elif face_rot == Vec3(0, 0, -90):
                    new_chunk_faces3.append(5)

        all_chunks[chunk_idx] = [new_chunk_faces, new_chunk_faces2, new_chunk_faces3]

        combined_terrains[chunk_idx].clear()
        terrain2 = Entity()
        for i, face_pos in enumerate(new_chunk_faces2):
            face = Entity(model="plane", position=face_pos, rotation=(cube_faces[new_chunk_faces3[i]][3],cube_faces[new_chunk_faces3[i]][4],cube_faces[new_chunk_faces3[i]][5]), parent=terrain2)
        p = terrain2.combine()
        terrains[chunk_idx] = terrain2
        combined_terrains[chunk_idx] = p
        terrain2.texture = texture

    c.y = -9999


def mine():
    global all_chunks, p, chunk_net, chunk_size, terrains, combined_terrains, texture

    cint = chunk_net.index(str(int(c.x // chunk_size)) + str(int(c.z // chunk_size)))

    affected_chunks = {}
    affected_chunks[cint] = {"faces": [], "to_remove": [], "to_add": []}  # Current chunk

    for cube_face in cube_faces2:
        pos___ = Vec3(cube_face[0], cube_face[1], cube_face[2]) + Vec3(c.position) + Vec3(0, -2.5, 0)

        face_chunk_x = int(pos___[0] // chunk_size)
        face_chunk_z = int(pos___[2] // chunk_size)
        face_chunk_key = f"{face_chunk_x}{face_chunk_z}"

        if face_chunk_key in chunk_net:
            face_chunk_idx = chunk_net.index(face_chunk_key)

            if face_chunk_idx not in affected_chunks:
                affected_chunks[face_chunk_idx] = {"faces": [], "to_remove": [], "to_add": []}

            affected_chunks[face_chunk_idx]["faces"].append((pos___, cube_face))

    for chunk_idx, data in affected_chunks.items():
        chunk_faces, chunk_faces2, chunk_faces3 = all_chunks[chunk_idx]

        combined_terrains[chunk_idx].clear()

        new_chunk_faces = []
        new_chunk_faces2 = []
        new_chunk_faces3 = []

        for pos___, cube_face in data["faces"]:
            if pos___ in chunk_faces2:
                cpos = chunk_faces2.index(pos___)
                data["to_remove"].append(pos___)
            else:
                data["to_add"].append((pos___, cube_faces2.index(cube_face)))

        pll = 0
        for element in chunk_faces2:
            if element not in data["to_remove"]:
                new_chunk_faces2.append(element)
                new_chunk_faces.append([element[0], element[2]])
                new_chunk_faces3.append(chunk_faces3[pll])
            pll += 1

        for face_pos, face_idx in data["to_add"]:
            new_chunk_faces2.append(face_pos)
            new_chunk_faces.append([face_pos[0], face_pos[2]])
            new_chunk_faces3.append(face_idx)

        all_chunks[chunk_idx] = [new_chunk_faces, new_chunk_faces2, new_chunk_faces3]

        terrain2 = Entity()
        for i, face_pos in enumerate(new_chunk_faces2):
            face = Entity(model="plane", position=face_pos,
                          rotation=(cube_faces[new_chunk_faces3[i]][3],cube_faces[new_chunk_faces3[i]][4],cube_faces[new_chunk_faces3[i]][5]), parent=terrain2)

        p = terrain2.combine()
        if hasattr(terrains[chunk_idx], 'disable'):
            terrains[chunk_idx].disable()
        terrains[chunk_idx] = terrain2
        combined_terrains[chunk_idx] = p
        terrain2.texture = texture

    c.y = -9999



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
    if key=="right mouse down" or key=="5":
        face_pos, normal = get_target_face()
        if face_pos:
            c.position = Vec3(face_pos) + normal + Vec3(0, 0.5, 0)
            save = 1
    #if save==3 and mouse.hovered_entity==c:

    if key=="left mouse down":
        face_pos, normal = get_target_face()
        if face_pos:
            c.position = Vec3(face_pos) + Vec3(0, 0.5, 0)
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
