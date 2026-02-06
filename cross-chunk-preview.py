from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np

app = Ursina()
player = FirstPersonController(gravity=0)

cube_faces = [
    (0, 1, 0, 180, 0, 0),
    (0, 2, 0, 0, 0, 0),
    (0, 1.5, 0.5, 90, 0, 0),
    (0, 1.5, -0.5, -90, 0, 0),
    (0.5, 1.5, 0, 0, 0, 90),
    (-0.5, 1.5, 0, 0, 0, -90),
]
cube_faces2 = [
    (0, 2, 0, 180, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 1.5, -0.5, 90, 0, 0),
    (0, 1.5, 0.5, -90, 0, 0),
    (-0.5, 1.5, 0, 0, 0, 90),
    (0.5, 1.5, 0, 0, 0, -90),
]

seed = ord("y") + ord("o")
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


chunk_rendering = 10
all_chunks = []
noise = Perlin()
texture = "sand"
xpos = 0
zpos = 0
chunk_size = 16
combined_terrains = []


def get_from_server_and_render():
    chunks_opened_ = list(eval(open("chunks.txt", "r").read()))
    for chunks_opened in chunks_opened_:
        terrain = Entity(texture="sand")
        print("yo")
        chunk_faces2 = []
        chunk_faces = []
        chunk_faces3 = []

        for face_position in chunks_opened[0]:
            chunk_faces2.append(face_position)
            chunk_faces.append([face_position[0], face_position[2]])
            chunk_faces3.append(chunks_opened[1][chunks_opened[0].index(face_position)])
            Entity(
                model="plane",
                position=face_position,
                rotation=Vec3(
                    cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][3],
                    cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][4],
                    cube_faces[chunks_opened[1][chunks_opened[0].index(face_position)]][5],
                ),
                parent=terrain,
            )

        all_chunks.append([chunk_faces, chunk_faces2, chunk_faces3])
        chunk_faces2 = []
        chunk_faces = []
        chunk_faces3 = []
        p = terrain.combine()
        terrain.clear()
        destroy(terrain)
        combined_terrains.append(p)

    return all_chunks


all_chunks = get_from_server_and_render()

try:
    player.y = all_chunks[1][all_chunks[0].index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]
except:
    pass

mode = 1
save = 0
q = (0, -9999, 0)
c = Entity(model="cube", color=color.clear, collider="box")
c2 = Entity(model="cube", texture="frame")

_FACE_NORMALS = {
    0: Vec3(0, -1, 0),
    1: Vec3(0, 1, 0),
    2: Vec3(0, 0, 1),
    3: Vec3(0, 0, -1),
    4: Vec3(1, 0, 0),
    5: Vec3(-1, 0, 0),
}

_FACE_OFFSETS = [Vec3(*cf[:3]) for cf in cube_faces]

_OPPOSITE_FACE = {
    0: 1, 1: 0,
    2: 3, 3: 2,
    4: 5, 5: 4,
}


def _vkey(v):
    return (round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4))


def _chunk_index_from_pos(pos):
    key = f"{int(pos[0] // chunk_size)}{int(pos[2] // chunk_size)}"
    return chunk_net.index(key) if key in chunk_net else None


def get_target_face(max_distance: int = 12):
    origin = camera.world_position
    direction = camera.forward

    for i in range(int(max_distance * 2)):
        step = i * 0.5
        point = origin + direction * step

        chunk_key = f"{int(point.x // chunk_size)}{int(point.z // chunk_size)}"
        if chunk_key not in chunk_net:
            continue

        chunk_idx = chunk_net.index(chunk_key)
        _, faces, face_indices = all_chunks[chunk_idx]

        closest_face = None
        closest_idx = None
        min_dist = 0.6
        for idx, fp in enumerate(faces):
            d = distance(fp, point)
            if d < min_dist:
                min_dist = d
                closest_face = fp
                closest_idx = face_indices[idx]

        if closest_face is not None:
            normal = _FACE_NORMALS.get(closest_idx, Vec3(0, 1, 0))
            return closest_face, normal, closest_idx

    return None, None, None


def update():
    global mode
    if mode == 0:
        x = player.x // chunk_size
        z = player.z // chunk_size
        try:
            chunk_faces2_ = x * z + 1
            chunk_faces_ = x * z
            chunk_faces3_ = x * z + 2
            player.y = chunk_faces2_[chunk_faces_.index([round(player.x * 2) / 2, round(player.z * 2) / 2])][1]
        except:
            pass

    face_pos, _, _ = get_target_face()
    if face_pos:
        c2.position = Vec3(round(face_pos[0]), round(face_pos[1]), round(face_pos[2])) + (0, -0.5, 0)
    else:
        c2.position = floor(player.position + player.forward * 4)


chunk_net = [f"{i}{j}" for i in range(4) for j in range(4)]
count = 0


def build():
    global all_chunks, c, chunk_net, chunk_size, combined_terrains, texture

    base_chunk_key = f"{int(c.x // chunk_size)}{int(c.z // chunk_size)}"
    if base_chunk_key not in chunk_net:
        c.y = -9999
        return

    pos = Vec3(c.position) + Vec3(0, -1.5, 0)

    # 6 potenzielle Flächen des neuen Blocks
    new_faces = []
    for face_idx, elem in enumerate(cube_faces):
        face_pos = Vec3(elem[0] + pos[0], elem[1] + pos[1], elem[2] + pos[2])
        new_faces.append((face_idx, face_pos))

    # Lookup: world-face-position -> [(chunk_idx, local_face_idx, face_idx), ...]
    face_lookup = {}
    for chunk_idx, (_, chunk_faces2, chunk_faces3) in enumerate(all_chunks):
        for local_idx, fp in enumerate(chunk_faces2):
            key = _vkey(fp)
            face_lookup.setdefault(key, []).append((chunk_idx, local_idx, chunk_faces3[local_idx]))

    remove_indices = {}  # chunk_idx -> set(local_face_idx)
    add_faces = {}       # chunk_idx -> [(face_pos, face_idx), ...]

    for face_idx, face_pos in new_faces:
        key = _vkey(face_pos)
        opposite_idx = _OPPOSITE_FACE[face_idx]
        entries = face_lookup.get(key, [])

        opposite_entry = next((e for e in entries if e[2] == opposite_idx), None)
        same_entry = next((e for e in entries if e[2] == face_idx), None)

        if opposite_entry is not None:
            # Innenfläche: vorhandene Gegenfläche entfernen, neue nicht hinzufügen
            chunk_i, local_i, _ = opposite_entry
            remove_indices.setdefault(chunk_i, set()).add(local_i)
            continue

        if same_entry is not None:
            # Fläche existiert bereits
            continue

        target_chunk_idx = _chunk_index_from_pos(face_pos)
        if target_chunk_idx is None:
            continue

        add_faces.setdefault(target_chunk_idx, []).append((face_pos, face_idx))

    affected_chunks = set(remove_indices.keys()) | set(add_faces.keys())

    for chunk_idx in affected_chunks:
        chunk_faces, chunk_faces2, chunk_faces3 = all_chunks[chunk_idx]
        remove_set = remove_indices.get(chunk_idx, set())

        new_chunk_faces = []
        new_chunk_faces2 = []
        new_chunk_faces3 = []

        for local_idx, element in enumerate(chunk_faces2):
            if local_idx in remove_set:
                continue
            new_chunk_faces2.append(element)
            new_chunk_faces.append([element[0], element[2]])
            new_chunk_faces3.append(chunk_faces3[local_idx])

        existing_keys = {_vkey(fp) for fp in new_chunk_faces2}
        for face_pos, face_idx in add_faces.get(chunk_idx, []):
            k = _vkey(face_pos)
            if k in existing_keys:
                continue
            new_chunk_faces2.append((face_pos[0], face_pos[1], face_pos[2]))
            new_chunk_faces.append([face_pos[0], face_pos[2]])
            new_chunk_faces3.append(face_idx)
            existing_keys.add(k)

        all_chunks[chunk_idx] = [new_chunk_faces, new_chunk_faces2, new_chunk_faces3]

        combined_terrains[chunk_idx].clear()

        terrain2 = Entity(texture="sand")
        for i, face_pos in enumerate(new_chunk_faces2):
            Entity(
                model="plane",
                position=face_pos,
                rotation=(
                    cube_faces[new_chunk_faces3[i]][3],
                    cube_faces[new_chunk_faces3[i]][4],
                    cube_faces[new_chunk_faces3[i]][5],
                ),
                parent=terrain2,
                color=color.brown,
            )

        combined_entity = terrain2.combine()
        combined_terrains[chunk_idx] = combined_entity
        combined_entity.texture = texture
        terrain2.clear()
        destroy(terrain2)

    c.y = -9999


def mine():
    global all_chunks, p, chunk_net, chunk_size, combined_terrains, texture

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
            Entity(
                model="plane",
                position=face_pos,
                rotation=(cube_faces[new_chunk_faces3[i]][3], cube_faces[new_chunk_faces3[i]][4], cube_faces[new_chunk_faces3[i]][5]),
                parent=terrain2,
            )

        p = terrain2.combine()
        combined_terrains[chunk_idx] = p
        p.texture = texture
        terrain2.clear()
        destroy(terrain2)

    c.y = -9999


player.speed = 20
print(len(all_chunks))


def input(key):
    global p, mode, save, q, count

    if key == "g":
        if len(p.vertices) != 0:
            p_verts = p.vertices
            p_norms = p.normals
            p_uvs = p.uvs
            p.clear()
            [p_verts.pop(3 + i) for i in range(6)]
            [p_uvs.pop(3 + i_) for i_ in range(6)]
            p.vertices = p_verts
            p.normals = p_norms
            p.uvs = p_uvs
            p.generate()

    if key == "o":
        mode = 1 - mode

    if key == "m":
        player.y += 1

    if key == "l":
        player.y -= 1

    if save == 1:
        if mouse.hovered_entity == c:
            build()
        else:
            c.y = -9999
        save = 0

    if save == 2:
        if mouse.hovered_entity == c:
            mine()
        c.y = -9999
        save = 0

    if key == "right mouse down" or key == "5":
        face_pos, normal, face_idx = get_target_face()
        if face_pos:
            base_pos = Vec3(face_pos) - _FACE_OFFSETS[face_idx] + normal
            c.position = base_pos + Vec3(0, 1.5, 0)
            save = 1

    if key in ("left mouse down", "4"):
        face_pos, normal, face_idx = get_target_face()
        if face_pos:
            base_pos = Vec3(face_pos) - _FACE_OFFSETS[face_idx] + normal
            c.position = base_pos + Vec3(0, 1.5, 0)
            mine()
            c.y = -9999

    if key == "up arrow":
        p_verts = p.vertices
        p_norms = p.normals
        p_uvs = p.uvs
        p.clear()
        try:
            [p_verts.pop(count * 6 + _i_) for _i_ in range(6)]
            [p_uvs.pop(count * 6 + _i_) for _i_ in range(6)]
        except:
            pass
        p.vertices = p_verts
        p.normals = p_norms
        p.uvs = p_uvs
        p.generate()
        count += 1

    if key == "e":
        player.enabled = not player.enabled
        print(len(scene.entities))


app.run()
