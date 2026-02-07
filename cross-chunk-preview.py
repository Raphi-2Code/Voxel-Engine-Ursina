from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np

app = Ursina()
player = FirstPersonController(gravity=0)
player.speed = 20

cube_faces = [
    (0, 1, 0, 180, 0, 0),      # 0 bottom
    (0, 2, 0, 0, 0, 0),        # 1 top
    (0, 1.5, 0.5, 90, 0, 0),   # 2 +z
    (0, 1.5, -0.5, -90, 0, 0), # 3 -z
    (0.5, 1.5, 0, 0, 0, 90),   # 4 +x
    (-0.5, 1.5, 0, 0, 0, -90), # 5 -x
]

seed = ord('y') + ord('o')
octaves = 0.5
frequency = 8
amplitude = 1

chunk_size = 16
texture = "sand"
chunk_net = [f"{i}{j}" for i in range(4) for j in range(4)]  # 4x4

# all_chunks[chunk_idx] = [chunk_faces(xz list), chunk_faces2(xyz list), chunk_faces3(face_idx list)]
all_chunks = [[[], [], []] for _ in chunk_net]

# Fast structures
chunk_face_sets = [set() for _ in chunk_net]   # set[((x,y,z), face_idx)] per chunk
world_faces = set()                            # union of all chunk_face_sets
face_to_chunk = {}                             # ((x,y,z), face_idx) -> chunk_idx

combined_terrains = [None for _ in chunk_net]

mode = 1
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


class Perlin:
    def __init__(self):
        self.seed = seed
        self.octaves = octaves
        self.freq = frequency
        self.amplitude = amplitude
        self.pNoise = PerlinNoise(seed=self.seed, octaves=self.octaves)

    def get_height(self, x, z):
        return self.pNoise([x / self.freq, z / self.freq]) * self.amplitude


noise = Perlin()


def _vkey(v):
    return (round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4))


def _face_key(pos, face_idx):
    return (_vkey(pos), int(face_idx))


def _face_rotation(face_idx):
    return (
        cube_faces[face_idx][3],
        cube_faces[face_idx][4],
        cube_faces[face_idx][5],
    )


def _chunk_index_from_pos(pos):
    key = f"{int(pos[0] // chunk_size)}{int(pos[2] // chunk_size)}"
    if key not in chunk_net:
        return None
    return chunk_net.index(key)


def _safe_clear_destroy(obj):
    if obj is None:
        return
    try:
        obj.clear()
    except:
        pass
    try:
        destroy(obj)
    except:
        pass


def _sync_chunk_lists(chunk_idx):
    faces2 = []
    faces3 = []
    for pos_key, fidx in chunk_face_sets[chunk_idx]:
        faces2.append((pos_key[0], pos_key[1], pos_key[2]))
        faces3.append(int(fidx))
    faces = [[fp[0], fp[2]] for fp in faces2]
    all_chunks[chunk_idx] = [faces, faces2, faces3]


def _rebuild_chunk_mesh(chunk_idx):
    old = combined_terrains[chunk_idx]
    _safe_clear_destroy(old)

    if len(chunk_face_sets[chunk_idx]) == 0:
        combined_terrains[chunk_idx] = None
        return

    terrain = Entity(texture=texture)

    for pos_key, fidx in chunk_face_sets[chunk_idx]:
        fp = Vec3(pos_key[0], pos_key[1], pos_key[2])
        Entity(
            model="plane",
            position=fp,
            rotation=_face_rotation(fidx),
            parent=terrain,
            # rote Farbe color=color.brown,
        )

    combined = terrain.combine()
    combined_terrains[chunk_idx] = combined
    try:
        combined.texture = texture
    except:
        pass

    terrain.clear()
    destroy(terrain)


def _refresh_chunks(affected_chunks):
    for chunk_idx in affected_chunks:
        if chunk_idx is None:
            continue
        _sync_chunk_lists(chunk_idx)
        _rebuild_chunk_mesh(chunk_idx)


def _remove_face(face_key, affected):
    if face_key not in world_faces:
        return False
    chunk_idx = face_to_chunk.get(face_key)
    if chunk_idx is None:
        return False

    world_faces.discard(face_key)
    face_to_chunk.pop(face_key, None)
    chunk_face_sets[chunk_idx].discard(face_key)
    affected.add(chunk_idx)
    return True


def _add_face(face_key, chunk_idx, affected):
    if chunk_idx is None:
        return False
    if face_key in world_faces:
        return False

    world_faces.add(face_key)
    face_to_chunk[face_key] = chunk_idx
    chunk_face_sets[chunk_idx].add(face_key)
    affected.add(chunk_idx)
    return True


def load_chunks():
    chunks_opened_ = list(eval(open("chunks.txt", "r").read()))

    for chunk_idx, chunk_data in enumerate(chunks_opened_):
        if chunk_idx >= len(chunk_net):
            break

        positions = chunk_data[0]
        indices = chunk_data[1]

        for i, face_pos in enumerate(positions):
            if i >= len(indices):
                break
            fidx = int(indices[i])

            key = _face_key(face_pos, fidx)
            if key in world_faces:
                continue

            world_faces.add(key)
            face_to_chunk[key] = chunk_idx
            chunk_face_sets[chunk_idx].add(key)

    for i in range(len(chunk_net)):
        _sync_chunk_lists(i)
        _rebuild_chunk_mesh(i)


load_chunks()

try:
    # simple spawn near first face
    first_face = next(iter(world_faces))
    player.position = Vec3(first_face[0][0], first_face[0][1] + 2, first_face[0][2])
except:
    player.position = Vec3(0, 6, 0)


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


def build():
    cube_base = Vec3(c.position) + Vec3(0, -1.5, 0)
    affected = set()

    for i, off in enumerate(_FACE_OFFSETS):
        fp = cube_base + off
        same = _face_key(fp, i)
        opp = _face_key(fp, _OPPOSITE_FACE[i])

        if opp in world_faces:
            _remove_face(opp, affected)  # opposite becomes internal
        elif same not in world_faces:
            tgt = _chunk_index_from_pos(fp)
            _add_face(same, tgt, affected)  # new outside face

    _refresh_chunks(affected)
    c.y = -9999


def mine(face_pos=None, face_idx=None):
    if face_pos is None or face_idx is None:
        face_pos, _, face_idx = get_target_face()
        if face_pos is None:
            c.y = -9999
            return

    # IMPORTANT: mine touched block itself (without +normal)
    cube_base = Vec3(face_pos) - _FACE_OFFSETS[face_idx]
    affected = set()

    for i, off in enumerate(_FACE_OFFSETS):
        fp = cube_base + off
        same = _face_key(fp, i)
        opp = _face_key(fp, _OPPOSITE_FACE[i])

        if same in world_faces:
            _remove_face(same, affected)      # remove mined cube face
        else:
            tgt = _chunk_index_from_pos(fp)
            _add_face(opp, tgt, affected)     # expose neighbor face

    _refresh_chunks(affected)
    c.y = -9999


def update():
    face_pos, _, _ = get_target_face()
    if face_pos:
        c2.position = Vec3(round(face_pos[0]), round(face_pos[1]), round(face_pos[2])) + (0, -0.5, 0)
    else:
        c2.position = floor(player.position + player.forward * 4)


def input(key):
    global mode

    if key == "o":
        mode = 1 - mode
    if key == "m":
        player.y += 1
    if key == "l":
        player.y -= 1
    if key == "e":
        player.enabled = not player.enabled
        print(len(scene.entities))

    if key in ("right mouse down", "5"):
        face_pos, normal, face_idx = get_target_face()
        if face_pos:
            # block next to clicked face
            cube_base = Vec3(face_pos) - _FACE_OFFSETS[face_idx] + normal
            c.position = cube_base + Vec3(0, 1.5, 0)
            build()

    if key in ("left mouse down", "4"):
        face_pos, _, face_idx = get_target_face()
        if face_pos:
            mine(face_pos, face_idx)


app.run()
