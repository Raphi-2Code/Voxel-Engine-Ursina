from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np
from bisect import bisect_left, bisect_right

app = Ursina()

PLAYER_WIDTH = 0.5
PLAYER_HEIGHT = 1.5

player = FirstPersonController(gravity=0)
player.speed = 5
player.height = PLAYER_HEIGHT
player.camera_pivot.y = 1.9

player.collider = None  # FIX: keine Ursina-Kollision

Sky(texture="clouds.png")

cube_faces = [
    (0, 1, 0, 180, 0, 0),
    (0, 2, 0, 0, 0, 0),
    (0, 1.5, 0.5, 90, 0, 0),
    (0, 1.5, -0.5, -90, 0, 0),
    (0.5, 1.5, 0, 0, 0, 90),
    (-0.5, 1.5, 0, 0, 0, -90),
]

seed = ord("y") + ord("o")
octaves = 0.5
frequency = 8
amplitude = 1

chunk_size = 16
chunk_height = 16
texture = "atlas1"

ATLAS_TILES_X = 4
ATLAS_TILES_Y = 4
ATLAS_BLEED = 0.0015

window.fps_counter.disable()
window.cog_menu.disable()

ATLAS_FLIP_Y = True
DEFAULT_ATLAS_TILE = (0, 0)
BLOCK_FACE_TILES = {
    "grass": {
        0: (2, 0),
        1: (0, 0),
        2: (1, 0),
        3: (1, 0),
        4: (1, 0),
        5: (1, 0),
    },
    "dirt": {
        0: (2, 0),
        1: (2, 0),
        2: (2, 0),
        3: (2, 0),
        4: (2, 0),
        5: (2, 0),
    },
    "stone": {
        0: (3, 0),
        1: (3, 0),
        2: (3, 0),
        3: (3, 0),
        4: (3, 0),
        5: (3, 0),
    },
    "sand": {
        0: (0, 1),
        1: (0, 1),
        2: (0, 1),
        3: (0, 1),
        4: (0, 1),
        5: (0, 1),
    },
    "planks": {
        0: (1, 1),
        1: (1, 1),
        2: (1, 1),
        3: (1, 1),
        4: (1, 1),
        5: (1, 1),
    },
    "leaves": {
        0: (2, 1),
        1: (2, 1),
        2: (2, 1),
        3: (2, 1),
        4: (2, 1),
        5: (2, 1),
    },
    "water": {
        0: (3, 1),
        1: (3, 1),
        2: (3, 1),
        3: (3, 1),
        4: (3, 1),
        5: (3, 1),
    },
}
DEFAULT_BLOCK_TYPE = "grass"
selected_block_type = DEFAULT_BLOCK_TYPE
BLOCK_SELECT_KEYS = {
    "1": "grass",
    "2": "stone",
    "3": "sand",
    "6": "planks",
    "7": "leaves",
    "8": "water",
    "9": "dirt",
}

atlas_texture = load_texture(texture)
if atlas_texture is not None:
    try:
        atlas_texture.filtering = None
    except:
        pass

GRID_X = 4
GRID_Z = 4
base_chunk_coords = [(cx, 0, cz) for cx in range(GRID_X) for cz in range(GRID_Z)]

all_chunks = {}
chunk_face_sets = {}
combined_terrains = {}
for coord in base_chunk_coords:
    all_chunks[coord] = [[], [], []]
    chunk_face_sets[coord] = set()
    combined_terrains[coord] = None

world_faces = set()
face_to_chunk = {}
face_block_types = {}

# WICHTIG: block_types bleibt jetzt persistent, auch wenn ein Block komplett gecullt ist
# (also 0 Faces hat). Gelöscht wird er nur beim tatsächlichen Abbauen.
block_types = {}

top_columns = {}
top_cells = {}
block_face_counts = {}

mode = 1
c = Entity(model="cube", color=color.clear)  # FIX: kein collider
c2 = Entity(model="cube", texture="frame", scale=1.05)

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
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
}

GRAVITY_ACCEL = 0.08#80.0
MAX_FALL_SPEED = 0.42#10.0
JUMP_SPEED = 1.5*3.92#20.0

# NEU: Sprung nur erlauben, wenn über dem Kopf genug Platz ist
MIN_HEADROOM_TO_JUMP = 1.0

PLAYER_STAND_HEIGHT = 0.0
GROUND_STICK = 0.08
MAX_STEP_UP = 0.35
PLAYER_COLLISION_RADIUS = PLAYER_WIDTH * 0.5
PLAYER_FOOT_RADIUS = PLAYER_COLLISION_RADIUS

# FIX (Claude): Clearance-Werte reduzieren
PLAYER_COLLISION_FOOT_CLEARANCE = 0.005
PLAYER_COLLISION_HEAD_CLEARANCE = 0.005

BLOCK_HALF_EXTENT = 0.5
BLOCK_HEIGHT = float(_FACE_OFFSETS[1].y - _FACE_OFFSETS[0].y)
WALL_EPS = 0.001
SWEEP_TOL = 0.005
MAX_PHYSICS_SUBSTEP = 1.0 / 120.0
MAX_PHYSICS_STEPS = 8

PROBE_GRID_STEP = 1.0
PROBE_YAW_STEP = 90.0
PROBE_FACE_SIZE = PLAYER_WIDTH * 2
PROBE_THICK = 0.06
PROBE_FRONT_OFFSET = PLAYER_COLLISION_RADIUS + 0.25
PROBE_SIDE_OFFSET = PLAYER_COLLISION_RADIUS + 0.25

PROBE_COLOR = color.clear  # white33
PROBE_HIT_COLOR = color.clear  # white33
EDGE_PROBE_NAMES = {
    "front_low",
    "front_high",
    "right_low",
    "right_high",
    "left_low",
    "left_high",
    "back_low",
    "back_high",
}
PLAYER_Y_SNAP_STEP = PROBE_GRID_STEP
PLAYER_Y_SNAP_ONLY_GROUNDED = True

player_probe_entities = {}
player_probe_hits = {}

vertical_velocity = 0.0
is_grounded = False
prev_horizontal_x = None
prev_horizontal_z = None


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
    return (cube_faces[face_idx][3], cube_faces[face_idx][4], cube_faces[face_idx][5])


def _normalize_block_type(block_type):
    key = str(block_type)
    if key in BLOCK_FACE_TILES:
        return key
    return DEFAULT_BLOCK_TYPE


def _block_tile_for_face(block_type, face_idx):
    btype = _normalize_block_type(block_type)
    face_map = BLOCK_FACE_TILES.get(btype, {})
    tile = face_map.get(int(face_idx))
    if tile is None:
        return DEFAULT_ATLAS_TILE
    return tile


def _atlas_rect(tile_x, tile_y):
    tx = int(clamp(tile_x, 0, ATLAS_TILES_X - 1))
    ty = int(clamp(tile_y, 0, ATLAS_TILES_Y - 1))

    w = 1.0 / ATLAS_TILES_X
    h = 1.0 / ATLAS_TILES_Y

    uv_row = ty
    if ATLAS_FLIP_Y:
        uv_row = (ATLAS_TILES_Y - 1) - ty

    u0 = tx * w + ATLAS_BLEED
    v0 = uv_row * h + ATLAS_BLEED
    u1 = (tx + 1) * w - ATLAS_BLEED
    v1 = (uv_row + 1) * h - ATLAS_BLEED
    return u0, v0, u1, v1


def _face_uvs(face_idx, block_type, quad_verts):
    tile = _block_tile_for_face(block_type, int(face_idx))
    u0, v0, u1, v1 = _atlas_rect(tile[0], tile[1])

    xs = [float(v.x) for v in quad_verts]
    ys = [float(v.y) for v in quad_verts]
    zs = [float(v.z) for v in quad_verts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    z0, z1 = min(zs), max(zs)

    du = max(u1 - u0, 1e-8)
    dv = max(v1 - v0, 1e-8)
    out = []

    for p in quad_verts:
        vx = float(p.x)
        vy = float(p.y)
        vz = float(p.z)

        if face_idx == 1:
            lu = vx - x0
            lv = z1 - vz
        elif face_idx == 0:
            lu = vx - x0
            lv = vz - z0
        elif face_idx == 2:
            lu = vx - x0
            lv = vy - y0
        elif face_idx == 3:
            lu = x1 - vx
            lv = vy - y0
        elif face_idx == 4:
            lu = z1 - vz
            lv = vy - y0
        else:
            lu = vz - z0
            lv = vy - y0

        lu = clamp(lu, 0.0, 1.0)
        lv = clamp(lv, 0.0, 1.0)
        out.append(Vec2(u0 + lu * du, v0 + lu * 0 + lv * dv))

    # NOTE: obige Zeile v0 + lu*0 ist äquivalent zu v0; lässt es explizit linear wirken.
    # Falls du lieber original willst: out.append(Vec2(u0 + lu * du, v0 + lv * dv))

    # Korrektur auf exakt original:
    out = []
    for p in quad_verts:
        vx = float(p.x)
        vy = float(p.y)
        vz = float(p.z)

        if face_idx == 1:
            lu = vx - x0
            lv = z1 - vz
        elif face_idx == 0:
            lu = vx - x0
            lv = vz - z0
        elif face_idx == 2:
            lu = vx - x0
            lv = vy - y0
        elif face_idx == 3:
            lu = x1 - vx
            lv = vy - y0
        elif face_idx == 4:
            lu = z1 - vz
            lv = vy - y0
        else:
            lu = vz - z0
            lv = vy - y0

        lu = clamp(lu, 0.0, 1.0)
        lv = clamp(lv, 0.0, 1.0)
        out.append(Vec2(u0 + lu * du, v0 + lv * dv))

    return out


def _face_vertices(pos_key, face_idx):
    base = _cube_base_from_face(pos_key, face_idx)
    x = float(base[0])
    y = float(base[1])
    z = float(base[2])

    x0 = x - BLOCK_HALF_EXTENT
    x1 = x + BLOCK_HALF_EXTENT
    y0 = y + float(_FACE_OFFSETS[0].y)
    y1 = y + float(_FACE_OFFSETS[1].y)
    z0 = z - BLOCK_HALF_EXTENT
    z1 = z + BLOCK_HALF_EXTENT

    if face_idx == 0:
        return [
            Vec3(x0, y0, z0),
            Vec3(x1, y0, z0),
            Vec3(x1, y0, z1),
            Vec3(x0, y0, z1),
        ]
    if face_idx == 1:
        return [
            Vec3(x0, y1, z0),
            Vec3(x0, y1, z1),
            Vec3(x1, y1, z1),
            Vec3(x1, y1, z0),
        ]
    if face_idx == 2:
        return [
            Vec3(x0, y0, z1),
            Vec3(x1, y0, z1),
            Vec3(x1, y1, z1),
            Vec3(x0, y1, z1),
        ]
    if face_idx == 3:
        return [
            Vec3(x0, y0, z0),
            Vec3(x0, y1, z0),
            Vec3(x1, y1, z0),
            Vec3(x1, y0, z0),
        ]
    if face_idx == 4:
        return [
            Vec3(x1, y0, z0),
            Vec3(x1, y1, z0),
            Vec3(x1, y1, z1),
            Vec3(x1, y0, z1),
        ]
    return [
        Vec3(x0, y0, z0),
        Vec3(x0, y0, z1),
        Vec3(x0, y1, z1),
        Vec3(x0, y1, z0),
    ]


def _chunk_coord_from_pos(pos):
    cx = math.floor(float(pos[0]) / chunk_size)
    cy = math.floor(float(pos[1]) / chunk_height)
    cz = math.floor(float(pos[2]) / chunk_size)
    return (cx, cy, cz)


def _legacy_chunk_coord_from_index(idx):
    cx = int(idx) // GRID_Z
    cz = int(idx) % GRID_Z
    return (cx, 0, cz)


def _ensure_chunk(chunk_coord):
    key = (int(chunk_coord[0]), int(chunk_coord[1]), int(chunk_coord[2]))
    if key not in chunk_face_sets:
        chunk_face_sets[key] = set()
        all_chunks[key] = [[], [], []]
        combined_terrains[key] = None
    return key


def _reset_chunk_storage():
    for obj in combined_terrains.values():
        _safe_clear_destroy(obj)
    all_chunks.clear()
    chunk_face_sets.clear()
    combined_terrains.clear()
    for coord in base_chunk_coords:
        all_chunks[coord] = [[], [], []]
        chunk_face_sets[coord] = set()
        combined_terrains[coord] = None


def _safe_clear_destroy(obj):
    if obj is None:
        return
    try:
        obj.enabled = False
    except:
        pass
    try:
        obj.model = None
    except:
        pass
    try:
        obj.collider = None
    except:
        pass
    try:
        obj.clear()
    except:
        pass
    try:
        destroy(obj)
    except:
        pass


def _sync_chunk_lists(chunk_coord):
    chunk_coord = _ensure_chunk(chunk_coord)
    faces2 = []
    faces3 = []
    for pos_key, fidx in chunk_face_sets[chunk_coord]:
        faces2.append((pos_key[0], pos_key[1], pos_key[2]))
        faces3.append(int(fidx))
    faces = [[fp[0], fp[2]] for fp in faces2]
    all_chunks[chunk_coord] = [faces, faces2, faces3]


def _set_block_type(base, block_type):
    btype = _normalize_block_type(block_type)
    block_types[base] = btype
    for i in range(len(_FACE_OFFSETS)):
        fp = _face_pos_from_base(base, i)
        fk = _face_key(fp, i)
        if fk in face_block_types:
            face_block_types[fk] = btype


def _apply_surface_layers():
    top_y_by_col = {}
    for base in block_types.keys():
        col = (base[0], base[2])
        y = base[1]
        prev = top_y_by_col.get(col)
        if prev is None or y > prev:
            top_y_by_col[col] = y

    for base, btype in list(block_types.items()):
        if _normalize_block_type(btype) != "grass":
            continue
        col = (base[0], base[2])
        if base[1] < top_y_by_col[col]:
            _set_block_type(base, "dirt")


def _block_type_from_face_key(face_key):
    base = _cube_base_from_face(face_key[0], face_key[1])
    return _normalize_block_type(block_types.get(base, DEFAULT_BLOCK_TYPE))


def _rebuild_chunk_mesh(chunk_coord):
    chunk_coord = _ensure_chunk(chunk_coord)
    old = combined_terrains.get(chunk_coord)

    if len(chunk_face_sets[chunk_coord]) == 0:
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return

    vertices = []
    triangles = []
    uvs = []
    normals = []

    for pos_key, fidx in chunk_face_sets[chunk_coord]:
        face_key = (pos_key, int(fidx))
        btype = _block_type_from_face_key(face_key)
        quad_verts = _face_vertices(pos_key, int(fidx))
        quad_uvs = _face_uvs(int(fidx), btype, quad_verts)
        n = _FACE_NORMALS.get(int(fidx), Vec3(0, 1, 0))
        idx0 = len(vertices)

        vertices.extend(quad_verts)
        uvs.extend(quad_uvs)
        normals.extend([n, n, n, n])
        triangles.extend([idx0, idx0 + 2, idx0 + 1, idx0, idx0 + 3, idx0 + 2])

    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        uvs=uvs,
        normals=normals,
        mode="triangle",
        static=True,
    )
    tex = atlas_texture if atlas_texture is not None else texture

    if old is None:
        combined_terrains[chunk_coord] = Entity(model=mesh, texture=tex)  # FIX: kein collider
        return

    try:
        old.model = mesh
        old.texture = tex
        old.collider = None  # FIX: kein collider
        old.enabled = True
        combined_terrains[chunk_coord] = old
    except:
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = Entity(model=mesh, texture=tex)  # FIX: kein collider


def _refresh_chunks(affected_chunks):
    for chunk_coord in affected_chunks:
        if chunk_coord is None:
            continue
        _rebuild_chunk_mesh(chunk_coord)


def _expand_chunk_neighborhood(chunks, radius=1):
    out = set()
    for chunk_coord in chunks:
        if chunk_coord is None:
            continue
        cx, cy, cz = chunk_coord
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key in chunk_face_sets:
                        out.add(key)
    if len(out) == 0:
        return set(chunks)
    return out


def _cube_base_from_face(pos_key, face_idx):
    off = _FACE_OFFSETS[int(face_idx)]
    return _vkey((pos_key[0] - off.x, pos_key[1] - off.y, pos_key[2] - off.z))


def _face_pos_from_base(base, face_idx):
    off = _FACE_OFFSETS[int(face_idx)]
    return _vkey((base[0] + off.x, base[1] + off.y, base[2] + off.z))


def _chunk_coord_from_face(pos_key, face_idx):
    base = _cube_base_from_face(pos_key, face_idx)
    return _chunk_coord_from_pos(base)


def _entry_face_from_axis(axis, step):
    if axis == 0:
        return 5 if step > 0 else 4
    if axis == 1:
        return 0 if step > 0 else 1
    return 3 if step > 0 else 2


def _register_top_face(pos_key, face_idx):
    base = _cube_base_from_face(pos_key, face_idx)
    prev = block_face_counts.get(base, 0)
    block_face_counts[base] = prev + 1
    if prev > 0:
        return

    x, yb, z = base
    y_top = round(yb + _FACE_OFFSETS[1].y, 4)
    col = (x, z)

    ys = top_columns.setdefault(col, [])
    idx = bisect_left(ys, y_top)
    if idx >= len(ys) or ys[idx] != y_top:
        ys.insert(idx, y_top)

    cell = (math.floor(x), math.floor(z))
    top_cells.setdefault(cell, set()).add(col)


def _unregister_top_face(pos_key, face_idx):
    base = _cube_base_from_face(pos_key, face_idx)
    prev = block_face_counts.get(base, 0)
    if prev == 0:
        return
    if prev > 1:
        block_face_counts[base] = prev - 1
        return
    block_face_counts.pop(base, None)

    x, yb, z = base
    y_top = round(yb + _FACE_OFFSETS[1].y, 4)
    col = (x, z)

    ys = top_columns.get(col)
    if not ys:
        return
    idx = bisect_left(ys, y_top)
    if idx >= len(ys) or ys[idx] != y_top:
        return
    ys.pop(idx)

    if ys:
        return

    top_columns.pop(col, None)
    cell = (math.floor(x), math.floor(z))
    cols = top_cells.get(cell)
    if cols is None:
        return
    cols.discard(col)
    if len(cols) == 0:
        top_cells.pop(cell, None)


def _find_support_y(px, pz, foot_y, max_up):
    reach = 0.5 + PLAYER_FOOT_RADIUS
    best = None
    ceiling = foot_y + max_up

    min_cx = math.floor(px - reach)
    max_cx = math.floor(px + reach)
    min_cz = math.floor(pz - reach)
    max_cz = math.floor(pz + reach)

    for cx in range(min_cx, max_cx + 1):
        for cz in range(min_cz, max_cz + 1):
            cols = top_cells.get((cx, cz))
            if not cols:
                continue
            for col in cols:
                x, z = col
                if abs(px - x) > reach or abs(pz - z) > reach:
                    continue
                ys = top_columns.get(col)
                if not ys:
                    continue
                idx = bisect_right(ys, ceiling)
                if idx == 0:
                    continue
                y = ys[idx - 1]
                if best is None or y > best:
                    best = y

    return best


def _find_support_y_fallback(px, pz, foot_y, max_up):
    reach = 0.5 + PLAYER_FOOT_RADIUS
    best = None
    ceiling = foot_y + max_up
    top_off = _FACE_OFFSETS[1].y

    for base in block_face_counts.keys():
        x, yb, z = base
        if abs(px - x) > reach or abs(pz - z) > reach:
            continue
        y_top = round(yb + top_off, 4)
        if y_top > ceiling:
            continue
        if best is None or y_top > best:
            best = y_top

    return best


def _player_body_y_span():
    y_min = float(player.y) + PLAYER_COLLISION_FOOT_CLEARANCE
    y_max = float(player.y) + float(player.height) - PLAYER_COLLISION_HEAD_CLEARANCE
    return y_min, y_max


def _iter_candidate_columns(min_x, max_x, min_z, max_z):
    seen = set()
    min_cx = math.floor(min_x - BLOCK_HALF_EXTENT)
    max_cx = math.floor(max_x + BLOCK_HALF_EXTENT)
    min_cz = math.floor(min_z - BLOCK_HALF_EXTENT)
    max_cz = math.floor(max_z + BLOCK_HALF_EXTENT)

    for cx in range(min_cx, max_cx + 1):
        for cz in range(min_cz, max_cz + 1):
            cols = top_cells.get((cx, cz))
            if not cols:
                continue
            for col in cols:
                if col in seen:
                    continue
                seen.add(col)
                x, z = col
                if x + BLOCK_HALF_EXTENT <= min_x or x - BLOCK_HALF_EXTENT >= max_x:
                    continue
                if z + BLOCK_HALF_EXTENT <= min_z or z - BLOCK_HALF_EXTENT >= max_z:
                    continue
                ys = top_columns.get(col)
                if ys:
                    yield col, ys


def _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
    for col, ys in _iter_candidate_columns(min_x, max_x, min_z, max_z):
        x, z = col
        bx0 = x - BLOCK_HALF_EXTENT
        bx1 = x + BLOCK_HALF_EXTENT
        bz0 = z - BLOCK_HALF_EXTENT
        bz1 = z + BLOCK_HALF_EXTENT
        for y_top in ys:
            by1 = y_top
            by0 = y_top - BLOCK_HEIGHT
            if y_max < by0 or y_min > by1:
                continue
            yield bx0, bx1, bz0, bz1


def _aabb_hit_info(min_x, max_x, min_y, max_y, min_z, max_z):
    hit_chunks = set()
    top_off = float(_FACE_OFFSETS[1].y)

    for col, ys in _iter_candidate_columns(min_x, max_x, min_z, max_z):
        x, z = col
        bx0 = x - BLOCK_HALF_EXTENT
        bx1 = x + BLOCK_HALF_EXTENT
        bz0 = z - BLOCK_HALF_EXTENT
        bz1 = z + BLOCK_HALF_EXTENT

        if max_x <= bx0 or min_x >= bx1:
            continue
        if max_z <= bz0 or min_z >= bz1:
            continue

        for y_top in ys:
            by1 = y_top
            by0 = y_top - BLOCK_HEIGHT
            if max_y <= by0 or min_y >= by1:
                continue
            base_y = y_top - top_off
            hit_chunks.add(_chunk_coord_from_pos((x, base_y, z)))

    return (len(hit_chunks) > 0), hit_chunks


def _aabb_hits_any_block(min_x, max_x, min_y, max_y, min_z, max_z):
    hit, _ = _aabb_hit_info(min_x, max_x, min_y, max_y, min_z, max_z)
    return hit


def _chunk_has_collider(chunk_coord):
    ent = combined_terrains.get(chunk_coord)
    if ent is None:
        return False
    if not getattr(ent, "enabled", True):
        return False
    return getattr(ent, "collider", None) is not None


# NEU: Jump blockieren, wenn direkt über dem Kopf ein Block ist
def _jump_blocked_by_ceiling():
    _, head_y = _player_body_y_span()
    px = float(player.x)
    pz = float(player.z)
    r = PLAYER_COLLISION_RADIUS
    return _aabb_hits_any_block(px - r, px + r, head_y, head_y + MIN_HEADROOM_TO_JUMP, pz - r, pz + r)


def _sweep_x(start_x, target_x, z, y_min, y_max):
    dx = target_x - start_x
    if abs(dx) < 1e-8:
        return target_x

    radius = PLAYER_COLLISION_RADIUS
    min_x = min(start_x, target_x) - radius
    max_x = max(start_x, target_x) + radius
    min_z = z - radius
    max_z = z + radius

    if dx > 0:
        limit = target_x
        for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
            if max_z <= bz0 or min_z >= bz1:
                continue
            boundary = bx0 - radius
            if start_x <= boundary + SWEEP_TOL and target_x > boundary and boundary < limit:
                limit = boundary - WALL_EPS
        return limit

    limit = target_x
    for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
        if max_z <= bz0 or min_z >= bz1:
            continue
        boundary = bx1 + radius
        if start_x >= boundary - SWEEP_TOL and target_x < boundary and boundary > limit:
            limit = boundary + WALL_EPS
    return limit


def _sweep_z(start_z, target_z, x, y_min, y_max):
    dz = target_z - start_z
    if abs(dz) < 1e-8:
        return target_z

    radius = PLAYER_COLLISION_RADIUS
    min_x = x - radius
    max_x = x + radius
    min_z = min(start_z, target_z) - radius
    max_z = max(start_z, target_z) + radius

    if dz > 0:
        limit = target_z
        for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
            if max_x <= bx0 or min_x >= bx1:
                continue
            boundary = bz0 - radius
            if start_z <= boundary + SWEEP_TOL and target_z > boundary and boundary < limit:
                limit = boundary - WALL_EPS
        return limit

    limit = target_z
    for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
        if max_x <= bx0 or min_x >= bx1:
            continue
        boundary = bz1 + radius
        if start_z >= boundary - SWEEP_TOL and target_z < boundary and boundary > limit:
            limit = boundary + WALL_EPS
    return limit


def _resolve_horizontal_penetration(px, pz, y_min, y_max):
    radius = PLAYER_COLLISION_RADIUS

    for _ in range(12):
        moved = False
        min_x = px - radius
        max_x = px + radius
        min_z = pz - radius
        max_z = pz + radius

        for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
            if max_x <= bx0 or min_x >= bx1 or max_z <= bz0 or min_z >= bz1:
                continue

            overlap_x = min(max_x - bx0, bx1 - min_x)
            overlap_z = min(max_z - bz0, bz1 - min_z)
            center_x = (bx0 + bx1) * 0.5
            center_z = (bz0 + bz1) * 0.5

            if overlap_x < overlap_z:
                direction = -1.0 if px < center_x else 1.0
                px += direction * (overlap_x + WALL_EPS)
            else:
                direction = -1.0 if pz < center_z else 1.0
                pz += direction * (overlap_z + WALL_EPS)

            moved = True
            break

        if not moved:
            break

    return px, pz


def _round_probe(value, step=PROBE_GRID_STEP):
    return round(float(value) / step) * step


def _round_probe_vec(v, step=PROBE_GRID_STEP):
    return Vec3(_round_probe(v.x, step), _round_probe(v.y, step), _round_probe(v.z, step))


def _snap_probe_yaw(yaw):
    return _round_probe(float(yaw), PROBE_YAW_STEP)


def _player_facing_vectors():
    yaw = math.radians(_snap_probe_yaw(player.rotation_y))
    forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
    right = Vec3(forward.z, 0, -forward.x)
    return forward, right


def _player_probe_layout():
    h = float(player.height)
    y_low = 0.65
    y_high = min(h - 0.2, 1.55)
    y_top = h + 0.05

    s = PROBE_FACE_SIZE * 0.5
    t = PROBE_THICK * 0.5

    side_half = Vec3(s, s, t)
    flat_half = Vec3(s, t, s)

    return [
        ("front_low", Vec3(0, y_low, PROBE_FRONT_OFFSET), side_half, 0),
        ("front_high", Vec3(0, y_high, PROBE_FRONT_OFFSET), side_half, 0),
        ("right_low", Vec3(PROBE_SIDE_OFFSET, y_low, 0), side_half, 90),
        ("right_high", Vec3(PROBE_SIDE_OFFSET, y_high, 0), side_half, 90),
        ("left_low", Vec3(-PROBE_SIDE_OFFSET, y_low, 0), side_half, -90),
        ("left_high", Vec3(-PROBE_SIDE_OFFSET, y_high, 0), side_half, -90),
        ("back_low", Vec3(0, y_low, -PROBE_FRONT_OFFSET), side_half, 180),
        ("back_high", Vec3(0, y_high, -PROBE_FRONT_OFFSET), side_half, 180),
        ("top", Vec3(0, y_top, 0), flat_half, 0),
        ("bottom", Vec3(0, -PROBE_THICK, 0), flat_half, 0),
    ]


def _ensure_player_probes():
    if player_probe_entities:
        return
    for name, _, half, _ in _player_probe_layout():
        e = Entity(
            model="cube",
            color=PROBE_COLOR,
            collider="box",  # FIX: Collider vorhanden
            scale=half * 2,
        )
        e.collision = False  # FIX: nur aktiv, wenn Hit auf Chunk-Face/Block
        player_probe_entities[name] = e
        player_probe_hits[name] = False


def _world_pos_from_local(base_pos, local_offset):
    forward, right = _player_facing_vectors()
    return base_pos + (right * local_offset.x) + Vec3(0, local_offset.y, 0) + (forward * local_offset.z)


def _player_stands_on_face_collider(base_position, hits):
    if hits.get("bottom", False):
        return True

    px = float(base_position.x)
    pz = float(base_position.z)
    foot_y = _round_probe(float(base_position.y) - PLAYER_STAND_HEIGHT, PLAYER_Y_SNAP_STEP)
    max_check = GROUND_STICK + 0.05

    support_y = _find_support_y(px, pz, foot_y, max_check)
    if support_y is None and len(block_face_counts) > 0:
        support_y = _find_support_y_fallback(px, pz, foot_y, max_check)
    if support_y is None:
        return False

    return abs(foot_y - support_y) <= max_check


def _snap_player_y_to_grid(force=False):
    if PLAYER_Y_SNAP_ONLY_GROUNDED and not force and not is_grounded:
        return
    snapped_y = _round_probe(float(player.y), PLAYER_Y_SNAP_STEP)
    if abs(float(player.y) - snapped_y) < 1e-6:
        return
    player.y = snapped_y
    _sample_player_probes_at(Vec3(float(player.x), float(player.y), float(player.z)), do_assign=True)


def _sample_player_probes_at(base_position, do_assign=True):
    _ensure_player_probes()
    hits = {}
    sampled = []

    snapped_base = _round_probe_vec(base_position)
    base_yaw = _snap_probe_yaw(player.rotation_y)

    for name, local_off, local_half, yaw_off in _player_probe_layout():
        raw_center = _world_pos_from_local(snapped_base, local_off)
        center = raw_center

        half_xz = max(float(local_half.x), float(local_half.z))
        min_x = float(center.x - half_xz)
        max_x = float(center.x + half_xz)
        min_y = float(center.y - local_half.y)
        max_y = float(center.y + local_half.y)
        min_z = float(center.z - half_xz)
        max_z = float(center.z + half_xz)

        hit, hit_chunks = _aabb_hit_info(min_x, max_x, min_y, max_y, min_z, max_z)
        has_chunk_collider = any(_chunk_has_collider(chunk_coord) for chunk_coord in hit_chunks)

        hits[name] = hit
        sampled.append((name, center, local_half, yaw_off, hit, has_chunk_collider))

    if do_assign:
        for name, center, local_half, yaw_off, hit, has_chunk_collider in sampled:
            probe = player_probe_entities[name]
            probe.position = center
            probe.scale = local_half * 2
            if name in ("top", "bottom"):
                probe.rotation = Vec3(0, 0, 0)
            else:
                probe.rotation = Vec3(0, base_yaw + yaw_off, 0)

            probe.collision = bool(hit)
            probe.color = PROBE_HIT_COLOR if hit else PROBE_COLOR

    player_probe_hits.clear()
    player_probe_hits.update(hits)
    return hits


def _apply_player_probe_horizontal():
    global prev_horizontal_x, prev_horizontal_z

    cur_x = float(player.x)
    cur_z = float(player.z)

    if prev_horizontal_x is None or prev_horizontal_z is None:
        prev_horizontal_x = cur_x
        prev_horizontal_z = cur_z
        _sample_player_probes_at(Vec3(cur_x, float(player.y), cur_z), do_assign=True)
        return

    y_min, y_max = _player_body_y_span()

    dx = cur_x - prev_horizontal_x  # FIX: größere Achse zuerst
    dz = cur_z - prev_horizontal_z
    if abs(dx) >= abs(dz):
        res_x = _sweep_x(prev_horizontal_x, cur_x, prev_horizontal_z, y_min, y_max)
        res_z = _sweep_z(prev_horizontal_z, cur_z, res_x, y_min, y_max)
    else:
        res_z = _sweep_z(prev_horizontal_z, cur_z, prev_horizontal_x, y_min, y_max)
        res_x = _sweep_x(prev_horizontal_x, cur_x, res_z, y_min, y_max)

    res_x, res_z = _resolve_horizontal_penetration(res_x, res_z, y_min, y_max)

    _sample_player_probes_at(Vec3(res_x, float(player.y), res_z), do_assign=False)
    player.x = res_x
    player.z = res_z
    prev_horizontal_x = float(player.x)
    prev_horizontal_z = float(player.z)
    _sample_player_probes_at(Vec3(prev_horizontal_x, float(player.y), prev_horizontal_z), do_assign=True)


def _block_bounds_from_base(base):
    x = float(base[0])
    y = float(base[1])
    z = float(base[2])
    by0 = y + float(_FACE_OFFSETS[0].y)
    by1 = y + float(_FACE_OFFSETS[1].y)
    bx0 = x - BLOCK_HALF_EXTENT
    bx1 = x + BLOCK_HALF_EXTENT
    bz0 = z - BLOCK_HALF_EXTENT
    bz1 = z + BLOCK_HALF_EXTENT
    return bx0, bx1, by0, by1, bz0, bz1


def _block_intersects_player(base):
    bx0, bx1, by0, by1, bz0, bz1 = _block_bounds_from_base(base)
    y_min, y_max = _player_body_y_span()

    if y_max <= by0 or y_min >= by1:
        return False

    px = float(player.x)
    pz = float(player.z)
    closest_x = clamp(px, bx0, bx1)
    closest_z = clamp(pz, bz0, bz1)
    dx = px - closest_x
    dz = pz - closest_z
    return (dx * dx + dz * dz) <= (PLAYER_COLLISION_RADIUS * PLAYER_COLLISION_RADIUS)


# FIX (Claude): Neue Funktion _can_stand_at hinzufügen
def _can_stand_at(px, pz, foot_y):
    """Prüft ob der Spieler an dieser Position frei von Blöcken ist."""
    y_min = foot_y + PLAYER_COLLISION_FOOT_CLEARANCE
    y_max = foot_y + PLAYER_HEIGHT - PLAYER_COLLISION_HEAD_CLEARANCE
    min_x = px - PLAYER_COLLISION_RADIUS
    max_x = px + PLAYER_COLLISION_RADIUS
    min_z = pz - PLAYER_COLLISION_RADIUS
    max_z = pz + PLAYER_COLLISION_RADIUS
    return not _aabb_hits_any_block(min_x, max_x, y_min, y_max, min_z, max_z)


def _has_block_in_direction(dir_vec, distance=PROBE_FRONT_OFFSET, half_y=0.35):
    d = Vec3(float(dir_vec.x), 0.0, float(dir_vec.z))
    if d.length_squared() < 1e-8:
        return False
    d = d.normalized()

    px = float(player.x) + d.x * distance
    pz = float(player.z) + d.z * distance
    r = PLAYER_COLLISION_RADIUS

    y_low = float(player.y) + 0.65
    y_high = float(player.y) + min(float(player.height) - 0.2, 1.55)

    low_hit = _aabb_hits_any_block(px - r, px + r, y_low - half_y, y_low + half_y, pz - r, pz + r)
    high_hit = _aabb_hits_any_block(px - r, px + r, y_high - half_y, y_high + half_y, pz - r, pz + r)
    return low_hit or high_hit


def get_neighbor_block_hits(distance=PROBE_FRONT_OFFSET):
    return {
        "front": _has_block_in_direction(player.forward, distance),
        "back": _has_block_in_direction(player.back, distance),
        "left": _has_block_in_direction(player.left, distance),
        "right": _has_block_in_direction(player.right, distance),
    }


def get_front_back_left_right_hits(direction=None, distance=PROBE_FRONT_OFFSET, half_y=0.35):
    r = PLAYER_COLLISION_RADIUS
    y_low = float(player.y) + 0.65
    y_high = float(player.y) + min(float(player.height) - 0.2, 1.55)

    hits = {}
    for name, dir_vec in (("front", player.forward), ("back", player.back), ("left", player.left), ("right", player.right)):
        d = Vec3(float(dir_vec.x), 0.0, float(dir_vec.z))
        if d.length_squared() < 1e-8:
            hits[name] = False
            continue

        d = d.normalized()
        px = float(player.x) + d.x * distance
        pz = float(player.z) + d.z * distance

        low_hit = _aabb_hits_any_block(px - r, px + r, y_low - half_y, y_low + half_y, pz - r, pz + r)
        high_hit = _aabb_hits_any_block(px - r, px + r, y_high - half_y, y_high + half_y, pz - r, pz + r)
        hits[name] = low_hit or high_hit

    if direction is None:
        return hits

    key = str(direction).strip().lower()
    return hits.get(key, False)


def _movement_input_dir_xz():
    yaw = math.radians(float(player.rotation_y))
    forward_x = math.sin(yaw)
    forward_z = math.cos(yaw)
    right_x = forward_z
    right_z = -forward_x

    mx = 0.0
    mz = 0.0

    if held_keys["w"]:
        mx += forward_x
        mz += forward_z
    if held_keys["s"]:
        mx -= forward_x
        mz -= forward_z
    if held_keys["d"]:
        mx += right_x
        mz += right_z
    if held_keys["a"]:
        mx -= right_x
        mz -= right_z

    l2 = mx * mx + mz * mz
    if l2 <= 1e-8:
        return None

    inv_len = 1.0 / math.sqrt(l2)
    return mx * inv_len, mz * inv_len


def _apply_vector_gravity():
    global vertical_velocity, is_grounded
    dt_total = time.dt
    if dt_total <= 0:
        return

    steps = max(1, int(math.ceil(dt_total / MAX_PHYSICS_SUBSTEP)))
    steps = min(steps, MAX_PHYSICS_STEPS)
    dt = dt_total / steps

    for _ in range(steps):
        px = float(player.x)
        pz = float(player.z)
        current_foot = float(player.y) - PLAYER_STAND_HEIGHT

        support_scan_up = MAX_STEP_UP
        if vertical_velocity < 0:
            support_scan_up = MAX_STEP_UP + max(BLOCK_HEIGHT, (-vertical_velocity * dt) + 0.05)

        support_y = _find_support_y(px, pz, current_foot, support_scan_up)
        if support_y is None and len(block_face_counts) > 0:
            support_y = _find_support_y_fallback(px, pz, current_foot, support_scan_up)

        if support_y is not None and current_foot < support_y:
            if not _can_stand_at(px, pz, support_y):
                continue
            player.y = support_y + PLAYER_STAND_HEIGHT
            vertical_velocity = 0.0
            is_grounded = True
            _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
            continue

        if support_y is not None and vertical_velocity <= 0:
            d = current_foot - support_y
            if 0 <= d <= GROUND_STICK:
                player.y = support_y + PLAYER_STAND_HEIGHT
                vertical_velocity = 0.0
                is_grounded = True
                _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
                continue

        vertical_velocity = max(vertical_velocity - GRAVITY_ACCEL, -MAX_FALL_SPEED/dt)#GRAVITY_ACCEL*dt
        next_y = float(player.y) + vertical_velocity * dt
        next_foot = next_y - PLAYER_STAND_HEIGHT

        probe_hits = _sample_player_probes_at(Vec3(px, next_y, pz), do_assign=False)

        if vertical_velocity > 0 and probe_hits.get("top", False):
            player.y = float(player.y)
            vertical_velocity = 0.0
            is_grounded = False
            _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
            continue

        if support_y is not None and vertical_velocity <= 0 and next_foot <= support_y:
            player.y = support_y + PLAYER_STAND_HEIGHT
            vertical_velocity = 0.0
            is_grounded = True
            _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
            continue

        if vertical_velocity <= 0 and probe_hits.get("bottom", False):
            snap = _find_support_y(px, pz, next_foot, support_scan_up)
            if snap is None:
                snap = support_y
            if snap is not None and next_foot <= snap + 0.25:
                if not _can_stand_at(px, pz, snap):
                    continue
                player.y = snap + PLAYER_STAND_HEIGHT
                vertical_velocity = 0.0
                is_grounded = True
                _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
                continue

        player.y = next_y
        is_grounded = False
        _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)


def _remove_face(face_key, affected):
    if face_key not in world_faces:
        return False
    chunk_coord = face_to_chunk.get(face_key)
    removed_from_chunk_set = False

    if chunk_coord is not None and chunk_coord in chunk_face_sets and face_key in chunk_face_sets[chunk_coord]:
        chunk_face_sets[chunk_coord].discard(face_key)
        affected.add(chunk_coord)
        removed_from_chunk_set = True
    else:
        for coord, fset in chunk_face_sets.items():
            if face_key in fset:
                fset.discard(face_key)
                affected.add(coord)
                removed_from_chunk_set = True
                break

    world_faces.discard(face_key)
    face_to_chunk.pop(face_key, None)
    face_block_types.pop(face_key, None)
    _unregister_top_face(face_key[0], face_key[1])

    if not removed_from_chunk_set:
        affected.update(chunk_face_sets.keys())

    return True


def _infer_block_type_for_hidden_block(base):
    above = _vkey((base[0], base[1] + BLOCK_HEIGHT, base[2]))
    above_type = block_types.get(above)

    if above_type is not None:
        above_type = _normalize_block_type(above_type)
        if above_type in ("grass", "dirt"):
            return "dirt"
        return above_type

    return "dirt"


def _add_face(face_key, chunk_coord, affected, block_type=None):
    chunk_coord = _ensure_chunk(chunk_coord)
    if face_key in world_faces:
        return False

    base = _cube_base_from_face(face_key[0], face_key[1])

    if block_type is None:
        if base in block_types:
            block_type = block_types[base]
        else:
            block_type = _infer_block_type_for_hidden_block(base)

    block_type = _normalize_block_type(block_type)

    world_faces.add(face_key)
    face_to_chunk[face_key] = chunk_coord
    face_block_types[face_key] = block_type
    if base not in block_types:
        block_types[base] = block_type
    chunk_face_sets[chunk_coord].add(face_key)
    _register_top_face(face_key[0], face_key[1])
    affected.add(chunk_coord)
    return True


def load_chunks():
    world_faces.clear()
    face_to_chunk.clear()
    face_block_types.clear()
    block_types.clear()
    top_columns.clear()
    top_cells.clear()
    block_face_counts.clear()
    _reset_chunk_storage()

    chunks_opened_ = list(eval(open("chunks.txt", "r").read()))

    for legacy_idx, chunk_data in enumerate(chunks_opened_):
        _ensure_chunk(_legacy_chunk_coord_from_index(legacy_idx))
        positions = chunk_data[0]
        indices = chunk_data[1]
        block_type_data = chunk_data[2] if len(chunk_data) > 2 else None

        for i, face_pos in enumerate(positions):
            if i >= len(indices):
                break
            fidx = int(indices[i])
            btype = DEFAULT_BLOCK_TYPE
            if block_type_data is not None and i < len(block_type_data):
                btype = _normalize_block_type(block_type_data[i])

            key = _face_key(face_pos, fidx)
            if key in world_faces:
                continue

            chunk_coord = _ensure_chunk(_chunk_coord_from_face(key[0], key[1]))
            world_faces.add(key)
            face_to_chunk[key] = chunk_coord
            face_block_types[key] = btype
            base = _cube_base_from_face(key[0], key[1])
            if base not in block_types:
                block_types[base] = btype
            chunk_face_sets[chunk_coord].add(key)
            _register_top_face(key[0], key[1])

    _apply_surface_layers()

    for chunk_coord in list(chunk_face_sets.keys()):
        _rebuild_chunk_mesh(chunk_coord)

    print(f"[gravity] faces={len(world_faces)} blocks={len(block_face_counts)} columns={len(top_columns)}")
    if len(world_faces) > 0 and len(block_face_counts) == 0:
        for face_key in world_faces:
            _register_top_face(face_key[0], face_key[1])
        print(f"[gravity] rebuilt blocks={len(block_face_counts)} columns={len(top_columns)}")


load_chunks()

try:
    if len(top_columns) > 0:
        col = next(iter(top_columns.keys()))
        y = top_columns[col][-1]
        player.position = Vec3(col[0], y + PLAYER_STAND_HEIGHT, col[1])
    else:
        first_face = next(iter(world_faces))
        player.position = Vec3(first_face[0][0], first_face[0][1] + 2, first_face[0][2])
except:
    player.position = Vec3(0, 6, 0)

_ensure_player_probes()
_sample_player_probes_at(Vec3(float(player.x), float(player.y), float(player.z)), do_assign=True)

prev_horizontal_x = float(player.x)
prev_horizontal_z = float(player.z)


def get_target_face(max_distance: int = 12):
    if len(block_face_counts) == 0:
        return None, None, None

    origin = Vec3(camera.world_position)
    direction = Vec3(camera.forward)

    dir_len2 = float(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z)
    if dir_len2 <= 1e-12:
        return None, None, None

    inv_len = 1.0 / math.sqrt(dir_len2)
    dx = float(direction.x) * inv_len
    dy = float(direction.y) * inv_len
    dz = float(direction.z) * inv_len

    bottom_off = float(_FACE_OFFSETS[0].y)
    ox = float(origin.x) + BLOCK_HALF_EXTENT
    oy = float(origin.y) - bottom_off
    oz = float(origin.z) + BLOCK_HALF_EXTENT

    ix = math.floor(ox)
    iy = math.floor(oy)
    iz = math.floor(oz)

    inf = float("inf")

    if dx > 0.0:
        step_x = 1
        t_max_x = (ix + 1.0 - ox) / dx
        t_delta_x = 1.0 / dx
    elif dx < 0.0:
        step_x = -1
        t_max_x = (ox - ix) / (-dx)
        t_delta_x = 1.0 / (-dx)
    else:
        step_x = 0
        t_max_x = inf
        t_delta_x = inf

    if dy > 0.0:
        step_y = 1
        t_max_y = (iy + 1.0 - oy) / dy
        t_delta_y = 1.0 / dy
    elif dy < 0.0:
        step_y = -1
        t_max_y = (oy - iy) / (-dy)
        t_delta_y = 1.0 / (-dy)
    else:
        step_y = 0
        t_max_y = inf
        t_delta_y = inf

    if dz > 0.0:
        step_z = 1
        t_max_z = (iz + 1.0 - oz) / dz
        t_delta_z = 1.0 / dz
    elif dz < 0.0:
        step_z = -1
        t_max_z = (oz - iz) / (-dz)
        t_delta_z = 1.0 / (-dz)
    else:
        step_z = 0
        t_max_z = inf
        t_delta_z = inf

    t = 0.0
    max_steps = int(max_distance * 8) + 32

    for _ in range(max_steps):
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            t = t_max_x
            t_max_x += t_delta_x
            ix += step_x
            face_idx = _entry_face_from_axis(0, step_x)
        elif t_max_y <= t_max_x and t_max_y <= t_max_z:
            t = t_max_y
            t_max_y += t_delta_y
            iy += step_y
            face_idx = _entry_face_from_axis(1, step_y)
        else:
            t = t_max_z
            t_max_z += t_delta_z
            iz += step_z
            face_idx = _entry_face_from_axis(2, step_z)

        if t > max_distance:
            break

        base = (ix, iy, iz)
        if block_face_counts.get(base, 0) <= 0:
            continue

        face_pos = _face_pos_from_base(base, face_idx)
        face_key = (face_pos, int(face_idx))
        if face_key not in world_faces:
            continue

        normal = _FACE_NORMALS.get(int(face_idx), Vec3(0, 1, 0))
        return face_pos, normal, int(face_idx)

    return None, None, None


def build():
    cube_base = Vec3(c.position) + Vec3(0, -1.5, 0)
    base_key = _vkey(cube_base)
    cube_base = Vec3(*base_key)

    if base_key in block_types:
        c.y = -9999
        return

    if _block_intersects_player(base_key):
        c.y = -9999
        return

    affected = set()

    _set_block_type(base_key, selected_block_type)

    below = _vkey((base_key[0], base_key[1] - BLOCK_HEIGHT, base_key[2]))
    if below in block_types and _normalize_block_type(block_types[below]) == "grass":
        _set_block_type(below, "dirt")

    for i, off in enumerate(_FACE_OFFSETS):
        fp = cube_base + off
        same = _face_key(fp, i)
        opp = _face_key(fp, _OPPOSITE_FACE[i])

        if opp in world_faces:
            _remove_face(opp, affected)
        elif same not in world_faces:
            tgt = _chunk_coord_from_face(fp, i)
            _add_face(same, tgt, affected)

    _refresh_chunks(affected)
    c.y = -9999


def mine(face_pos=None, face_idx=None):
    if face_pos is None or face_idx is None:
        face_pos, _, face_idx = get_target_face()
        if face_pos is None:
            c.y = -9999
            return

    cube_base = _cube_base_from_face(face_pos, face_idx)
    affected = set()

    for i in range(len(_FACE_OFFSETS)):
        fp = _face_pos_from_base(cube_base, i)
        same = _face_key(fp, i)
        opp = _face_key(fp, _OPPOSITE_FACE[i])

        if same in world_faces:
            _remove_face(same, affected)
        else:
            tgt = _chunk_coord_from_face(fp, _OPPOSITE_FACE[i])
            _add_face(opp, tgt, affected)

    for i in range(len(_FACE_OFFSETS)):
        fp = _face_pos_from_base(cube_base, i)
        same = _face_key(fp, i)
        if same in world_faces:
            _remove_face(same, affected)

    block_types.pop(cube_base, None)

    below = _vkey((cube_base[0], cube_base[1] - BLOCK_HEIGHT, cube_base[2]))
    if below in block_types and _normalize_block_type(block_types[below]) == "grass":
        _set_block_type(below, "dirt")

    _refresh_chunks(_expand_chunk_neighborhood(affected, radius=1))
    c.y = -9999


def _frame_position_for_target(face_pos, face_idx):
    hit_base = _cube_base_from_face(face_pos, face_idx)
    return Vec3(hit_base[0], hit_base[1] + 1.5, hit_base[2])


def update():
    _apply_player_probe_horizontal()
    _apply_vector_gravity()
    _snap_player_y_to_grid()

    face_pos, _, face_idx = get_target_face()
    if face_pos:
        c2.position = _frame_position_for_target(face_pos, face_idx)
    else:
        c2.position = floor(player.position + (0, 10000, 0))


def input(key):
    global mode, vertical_velocity, is_grounded, selected_block_type
    global prev_horizontal_x, prev_horizontal_z

    print("Gay")
    move_dir = True
    if get_front_back_left_right_hits("back"):
        print("back")
        move_dir = (
            Vec3(player.forward)
            if key == "w"
            else Vec3(player.forward)
            if key == "s"
            else Vec3(player.forward)
            if key == "d"
            else Vec3(player.forward)
            if key == "a"
            else True
        )
    if get_front_back_left_right_hits("front"):
        print("front")
        move_dir = (
            Vec3(player.back)
            if key == "w"
            else Vec3(player.back)
            if key == "s"
            else Vec3(player.back)
            if key == "d"
            else Vec3(player.back)
            if key == "a"
            else True
        )
    if get_front_back_left_right_hits("right"):
        print("right")
        move_dir = (
            Vec3(player.left)
            if key == "s"
            else Vec3(player.left)
            if key == "w"
            else Vec3(player.left)
            if key == "a"
            else Vec3(player.left)
            if key == "d"
            else True
        )
    if get_front_back_left_right_hits("left"):
        print("left")
        move_dir = (
            Vec3(player.right)
            if key == "w"
            else Vec3(player.right)
            if key == "s"
            else Vec3(player.right)
            if key == "d"
            else Vec3(player.right)
            if key == "a"
            else True
        )
    print(move_dir)

    if isinstance(move_dir, Vec3):
        px = float(player.x)
        pz = float(player.z)

        print(move_dir)
        nx = px + move_dir[0] * player.speed * time.dt
        nz = pz + move_dir[2] * player.speed * time.dt

        print("lol")
        player.x = nx
        player.z = nz
        prev_horizontal_x = nx
        prev_horizontal_z = nz
        _sample_player_probes_at(Vec3(nx, float(player.y), nz), do_assign=True)
        return True

    if key == "o":
        mode = 1 - mode
    if key == "m" or key == "m hold":
        player.y += 1
        vertical_velocity = 0.0
        _snap_player_y_to_grid(force=True)
    if key == "l":
        player.y -= 1
        vertical_velocity = 0.0
        _snap_player_y_to_grid(force=True)

    if key == "space" and is_grounded:
        if _jump_blocked_by_ceiling():
            vertical_velocity = 0.0
        else:
            vertical_velocity = JUMP_SPEED
            is_grounded = False

    if key == "e":
        player.enabled = not player.enabled
        print(len(scene.entities))
    if key in BLOCK_SELECT_KEYS:
        selected_block_type = BLOCK_SELECT_KEYS[key]
        print(f"[build] selected block: {selected_block_type}")

    if key in ("right mouse down", "5"):
        face_pos, normal, face_idx = get_target_face()
        if face_pos:
            cube_base = Vec3(face_pos) - _FACE_OFFSETS[face_idx] + normal
            c.position = cube_base + Vec3(0, 1.5, 0)
            build()

    if key in ("left mouse down", "4"):
        face_pos, _, face_idx = get_target_face()
        if face_pos:
            mine(face_pos, face_idx)

    if key == "r":
        player.y = 10
        _snap_player_y_to_grid(force=True)
    if key == "n":
        player.rotation_x = -90
        player.rotation_y = 90
        player.rotation_z = 90
        window.exit_button.disable()
        window.cog_menu.disable()
        c2.disable()
    if key == "z":
        player.cursor.disable()


app.run()
