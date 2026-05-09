from ursina import *
from panda3d.core import LVecBase3f, Vec4, Vec3, Vec2
from ursina.prefabs.first_person_controller import *
from itertools import *
import math
import numpy as np
from bisect import bisect_left, bisect_right

app = Ursina()

PLAYER_WIDTH = 0.5
PLAYER_HEIGHT = 1.5

player = FirstPersonController(gravity=0)
player.cursor.color = color.white
player.cursor.rotation = (0, 0, 0)
player.cursor.texture = "cursor"
player.cursor.scale = 0.04
PLAYER_MOVE_SPEED = 5
player.speed = 0  # Bewegung läuft unten über unsere eigene Sweep-Physik.
player.height = PLAYER_HEIGHT
player.camera_pivot.y = 1.28
camera.fov = 120
player.collider = None

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

chunk_size = 16
chunk_height = 16
texture = "atlas2"

ATLAS_TILES_X = 4
ATLAS_TILES_Y = 5
ATLAS_BLEED = 0.0015

window.fps_counter.disable()
window.cog_menu.disable()

ATLAS_FLIP_Y = True
DEFAULT_ATLAS_TILE = (0, 0)
BLOCK_FACE_TILES = {
    "grass": {0: (2, 0), 1: (0, 0), 2: (1, 0), 3: (1, 0), 4: (1, 0), 5: (1, 0)},
    "dirt": {0: (2, 0), 1: (2, 0), 2: (2, 0), 3: (2, 0), 4: (2, 0), 5: (2, 0)},
    "cobblestone": {0: (3, 0), 1: (3, 0), 2: (3, 0), 3: (3, 0), 4: (3, 0), 5: (3, 0)},
    "sand": {0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0, 1)},
    "planks": {0: (1, 1), 1: (1, 1), 2: (1, 1), 3: (1, 1), 4: (1, 1), 5: (1, 1)},
    "leaves": {0: (2, 1), 1: (2, 1), 2: (2, 1), 3: (2, 1), 4: (2, 1), 5: (2, 1)},
    "water": {0: (3, 1), 1: (3, 1), 2: (3, 1), 3: (3, 1), 4: (3, 1), 5: (3, 1)},
    "log": {0: (1, 4), 1: (1, 4), 2: (0, 4), 3: (0, 4), 4: (0, 4), 5: (0, 4)},
    "brick": {0: (0, 2), 1: (0, 2), 2: (0, 2), 3: (0, 2), 4: (0, 2), 5: (0, 2)},
    "wool": {0: (1, 2), 1: (1, 2), 2: (1, 2), 3: (1, 2), 4: (1, 2), 5: (1, 2)},
    "stone": {0: (2, 2), 1: (2, 2), 2: (2, 2), 3: (2, 2), 4: (2, 2), 5: (2, 2)},
}
DEFAULT_BLOCK_TYPE = "grass"
selected_block_type = DEFAULT_BLOCK_TYPE
BLOCK_SELECT_KEYS = {
    "1": "grass", "2": "cobblestone", "3": "sand", "6": "planks",
    "7": "leaves", "8": "water", "9": "dirt", "0": "log", "`": "brick",
    "-": "wool", "=": "stone"
}

# Block-Rotation: Nur Y-Achse/Yaw.
# Gespeichert wird die horizontale Richtung, in die die lokale Vorderseite zeigt.
# 2 = +Z, 3 = -Z, 4 = +X, 5 = -X.
# Die lokale Oberseite bleibt IMMER Welt-Oberseite (Face 1) und die Unterseite
# bleibt IMMER Face 0. Dadurch können Gras/Erde/etc. nicht mehr seitlich kippen.
DEFAULT_BLOCK_ROTATION = 2
HORIZONTAL_ROTATION_FACES = (2, 3, 4, 5)
ROTATABLE_BLOCK_TYPES = set(BLOCK_FACE_TILES.keys())

atlas_texture = load_texture(texture)
if atlas_texture is not None:
    try:
        atlas_texture.filtering = None
    except:
        pass

atlas_repeat_shader = Shader(
    language=Shader.GLSL,
    vertex=
    "#version 120\n"
    "uniform mat4 p3d_ModelViewProjectionMatrix;\n"
    "attribute vec4 p3d_Vertex;\n"
    "attribute vec2 p3d_MultiTexCoord0;\n"
    "attribute vec4 p3d_Color;\n"
    "varying vec2 v_local_uv;\n"
    "varying vec4 v_tile_rect;\n"
    "void main(){\n"
    "    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;\n"
    "    v_local_uv = p3d_MultiTexCoord0;\n"
    "    v_tile_rect = p3d_Color;\n"
    "}\n",
    fragment=
    "#version 120\n"
    "uniform sampler2D p3d_Texture0;\n"
    "varying vec2 v_local_uv;\n"
    "varying vec4 v_tile_rect;\n"
    "void main(){\n"
    "    vec2 f = fract(v_local_uv);\n"
    "    vec2 uv = mix(v_tile_rect.xy, v_tile_rect.zw, f);\n"
    "    gl_FragColor = texture2D(p3d_Texture0, uv);\n"
    "}\n",
)

GRID_X = 8
GRID_Z = 8
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

block_types = {}
block_rotations = {}

top_columns = {}
top_cells = {}
block_face_counts = {}

chunk_update_queue = []

# NEU: Speichert die Höhen der ursprünglichen natürlichen Oberfläche
surface_heights = {}

mode = 1
c = Entity(model="cube", color=color.clear)
c2 = Entity(model="cube", texture="2", scale=1.01)

_FACE_NORMALS_TUPLES = {
    0: (0, -1, 0), 1: (0, 1, 0), 2: (0, 0, 1),
    3: (0, 0, -1), 4: (1, 0, 0), 5: (-1, 0, 0),
}
_FACE_NORMALS = {k: Vec3(*v) for k, v in _FACE_NORMALS_TUPLES.items()}
_FACE_INDEX_BY_NORMAL = {v: k for k, v in _FACE_NORMALS_TUPLES.items()}
_ROTATION_AXES_CACHE = {}
_ROTATION_FACE_MAP_CACHE = {}

_FACE_OFFSETS = [Vec3(*cf[:3]) for cf in cube_faces]
_OPPOSITE_FACE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

GRAVITY_ACCEL = 0.08 * 100
MAX_FALL_SPEED = 0.42 * 1000
JUMP_SPEED = 1.7 * 3.92

MIN_HEADROOM_TO_JUMP = 1.0
JUMP_START_HEADROOM = 0.05

PLAYER_STAND_HEIGHT = 0.0
GROUND_STICK = 0.08
MAX_STEP_UP = 0.35
PLAYER_COLLISION_RADIUS = PLAYER_WIDTH * 0.5
PLAYER_FOOT_RADIUS = PLAYER_COLLISION_RADIUS

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

PROBE_COLOR = color.clear
PROBE_HIT_COLOR = color.clear
EDGE_PROBE_NAMES = {
    "front_low", "front_high", "right_low", "right_high",
    "left_low", "left_high", "back_low", "back_high",
}
PLAYER_Y_SNAP_STEP = PROBE_GRID_STEP
PLAYER_Y_SNAP_ONLY_GROUNDED = True

player_probe_entities = {}
player_probe_hits = {}

vertical_velocity = 0.0
is_grounded = False
prev_horizontal_x = None
prev_horizontal_z = None



def _vkey(v):
    return (round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4))


def _face_key(pos, face_idx):
    return (_vkey(pos), int(face_idx))


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


def _normalize_face_index(face_idx, default=DEFAULT_BLOCK_ROTATION):
    try:
        idx = int(face_idx)
    except:
        idx = int(default)
    if idx in _FACE_NORMALS_TUPLES:
        return idx
    return int(default)


def _vec_neg(v):
    return (-v[0], -v[1], -v[2])


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _default_forward_face_for_up(up_face_idx):
    # Rotation ist jetzt absichtlich nur noch Y-Achse.
    # Diese Funktion bleibt als Fallback für ältere Aufrufe erhalten.
    return DEFAULT_BLOCK_ROTATION


def _normalize_horizontal_rotation_face(face_idx, default=DEFAULT_BLOCK_ROTATION):
    idx = _normalize_face_index(face_idx, default)
    if idx in HORIZONTAL_ROTATION_FACES:
        return idx
    return int(default)


def _normalize_block_rotation(rotation):
    # Nur Y-Achse/Yaw: Rückgabe bleibt aus Kompatibilitätsgründen ein Tupel
    # (up_face, forward_face), aber up_face ist IMMER 1.
    #
    # Kompatibilität mit der vorherigen Version:
    # - (1, horizontal) bleibt erhalten.
    # - (horizontal, 1) aus der alten "gekippte Blöcke"-Variante wird zu
    #   (1, horizontal), damit alte Wand-Platzierungen nicht mehr kippen.
    # - einzelne horizontale Face-Indizes werden direkt als Yaw benutzt.
    forward_face = DEFAULT_BLOCK_ROTATION

    if isinstance(rotation, (tuple, list)) and len(rotation) >= 2:
        first = _normalize_face_index(rotation[0], DEFAULT_BLOCK_ROTATION)
        second = _normalize_face_index(rotation[1], DEFAULT_BLOCK_ROTATION)
        if second in HORIZONTAL_ROTATION_FACES:
            forward_face = second
        elif first in HORIZONTAL_ROTATION_FACES:
            forward_face = first
    else:
        idx = _normalize_face_index(rotation, DEFAULT_BLOCK_ROTATION)
        if idx in HORIZONTAL_ROTATION_FACES:
            forward_face = idx

    return (1, forward_face)


def _rotation_axes(rotation):
    rotation = _normalize_block_rotation(rotation)
    cached = _ROTATION_AXES_CACHE.get(rotation)
    if cached is not None:
        return cached

    up_face, forward_face = rotation
    up_axis = _FACE_NORMALS_TUPLES[up_face]
    preferred_forward = _FACE_NORMALS_TUPLES[forward_face]

    right_axis = _vec_cross(up_axis, preferred_forward)
    if right_axis == (0, 0, 0):
        preferred_forward = _FACE_NORMALS_TUPLES[_default_forward_face_for_up(up_face)]
        right_axis = _vec_cross(up_axis, preferred_forward)
    forward_axis = _vec_cross(right_axis, up_axis)

    cached = (right_axis, up_axis, forward_axis)
    _ROTATION_AXES_CACHE[rotation] = cached
    return cached


def _transform_local_axis_to_world(axis, rotation):
    right_axis, up_axis, forward_axis = _rotation_axes(rotation)
    return (
        axis[0] * right_axis[0] + axis[1] * up_axis[0] + axis[2] * forward_axis[0],
        axis[0] * right_axis[1] + axis[1] * up_axis[1] + axis[2] * forward_axis[1],
        axis[0] * right_axis[2] + axis[1] * up_axis[2] + axis[2] * forward_axis[2],
    )


def _rotation_world_to_local_face_map(rotation):
    rotation = _normalize_block_rotation(rotation)
    cached = _ROTATION_FACE_MAP_CACHE.get(rotation)
    if cached is not None:
        return cached

    right_axis, up_axis, forward_axis = _rotation_axes(rotation)
    local_to_world = {
        0: _FACE_INDEX_BY_NORMAL[_vec_neg(up_axis)],
        1: _FACE_INDEX_BY_NORMAL[up_axis],
        2: _FACE_INDEX_BY_NORMAL[forward_axis],
        3: _FACE_INDEX_BY_NORMAL[_vec_neg(forward_axis)],
        4: _FACE_INDEX_BY_NORMAL[right_axis],
        5: _FACE_INDEX_BY_NORMAL[_vec_neg(right_axis)],
    }
    world_to_local = {world_face: local_face for local_face, world_face in local_to_world.items()}
    _ROTATION_FACE_MAP_CACHE[rotation] = world_to_local
    return world_to_local


def _local_face_for_world_face(world_face_idx, rotation):
    world_face_idx = int(world_face_idx)
    return _rotation_world_to_local_face_map(rotation).get(world_face_idx, world_face_idx)


def _block_rotation_from_base(base, block_type=None):
    base = _vkey(base)
    if block_type is None:
        block_type = block_types.get(base, DEFAULT_BLOCK_TYPE)
    btype = _normalize_block_type(block_type)
    raw_rotation = _normalize_block_rotation(block_rotations.get(base, DEFAULT_BLOCK_ROTATION))

    # Nur Y-Achse: Top/Bottom bleiben immer Top/Bottom. Die gespeicherte Rotation
    # dreht nur die horizontale Ausrichtung der Textur/Face-Zuordnung.
    if btype not in ROTATABLE_BLOCK_TYPES:
        return _normalize_block_rotation(DEFAULT_BLOCK_ROTATION)
    return raw_rotation


def _block_rotation_from_face_key(face_key):
    base = _cube_base_from_face(face_key[0], face_key[1])
    btype = block_types.get(base, DEFAULT_BLOCK_TYPE)
    return _block_rotation_from_base(base, btype)


def _block_tile_for_world_face(block_type, rotation, world_face_idx):
    local_face_idx = _local_face_for_world_face(world_face_idx, rotation)
    return _block_tile_for_face(block_type, local_face_idx)


def _axis_coord_for_uv(vertex, axis):
    x, y, z = vertex
    if axis[0] != 0:
        return float(x) * axis[0]
    if axis[1] != 0:
        return (float(y) / max(BLOCK_HEIGHT, 1e-8)) * axis[1]
    if axis[2] != 0:
        return float(z) * axis[2]
    return 0.0


_LOCAL_FACE_UV_AXES = {
    0: ((1, 0, 0), (0, 0, 1)),
    1: ((1, 0, 0), (0, 0, -1)),
    2: ((1, 0, 0), (0, 1, 0)),
    3: ((-1, 0, 0), (0, 1, 0)),
    4: ((0, 0, -1), (0, 1, 0)),
    5: ((0, 0, 1), (0, 1, 0)),
}


def _rotated_uvs(world_face_idx, rotation, quad_verts):
    local_face_idx = _local_face_for_world_face(world_face_idx, rotation)
    local_u_axis, local_v_axis = _LOCAL_FACE_UV_AXES.get(
        local_face_idx,
        ((1, 0, 0), (0, 1, 0)),
    )

    world_u_axis = _transform_local_axis_to_world(local_u_axis, rotation)
    world_v_axis = _transform_local_axis_to_world(local_v_axis, rotation)

    raw_us = [_axis_coord_for_uv(v, world_u_axis) for v in quad_verts]
    raw_vs = [_axis_coord_for_uv(v, world_v_axis) for v in quad_verts]
    min_u = min(raw_us)
    min_v = min(raw_vs)
    return [(raw_us[i] - min_u, raw_vs[i] - min_v) for i in range(len(quad_verts))]


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


def _set_block_rotation(base, rotation):
    block_rotations[_vkey(base)] = _normalize_block_rotation(rotation)


def _set_block_type(base, block_type, rotation=None):
    base = _vkey(base)
    btype = _normalize_block_type(block_type)
    block_types[base] = btype

    if rotation is not None:
        _set_block_rotation(base, rotation)
    elif base not in block_rotations:
        block_rotations[base] = DEFAULT_BLOCK_ROTATION

    for i in range(len(_FACE_OFFSETS)):
        fp = _face_pos_from_base(base, i)
        fk = _face_key(fp, i)
        if fk in face_block_types:
            face_block_types[fk] = btype


def _infer_natural_block_type(base):
    """Berechnet deterministisch den natürlichen Blocktyp (Gras, Erde, Stein) anhand der Tiefe."""
    x, y, z = base
    col = (x, z)

    # Nimm die Ursprungsoberfläche dieser Säule. Falls wir die Spalte nicht kennen (sollte
    # eigentlich nicht passieren), nimm an der momentane Block ist die Spitze.
    top_y = surface_heights.get(col, y)

    # Bestimme die Tiefe (0 = Spitze, >0 = Untergrund)
    depth = int(round((top_y - y) / BLOCK_HEIGHT))

    if depth <= 0:
        return "grass"

    # Berechne wie dick die Erdschicht an dieser (x, z) Stelle ist (1 bis 5 Blöcke)
    rand_val = abs(math.sin(x * 12.9898 + z * 78.233 + seed) * 43758.5453)
    dirt_thickness = 1 + int((rand_val - math.floor(rand_val)) * 5)

    # Alles ab der berechneten Erdschicht und tiefer wird durchgehend Stein
    if depth >= dirt_thickness:
        return "stone"

    return "dirt"


def _apply_surface_layers():
    global surface_heights
    surface_heights.clear()

    # 1. Ermittle den absolut höchsten natürlichen Punkt für jede X/Z Spalte und speichere ihn global ab
    for base in block_types.keys():
        col = (base[0], base[2])
        y = base[1]
        prev = surface_heights.get(col)
        if prev is None or y > prev:
            surface_heights[col] = y

    # 2. Block-Typen deterministisch aufsetzen (sodass ab Tiefe 1-5 *alles* aus Stein ist)
    for base, btype in list(block_types.items()):
        current_btype = _normalize_block_type(btype)
        if current_btype not in ("grass", "dirt", "stone"):
            continue

        new_type = _infer_natural_block_type(base)
        _set_block_type(base, new_type)


def _infer_block_type_for_hidden_block(base):
    # Wird aufgerufen, wenn ein bisher komplett versteckter Block durch Abbauen freigelegt wird.
    # Nutzt exakt dieselbe Logik, damit alles ab einer gewissen Tiefe dauerhaft Stein ist.
    return _infer_natural_block_type(base)


def _block_type_from_face_key(face_key):
    base = _cube_base_from_face(face_key[0], face_key[1])
    return _normalize_block_type(block_types.get(base, DEFAULT_BLOCK_TYPE))


def _fast_uvs(face_idx, w, h, d_ext):
    hb = h / max(BLOCK_HEIGHT, 1e-8)
    if face_idx == 0: return [(0, 0), (w, 0), (w, d_ext), (0, d_ext)]
    if face_idx == 1: return [(0, d_ext), (0, 0), (w, 0), (w, d_ext)]
    if face_idx == 2: return [(0, 0), (w, 0), (w, hb), (0, hb)]
    if face_idx == 3: return [(w, 0), (w, hb), (0, hb), (0, 0)]
    if face_idx == 4: return [(d_ext, 0), (d_ext, hb), (0, hb), (0, 0)]
    if face_idx == 5: return [(0, 0), (d_ext, 0), (d_ext, hb), (0, hb)]
    return [(0, 0), (1, 0), (1, 1), (0, 1)]


def _rebuild_chunk_mesh(chunk_coord):
    chunk_coord = _ensure_chunk(chunk_coord)
    old = combined_terrains.get(chunk_coord)

    faces_snapshot = chunk_face_sets[chunk_coord]
    if not faces_snapshot:
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return

    vertices = []
    triangles = []
    uvs = []
    normals = []
    colors = []

    faces_by_dir = {i: [] for i in range(6)}
    for fk in faces_snapshot:
        faces_by_dir[int(fk[1])].append(fk)

    for d in range(6):
        if not faces_by_dir[d]:
            continue

        slices = {}
        for fk in faces_by_dir[d]:
            pos_key, fidx = fk
            base = _cube_base_from_face(pos_key, fidx)
            btype = _block_type_from_face_key(fk)
            brot = _block_rotation_from_base(base, btype)

            lx = int(round(base[0]))
            ly = int(round(base[1] / BLOCK_HEIGHT))
            lz = int(round(base[2]))

            if d in (0, 1):
                slice_idx, u, v = ly, lx, lz
            elif d in (2, 3):
                slice_idx, u, v = lz, lx, ly
            elif d in (4, 5):
                slice_idx, u, v = lx, lz, ly

            if slice_idx not in slices:
                slices[slice_idx] = {}
            slices[slice_idx][(u, v)] = (btype, brot)

        for slice_idx, grid in slices.items():
            if not grid: continue

            visited = set()
            keys = grid.keys()
            min_u, max_u = min(k[0] for k in keys), max(k[0] for k in keys)
            min_v, max_v = min(k[1] for k in keys), max(k[1] for k in keys)

            for v in range(min_v, max_v + 1):
                for u in range(min_u, max_u + 1):
                    if (u, v) in visited or (u, v) not in grid:
                        continue

                    cell_data = grid[(u, v)]
                    btype, brot = cell_data

                    w = 1
                    while (u + w) <= max_u and (u + w, v) not in visited and grid.get((u + w, v)) == cell_data:
                        w += 1

                    h = 1
                    can_expand = True
                    while (v + h) <= max_v and can_expand:
                        for du in range(w):
                            if (u + du, v + h) in visited or grid.get((u + du, v + h)) != cell_data:
                                can_expand = False
                                break
                        if can_expand:
                            h += 1

                    for du in range(w):
                        for dv in range(h):
                            visited.add((u + du, v + dv))

                    if d in (0, 1):
                        bx = float(u);
                        by = float(slice_idx) * BLOCK_HEIGHT;
                        bz = float(v)
                        W_ext, H_ext, D_ext = w, BLOCK_HEIGHT, h
                    elif d in (2, 3):
                        bx = float(u);
                        by = float(v) * BLOCK_HEIGHT;
                        bz = float(slice_idx)
                        W_ext, H_ext, D_ext = w, h * BLOCK_HEIGHT, 1.0
                    else:
                        bx = float(slice_idx);
                        by = float(v) * BLOCK_HEIGHT;
                        bz = float(u)
                        W_ext, H_ext, D_ext = 1.0, h * BLOCK_HEIGHT, w

                    X0 = bx - 0.5
                    X1 = bx - 0.5 + W_ext
                    Y0 = by + float(_FACE_OFFSETS[0].y)
                    Y1 = by + float(_FACE_OFFSETS[0].y) + H_ext
                    Z0 = bz - 0.5
                    Z1 = bz - 0.5 + D_ext

                    if d == 0:
                        quad_verts = [(X0, Y0, Z0), (X1, Y0, Z0), (X1, Y0, Z1), (X0, Y0, Z1)]
                    elif d == 1:
                        quad_verts = [(X0, Y1, Z0), (X0, Y1, Z1), (X1, Y1, Z1), (X1, Y1, Z0)]
                    elif d == 2:
                        quad_verts = [(X0, Y0, Z1), (X1, Y0, Z1), (X1, Y1, Z1), (X0, Y1, Z1)]
                    elif d == 3:
                        quad_verts = [(X0, Y0, Z0), (X0, Y1, Z0), (X1, Y1, Z0), (X1, Y0, Z0)]
                    elif d == 4:
                        quad_verts = [(X1, Y0, Z0), (X1, Y1, Z0), (X1, Y1, Z1), (X1, Y0, Z1)]
                    else:
                        quad_verts = [(X0, Y0, Z0), (X0, Y0, Z1), (X0, Y1, Z1), (X0, Y1, Z0)]

                    tile = _block_tile_for_world_face(btype, brot, int(d))
                    u0, v0, u1, v1 = _atlas_rect(tile[0], tile[1])
                    rect = (u0, v0, u1, v1)

                    quad_uvs = _rotated_uvs(d, brot, quad_verts)
                    n = _FACE_NORMALS_TUPLES.get(d, (0, 1, 0))

                    idx0 = len(vertices)
                    vertices.extend(quad_verts)
                    uvs.extend(quad_uvs)
                    colors.extend([rect, rect, rect, rect])
                    normals.extend([n, n, n, n])
                    triangles.extend([idx0, idx0 + 2, idx0 + 1, idx0, idx0 + 3, idx0 + 2])

    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        uvs=uvs,
        normals=normals,
        colors=colors,
        mode="triangle",
        static=True,
    )

    tex = atlas_texture if atlas_texture is not None else texture

    if old is None:
        ent = Entity(model=mesh, texture=tex, shader=atlas_repeat_shader)
        ent.collider = None  # Chunks bleiben absichtlich ohne Ursina-Collider.
        combined_terrains[chunk_coord] = ent
        return

    try:
        old.model = mesh
        old.texture = tex
        old.shader = atlas_repeat_shader
        old.collider = None
        old.enabled = True
        combined_terrains[chunk_coord] = old
    except:
        _safe_clear_destroy(old)
        ent = Entity(model=mesh, texture=tex, shader=atlas_repeat_shader)
        ent.collider = None  # Chunks bleiben absichtlich ohne Ursina-Collider.
        combined_terrains[chunk_coord] = ent


def _refresh_chunks(affected_chunks):
    for chunk_coord in affected_chunks:
        if chunk_coord is not None and chunk_coord not in chunk_update_queue:
            chunk_update_queue.append(chunk_coord)


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
    if axis == 0: return 5 if step > 0 else 4
    if axis == 1: return 0 if step > 0 else 1
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


def _iter_solid_block_bounds(min_x, max_x, min_z, max_z, min_y, max_y):
    """Wie _iter_solid_blocks, aber mit Y-Grenzen.

    Wird für vertikale Sweep-Kollision benutzt. Dadurch entscheidet beim Springen
    nicht mehr die breite Top-Probe, sondern der echte Player-AABB.
    """
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
            yield bx0, bx1, by0, by1, bz0, bz1


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


def _jump_blocked_by_ceiling():
    _, head_y = _player_body_y_span()
    px = float(player.x)
    pz = float(player.z)
    r = PLAYER_COLLISION_RADIUS

    # Nur direkt blockierte Kopffreiheit verhindert den Start des Sprungs.
    # Eine Decke weiter oben wird während des Sprungs per _sweep_y sauber getroffen.
    return _aabb_hits_any_block(px - r, px + r, head_y, head_y + JUMP_START_HEADROOM, pz - r, pz + r)


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


def _sweep_y(start_y, target_y, px, pz):
    """Vertikale Sweep-Kollision für Springen und Fallen.

    Vorher hat die breite Top-Probe den Sprung gestoppt. An Blockkanten konnte sie
    seitliche/benachbarte Blöcke als Decke werten. Diese Funktion benutzt stattdessen
    den echten Player-AABB und gibt optional "ceiling" oder "ground" zurück.
    """
    dy = target_y - start_y
    if abs(dy) < 1e-8:
        return target_y, None

    r = PLAYER_COLLISION_RADIUS
    min_x = float(px) - r
    max_x = float(px) + r
    min_z = float(pz) - r
    max_z = float(pz) + r

    if dy > 0:
        head_start = start_y + PLAYER_HEIGHT - PLAYER_COLLISION_HEAD_CLEARANCE
        head_target = target_y + PLAYER_HEIGHT - PLAYER_COLLISION_HEAD_CLEARANCE
        sweep_min_y = start_y + PLAYER_COLLISION_FOOT_CLEARANCE
        sweep_max_y = head_target

        limit = target_y
        hit = False
        for _, _, by0, _, _, _ in _iter_solid_block_bounds(min_x, max_x, min_z, max_z, sweep_min_y, sweep_max_y):
            if head_start <= by0 + SWEEP_TOL and head_target >= by0:
                candidate = by0 - PLAYER_HEIGHT + PLAYER_COLLISION_HEAD_CLEARANCE - WALL_EPS
                if candidate < limit:
                    limit = candidate
                    hit = True

        if hit:
            return max(start_y, limit), "ceiling"
        return target_y, None

    foot_start = start_y - PLAYER_STAND_HEIGHT
    foot_target = target_y - PLAYER_STAND_HEIGHT
    sweep_min_y = target_y + PLAYER_COLLISION_FOOT_CLEARANCE
    sweep_max_y = start_y + PLAYER_HEIGHT - PLAYER_COLLISION_HEAD_CLEARANCE

    limit = target_y
    hit = False
    for _, _, _, by1, _, _ in _iter_solid_block_bounds(min_x, max_x, min_z, max_z, sweep_min_y, sweep_max_y):
        if foot_start >= by1 - SWEEP_TOL and foot_target <= by1:
            candidate = by1 + PLAYER_STAND_HEIGHT
            if candidate > limit:
                limit = candidate
                hit = True

    if hit:
        return min(start_y, limit), "ground"
    return target_y, None


def _round_probe(value, step=PROBE_GRID_STEP):
    return round(float(value) / step) * step


def _round_probe_vec(v, step=PROBE_GRID_STEP):
    return Vec3(_round_probe(v.x, step), _round_probe(v.y, step), _round_probe(v.z, step))


def _snap_probe_yaw(yaw):
    return _round_probe(float(yaw), PROBE_YAW_STEP)


def _player_facing_vectors():
    yaw = math.radians(float(player.rotation_y))
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
            scale=half * 2,
        )
        # Probes sind nur Debug-/Abtast-Entities. Sie dürfen keinen Collider haben,
        # sonst können Ursina-Raycasts sie als Hindernis treffen.
        e.collider = None
        player_probe_entities[name] = e
        player_probe_hits[name] = False


def _world_pos_from_local(base_pos, local_offset):
    forward, right = _player_facing_vectors()
    return base_pos + (right * local_offset.x) + Vec3(0, local_offset.y, 0) + (forward * local_offset.z)


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

    # Für Kollisionsabfragen die echte Player-Position verwenden, nicht auf das Blockraster runden.
    # Gerundete Probes können sonst an Kanten falsche Treffer liefern.
    snapped_base = Vec3(float(base_position.x), float(base_position.y), float(base_position.z))
    base_yaw = float(player.rotation_y)

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

            probe.color = PROBE_HIT_COLOR if hit else PROBE_COLOR

    player_probe_hits.clear()
    player_probe_hits.update(hits)
    return hits


def _apply_player_probe_horizontal():
    """Bewegt den Spieler horizontal per Sweep-Kollision.

    Der FirstPersonController darf nicht mehr selbst laufen, weil er sonst erst in den
    Block hineinbewegt und unsere Korrektur dagegen ankämpfen muss. Stattdessen wird
    die gewünschte WASD-Bewegung hier berechnet und vor dem Setzen der Position gegen
    die Block-AABBs gesweept.
    """
    global prev_horizontal_x, prev_horizontal_z

    cur_x = float(player.x)
    cur_z = float(player.z)

    if not getattr(player, "enabled", True):
        prev_horizontal_x = cur_x
        prev_horizontal_z = cur_z
        _sample_player_probes_at(Vec3(cur_x, float(player.y), cur_z), do_assign=True)
        return

    forward = Vec3(player.forward)
    forward.y = 0
    if forward.length_squared() > 1e-8:
        forward = forward.normalized()

    right = Vec3(player.right)
    right.y = 0
    if right.length_squared() > 1e-8:
        right = right.normalized()

    move_dir = (
        forward * (held_keys["w"] - held_keys["s"])
        + right * (held_keys["d"] - held_keys["a"])
    )

    if move_dir.length_squared() > 1e-8:
        move_dir = move_dir.normalized()
        target_x = cur_x + move_dir.x * PLAYER_MOVE_SPEED * time.dt
        target_z = cur_z + move_dir.z * PLAYER_MOVE_SPEED * time.dt
    else:
        target_x = cur_x
        target_z = cur_z

    y_min, y_max = _player_body_y_span()

    dx = target_x - cur_x
    dz = target_z - cur_z
    if abs(dx) >= abs(dz):
        res_x = _sweep_x(cur_x, target_x, cur_z, y_min, y_max)
        res_z = _sweep_z(cur_z, target_z, res_x, y_min, y_max)
    else:
        res_z = _sweep_z(cur_z, target_z, cur_x, y_min, y_max)
        res_x = _sweep_x(cur_x, target_x, res_z, y_min, y_max)

    # Normale Überlappungsauflösung, falls z.B. ein Block direkt am Spieler
    # gebaut wurde oder Rundungsfehler eine minimale Penetration erzeugen.
    res_x, res_z = _resolve_horizontal_penetration(res_x, res_z, y_min, y_max)

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


def _can_stand_at(px, pz, foot_y):
    y_min = foot_y + PLAYER_COLLISION_FOOT_CLEARANCE
    y_max = foot_y + PLAYER_HEIGHT - PLAYER_COLLISION_HEAD_CLEARANCE
    min_x = px - PLAYER_COLLISION_RADIUS
    max_x = px + PLAYER_COLLISION_RADIUS
    min_z = pz - PLAYER_COLLISION_RADIUS
    max_z = pz + PLAYER_COLLISION_RADIUS
    return not _aabb_hits_any_block(min_x, max_x, y_min, y_max, min_z, max_z)


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
        current_y = float(player.y)
        current_foot = current_y - PLAYER_STAND_HEIGHT

        support_scan_up = MAX_STEP_UP
        if vertical_velocity < 0:
            support_scan_up = MAX_STEP_UP + max(BLOCK_HEIGHT, (-vertical_velocity * dt) + 0.05)

        support_y = _find_support_y(px, pz, current_foot, support_scan_up)
        if support_y is None and len(block_face_counts) > 0:
            support_y = _find_support_y_fallback(px, pz, current_foot, support_scan_up)

        # Boden-Snap/Step-Up nur beim Fallen oder Stehen. Beim Springen darf kein
        # oberer Block den Spieler per Support-Suche nach oben "festziehen".
        if support_y is not None and vertical_velocity <= 0:
            if current_foot < support_y:
                if _can_stand_at(px, pz, support_y):
                    player.y = support_y + PLAYER_STAND_HEIGHT
                    vertical_velocity = 0.0
                    is_grounded = True
                    _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
                    continue

            d = current_foot - support_y
            if 0 <= d <= GROUND_STICK:
                player.y = support_y + PLAYER_STAND_HEIGHT
                vertical_velocity = 0.0
                is_grounded = True
                _sample_player_probes_at(Vec3(px, float(player.y), pz), do_assign=True)
                continue

        vertical_velocity = max(vertical_velocity - GRAVITY_ACCEL * time.dt, -MAX_FALL_SPEED / 60)
        next_y = float(player.y) + vertical_velocity * dt

        swept_y, vertical_hit = _sweep_y(float(player.y), next_y, px, pz)
        player.y = swept_y

        if vertical_hit == "ceiling":
            vertical_velocity = 0.0
            is_grounded = False
        elif vertical_hit == "ground":
            vertical_velocity = 0.0
            is_grounded = True
        else:
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
    if base not in block_rotations:
        block_rotations[base] = DEFAULT_BLOCK_ROTATION
    chunk_face_sets[chunk_coord].add(face_key)
    _register_top_face(face_key[0], face_key[1])
    affected.add(chunk_coord)
    return True


def place_block_programmatically(base_tuple, btype, affected, rotation=DEFAULT_BLOCK_ROTATION):
    base_key = _vkey(base_tuple)
    if base_key in block_types:
        return

    _set_block_type(base_key, btype, rotation=rotation)
    cube_base = Vec3(*base_key)

    for i, off in enumerate(_FACE_OFFSETS):
        fp = cube_base + off
        same = _face_key(fp, i)
        opp = _face_key(fp, _OPPOSITE_FACE[i])

        if opp in world_faces:
            _remove_face(opp, affected)
        elif same not in world_faces:
            tgt = _chunk_coord_from_face(fp, i)
            _add_face(same, tgt, affected, block_type=btype)


def generate_tree(x, y, z, affected):
    height_val = math.sin(x * 78.233 + z * 12.9898 + seed) * 31337.1337
    fraction = abs(height_val) - math.floor(abs(height_val))
    tree_height = 4 + int(fraction * 3)

    for i in range(tree_height):
        place_block_programmatically((x, y + i * BLOCK_HEIGHT, z), "log", affected)

    top_y = y + (tree_height - 1) * BLOCK_HEIGHT
    for dx in range(-2, 3):
        for dy in range(-1, 2):
            for dz in range(-2, 3):
                if dx == 0 and dz == 0 and dy <= 0:
                    continue
                if abs(dx) + abs(dy) + abs(dz) > 3:
                    continue

                lx = x + dx * 1.0
                ly = top_y + dy * BLOCK_HEIGHT
                lz = z + dz * 1.0
                place_block_programmatically((lx, ly, lz), "leaves", affected)


def load_chunks():
    global surface_heights
    world_faces.clear()
    face_to_chunk.clear()
    face_block_types.clear()
    block_types.clear()
    block_rotations.clear()
    top_columns.clear()
    top_cells.clear()
    block_face_counts.clear()
    surface_heights.clear()
    _reset_chunk_storage()

    try:
        chunks_opened_ = list(eval(open("chunks.txt", "r").read()))

        for legacy_idx, chunk_data in enumerate(chunks_opened_):
            _ensure_chunk(_legacy_chunk_coord_from_index(legacy_idx))
            positions = chunk_data[0]
            indices = chunk_data[1]
            block_type_data = chunk_data[2] if len(chunk_data) > 2 else None
            block_rotation_data = chunk_data[3] if len(chunk_data) > 3 else None

            for i, face_pos in enumerate(positions):
                if i >= len(indices):
                    break
                fidx = int(indices[i])
                btype = DEFAULT_BLOCK_TYPE
                if block_type_data is not None and i < len(block_type_data):
                    btype = _normalize_block_type(block_type_data[i])

                brot = DEFAULT_BLOCK_ROTATION
                if block_rotation_data is not None and i < len(block_rotation_data):
                    brot = _normalize_block_rotation(block_rotation_data[i])

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
                    block_rotations[base] = brot
                elif base not in block_rotations:
                    block_rotations[base] = DEFAULT_BLOCK_ROTATION
                chunk_face_sets[chunk_coord].add(key)
                _register_top_face(key[0], key[1])
    except Exception as e:
        print(f"Error loading chunks.txt: {e}")

    # Aktualisiert die Oberflächen und bestimmt deterministisch Tiefe/Stein-Schichten
    _apply_surface_layers()

    affected_by_trees = set()
    grass_blocks = [base for base, btype in block_types.items() if btype == "grass"]

    for base in grass_blocks:
        x, yb, z = base
        placement_val = math.sin(x * 12.9898 + z * 78.233 + seed) * 43758.5453
        tree_chance = abs(placement_val) - math.floor(abs(placement_val))

        if tree_chance < 0.02:
            tree_y = yb + BLOCK_HEIGHT
            generate_tree(x, tree_y, z, affected_by_trees)

    all_chunks_to_rebuild = set(chunk_face_sets.keys()).union(affected_by_trees)

    for chunk_coord in all_chunks_to_rebuild:
        _rebuild_chunk_mesh(chunk_coord)


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


def _horizontal_face_from_direction(direction):
    dx = float(direction.x)
    dz = float(direction.z)

    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        yaw = math.radians(float(player.rotation_y))
        dx = math.sin(yaw)
        dz = math.cos(yaw)

    if abs(dx) >= abs(dz):
        return 4 if dx >= 0 else 5
    return 2 if dz >= 0 else 3


def _placement_rotation_from_face(placement_face_idx):
    face_idx = _normalize_face_index(placement_face_idx)

    # Nur Y-Achse:
    # - Beim Platzieren an einer Wand nimmt der neue Block die Wand-Richtung als Yaw.
    # - Beim Platzieren auf Boden/Decke nimmt er die horizontale Blickrichtung.
    # Dadurch bleiben Gras-Oberseite und Unterseite immer korrekt oben/unten.
    if face_idx in HORIZONTAL_ROTATION_FACES:
        forward_face = face_idx
    else:
        forward = Vec3(camera.forward)
        forward.y = 0
        if forward.length_squared() <= 1e-8:
            forward = Vec3(player.forward)
            forward.y = 0
        forward_face = _horizontal_face_from_direction(forward)

    return _normalize_block_rotation(forward_face)


def build(placement_rotation=DEFAULT_BLOCK_ROTATION):
    placement_rotation = _normalize_block_rotation(placement_rotation)
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

    _set_block_type(base_key, selected_block_type, rotation=placement_rotation)

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

    main_chunk = _chunk_coord_from_pos(cube_base)
    if main_chunk in affected:
        _rebuild_chunk_mesh(main_chunk)
        affected.discard(main_chunk)

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

    block_types.pop(cube_base, None)
    block_rotations.pop(cube_base, None)

    below = _vkey((cube_base[0], cube_base[1] - BLOCK_HEIGHT, cube_base[2]))
    if below in block_types and _normalize_block_type(block_types[below]) == "grass":
        _set_block_type(below, "dirt")

    main_chunk = _chunk_coord_from_pos(cube_base)
    if main_chunk in affected:
        _rebuild_chunk_mesh(main_chunk)
        affected.discard(main_chunk)

    _refresh_chunks(affected)
    c.y = -9999


def _frame_position_for_target(face_pos, face_idx):
    hit_base = _cube_base_from_face(face_pos, face_idx)
    return Vec3(hit_base[0], hit_base[1] + 1.5, hit_base[2])


def update():
    if chunk_update_queue:
        chunk_to_update = chunk_update_queue.pop(0)
        _rebuild_chunk_mesh(chunk_to_update)


class PlayerPhysicsController(Entity):
    def update(self):
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

    if key == "o":
        mode = 1 - mode

    if key == "c hold" or key == "c":
        camera.fov = 30
    if key == "c up":
        camera.fov = 120

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

    if key in BLOCK_SELECT_KEYS:
        selected_block_type = BLOCK_SELECT_KEYS[key]

    if key in ("right mouse down", "5"):
        face_pos, normal, face_idx = get_target_face()
        if face_pos:
            cube_base = Vec3(face_pos) - _FACE_OFFSETS[face_idx] + normal
            c.position = cube_base + Vec3(0, 1.5, 0)
            build(_placement_rotation_from_face(face_idx))

    if key in ("left mouse down", "4"):
        face_pos, _, face_idx = get_target_face()
        if face_pos:
            mine(face_pos, face_idx)

    if key == "r":
        player.y += 10
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


player_physics_controller = PlayerPhysicsController()

app.run()
