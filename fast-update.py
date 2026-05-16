from ursina import *
from panda3d.core import LVecBase3f, Vec4, Vec3, Vec2, PNMImage, Filename
from panda3d.core import Texture as PandaTexture, SamplerState as PandaSamplerState
from ursina.texture import Texture as UrsinaTexture
from ursina.prefabs.first_person_controller import *
from itertools import *
import math
import numpy as np
from bisect import bisect_left, bisect_right
from time import perf_counter

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

# Time-sliced Lazy Greedy-Meshing:
# Beim Bauen/Abbauen wird sofort nur ein billiges Dirty-Preview-Mesh angezeigt.
# Das teure Cross-Type-Greedy-Mesh mit gebackener Chunk-Textur wird danach
# NICHT nur verzögert, sondern wirklich über mehrere Frames aufgebaut.
chunk_update_set = set()
chunk_update_due = {}
chunk_rebuild_versions = {}
active_chunk_rebuild_job = None
dirty_chunk_previews = {}

# Nur sehr kleine Sammelzeit für mehrere schnelle Änderungen. Das ist KEINE
# "Bauzeit". Die eigentliche Arbeit wird durch LAZY_REBUILD_FRAME_BUDGET verteilt.
LAZY_REBUILD_SETTLE_DELAY = 0.03

# Maximal erlaubte Mesh-/Bake-Arbeit pro Frame in Sekunden.
# 0.0025-0.0045 ist meistens angenehm. Höher = schneller fertig, aber mehr Frame-Spikes.
LAZY_REBUILD_FRAME_BUDGET = 0.0035
DIRTY_PREVIEW_ENABLED = True
_REBUILD_JOB_DEADLINE = 0.0

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
    global active_chunk_rebuild_job
    for obj in combined_terrains.values():
        _safe_clear_destroy(obj)
    for obj in dirty_chunk_previews.values():
        _safe_clear_destroy(obj)
    dirty_chunk_previews.clear()
    chunk_update_queue.clear()
    chunk_update_set.clear()
    chunk_update_due.clear()
    chunk_rebuild_versions.clear()
    active_chunk_rebuild_job = None
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


# Cross-type Greedy-Meshing:
# Ein einziges großes Quad kann mit normalen Atlas-UVs immer nur eine Textur
# wiederholen. Damit ein Quad trotzdem mehrere Blocktypen/Rotationen enthalten
# kann, wird pro Chunk eine kleine gebackene Textur erzeugt. In diese Textur
# werden die Atlas-Kacheln der einzelnen Blockzellen hineinkopiert; das Mesh
# benutzt danach normale UVs auf diese Chunk-Textur und braucht keinen Lookup-
# Shader mehr.
_BAKED_SOURCE_ATLAS_IMAGE = None
_BAKED_SOURCE_TILE_SIZE = None
_BAKED_TEXTURE_PADDING = 1
_BAKED_TEXTURE_VERSION = 0
_BAKED_CHUNK_TEXTURE_KEEPALIVE = {}


class _BakedTextureWrapper:
    # Minimaler Fallback für Entity.texture: Ursinas texture_setter greift auf ._texture zu.
    def __init__(self, panda_texture):
        self._texture = panda_texture


baked_texture_shader = Shader(
    language=Shader.GLSL,
    vertex=
    "#version 120\n"
    "uniform mat4 p3d_ModelViewProjectionMatrix;\n"
    "attribute vec4 p3d_Vertex;\n"
    "attribute vec2 p3d_MultiTexCoord0;\n"
    "attribute vec4 p3d_Color;\n"
    "varying vec2 v_uv;\n"
    "varying vec4 v_color;\n"
    "void main(){\n"
    "    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;\n"
    "    v_uv = p3d_MultiTexCoord0;\n"
    "    v_color = p3d_Color;\n"
    "}\n",
    fragment=
    "#version 120\n"
    "uniform sampler2D p3d_Texture0;\n"
    "varying vec2 v_uv;\n"
    "varying vec4 v_color;\n"
    "void main(){\n"
    "    gl_FragColor = texture2D(p3d_Texture0, v_uv) * v_color;\n"
    "}\n",
)


def _next_power_of_two(value):
    value = max(1, int(math.ceil(float(value))))
    return 1 << (value - 1).bit_length()


def _ensure_pnm_alpha(img):
    try:
        if img.get_num_channels() < 4:
            img.add_alpha()
    except:
        try:
            img.add_alpha()
        except:
            pass
    return img


def _read_pnm_candidate(candidate):
    img = PNMImage()
    try:
        if img.read(Filename(candidate)) and img.get_x_size() > 0 and img.get_y_size() > 0:
            return _ensure_pnm_alpha(img)
    except:
        pass
    return None


def _source_atlas_pnm():
    global _BAKED_SOURCE_ATLAS_IMAGE, _BAKED_SOURCE_TILE_SIZE

    if _BAKED_SOURCE_ATLAS_IMAGE is not None and _BAKED_SOURCE_TILE_SIZE is not None:
        return _BAKED_SOURCE_ATLAS_IMAGE, _BAKED_SOURCE_TILE_SIZE[0], _BAKED_SOURCE_TILE_SIZE[1]

    img = None
    base_name = str(texture)

    # Erst direkt aus Dateien lesen. Das ist am stabilsten, weil die Zeilenrichtung
    # dann der echten Atlas-Datei entspricht. Zusätzlich werden Ursinas asset_folder
    # und ein lokaler assets/-Ordner probiert, bevor Texture.store als Fallback kommt.
    base_candidates = [base_name]
    try:
        asset_folder = getattr(application, "asset_folder", None)
        if asset_folder:
            try:
                base_candidates.append(str(asset_folder / base_name))
            except:
                pass
            base_candidates.append(str(asset_folder) + "/" + base_name)
    except:
        pass
    base_candidates.append("assets/" + base_name)

    candidates = []
    for base_candidate in base_candidates:
        candidates.append(base_candidate)
        if "." not in base_candidate.rsplit("/", 1)[-1]:
            candidates.extend([base_candidate + ".png", base_candidate + ".jpg", base_candidate + ".jpeg"])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        img = _read_pnm_candidate(candidate)
        if img is not None:
            break

    if img is None and atlas_texture is not None:
        try:
            temp = PNMImage()
            # load_texture() gibt in Ursina normalerweise einen UrsinaTexture-Wrapper zurück.
            # Der eigentliche Panda3D-Texture liegt dann in ._texture und nur der hat store().
            raw_atlas_texture = getattr(atlas_texture, "_texture", atlas_texture)
            if raw_atlas_texture is not None and raw_atlas_texture.store(temp) and temp.get_x_size() > 0 and temp.get_y_size() > 0:
                img = _ensure_pnm_alpha(temp)
        except:
            img = None

    if img is None:
        print("Cross-type texture baking failed: atlas image could not be read.")
        return None, None, None

    tile_w = max(1, int(img.get_x_size()) // max(1, int(ATLAS_TILES_X)))
    tile_h = max(1, int(img.get_y_size()) // max(1, int(ATLAS_TILES_Y)))

    _BAKED_SOURCE_ATLAS_IMAGE = img
    _BAKED_SOURCE_TILE_SIZE = (tile_w, tile_h)
    return img, tile_w, tile_h


def _new_pnm_image(width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    try:
        img = PNMImage(width, height, 4)
    except:
        img = PNMImage(width, height)
        try:
            img.add_alpha()
        except:
            pass

    try:
        img.fill(0, 0, 0)
    except:
        pass
    try:
        img.alpha_fill(0)
    except:
        pass
    return img


def _get_pnm_pixel(img, x, y):
    x = int(x)
    y = int(y)
    col = img.get_xel(x, y)
    try:
        alpha = img.get_alpha(x, y)
    except:
        alpha = 1.0
    return col, alpha


def _set_pnm_pixel(img, x, y, col, alpha=1.0):
    x = int(x)
    y = int(y)
    try:
        img.set_xel(x, y, col)
    except:
        try:
            img.set_xel(x, y, float(col[0]), float(col[1]), float(col[2]))
        except:
            img.set_xel(x, y, 1, 1, 1)
    try:
        img.set_alpha(x, y, float(alpha))
    except:
        pass


def _copy_pnm_pixel(img, sx, sy, dx, dy):
    if dx < 0 or dy < 0 or dx >= img.get_x_size() or dy >= img.get_y_size():
        return
    if sx < 0 or sy < 0 or sx >= img.get_x_size() or sy >= img.get_y_size():
        return
    col, alpha = _get_pnm_pixel(img, sx, sy)
    _set_pnm_pixel(img, dx, dy, col, alpha)


def _pad_pnm_rect(img, x, y, w, h, padding):
    padding = int(padding)
    if padding <= 0 or w <= 0 or h <= 0:
        return

    # Links/rechts die Randpixel kopieren.
    for p in range(1, padding + 1):
        for yy in range(y, y + h):
            _copy_pnm_pixel(img, x, yy, x - p, yy)
            _copy_pnm_pixel(img, x + w - 1, yy, x + w - 1 + p, yy)

    # Oben/unten inklusive der gerade erzeugten Seitenränder kopieren.
    for p in range(1, padding + 1):
        for xx in range(x - padding, x + w + padding):
            src_x = min(max(xx, x), x + w - 1)
            _copy_pnm_pixel(img, src_x, y, xx, y - p)
            _copy_pnm_pixel(img, src_x, y + h - 1, xx, y + h - 1 + p)


def _frac(value):
    value = float(value)
    return value - math.floor(value)


def _single_block_face_verts(base, face_idx):
    x = float(base[0])
    y = float(base[1])
    z = float(base[2])
    x0 = x - 0.5
    x1 = x + 0.5
    y0 = y + float(_FACE_OFFSETS[0].y)
    y1 = y + float(_FACE_OFFSETS[1].y)
    z0 = z - 0.5
    z1 = z + 0.5

    face_idx = int(face_idx)
    if face_idx == 0:
        return [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    if face_idx == 1:
        return [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]
    if face_idx == 2:
        return [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    if face_idx == 3:
        return [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]
    if face_idx == 4:
        return [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
    return [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]


def _cell_material_info(base, block_type, rotation, world_face_idx):
    local_face_idx = _local_face_for_world_face(world_face_idx, rotation)
    local_u_axis, local_v_axis = _LOCAL_FACE_UV_AXES.get(
        local_face_idx,
        ((1, 0, 0), (0, 1, 0)),
    )

    world_u_axis = _transform_local_axis_to_world(local_u_axis, rotation)
    world_v_axis = _transform_local_axis_to_world(local_v_axis, rotation)
    single_verts = _single_block_face_verts(base, world_face_idx)
    raw_us = [_axis_coord_for_uv(v, world_u_axis) for v in single_verts]
    raw_vs = [_axis_coord_for_uv(v, world_v_axis) for v in single_verts]

    return {
        "base": base,
        "btype": block_type,
        "brot": rotation,
        "tile": _block_tile_for_world_face(block_type, rotation, int(world_face_idx)),
        "u_axis": world_u_axis,
        "v_axis": world_v_axis,
        "u_min": min(raw_us),
        "v_min": min(raw_vs),
    }


def _grid_cell_to_baked_cell(face_idx, width_cells, height_cells, du, dv):
    face_idx = int(face_idx)
    if face_idx == 1:
        return int(du), int(height_cells - 1 - dv)
    if face_idx in (3, 4):
        return int(width_cells - 1 - du), int(dv)
    return int(du), int(dv)


def _world_point_from_baked_local(face_idx, slice_idx, grid_u0, grid_v0, width_cells, height_cells, local_u, local_v):
    face_idx = int(face_idx)
    slice_idx = float(slice_idx)
    grid_u0 = float(grid_u0)
    grid_v0 = float(grid_v0)
    width_cells = float(width_cells)
    height_cells = float(height_cells)
    local_u = float(local_u)
    local_v = float(local_v)
    y_bottom = float(_FACE_OFFSETS[0].y)
    y_top = float(_FACE_OFFSETS[1].y)

    if face_idx == 0:
        return (
            grid_u0 - 0.5 + local_u,
            slice_idx * BLOCK_HEIGHT + y_bottom,
            grid_v0 - 0.5 + local_v,
        )
    if face_idx == 1:
        return (
            grid_u0 - 0.5 + local_u,
            slice_idx * BLOCK_HEIGHT + y_top,
            grid_v0 - 0.5 + (height_cells - local_v),
        )
    if face_idx == 2:
        return (
            grid_u0 - 0.5 + local_u,
            grid_v0 * BLOCK_HEIGHT + y_bottom + local_v * BLOCK_HEIGHT,
            slice_idx + 0.5,
        )
    if face_idx == 3:
        return (
            grid_u0 - 0.5 + (width_cells - local_u),
            grid_v0 * BLOCK_HEIGHT + y_bottom + local_v * BLOCK_HEIGHT,
            slice_idx - 0.5,
        )
    if face_idx == 4:
        return (
            slice_idx + 0.5,
            grid_v0 * BLOCK_HEIGHT + y_bottom + local_v * BLOCK_HEIGHT,
            grid_u0 - 0.5 + (width_cells - local_u),
        )
    return (
        slice_idx - 0.5,
        grid_v0 * BLOCK_HEIGHT + y_bottom + local_v * BLOCK_HEIGHT,
        grid_u0 - 0.5 + local_u,
    )


def _source_pixel_for_cell(source_img, tile_w, tile_h, tile, f_u, f_v):
    tx = max(0, min(int(ATLAS_TILES_X) - 1, int(tile[0])))
    ty = max(0, min(int(ATLAS_TILES_Y) - 1, int(tile[1])))
    f_u = max(0.0, min(0.999999, float(f_u)))
    f_v = max(0.0, min(0.999999, float(f_v)))

    sx = tx * tile_w + int(f_u * tile_w)
    # Tile-Koordinaten in BLOCK_FACE_TILES sind wie im Bild: y=0 ist die obere Atlas-Zeile.
    # UV-v=0 ist dagegen die Unterkante einer Kachel, deshalb wird die y-Achse hier gedreht.
    sy = ty * tile_h + (tile_h - 1 - int(f_v * tile_h))
    sx = max(0, min(source_img.get_x_size() - 1, sx))
    sy = max(0, min(source_img.get_y_size() - 1, sy))
    return _get_pnm_pixel(source_img, sx, sy)


def _copy_cell_to_baked_texture(dst_img, source_img, tile_w, tile_h, job, du, dv):
    info = job["grid"][(job["u"] + du, job["v"] + dv)]
    baked_col, baked_row_bottom = _grid_cell_to_baked_cell(job["face"], job["w"], job["h"], du, dv)

    dst_x0 = job["pack_x"] + baked_col * tile_w
    dst_y0 = job["pack_y"] + (job["h"] - 1 - baked_row_bottom) * tile_h

    for py in range(tile_h):
        local_v = baked_row_bottom + 1.0 - ((py + 0.5) / tile_h)
        for px in range(tile_w):
            local_u = baked_col + ((px + 0.5) / tile_w)
            world_point = _world_point_from_baked_local(
                job["face"], job["slice_idx"], job["u"], job["v"], job["w"], job["h"], local_u, local_v
            )

            raw_u = _axis_coord_for_uv(world_point, info["u_axis"])
            raw_v = _axis_coord_for_uv(world_point, info["v_axis"])
            f_u = _frac(raw_u - info["u_min"])
            f_v = _frac(raw_v - info["v_min"])

            col, alpha = _source_pixel_for_cell(source_img, tile_w, tile_h, info["tile"], f_u, f_v)
            _set_pnm_pixel(dst_img, dst_x0 + px, dst_y0 + py, col, alpha)


def _build_baked_chunk_texture(quad_jobs, chunk_coord=None):
    global _BAKED_TEXTURE_VERSION
    if not quad_jobs:
        return None, None, None, None, None

    source_img, tile_w, tile_h = _source_atlas_pnm()
    if source_img is None:
        return None, None, None, None, None

    padding = int(_BAKED_TEXTURE_PADDING)
    total_area = 0
    max_rect_w = 1
    for job in quad_jobs:
        job["tile_w"] = tile_w
        job["tile_h"] = tile_h
        job["tex_w"] = max(1, int(job["w"]) * tile_w)
        job["tex_h"] = max(1, int(job["h"]) * tile_h)
        packed_w = job["tex_w"] + padding * 2
        packed_h = job["tex_h"] + padding * 2
        total_area += packed_w * packed_h
        max_rect_w = max(max_rect_w, packed_w)

    target_w = _next_power_of_two(max(max_rect_w, math.sqrt(max(1, total_area))))

    # Einfaches Shelf-Packing. Große Rechtecke zuerst packen weniger schlecht.
    packing_order = sorted(range(len(quad_jobs)), key=lambda i: quad_jobs[i]["tex_h"] * quad_jobs[i]["tex_w"], reverse=True)
    x = 0
    y = 0
    row_h = 0
    for idx in packing_order:
        job = quad_jobs[idx]
        rect_w = job["tex_w"] + padding * 2
        rect_h = job["tex_h"] + padding * 2
        if x > 0 and x + rect_w > target_w:
            x = 0
            y += row_h
            row_h = 0
        job["pack_outer_x"] = x
        job["pack_outer_y"] = y
        job["pack_x"] = x + padding
        job["pack_y"] = y + padding
        x += rect_w
        row_h = max(row_h, rect_h)

    target_h = _next_power_of_two(y + row_h)
    baked_img = _new_pnm_image(target_w, target_h)

    for job in quad_jobs:
        for du in range(job["w"]):
            for dv in range(job["h"]):
                _copy_cell_to_baked_texture(baked_img, source_img, tile_w, tile_h, job, du, dv)
        _pad_pnm_rect(baked_img, job["pack_x"], job["pack_y"], job["tex_w"], job["tex_h"], padding)

    # Wichtig: Nicht Ursinas Texture("name") benutzen — das interpretiert den String
    # als Dateipfad. Außerdem bekommt jede Rebuild-Textur einen eindeutigen Namen;
    # Panda/Ursina können sonst bei wiederverwendeten Chunk-Entities alte Texture-States
    # oder Cache-Einträge behalten, was nach Abbauen/Bauen zu schwarzen Chunks führen kann.
    _BAKED_TEXTURE_VERSION += 1
    if chunk_coord is None:
        tex_name = f"chunk_baked_mixed_tiles_{_BAKED_TEXTURE_VERSION}"
    else:
        cx, cy, cz = chunk_coord
        tex_name = f"chunk_baked_mixed_tiles_{cx}_{cy}_{cz}_{_BAKED_TEXTURE_VERSION}"
    panda_tex = PandaTexture(tex_name)
    try:
        panda_tex.load(baked_img)
    except:
        print("Cross-type texture baking failed: could not upload baked texture.")
        return None, None, None, None, None

    nearest = None
    clamp_mode = None
    for enum_owner in (PandaSamplerState, PandaTexture):
        if nearest is None:
            for name in ("FT_nearest", "FTNearest"):
                if hasattr(enum_owner, name):
                    nearest = getattr(enum_owner, name)
                    break
        if clamp_mode is None:
            for name in ("WM_clamp", "WMClamp"):
                if hasattr(enum_owner, name):
                    clamp_mode = getattr(enum_owner, name)
                    break

    if nearest is not None:
        for method_name in ("set_minfilter", "setMinfilter"):
            try:
                getattr(panda_tex, method_name)(nearest)
                break
            except:
                pass
        for method_name in ("set_magfilter", "setMagfilter"):
            try:
                getattr(panda_tex, method_name)(nearest)
                break
            except:
                pass

    if clamp_mode is not None:
        for method_name in ("set_wrap_u", "setWrapU"):
            try:
                getattr(panda_tex, method_name)(clamp_mode)
                break
            except:
                pass
        for method_name in ("set_wrap_v", "setWrapV"):
            try:
                getattr(panda_tex, method_name)(clamp_mode)
                break
            except:
                pass

    try:
        baked_tex = UrsinaTexture(panda_tex, filtering=None)
    except:
        # Fallback für Entity.texture: Der Setter erwartet ein Objekt mit ._texture.
        baked_tex = _BakedTextureWrapper(panda_tex)

    return baked_tex, panda_tex, tile_w, tile_h, (target_w, target_h)


def _apply_baked_chunk_texture(ent, baked_tex, panda_tex):
    """Bindet die gebackene Panda3D-Texture direkt und hält Referenzen am Entity.

    Entity.texture alleine ist bei dynamisch erzeugten Texturen in manchen Ursina/Panda-
    Kombinationen nicht zuverlässig, besonders wenn ein Chunk-Entity wiederverwendet wird.
    Deshalb setzen wir die Texture zusätzlich direkt auf der NodePath und speichern starke
    Referenzen, damit Python die Texture nicht freigibt.
    """
    try:
        ent.shader = baked_texture_shader
    except:
        pass

    try:
        ent.color = color.white
    except:
        pass
    for method_name, args in (
        ("set_color", (1, 1, 1, 1)),
        ("setColor", (1, 1, 1, 1)),
        ("set_color_scale", (1, 1, 1, 1)),
        ("setColorScale", (1, 1, 1, 1)),
    ):
        try:
            getattr(ent, method_name)(*args)
        except:
            pass

    # Alte Texture-States weg, dann die neue Panda-Texture direkt binden.
    for method_name in ("clear_texture", "clearTexture"):
        try:
            getattr(ent, method_name)()
            break
        except:
            pass

    applied_directly = False
    for method_name in ("set_texture", "setTexture"):
        try:
            getattr(ent, method_name)(panda_tex, 1)
            applied_directly = True
            break
        except TypeError:
            try:
                getattr(ent, method_name)(panda_tex)
                applied_directly = True
                break
            except:
                pass
        except:
            pass

    if not applied_directly:
        try:
            ent.texture = baked_tex
        except:
            pass

    # Starke Referenzen: verhindert schwarze Chunks durch Garbage Collection der
    # dynamisch erzeugten Texture/Wrapper nach dem Rebuild.
    ent._baked_texture_ref = baked_tex
    ent._baked_panda_texture_ref = panda_tex
    try:
        ent._texture = baked_tex
    except:
        pass


def _make_baked_chunk_entity(mesh, baked_tex, panda_tex):
    ent = Entity(model=mesh)
    _apply_baked_chunk_texture(ent, baked_tex, panda_tex)
    ent.collider = None  # Chunks bleiben absichtlich ohne Ursina-Collider.
    ent.enabled = True
    return ent


def _baked_uv(job, local_u, local_v, baked_size):
    tex_w, tex_h = baked_size
    u = (job["pack_x"] + float(local_u) * job["tile_w"]) / max(1.0, float(tex_w))
    # PNMImage-Zeilen laufen von oben nach unten; Texture-v läuft von unten nach oben.
    v = 1.0 - ((job["pack_y"] + job["tex_h"] - float(local_v) * job["tile_h"]) / max(1.0, float(tex_h)))
    return (u, v)


def _rebuild_chunk_mesh(chunk_coord):
    chunk_coord = _ensure_chunk(chunk_coord)
    old = combined_terrains.get(chunk_coord)

    faces_snapshot = chunk_face_sets[chunk_coord]
    if not faces_snapshot:
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return

    quad_jobs = []

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
            else:
                slice_idx, u, v = lx, lz, ly

            if slice_idx not in slices:
                slices[slice_idx] = {}
            slices[slice_idx][(u, v)] = _cell_material_info(base, btype, brot, int(d))

        for slice_idx, grid in slices.items():
            if not grid:
                continue

            visited = set()
            keys = grid.keys()
            min_u, max_u = min(k[0] for k in keys), max(k[0] for k in keys)
            min_v, max_v = min(k[1] for k in keys), max(k[1] for k in keys)

            for v in range(min_v, max_v + 1):
                for u in range(min_u, max_u + 1):
                    if (u, v) in visited or (u, v) not in grid:
                        continue

                    # Wichtig: Blocktyp, Textur-Kachel und Rotation sind hier ABSICHTLICH
                    # keine Grenze mehr. Alles, was auf derselben Ebene ein sichtbares Face
                    # hat, darf zu einem Quad werden. Die verschiedenen Texturen werden danach
                    # in die Chunk-Textur gebacken.
                    w = 1
                    while (u + w) <= max_u and (u + w, v) not in visited and (u + w, v) in grid:
                        w += 1

                    h = 1
                    can_expand = True
                    while (v + h) <= max_v and can_expand:
                        for du in range(w):
                            if (u + du, v + h) in visited or (u + du, v + h) not in grid:
                                can_expand = False
                                break
                        if can_expand:
                            h += 1

                    for du in range(w):
                        for dv in range(h):
                            visited.add((u + du, v + dv))

                    if d in (0, 1):
                        bx = float(u)
                        by = float(slice_idx) * BLOCK_HEIGHT
                        bz = float(v)
                        W_ext, H_ext, D_ext = w, BLOCK_HEIGHT, h
                    elif d in (2, 3):
                        bx = float(u)
                        by = float(v) * BLOCK_HEIGHT
                        bz = float(slice_idx)
                        W_ext, H_ext, D_ext = w, h * BLOCK_HEIGHT, 1.0
                    else:
                        bx = float(slice_idx)
                        by = float(v) * BLOCK_HEIGHT
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

                    quad_jobs.append({
                        "face": int(d),
                        "slice_idx": int(slice_idx),
                        "u": int(u),
                        "v": int(v),
                        "w": int(w),
                        "h": int(h),
                        "grid": grid,
                        "quad_verts": quad_verts,
                        "W_ext": W_ext,
                        "H_ext": H_ext,
                        "D_ext": D_ext,
                        "normal": _FACE_NORMALS_TUPLES.get(d, (0, 1, 0)),
                    })

    if not quad_jobs:
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return

    baked_tex, panda_tex, tile_w, tile_h, baked_size = _build_baked_chunk_texture(quad_jobs, chunk_coord)
    if baked_tex is None or panda_tex is None:
        # Ohne lesbares Atlas-Bild kann ein einzelnes Quad nicht korrekt mehrere Texturen tragen.
        # Lieber den Chunk leer lassen als wieder die kaputten smeared Textures zu zeichnen.
        print("Chunk mesh skipped: baked texture was unavailable.")
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return

    vertices = []
    triangles = []
    uvs = []
    normals = []
    colors = []

    for job in quad_jobs:
        quad_verts = job["quad_verts"]
        local_uvs = _fast_uvs(job["face"], job["W_ext"], job["H_ext"], job["D_ext"])
        quad_uvs = [_baked_uv(job, uv[0], uv[1], baked_size) for uv in local_uvs]
        n = job["normal"]

        idx0 = len(vertices)
        vertices.extend(quad_verts)
        uvs.extend(quad_uvs)
        normals.extend([n, n, n, n])
        colors.extend([(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)])
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

    # Dynamisch gebackene Chunk-Texturen sind empfindlich gegenüber altem Render-State.
    # Darum wird ein Chunk beim Rebuild neu erstellt statt das Entity in-place zu recyclen.
    # Das verhindert den typischen "nach Bauen/Abbauen schwarz"-Bug.
    if old is not None:
        _safe_clear_destroy(old)

    ent = _make_baked_chunk_entity(mesh, baked_tex, panda_tex)
    ent._baked_chunk_coord = chunk_coord
    _BAKED_CHUNK_TEXTURE_KEEPALIVE[chunk_coord] = (baked_tex, panda_tex)
    combined_terrains[chunk_coord] = ent



def _time_slice_should_yield():
    return LAZY_REBUILD_FRAME_BUDGET > 0.0 and perf_counter() >= _REBUILD_JOB_DEADLINE


def _chunk_rebuild_job_is_stale(chunk_coord, version):
    return chunk_rebuild_versions.get(chunk_coord, 0) != version


def _copy_cell_to_baked_texture_sliced(dst_img, source_img, tile_w, tile_h, job, du, dv):
    """Wie _copy_cell_to_baked_texture(), aber mit Yield-Punkten pro Pixelzeile."""
    info = job["grid"][(job["u"] + du, job["v"] + dv)]
    baked_col, baked_row_bottom = _grid_cell_to_baked_cell(job["face"], job["w"], job["h"], du, dv)

    dst_x0 = job["pack_x"] + baked_col * tile_w
    dst_y0 = job["pack_y"] + (job["h"] - 1 - baked_row_bottom) * tile_h

    for py in range(tile_h):
        local_v = baked_row_bottom + 1.0 - ((py + 0.5) / tile_h)
        for px in range(tile_w):
            local_u = baked_col + ((px + 0.5) / tile_w)
            world_point = _world_point_from_baked_local(
                job["face"], job["slice_idx"], job["u"], job["v"], job["w"], job["h"], local_u, local_v
            )

            raw_u = _axis_coord_for_uv(world_point, info["u_axis"])
            raw_v = _axis_coord_for_uv(world_point, info["v_axis"])
            f_u = _frac(raw_u - info["u_min"])
            f_v = _frac(raw_v - info["v_min"])

            col, alpha = _source_pixel_for_cell(source_img, tile_w, tile_h, info["tile"], f_u, f_v)
            _set_pnm_pixel(dst_img, dst_x0 + px, dst_y0 + py, col, alpha)

        if _time_slice_should_yield():
            yield


def _build_baked_chunk_texture_sliced(quad_jobs, chunk_coord=None, version=0):
    """Time-sliced Version von _build_baked_chunk_texture().

    Der teure Teil ist das pixelweise Kopieren in die gebackene Chunk-Textur.
    Diese Version stoppt nach dem Frame-Budget und macht im nächsten Frame weiter,
    statt nach einer festen Verzögerung alles auf einmal zu bauen.
    """
    global _BAKED_TEXTURE_VERSION
    if not quad_jobs:
        return None, None, None, None, None

    source_img, tile_w, tile_h = _source_atlas_pnm()
    if source_img is None:
        return None, None, None, None, None

    padding = int(_BAKED_TEXTURE_PADDING)
    total_area = 0
    max_rect_w = 1
    for job in quad_jobs:
        if chunk_coord is not None and _chunk_rebuild_job_is_stale(chunk_coord, version):
            return None, None, None, None, None

        job["tile_w"] = tile_w
        job["tile_h"] = tile_h
        job["tex_w"] = max(1, int(job["w"]) * tile_w)
        job["tex_h"] = max(1, int(job["h"]) * tile_h)
        packed_w = job["tex_w"] + padding * 2
        packed_h = job["tex_h"] + padding * 2
        total_area += packed_w * packed_h
        max_rect_w = max(max_rect_w, packed_w)

        if _time_slice_should_yield():
            yield

    target_w = _next_power_of_two(max(max_rect_w, math.sqrt(max(1, total_area))))

    packing_order = sorted(range(len(quad_jobs)), key=lambda i: quad_jobs[i]["tex_h"] * quad_jobs[i]["tex_w"], reverse=True)
    x = 0
    y = 0
    row_h = 0
    for idx in packing_order:
        if chunk_coord is not None and _chunk_rebuild_job_is_stale(chunk_coord, version):
            return None, None, None, None, None

        job = quad_jobs[idx]
        rect_w = job["tex_w"] + padding * 2
        rect_h = job["tex_h"] + padding * 2
        if x > 0 and x + rect_w > target_w:
            x = 0
            y += row_h
            row_h = 0
        job["pack_outer_x"] = x
        job["pack_outer_y"] = y
        job["pack_x"] = x + padding
        job["pack_y"] = y + padding
        x += rect_w
        row_h = max(row_h, rect_h)

        if _time_slice_should_yield():
            yield

    target_h = _next_power_of_two(y + row_h)
    baked_img = _new_pnm_image(target_w, target_h)

    if _time_slice_should_yield():
        yield

    for job in quad_jobs:
        if chunk_coord is not None and _chunk_rebuild_job_is_stale(chunk_coord, version):
            return None, None, None, None, None

        for du in range(job["w"]):
            for dv in range(job["h"]):
                if chunk_coord is not None and _chunk_rebuild_job_is_stale(chunk_coord, version):
                    return None, None, None, None, None

                yield from _copy_cell_to_baked_texture_sliced(baked_img, source_img, tile_w, tile_h, job, du, dv)

                if _time_slice_should_yield():
                    yield

        # Padding ist deutlich günstiger als das eigentliche Pixel-Baking, aber bei
        # großen Quads trotzdem nicht im selben Budget erzwingen.
        if _time_slice_should_yield():
            yield
        _pad_pnm_rect(baked_img, job["pack_x"], job["pack_y"], job["tex_w"], job["tex_h"], padding)
        if _time_slice_should_yield():
            yield

    # Texture-Upload kann nicht sinnvoll in Python auf mehrere Frames geteilt werden.
    # Deshalb starten wir ihn wenigstens am Anfang eines frischen Frame-Budgets.
    if _time_slice_should_yield():
        yield

    if chunk_coord is not None and _chunk_rebuild_job_is_stale(chunk_coord, version):
        return None, None, None, None, None

    _BAKED_TEXTURE_VERSION += 1
    if chunk_coord is None:
        tex_name = f"chunk_baked_mixed_tiles_{_BAKED_TEXTURE_VERSION}"
    else:
        cx, cy, cz = chunk_coord
        tex_name = f"chunk_baked_mixed_tiles_{cx}_{cy}_{cz}_{_BAKED_TEXTURE_VERSION}"
    panda_tex = PandaTexture(tex_name)
    try:
        panda_tex.load(baked_img)
    except:
        print("Cross-type texture baking failed: could not upload baked texture.")
        return None, None, None, None, None

    nearest = None
    clamp_mode = None
    for enum_owner in (PandaSamplerState, PandaTexture):
        if nearest is None:
            for name in ("FT_nearest", "FTNearest"):
                if hasattr(enum_owner, name):
                    nearest = getattr(enum_owner, name)
                    break
        if clamp_mode is None:
            for name in ("WM_clamp", "WMClamp"):
                if hasattr(enum_owner, name):
                    clamp_mode = getattr(enum_owner, name)
                    break

    if nearest is not None:
        for method_name in ("set_minfilter", "setMinfilter"):
            try:
                getattr(panda_tex, method_name)(nearest)
                break
            except:
                pass
        for method_name in ("set_magfilter", "setMagfilter"):
            try:
                getattr(panda_tex, method_name)(nearest)
                break
            except:
                pass

    if clamp_mode is not None:
        for method_name in ("set_wrap_u", "setWrapU"):
            try:
                getattr(panda_tex, method_name)(clamp_mode)
                break
            except:
                pass
        for method_name in ("set_wrap_v", "setWrapV"):
            try:
                getattr(panda_tex, method_name)(clamp_mode)
                break
            except:
                pass

    try:
        baked_tex = UrsinaTexture(panda_tex, filtering=None)
    except:
        baked_tex = _BakedTextureWrapper(panda_tex)

    return baked_tex, panda_tex, tile_w, tile_h, (target_w, target_h)


def _build_chunk_quad_jobs_sliced(chunk_coord, faces_snapshot, version):
    quad_jobs = []

    faces_by_dir = {i: [] for i in range(6)}
    for fk in faces_snapshot:
        if _chunk_rebuild_job_is_stale(chunk_coord, version):
            return None
        faces_by_dir[int(fk[1])].append(fk)
        if _time_slice_should_yield():
            yield

    for d in range(6):
        if _chunk_rebuild_job_is_stale(chunk_coord, version):
            return None
        if not faces_by_dir[d]:
            continue

        slices = {}
        for fk in faces_by_dir[d]:
            if _chunk_rebuild_job_is_stale(chunk_coord, version):
                return None

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
            else:
                slice_idx, u, v = lx, lz, ly

            if slice_idx not in slices:
                slices[slice_idx] = {}
            slices[slice_idx][(u, v)] = _cell_material_info(base, btype, brot, int(d))

            if _time_slice_should_yield():
                yield

        for slice_idx, grid in slices.items():
            if _chunk_rebuild_job_is_stale(chunk_coord, version):
                return None
            if not grid:
                continue

            visited = set()
            keys = grid.keys()
            min_u, max_u = min(k[0] for k in keys), max(k[0] for k in keys)
            min_v, max_v = min(k[1] for k in keys), max(k[1] for k in keys)

            for v in range(min_v, max_v + 1):
                for u in range(min_u, max_u + 1):
                    if _chunk_rebuild_job_is_stale(chunk_coord, version):
                        return None
                    if (u, v) in visited or (u, v) not in grid:
                        continue

                    w = 1
                    while (u + w) <= max_u and (u + w, v) not in visited and (u + w, v) in grid:
                        w += 1

                    h = 1
                    can_expand = True
                    while (v + h) <= max_v and can_expand:
                        for du in range(w):
                            if (u + du, v + h) in visited or (u + du, v + h) not in grid:
                                can_expand = False
                                break
                        if can_expand:
                            h += 1

                    for du in range(w):
                        for dv in range(h):
                            visited.add((u + du, v + dv))

                    if d in (0, 1):
                        bx = float(u)
                        by = float(slice_idx) * BLOCK_HEIGHT
                        bz = float(v)
                        W_ext, H_ext, D_ext = w, BLOCK_HEIGHT, h
                    elif d in (2, 3):
                        bx = float(u)
                        by = float(v) * BLOCK_HEIGHT
                        bz = float(slice_idx)
                        W_ext, H_ext, D_ext = w, h * BLOCK_HEIGHT, 1.0
                    else:
                        bx = float(slice_idx)
                        by = float(v) * BLOCK_HEIGHT
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

                    quad_jobs.append({
                        "face": int(d),
                        "slice_idx": int(slice_idx),
                        "u": int(u),
                        "v": int(v),
                        "w": int(w),
                        "h": int(h),
                        "grid": grid,
                        "quad_verts": quad_verts,
                        "W_ext": W_ext,
                        "H_ext": H_ext,
                        "D_ext": D_ext,
                        "normal": _FACE_NORMALS_TUPLES.get(d, (0, 1, 0)),
                    })

                    if _time_slice_should_yield():
                        yield

    return quad_jobs


def _rebuild_chunk_mesh_sliced(chunk_coord, version):
    """Generator-Version von _rebuild_chunk_mesh().

    Diese Funktion erledigt Greedy-Suche, Texture-Baking und Mesh-Listenaufbau in
    kleinen Stücken. Nur der finale Texture-Upload/Mesh-Konstruktor bleibt ein kurzer
    atomarer Schritt, weil Panda3D/Ursina das nicht zeilenweise übernehmen können.
    """
    chunk_coord = _ensure_chunk(chunk_coord)
    old = combined_terrains.get(chunk_coord)

    faces_snapshot = list(chunk_face_sets.get(chunk_coord, ()))
    if not faces_snapshot:
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return {"finished": True}

    if _time_slice_should_yield():
        yield

    quad_jobs = yield from _build_chunk_quad_jobs_sliced(chunk_coord, faces_snapshot, version)
    if quad_jobs is None or _chunk_rebuild_job_is_stale(chunk_coord, version):
        return {"cancelled": True}

    if not quad_jobs:
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return {"finished": True}

    baked_tex, panda_tex, tile_w, tile_h, baked_size = yield from _build_baked_chunk_texture_sliced(
        quad_jobs, chunk_coord=chunk_coord, version=version
    )
    if _chunk_rebuild_job_is_stale(chunk_coord, version):
        return {"cancelled": True}

    if baked_tex is None or panda_tex is None:
        print("Chunk mesh skipped: baked texture was unavailable.")
        _BAKED_CHUNK_TEXTURE_KEEPALIVE.pop(chunk_coord, None)
        _safe_clear_destroy(old)
        combined_terrains[chunk_coord] = None
        return {"finished": True, "bake_failed": True}

    vertices = []
    triangles = []
    uvs = []
    normals = []
    colors = []

    for job in quad_jobs:
        if _chunk_rebuild_job_is_stale(chunk_coord, version):
            return {"cancelled": True}

        quad_verts = job["quad_verts"]
        local_uvs = _fast_uvs(job["face"], job["W_ext"], job["H_ext"], job["D_ext"])
        quad_uvs = [_baked_uv(job, uv[0], uv[1], baked_size) for uv in local_uvs]
        n = job["normal"]

        idx0 = len(vertices)
        vertices.extend(quad_verts)
        uvs.extend(quad_uvs)
        normals.extend([n, n, n, n])
        colors.extend([(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)])
        triangles.extend([idx0, idx0 + 2, idx0 + 1, idx0, idx0 + 3, idx0 + 2])

        if _time_slice_should_yield():
            yield

    # Mesh-Konstruktion/Entity-Swap am Anfang eines neuen Budgets starten.
    if _time_slice_should_yield():
        yield

    if _chunk_rebuild_job_is_stale(chunk_coord, version):
        return {"cancelled": True}

    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        uvs=uvs,
        normals=normals,
        colors=colors,
        mode="triangle",
        static=True,
    )

    if old is not None:
        _safe_clear_destroy(old)

    ent = _make_baked_chunk_entity(mesh, baked_tex, panda_tex)
    ent._baked_chunk_coord = chunk_coord
    _BAKED_CHUNK_TEXTURE_KEEPALIVE[chunk_coord] = (baked_tex, panda_tex)
    combined_terrains[chunk_coord] = ent
    return {"finished": True}

def _clear_dirty_chunk_preview(chunk_coord):
    chunk_coord = _ensure_chunk(chunk_coord)
    old_preview = dirty_chunk_previews.pop(chunk_coord, None)
    _safe_clear_destroy(old_preview)


def _hide_final_chunk_for_preview(chunk_coord):
    ent = combined_terrains.get(chunk_coord)
    if ent is None:
        return
    try:
        ent.enabled = False
    except:
        pass


def _make_dirty_chunk_preview(chunk_coord):
    """Billiges Sofort-Mesh für einen dirty Chunk.

    Dieses Mesh ist NICHT greedy und backt keine Textur. Es baut nur die aktuell
    sichtbaren Faces einzeln mit dem vorhandenen Atlas-Shader. Dadurch sieht der
    Chunk direkt nach Bauen/Abbauen korrekt aus, während der echte Greedy-Rebuild
    später im Hintergrund/Queue nachzieht.
    """
    if not DIRTY_PREVIEW_ENABLED:
        return

    chunk_coord = _ensure_chunk(chunk_coord)

    # Altes Preview ersetzen.
    old_preview = dirty_chunk_previews.pop(chunk_coord, None)
    _safe_clear_destroy(old_preview)

    # Den alten Greedy-Chunk ausblenden, sonst würden entfernte Blöcke/Faces
    # bis zum Rebuild noch im alten gebackenen Mesh sichtbar bleiben.
    _hide_final_chunk_for_preview(chunk_coord)

    faces = list(chunk_face_sets.get(chunk_coord, ()))
    if not faces:
        return

    vertices = []
    triangles = []
    uvs = []
    normals = []
    colors = []

    for fk in faces:
        pos_key, fidx = fk
        face_idx = int(fidx)
        base = _cube_base_from_face(pos_key, face_idx)
        btype = _block_type_from_face_key(fk)
        brot = _block_rotation_from_base(base, btype)

        quad_verts = _single_block_face_verts(base, face_idx)
        quad_uvs = _rotated_uvs(face_idx, brot, quad_verts)
        tile = _block_tile_for_world_face(btype, brot, face_idx)
        tile_rect = _atlas_rect(tile[0], tile[1])
        n = _FACE_NORMALS_TUPLES.get(face_idx, (0, 1, 0))

        idx0 = len(vertices)
        vertices.extend(quad_verts)
        uvs.extend(quad_uvs)
        normals.extend([n, n, n, n])
        colors.extend([tile_rect, tile_rect, tile_rect, tile_rect])
        triangles.extend([idx0, idx0 + 2, idx0 + 1, idx0, idx0 + 3, idx0 + 2])

    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
        uvs=uvs,
        normals=normals,
        colors=colors,
        mode="triangle",
        static=False,
    )

    ent = Entity(model=mesh)
    try:
        ent.texture = atlas_texture
    except:
        pass
    try:
        ent.shader = atlas_repeat_shader
    except:
        pass
    ent.collider = None
    ent.enabled = True
    dirty_chunk_previews[chunk_coord] = ent


def _queue_lazy_chunk_rebuild(chunk_coord, settle_delay=LAZY_REBUILD_SETTLE_DELAY, make_preview=True):
    """Chunk dirty markieren und einen time-sliced Greedy-Rebuild planen.

    settle_delay ist nur die kurze Sammelzeit für mehrere schnelle Änderungen.
    Die eigentliche Mesh-Arbeit passiert später über LAZY_REBUILD_FRAME_BUDGET.
    """
    if chunk_coord is None:
        return

    chunk_coord = _ensure_chunk(chunk_coord)

    if make_preview:
        _make_dirty_chunk_preview(chunk_coord)

    chunk_rebuild_versions[chunk_coord] = chunk_rebuild_versions.get(chunk_coord, 0) + 1
    chunk_update_due[chunk_coord] = perf_counter() + float(settle_delay)

    if chunk_coord not in chunk_update_set:
        chunk_update_queue.append(chunk_coord)
        chunk_update_set.add(chunk_coord)


def _refresh_chunks(affected_chunks, settle_delay=LAZY_REBUILD_SETTLE_DELAY, make_preview=True):
    """Markiert Chunks dirty, ohne sofort das teure Greedy-Mesh zu bauen."""
    for chunk_coord in affected_chunks:
        _queue_lazy_chunk_rebuild(chunk_coord, settle_delay=settle_delay, make_preview=make_preview)


def _finish_lazy_chunk_rebuild_result(chunk_coord, result):
    # Wenn der Greedy-Rebuild erfolgreich war oder der Chunk leer ist, kann das
    # Preview weg. Wenn Baking fehlschlägt und noch Faces existieren, bleibt das
    # Preview als sichtbarer Fallback erhalten.
    if result is None:
        result = {}
    if result.get("cancelled"):
        return

    has_faces = bool(chunk_face_sets.get(chunk_coord))
    final_ent = combined_terrains.get(chunk_coord)

    if final_ent is not None:
        try:
            final_ent.enabled = True
        except:
            pass

    if final_ent is not None or not has_faces:
        _clear_dirty_chunk_preview(chunk_coord)


def _pop_due_chunk_for_rebuild(now):
    checks_left = len(chunk_update_queue)
    while chunk_update_queue and checks_left > 0:
        chunk_coord = chunk_update_queue.pop(0)
        chunk_update_set.discard(chunk_coord)
        checks_left -= 1

        due = chunk_update_due.get(chunk_coord, 0.0)
        if now < due:
            if chunk_coord not in chunk_update_set:
                chunk_update_queue.append(chunk_coord)
                chunk_update_set.add(chunk_coord)
            continue

        chunk_update_due.pop(chunk_coord, None)
        return chunk_coord

    return None


def _process_lazy_chunk_rebuilds():
    """Verteilt den teuren Greedy-/Bake-Rebuild auf mehrere Frames."""
    global active_chunk_rebuild_job, _REBUILD_JOB_DEADLINE

    frame_start = perf_counter()
    frame_budget = max(0.0005, float(LAZY_REBUILD_FRAME_BUDGET))
    _REBUILD_JOB_DEADLINE = frame_start + frame_budget

    while perf_counter() < _REBUILD_JOB_DEADLINE:
        if active_chunk_rebuild_job is not None:
            coord = active_chunk_rebuild_job["coord"]
            version = active_chunk_rebuild_job["version"]
            if _chunk_rebuild_job_is_stale(coord, version):
                active_chunk_rebuild_job = None
                continue

        if active_chunk_rebuild_job is None:
            chunk_coord = _pop_due_chunk_for_rebuild(perf_counter())
            if chunk_coord is None:
                break

            version = chunk_rebuild_versions.get(chunk_coord, 0)
            active_chunk_rebuild_job = {
                "coord": chunk_coord,
                "version": version,
                "gen": _rebuild_chunk_mesh_sliced(chunk_coord, version),
            }

        job = active_chunk_rebuild_job
        try:
            next(job["gen"])
        except StopIteration as stop:
            coord = job["coord"]
            version = job["version"]
            result = stop.value if stop.value is not None else {}
            active_chunk_rebuild_job = None

            if result.get("cancelled") or _chunk_rebuild_job_is_stale(coord, version):
                # Eine neuere Änderung hat diesen Job überholt. Der neuere Job wurde
                # beim Dirty-Markieren bereits wieder in die Queue gelegt.
                continue

            _finish_lazy_chunk_rebuild_result(coord, result)
            continue

        # Der Generator hat wegen Budget-Ende freiwillig pausiert.
        if perf_counter() >= _REBUILD_JOB_DEADLINE:
            break


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

    _refresh_chunks(affected)
    c.y = -9999


def _frame_position_for_target(face_pos, face_idx):
    hit_base = _cube_base_from_face(face_pos, face_idx)
    return Vec3(hit_base[0], hit_base[1] + 1.5, hit_base[2])


def update():
    _process_lazy_chunk_rebuilds()


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
