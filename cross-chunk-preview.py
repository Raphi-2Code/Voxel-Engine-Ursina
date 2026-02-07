from ursina import *
from panda3d.core import LVecBase3f
from ursina.prefabs.first_person_controller import *
from perlin_noise import *
from itertools import *
import math
import numpy as np
from bisect import bisect_left, bisect_right

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

GRID_X = 4
GRID_Z = 4
chunk_keys = [(cx, cz) for cx in range(GRID_X) for cz in range(GRID_Z)]
chunk_index = {k: i for i, k in enumerate(chunk_keys)}


all_chunks = [[[], [], []] for _ in chunk_keys]
chunk_face_sets = [set() for _ in chunk_keys]
combined_terrains = [None for _ in chunk_keys]


world_faces = set()                            # union of all chunk_face_sets
face_to_chunk = {}                             # ((x,y,z), face_idx) -> chunk_idx

# Block-derived acceleration structure for collider-free FPS gravity.
# We infer each visible block from any visible face, then index its top y.
# top_columns[(x,z)] -> sorted [top_y1, top_y2, ...]
# top_cells[(floor(x), floor(z))] -> {(x,z), ...} for fast local lookup.
top_columns = {}
top_cells = {}
block_face_counts = {}                         # (base_x, base_y, base_z) -> visible face count


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

# Collider-free gravity settings.
GRAVITY_ACCEL = 35.0
MAX_FALL_SPEED = 55.0
JUMP_SPEED = 11.5
# In Ursina's FirstPersonController, player.y is already at foot/ground level.
PLAYER_STAND_HEIGHT = 0.0
GROUND_STICK = 0.08
MAX_STEP_UP = 0.35
PLAYER_COLLISION_RADIUS = 0.33
PLAYER_FOOT_RADIUS = PLAYER_COLLISION_RADIUS
PLAYER_COLLISION_FOOT_CLEARANCE = 0.02
PLAYER_COLLISION_HEAD_CLEARANCE = 0.1
BLOCK_HALF_EXTENT = 0.5
BLOCK_HEIGHT = float(_FACE_OFFSETS[1].y - _FACE_OFFSETS[0].y)
WALL_EPS = 0.001
MAX_PHYSICS_SUBSTEP = 1.0 / 120.0
MAX_PHYSICS_STEPS = 8

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
    return (
        cube_faces[face_idx][3],
        cube_faces[face_idx][4],
        cube_faces[face_idx][5],
    )


def _chunk_coord_from_pos(pos):
    cx = math.floor(float(pos[0]) / chunk_size)
    cz = math.floor(float(pos[2]) / chunk_size)
    return (cx, cz)


def _chunk_index_from_pos(pos):
    return chunk_index.get(_chunk_coord_from_pos(pos))


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


def _cube_base_from_face(pos_key, face_idx):
    off = _FACE_OFFSETS[int(face_idx)]
    return _vkey((pos_key[0] - off.x, pos_key[1] - off.y, pos_key[2] - off.z))


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
    # Safety net: scan inferred blocks directly if fast cell index misses.
    # This is still math-based and avoids colliders/raycast collision checks.
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
    min_cx = math.floor(min_x)
    max_cx = math.floor(max_x)
    min_cz = math.floor(min_z)
    max_cz = math.floor(max_z)

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
            if start_x <= boundary and target_x > boundary and boundary < limit:
                limit = boundary - WALL_EPS
        return limit

    limit = target_x
    for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
        if max_z <= bz0 or min_z >= bz1:
            continue
        boundary = bx1 + radius
        if start_x >= boundary and target_x < boundary and boundary > limit:
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
            if start_z <= boundary and target_z > boundary and boundary < limit:
                limit = boundary - WALL_EPS
        return limit

    limit = target_z
    for bx0, bx1, bz0, bz1 in _iter_solid_blocks(min_x, max_x, min_z, max_z, y_min, y_max):
        if max_x <= bx0 or min_x >= bx1:
            continue
        boundary = bz1 + radius
        if start_z >= boundary and target_z < boundary and boundary > limit:
            limit = boundary + WALL_EPS
    return limit


def _resolve_horizontal_penetration(px, pz, y_min, y_max):
    radius = PLAYER_COLLISION_RADIUS

    for _ in range(4):
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


def _apply_vector_horizontal_collisions():
    global prev_horizontal_x, prev_horizontal_z

    cur_x = float(player.x)
    cur_z = float(player.z)

    if prev_horizontal_x is None or prev_horizontal_z is None:
        prev_horizontal_x = cur_x
        prev_horizontal_z = cur_z
        return

    y_min, y_max = _player_body_y_span()
    res_x = _sweep_x(prev_horizontal_x, cur_x, prev_horizontal_z, y_min, y_max)
    res_z = _sweep_z(prev_horizontal_z, cur_z, res_x, y_min, y_max)
    res_x, res_z = _resolve_horizontal_penetration(res_x, res_z, y_min, y_max)

    player.x = res_x
    player.z = res_z
    prev_horizontal_x = float(player.x)
    prev_horizontal_z = float(player.z)


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

        # While falling, scan farther up so we still catch top faces even if one frame
        # already moved the feet below the surface.
        support_scan_up = MAX_STEP_UP
        if vertical_velocity < 0:
            support_scan_up = MAX_STEP_UP + max(BLOCK_HEIGHT, (-vertical_velocity * dt) + 0.05)

        support_y = _find_support_y(px, pz, current_foot, support_scan_up)
        if support_y is None and len(block_face_counts) > 0:
            support_y = _find_support_y_fallback(px, pz, current_foot, support_scan_up)

        if support_y is not None and current_foot < support_y:
            player.y = support_y + PLAYER_STAND_HEIGHT
            vertical_velocity = 0.0
            is_grounded = True
            continue

        if support_y is not None and vertical_velocity <= 0:
            d = current_foot - support_y
            if 0 <= d <= GROUND_STICK:
                player.y = support_y + PLAYER_STAND_HEIGHT
                vertical_velocity = 0.0
                is_grounded = True
                continue

        vertical_velocity = max(vertical_velocity - GRAVITY_ACCEL * dt, -MAX_FALL_SPEED)
        next_y = float(player.y) + vertical_velocity * dt
        next_foot = next_y - PLAYER_STAND_HEIGHT

        if support_y is not None and vertical_velocity <= 0 and next_foot <= support_y:
            player.y = support_y + PLAYER_STAND_HEIGHT
            vertical_velocity = 0.0
            is_grounded = True
        else:
            player.y = next_y
            is_grounded = False


def _remove_face(face_key, affected):
    if face_key not in world_faces:
        return False
    chunk_idx = face_to_chunk.get(face_key)
    if chunk_idx is None:
        return False

    world_faces.discard(face_key)
    face_to_chunk.pop(face_key, None)
    chunk_face_sets[chunk_idx].discard(face_key)
    _unregister_top_face(face_key[0], face_key[1])
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
    _register_top_face(face_key[0], face_key[1])
    affected.add(chunk_idx)
    return True


def load_chunks():
    world_faces.clear()
    face_to_chunk.clear()
    top_columns.clear()
    top_cells.clear()
    block_face_counts.clear()
    for i in range(len(chunk_face_sets)):
        chunk_face_sets[i].clear()

    chunks_opened_ = list(eval(open("chunks.txt", "r").read()))

    for chunk_idx, chunk_data in enumerate(chunks_opened_):
        if chunk_idx >= len(chunk_keys):
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
            _register_top_face(key[0], key[1])

    for i in range(len(chunk_keys)):
        _sync_chunk_lists(i)
        _rebuild_chunk_mesh(i)

    print(
        f"[gravity] faces={len(world_faces)} blocks={len(block_face_counts)} columns={len(top_columns)}"
    )
    if len(world_faces) > 0 and len(block_face_counts) == 0:
        # Last-resort rebuild in case an earlier load path skipped support indexing.
        for face_key in world_faces:
            _register_top_face(face_key[0], face_key[1])
        print(
            f"[gravity] rebuilt blocks={len(block_face_counts)} columns={len(top_columns)}"
        )


load_chunks()

try:
    # simple spawn near first top face
    if len(top_columns) > 0:
        col = next(iter(top_columns.keys()))
        y = top_columns[col][-1]
        player.position = Vec3(col[0], y + PLAYER_STAND_HEIGHT, col[1])
    else:
        first_face = next(iter(world_faces))
        player.position = Vec3(first_face[0][0], first_face[0][1] + 2, first_face[0][2])
except:
    player.position = Vec3(0, 6, 0)

prev_horizontal_x = float(player.x)
prev_horizontal_z = float(player.z)


def get_target_face(max_distance: int = 12):
    origin = camera.world_position
    direction = camera.forward

    for i in range(int(max_distance * 2)):
        step = i * 0.5
        point = origin + direction * step

        chunk_key = _chunk_coord_from_pos(point)
        chunk_idx = chunk_index.get(chunk_key)
        if chunk_idx is None:
            continue
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
    if _block_intersects_player(cube_base):
        c.y = -9999
        return

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
    _apply_vector_horizontal_collisions()
    _apply_vector_gravity()

    face_pos, _, _ = get_target_face()
    if face_pos:
        c2.position = Vec3(round(face_pos[0]), round(face_pos[1]), round(face_pos[2])) + (0, -0.5, 0)
    else:
        c2.position = floor(player.position + player.forward * 4)


def input(key):
    global mode, vertical_velocity, is_grounded

    if key == "o":
        mode = 1 - mode
    if key == "m":
        player.y += 1
        vertical_velocity = 0.0
    if key == "l":
        player.y -= 1
        vertical_velocity = 0.0
    if key == "space" and is_grounded:
        vertical_velocity = JUMP_SPEED
        is_grounded = False
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
