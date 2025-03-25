import math
import numpy as np
import numba

@numba.njit(parallel=True)
def update_positions(pos, prev, radii, n_balls, dt, dt2, width, height, gravity, settle):
    """
    Update positions using Verlet integration.
    If settle is True, apply a damping factor and reduced effective gravity.
    """
    if settle:
        v_damp = 0.95
        g_eff = gravity * 0.2
    else:
        v_damp = 1.0
        g_eff = gravity
    for i in numba.prange(n_balls):
        x = pos[i, 0]
        y = pos[i, 1]
        prev_x = prev[i, 0]
        prev_y = prev[i, 1]
        new_x = x + (x - prev_x) * v_damp
        new_y = y + (y - prev_y) * v_damp + g_eff * dt2
        prev[i, 0] = x
        prev[i, 1] = y
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        r = radii[i]
        if pos[i, 0] < r:
            pos[i, 0] = r
            prev[i, 0] = pos[i, 0] + (pos[i, 0] - prev[i, 0]) * -0.8
        if pos[i, 0] > width - r:
            pos[i, 0] = width - r
            prev[i, 0] = pos[i, 0] + (pos[i, 0] - prev[i, 0]) * -0.8
        if pos[i, 1] < r:
            pos[i, 1] = r
            prev[i, 1] = pos[i, 1] + (pos[i, 1] - prev[i, 1]) * -0.8
        if pos[i, 1] > height - r:
            pos[i, 1] = height - r
            prev[i, 1] = pos[i, 1] + (pos[i, 1] - prev[i, 1]) * -0.8

@numba.njit
def collision_detection(pos, radii, n_balls, cell_size, cells_x, cells_y):
    """
    Grid–based collision detection and response.
    """
    total_cells = cells_x * cells_y
    cell_ids = np.empty(n_balls, dtype=np.int32)
    for i in range(n_balls):
        cx = int(pos[i, 0] // cell_size)
        cy = int(pos[i, 1] // cell_size)
        if cx < 0:
            cx = 0
        elif cx >= cells_x:
            cx = cells_x - 1
        if cy < 0:
            cy = 0
        elif cy >= cells_y:
            cy = cells_y - 1
        cell_ids[i] = cx + cy * cells_x

    sorted_indices = np.argsort(cell_ids)
    cell_start = -np.ones(total_cells, dtype=np.int32)
    cell_end = -np.ones(total_cells, dtype=np.int32)

    if n_balls > 0:
        current_cell = cell_ids[sorted_indices[0]]
        cell_start[current_cell] = 0
        for k in range(n_balls):
            cell_val = cell_ids[sorted_indices[k]]
            if cell_val != current_cell:
                cell_end[current_cell] = k
                current_cell = cell_val
                cell_start[current_cell] = k
        cell_end[current_cell] = n_balls

    factor = 0.3
    # Process collisions within the same cell.
    for cell in range(total_cells):
        if cell_start[cell] == -1:
            continue
        cx = cell % cells_x
        cy = cell // cells_x
        start_i = cell_start[cell]
        end_i = cell_end[cell]
        for a in range(start_i, end_i):
            i = sorted_indices[a]
            for b in range(a + 1, end_i):
                j = sorted_indices[b]
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist > 0.0:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        shift = overlap * factor
                        pos[i, 0] -= nx * shift
                        pos[i, 1] -= ny * shift
                        pos[j, 0] += nx * shift
                        pos[j, 1] += ny * shift
                    else:
                        shift = min_dist * factor
                        pos[i, 0] -= shift
                        pos[j, 0] += shift
        # Process neighbor-cell collisions.
        for off_x, off_y in ((1, -1), (1, 0), (1, 1), (0, 1)):
            ncx = cx + off_x
            ncy = cy + off_y
            if ncx < 0 or ncx >= cells_x or ncy < 0 or ncy >= cells_y:
                continue
            neighbor_cell = ncx + ncy * cells_x
            if cell_start[neighbor_cell] == -1:
                continue
            start_j = cell_start[neighbor_cell]
            end_j = cell_end[neighbor_cell]
            for a in range(start_i, end_i):
                i = sorted_indices[a]
                for b in range(start_j, end_j):
                    j = sorted_indices[b]
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        if dist > 0.0:
                            overlap = min_dist - dist
                            nx = dx / dist
                            ny = dy / dist
                            shift = overlap * factor
                            pos[i, 0] -= nx * shift
                            pos[i, 1] -= ny * shift
                            pos[j, 0] += nx * shift
                            pos[j, 1] += ny * shift
                        else:
                            shift = min_dist * factor
                            pos[i, 0] -= shift
                            pos[j, 0] += shift
