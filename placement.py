"""
VLSI Cell Placement Optimization Challenge
==========================================

CHALLENGE OVERVIEW:
You are tasked with implementing a critical component of a chip placement optimizer.
Given a set of cells (circuit components) with fixed sizes and connectivity requirements,
you need to find positions for these cells that:
1. Minimize total wirelength (wiring cost between connected pins)
2. Eliminate all overlaps between cells

YOUR TASK:
Implement the `overlap_repulsion_loss()` function to prevent cells from overlapping.
The function must:
- Be differentiable (uses PyTorch operations for gradient descent)
- Detect when cells overlap in 2D space
- Apply increasing penalties for larger overlaps
- Work efficiently with vectorized operations

SUCCESS CRITERIA:
After running the optimizer with your implementation:
- overlap_count should be 0 (no overlapping cell pairs)
- total_overlap_area should be 0.0 (no overlap)
- wirelength should be minimized
- Visualization should show clean, non-overlapping placement

GETTING STARTED:
1. Read through the existing code to understand the data structures
2. Look at wirelength_attraction_loss() as a reference implementation
3. Implement overlap_repulsion_loss() following the TODO instructions
4. Run main() and check the overlap metrics in the output
5. Tune hyperparameters (lambda_overlap, lambda_wirelength) if needed
6. Generate visualization to verify your solution

BONUS CHALLENGES:
- Improve convergence speed by tuning learning rate or adding momentum
- Implement better initial placement strategy
- Add visualization of optimization progress over time
"""

import os
from enum import IntEnum

import torch
import torch.optim as optim

import math
import torch.nn.functional as F
import numpy as np


# Feature index enums for cleaner code access
class CellFeatureIdx(IntEnum):
    """Indices for cell feature tensor columns."""
    AREA = 0
    NUM_PINS = 1
    X = 2
    Y = 3
    WIDTH = 4
    HEIGHT = 5


class PinFeatureIdx(IntEnum):
    """Indices for pin feature tensor columns."""
    CELL_IDX = 0
    PIN_X = 1  # Relative to cell corner
    PIN_Y = 2  # Relative to cell corner
    X = 3  # Absolute position
    Y = 4  # Absolute position
    WIDTH = 5
    HEIGHT = 6


# Configuration constants
# Macro parameters
MIN_MACRO_AREA = 100.0
MAX_MACRO_AREA = 10000.0

# Standard cell parameters (areas can be 1, 2, or 3)
STANDARD_CELL_AREAS = [1.0, 2.0, 3.0]
STANDARD_CELL_HEIGHT = 1.0

# Pin count parameters
MIN_STANDARD_CELL_PINS = 3
MAX_STANDARD_CELL_PINS = 6

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======= SETUP =======

def generate_placement_input(num_macros, num_std_cells):
    """Generate synthetic placement input data.

    Args:
        num_macros: Number of macros to generate
        num_std_cells: Number of standard cells to generate

    Returns:
        Tuple of (cell_features, pin_features, edge_list):
            - cell_features: torch.Tensor of shape [N, 6] with columns [area, num_pins, x, y, width, height]
            - pin_features: torch.Tensor of shape [total_pins, 7] with columns
              [cell_instance_index, pin_x, pin_y, x, y, pin_width, pin_height]
            - edge_list: torch.Tensor of shape [E, 2] with [src_pin_idx, tgt_pin_idx]
    """
    total_cells = num_macros + num_std_cells

    # Step 1: Generate macro areas (uniformly distributed between min and max)
    macro_areas = (
        torch.rand(num_macros) * (MAX_MACRO_AREA - MIN_MACRO_AREA) + MIN_MACRO_AREA
    )

    # Step 2: Generate standard cell areas (randomly pick from 1, 2, or 3)
    std_cell_areas = torch.tensor(STANDARD_CELL_AREAS)[
        torch.randint(0, len(STANDARD_CELL_AREAS), (num_std_cells,))
    ]

    # Combine all areas
    areas = torch.cat([macro_areas, std_cell_areas])

    # Step 3: Calculate cell dimensions
    # Macros are square
    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)

    # Standard cells have fixed height = 1, width = area
    std_cell_widths = std_cell_areas / STANDARD_CELL_HEIGHT
    std_cell_heights = torch.full((num_std_cells,), STANDARD_CELL_HEIGHT)

    # Combine dimensions
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    # Step 4: Calculate number of pins per cell
    num_pins_per_cell = torch.zeros(total_cells, dtype=torch.int)

    # Macros: between sqrt(area) and 2*sqrt(area) pins
    for i in range(num_macros):
        sqrt_area = int(torch.sqrt(macro_areas[i]).item())
        num_pins_per_cell[i] = torch.randint(sqrt_area, 2 * sqrt_area + 1, (1,)).item()

    # Standard cells: between 3 and 6 pins
    num_pins_per_cell[num_macros:] = torch.randint(
        MIN_STANDARD_CELL_PINS, MAX_STANDARD_CELL_PINS + 1, (num_std_cells,)
    )

    # Step 5: Create cell features tensor [area, num_pins, x, y, width, height]
    cell_features = torch.zeros(total_cells, 6)
    cell_features[:, CellFeatureIdx.AREA] = areas
    cell_features[:, CellFeatureIdx.NUM_PINS] = num_pins_per_cell.float()
    cell_features[:, CellFeatureIdx.X] = 0.0  # x position (initialized to 0)
    cell_features[:, CellFeatureIdx.Y] = 0.0  # y position (initialized to 0)
    cell_features[:, CellFeatureIdx.WIDTH] = cell_widths
    cell_features[:, CellFeatureIdx.HEIGHT] = cell_heights

    # Step 6: Generate pins for each cell
    total_pins = num_pins_per_cell.sum().item()
    pin_features = torch.zeros(total_pins, 7)

    # Fixed pin size for all pins (square pins)
    PIN_SIZE = 0.1  # All pins are 0.1 x 0.1

    pin_idx = 0
    for cell_idx in range(total_cells):
        n_pins = num_pins_per_cell[cell_idx].item()
        cell_width = cell_widths[cell_idx].item()
        cell_height = cell_heights[cell_idx].item()

        # Generate random pin positions within the cell
        # Offset from edges to ensure pins are fully inside
        margin = PIN_SIZE / 2
        if cell_width > 2 * margin and cell_height > 2 * margin:
            pin_x = torch.rand(n_pins) * (cell_width - 2 * margin) + margin
            pin_y = torch.rand(n_pins) * (cell_height - 2 * margin) + margin
        else:
            # For very small cells, just center the pins
            pin_x = torch.full((n_pins,), cell_width / 2)
            pin_y = torch.full((n_pins,), cell_height / 2)

        # Fill pin features
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.CELL_IDX] = cell_idx
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_X] = (
            pin_x  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.PIN_Y] = (
            pin_y  # relative to cell
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.X] = (
            pin_x  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.Y] = (
            pin_y  # absolute (same as relative initially)
        )
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.WIDTH] = PIN_SIZE
        pin_features[pin_idx : pin_idx + n_pins, PinFeatureIdx.HEIGHT] = PIN_SIZE

        pin_idx += n_pins

    # Step 7: Generate edges with simple random connectivity
    # Each pin connects to 1-3 random pins (preferring different cells)
    edge_list = []
    avg_edges_per_pin = 2.0

    pin_to_cell = torch.zeros(total_pins, dtype=torch.long)
    pin_idx = 0
    for cell_idx, n_pins in enumerate(num_pins_per_cell):
        pin_to_cell[pin_idx : pin_idx + n_pins] = cell_idx
        pin_idx += n_pins

    # Create adjacency set to avoid duplicate edges
    adjacency = [set() for _ in range(total_pins)]

    for pin_idx in range(total_pins):
        pin_cell = pin_to_cell[pin_idx].item()
        num_connections = torch.randint(1, 4, (1,)).item()  # 1-3 connections per pin

        # Try to connect to pins from different cells
        for _ in range(num_connections):
            # Random candidate
            other_pin = torch.randint(0, total_pins, (1,)).item()

            # Skip self-connections and existing connections
            if other_pin == pin_idx or other_pin in adjacency[pin_idx]:
                continue

            # Add edge (always store smaller index first for consistency)
            if pin_idx < other_pin:
                edge_list.append([pin_idx, other_pin])
            else:
                edge_list.append([other_pin, pin_idx])

            # Update adjacency
            adjacency[pin_idx].add(other_pin)
            adjacency[other_pin].add(pin_idx)

    # Convert to tensor and remove duplicates
    if edge_list:
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_list = torch.unique(edge_list, dim=0)
    else:
        edge_list = torch.zeros((0, 2), dtype=torch.long)

    print(f"\nGenerated placement data:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total pins: {total_pins}")
    print(f"  Total edges: {len(edge_list)}")
    print(f"  Average edges per pin: {2 * len(edge_list) / total_pins:.2f}")

    return cell_features, pin_features, edge_list

# ======= OPTIMIZATION CODE (edit this part) =======

def initialize_smart_placement(cell_features, pin_features, edge_list):
    """Initializes placement by solving a connectivity-only attraction phase."""
    # Start all at center with tiny noise
    N = cell_features.shape[0]
    pos = torch.randn((N, 2)) * 0.1
    pos.requires_grad_(True)

    # Quick 50-step optimization for pure wirelength to 'cluster' components
    opt = torch.optim.Adam([pos], lr=2.0)
    for _ in range(50):
        opt.zero_grad()
        feat = cell_features.clone()
        feat[:, CellFeatureIdx.X:CellFeatureIdx.WIDTH] = pos
        loss = wirelength_attraction_loss(feat, pin_features, edge_list)
        loss.backward()
        opt.step()

    return pos.detach().requires_grad_(True)


def wirelength_attraction_loss(cell_features, pin_features, edge_list):
    """Calculate loss based on total wirelength to minimize routing.

    This is a REFERENCE IMPLEMENTATION showing how to write a differentiable loss function.

    The loss computes the Manhattan distance between connected pins and minimizes
    the total wirelength across all edges.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        pin_features: [P, 7] tensor with pin information
        edge_list: [E, 2] tensor with edges

    Returns:
        Scalar loss value
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Update absolute pin positions based on cell positions
    cell_positions = cell_features[:, CellFeatureIdx.X:CellFeatureIdx.WIDTH]  # [N, 2] X_position and Y_position of all cells
    cell_indices = pin_features[:, PinFeatureIdx.CELL_IDX].long()

    # Calculate absolute pin positions
    pin_absolute_x = cell_positions[cell_indices, 0] + pin_features[:, PinFeatureIdx.PIN_X]
    pin_absolute_y = cell_positions[cell_indices, 1] + pin_features[:, PinFeatureIdx.PIN_Y]

    # Get source and target pin positions for each edge
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()

    src_x = pin_absolute_x[src_pins]
    src_y = pin_absolute_y[src_pins]
    tgt_x = pin_absolute_x[tgt_pins]
    tgt_y = pin_absolute_y[tgt_pins]

    # Calculate smooth approximation of Manhattan distance
    # Using log-sum-exp approximation for differentiability
    alpha = 0.1  # Smoothing parameter
    dx = torch.abs(src_x - tgt_x)
    dy = torch.abs(src_y - tgt_y)

    # Smooth L1 distance with numerical stability
    smooth_manhattan = alpha * torch.logsumexp(
        torch.stack([dx / alpha, dy / alpha], dim=0), dim=0
    )

    # Total wirelength
    total_wirelength = torch.sum(smooth_manhattan)

    return total_wirelength / edge_list.shape[0]  # Normalize by number of edges


# Vectorized Pairwise for Small N
def overlap_repulsion_loss(cell_features, pin_features,edge_list):
  N = cell_features.shape[0]
  device = cell_features.device

  pos = cell_features[:, CellFeatureIdx.X:CellFeatureIdx.WIDTH]    # (x, y)
  dims = cell_features[:, CellFeatureIdx.WIDTH:CellFeatureIdx.HEIGHT+1]   # (w, h)
  areas = cell_features[:, CellFeatureIdx.AREA]

  dist = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
  min_sep = (dims.unsqueeze(1) + dims.unsqueeze(0)) * 0.5

  # Smooth overlap per axis (softplus ~ ReLU with gradient near zero)
  overlap_vec = F.softplus(min_sep - dist, beta=30.0)
  overlap_area = overlap_vec[..., 0] * overlap_vec[..., 1]

  weights = areas.unsqueeze(1) + areas.unsqueeze(0)
  mask = torch.triu(torch.ones(N, N, device=device),diagonal=1)

  return 2* (overlap_area * weights * mask).sum()  / (N * (N-1))


# Spatial Hashing for large N
def overlap_repulsion_loss_lite(cell_features, pin_features, edge_list, epoch_progress=1.0, grid_bucket_size=None, max_pairs=2_000_000, device=None):
    """
    Computes a differentiable overlap penalty.
    Uses Spatial Hashing for large ones.
    """
    if device is None:
        device = cell_features.device
    N = cell_features.shape[0]

    if N <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    x = cell_features[:, CellFeatureIdx.X]
    y = cell_features[:, CellFeatureIdx.Y]
    w = cell_features[:, CellFeatureIdx.WIDTH]
    h = cell_features[:, CellFeatureIdx.HEIGHT]

    # Bucket size based on typical cell dimensions
    # Median keeps buckets local for standard cells
    if grid_bucket_size is None:
        median_w = torch.median(w).item()
        median_h = torch.median(h).item()
        grid_bucket_size = max(1e-3, 0.5 * max(median_w, median_h))

    xmin = x.min().detach()
    ymin = y.min().detach()

    bx = torch.floor((x - xmin) / grid_bucket_size).to(torch.int64)
    by = torch.floor((y - ymin) / grid_bucket_size).to(torch.int64)

    # Build bucket map
    buckets = {}
    for idx, (ix, iy) in enumerate(zip(bx.cpu().tolist(), by.cpu().tolist())):
        buckets.setdefault((ix, iy), []).append(idx)

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # Search Moore neighborhood (self + 8 neighbors) for candidate overlaps
    pairs = []
    for (ix, iy), indices in buckets.items():
        neighbors = []
        for dx, dy in neighbor_offsets:
            neighbors.extend(buckets.get((ix + dx, iy + dy), []))
        for i in indices:
            for j in neighbors:
                if j > i:
                    pairs.append((i, j))
        if len(pairs) > max_pairs:
            break

    if not pairs:
        return torch.tensor(0.0, device=device, requires_grad=True)


    pairs = torch.tensor(pairs, dtype=torch.long, device=device)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    dx = torch.abs(x[i_idx] - x[j_idx])
    dy = torch.abs(y[i_idx] - y[j_idx])
    min_sep_x = 0.5 * (w[i_idx] + w[j_idx])
    min_sep_y = 0.5 * (h[i_idx] + h[j_idx])

    overlap_x = F.softplus(min_sep_x - dx, beta=20.0)
    overlap_y = F.softplus(min_sep_y - dy, beta=20.0)

    return (overlap_x * overlap_y).sum()


def detect_macros_by_area(cell_features, ratio_threshold=30.0, max_macros=50):
    """
    Detect macros by finding a large area gap.
    Returns macro_indices, stdcell_indices
    """
    areas = cell_features[:, CellFeatureIdx.AREA]
    sorted_idx = torch.argsort(areas, descending=True)
    sorted_areas = areas[sorted_idx]

    # Find first big ratio drop
    ratios = sorted_areas[:-1] / (sorted_areas[1:] + 1e-9)

    macro_cut = 0
    for i, r in enumerate(ratios):
        if r > ratio_threshold:
            macro_cut = i + 1
            break

    macro_cut = min(macro_cut, max_macros)
    macro_indices = sorted_idx[:macro_cut].tolist()
    stdcell_indices = sorted_idx[macro_cut:].tolist()

    return macro_indices, stdcell_indices


def macro_cell_repulsion_loss(cell_features, macro_indices, stdcell_mask):
    """
    Penalize overlap between macros and standard cells.
    """
    device = cell_features.device

    macros = cell_features[macro_indices]
    stds = cell_features[stdcell_mask]

    if macros.shape[0] == 0 or stds.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    mx = macros[:, CellFeatureIdx.X][:, None]
    my = macros[:, CellFeatureIdx.Y][:, None]
    mw = macros[:, CellFeatureIdx.WIDTH][:, None]
    mh = macros[:, CellFeatureIdx.HEIGHT][:, None]

    sx = stds[:, CellFeatureIdx.X][None, :]
    sy = stds[:, CellFeatureIdx.Y][None, :]
    sw = stds[:, CellFeatureIdx.WIDTH][None, :]
    sh = stds[:, CellFeatureIdx.HEIGHT][None, :]

    dx = torch.abs(mx - sx)
    dy = torch.abs(my - sy)

    min_x = 0.5 * (mw + sw)
    min_y = 0.5 * (mh + sh)

    ox = F.softplus(min_x - dx, beta=20.0)
    oy = F.softplus(min_y - dy, beta=20.0)

    overlap = ox * oy
    return overlap.sum()


def macro_macro_overlap_exact(cell_features, macro_indices, beta=50.0):
    """
    macro-macro overlap loss (NO BUCKETS).
    """
    device = cell_features.device
    M = len(macro_indices)
    if M <= 1:
        return torch.tensor(0.0, device=device)

    macros = cell_features[macro_indices]
    x = macros[:, CellFeatureIdx.X]
    y = macros[:, CellFeatureIdx.Y]
    w = macros[:, CellFeatureIdx.WIDTH]
    h = macros[:, CellFeatureIdx.HEIGHT]

    dx = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    dy = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))

    min_sep_x = 0.5 * (w.unsqueeze(0) + w.unsqueeze(1))
    min_sep_y = 0.5 * (h.unsqueeze(0) + h.unsqueeze(1))

    ox = torch.nn.functional.softplus(min_sep_x - dx, beta=beta)
    oy = torch.nn.functional.softplus(min_sep_y - dy, beta=beta)

    overlap = ox * oy
    overlap = overlap * (1.0 - torch.eye(M, device=device))  # remove self-pairs
    mask = torch.triu(torch.ones(M, M, device=device), diagonal=1)
    overlap = overlap * mask
    return overlap.sum()


def legalize_std_cells_row_based_two_pass(
    cell_features,
    stdcell_mask,
    macro_indices,
    row_height=None,
    x_margin=1e-3,
    do_final_cleanup=False,
):
    cf = cell_features.clone()
    device = cf.device

    std_idxs = torch.where(stdcell_mask)[0].cpu().numpy()
    if len(std_idxs) <= 1:
        return cf

    x = cf[:, CellFeatureIdx.X].cpu().numpy()
    y = cf[:, CellFeatureIdx.Y].cpu().numpy()
    w = cf[:, CellFeatureIdx.WIDTH].cpu().numpy()
    h = cf[:, CellFeatureIdx.HEIGHT].cpu().numpy()

    if row_height is None:
        row_height = float(np.median(h[std_idxs]))

    ymin = float((y[std_idxs] - 0.5 * row_height).min())
    ymax = float((y[std_idxs] + 0.5 * row_height).max())
    num_rows = max(1, int(np.ceil((ymax - ymin) / row_height)))
    row_centers = ymin + row_height * (np.arange(num_rows) + 0.5)

    # assign to rows and snap Y
    row_bins = {r: [] for r in range(num_rows)}
    for i in std_idxs:
        r = int(np.argmin(np.abs(row_centers - y[i])))
        y[i] = row_centers[r]
        row_bins[r].append(i)

    # obstacles per row
    macro_obs = {r: [] for r in range(num_rows)}
    for mi in macro_indices:
        mx, my, mw, mh = x[mi], y[mi], w[mi], h[mi]
        for r, rc in enumerate(row_centers):
            if abs(rc - my) <= (mh / 2 + row_height / 2):
                macro_obs[r].append((mx - mw / 2, mx + mw / 2))

    # merge obstacles
    for r in macro_obs:
        obs = sorted(macro_obs[r])
        merged = []
        for a, b in obs:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        macro_obs[r] = merged

    # per-row packing: left->right pass (use left-edge sorting & left-edge cursor)
    for r, cells in row_bins.items():
        if not cells:
            continue
        # sort by left edge (stronger than center)
        cells.sort(key=lambda i: x[i] - 0.5 * w[i])

        cursor = -1e9
        obs = macro_obs[r]
        obs_idx = 0

        for i in cells:
            wi = w[i]
            # ensure cursor is not inside a macro; move past obstacles as needed
            while obs_idx < len(obs) and cursor + wi > obs[obs_idx][0]:
                cursor = obs[obs_idx][1] + x_margin
                obs_idx += 1

            left_pos = max(cursor, x[i] - wi / 2)   # do not move left of original left edge
            x[i] = left_pos + wi / 2
            cursor = left_pos + wi + x_margin

    # right->left pass: compact and fix remaining overlaps (mirror logic)
    for r, cells in row_bins.items():
        if not cells:
            continue
        # sort by right edge descending
        cells.sort(key=lambda i: x[i] + 0.5 * w[i], reverse=True)

        cursor = 1e9
        obs = macro_obs[r]
        # iterate obs in reverse for right->left
        obs_rev = obs[::-1]
        obs_idx = 0

        for i in cells:
            wi = w[i]
            # while placing cell at center = cursor - wi/2 would hit obstacle on the left, push cursor left of obstacle
            while obs_idx < len(obs_rev) and cursor - wi < obs_rev[obs_idx][1]:
                cursor = obs_rev[obs_idx][0] - x_margin
                obs_idx += 1

            right_pos = min(cursor, x[i] + wi / 2)   # do not move right of original right edge
            x[i] = right_pos - wi / 2
            cursor = right_pos - wi - x_margin

    cf[:, 2] = torch.tensor(x, device=device)
    cf[:, 3] = torch.tensor(y, device=device)

    # optional final exact cleanup on any remaining overlapping pairs
    if do_final_cleanup:
        pairs = find_overlapping_pairs_bucketed(cf)
        if pairs:
            active = unique_indices_from_pairs(pairs)
            cf = exact_cleanup_on_active(cf, active)

    return cf


def find_overlapping_pairs_bucketed(cell_features, bucket_size=None):
    cf = cell_features
    N = cf.shape[0]
    if N <= 1:
        return []

    x = cf[:, CellFeatureIdx.X].detach()
    y = cf[:, CellFeatureIdx.Y].detach()
    w = cf[:, CellFeatureIdx.WIDTH].detach()
    h = cf[:, CellFeatureIdx.HEIGHT].detach()

    if bucket_size is None:
        median_w = float(torch.median(w).item())
        median_h = float(torch.median(h).item())
        bucket_size = max(1e-6, 0.75 * max(median_w, median_h))

    xmin = float((x - 0.5 * w).min().item())
    ymin = float((y - 0.5 * h).min().item())

    # Compute bounding-box extents in bucket coordinates
    left = (x - 0.5 * w - xmin)  # relative coord
    right = (x + 0.5 * w - xmin)
    bottom = (y - 0.5 * h - ymin)
    top = (y + 0.5 * h - ymin)

    bx_min = torch.floor(left / bucket_size).to(torch.int64)
    bx_max = torch.floor(right / bucket_size).to(torch.int64)
    by_min = torch.floor(bottom / bucket_size).to(torch.int64)
    by_max = torch.floor(top / bucket_size).to(torch.int64)

    # build bucket map
    buckets = {}
    for i in range(N):
        imin = int(bx_min[i].item()); imax = int(bx_max[i].item())
        jmin = int(by_min[i].item()); jmax = int(by_max[i].item())
        for ix in range(imin, imax + 1):
            for iy in range(jmin, jmax + 1):
                key = (ix, iy)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(i)

    # Collect overlapping pairs using local neighborhood checks
    pairs_set = set()
    for (ix, iy), ids in buckets.items():
        # collect indices in neighbor 3x3 buckets
        neighbor_ids = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_ids.extend(buckets.get((ix + dx, iy + dy), []))

        if not neighbor_ids:
            continue
        neighbor_ids = list(set(neighbor_ids))

        # Exact overlap check for candidate pairs
        for i in ids:
            for j in neighbor_ids:
                if j <= i:
                    continue
                dxv = abs(float(x[i].item()) - float(x[j].item()))
                dyv = abs(float(y[i].item()) - float(y[j].item()))
                min_x = 0.5 * (float(w[i].item()) + float(w[j].item()))
                min_y = 0.5 * (float(h[i].item()) + float(h[j].item()))
                if (dxv < min_x) and (dyv < min_y):
                    pairs_set.add((i, j))

    # return sorted list
    pairs = sorted(list(pairs_set))
    return pairs


def unique_indices_from_pairs(pairs):
    s = set()
    for a, b in pairs:
        s.add(a); s.add(b)
    return sorted(list(s))


def exact_cleanup_on_active(cell_features, active_indices, eps=1e-2):
    """
    resolves overlaps deterministically by directly shifting
    cell centers along the minimum penetration axis. It is intended to be
    used only on a very small subset of cells (the final few overlapping
    pairs after optimization and legalization.
    """
    if len(active_indices) <= 1:
        return cell_features

    cf = cell_features.clone()
    k = len(active_indices)

    xs = cf[active_indices, CellFeatureIdx.X].clone()
    ys = cf[active_indices, CellFeatureIdx.Y].clone()
    ws = cf[active_indices, CellFeatureIdx.WIDTH]
    hs = cf[active_indices, CellFeatureIdx.HEIGHT]

    # exact pairwise resolve few elements
    for i in range(k):
        for j in range(i + 1, k):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            minx = 0.5 * (ws[i] + ws[j])
            miny = 0.5 * (hs[i] + hs[j])

            ox = (minx - torch.abs(dx)).item()
            oy = (miny - torch.abs(dy)).item()

            if ox > 0 and oy > 0:
                if ox < oy:
                    shift = 0.5 * ox + eps
                    sign = 1.0 if dx >= 0 else -1.0
                    xs[i] = xs[i] + sign * shift
                    xs[j] = xs[j] - sign * shift
                else:
                    shift = 0.5 * oy + eps
                    sign = 1.0 if dy >= 0 else -1.0
                    ys[i] = ys[i] + sign * shift
                    ys[j] = ys[j] - sign * shift

    for ii, idx in enumerate(active_indices):
        cf[idx, 2] = xs[ii]
        cf[idx, 3] = ys[ii]

    return cf


def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.5,
    lambda_wirelength=100.0,
    lambda_overlap=200.0,
    verbose=True,
    log_interval=100,
):
    """
    Performs continuous wirelength-driven placement with scalable overlap handling.
    A final deterministic cleanup to ensure strictly zero overlaps.
    """

    device = cell_features.device
    initial_cell_features = cell_features.clone()
    N = cell_features.shape[0]

    # Detect macros
    macro_indices, std_cell_indices = detect_macros_by_area(cell_features)
    # print("done macro_indices")

    stdcell_mask = torch.zeros(cell_features.shape[0], dtype=torch.bool, device=device)
    stdcell_mask[std_cell_indices] = True

    # Smart initialization (wirelength clustering only)
    cell_positions = initialize_smart_placement(cell_features, pin_features, edge_list)
    cell_features[:, 2:4] = cell_positions.detach()
    # print("done Smart initialization")

    cell_positions = cell_features[:, CellFeatureIdx.X:CellFeatureIdx.WIDTH].detach().clone().requires_grad_(True)

    optimizer = torch.optim.AdamW([cell_positions], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    # Continuous optimization loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        cur = cell_features.clone()
        cur[:, 2:4] = cell_positions

        # Wirelength
        wl_loss = wirelength_attraction_loss(cur, pin_features, edge_list)
        # print("Done wirlength loss calculation")

        progress = epoch / num_epochs
        progress_threshold = 0.75

        if N > 500:
          # split overlap losses (FAST)
          # std–std
          ov_std = overlap_repulsion_loss_lite(cur[stdcell_mask], pin_features, edge_list, epoch_progress=progress,)
          # print("Done overlap_repulsion_loss_lite")

          # macro–std
          ov_macro_std = macro_cell_repulsion_loss(cur, macro_indices, stdcell_mask)
          # print("Done macro_cell_repulsion_loss")

          # macro–macro
          if len(macro_indices) > 1:
              ov_macro_macro = macro_macro_overlap_exact(cur, macro_indices)
          else:
              ov_macro_macro = torch.tensor(0.0, device=device)
          # print("Done macro_macro_overlap_exact")


          overlap_loss = (1000 * ov_std + 100.0 * ov_macro_std + 100.0 * ov_macro_macro)
          progress_threshold = 0.6

        else:
          overlap_loss = overlap_repulsion_loss(cur, pin_features, edge_list)

        if progress < progress_threshold:
            cur_lambda_wl = lambda_wirelength / max(progress, 0.1)
            cur_lambda_ov = lambda_overlap * progress
        else:
            cur_lambda_wl = 10.0
            cur_lambda_ov = 100000.0

        total_loss = (cur_lambda_wl * wl_loss + cur_lambda_ov * overlap_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], 1.0)

        optimizer.step()
        scheduler.step()

        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"WL: {wl_loss.item():.4f} | "
                f"OV: {overlap_loss.item():.4f} | "
                f"λ_ov: {cur_lambda_ov:.1f}"
            )

    cell_features[:, 2:4] = cell_positions.detach()

    if N > 500 : cell_features = legalize_std_cells_row_based_two_pass(cell_features, stdcell_mask, macro_indices)
    
    return {
        "final_cell_features": cell_features.clone(),
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }

# ======= FINAL EVALUATION CODE (Don't edit this part) =======

def calculate_overlap_metrics(cell_features):
    """Calculate ground truth overlap statistics (non-differentiable).

    This function provides exact overlap measurements for evaluation and reporting.
    Unlike the loss function, this does NOT need to be differentiable.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]

    Returns:
        Dictionary with:
            - overlap_count: number of overlapping cell pairs (int)
            - total_overlap_area: sum of all overlap areas (float)
            - max_overlap_area: largest single overlap area (float)
            - overlap_percentage: percentage of total area that overlaps (float)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()  # [N, 2]
    widths = cell_features[:, 4].detach().numpy()  # [N]
    heights = cell_features[:, 5].detach().numpy()  # [N]
    areas = cell_features[:, 0].detach().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    overlap_areas = []

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                overlap_areas.append(overlap_area)

    # Calculate percentage of total area
    total_area = sum(areas)
    overlap_percentage = (overlap_count / N * 100) if total_area > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "overlap_percentage": overlap_percentage,
    }


def calculate_cells_with_overlaps(cell_features):
    """Calculate number of cells involved in at least one overlap.

    This metric matches the test suite evaluation criteria.

    Args:
        cell_features: [N, 6] tensor with cell properties

    Returns:
        Set of cell indices that have overlaps with other cells
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    # Extract cell properties
    positions = cell_features[:, 2:4].detach().numpy()
    widths = cell_features[:, 4].detach().numpy()
    heights = cell_features[:, 5].detach().numpy()

    cells_with_overlaps = set()

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap
            min_sep_x = (widths[i] + widths[j]) / 2
            min_sep_y = (heights[i] + heights[j]) / 2

            # Calculate overlap amounts
            overlap_x = max(0, min_sep_x - dx)
            overlap_y = max(0, min_sep_y - dy)

            # Overlap occurs only if both x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                cells_with_overlaps.add(i)
                cells_with_overlaps.add(j)

    return cells_with_overlaps


def calculate_normalized_metrics(cell_features, pin_features, edge_list):
    """Calculate normalized overlap and wirelength metrics for test suite.

    These metrics match the evaluation criteria in the test suite.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity

    Returns:
        Dictionary with:
            - overlap_ratio: (num cells with overlaps / total cells)
            - normalized_wl: (wirelength / num nets) / sqrt(total area)
            - num_cells_with_overlaps: number of unique cells involved in overlaps
            - total_cells: total number of cells
            - num_nets: number of nets (edges)
    """
    N = cell_features.shape[0]

    # Calculate overlap metric: num cells with overlaps / total cells
    cells_with_overlaps = calculate_cells_with_overlaps(cell_features)
    num_cells_with_overlaps = len(cells_with_overlaps)
    overlap_ratio = num_cells_with_overlaps / N if N > 0 else 0.0

    # Calculate wirelength metric: (wirelength / num nets) / sqrt(total area)
    if edge_list.shape[0] == 0:
        normalized_wl = 0.0
        num_nets = 0
    else:
        # Calculate total wirelength using the loss function (unnormalized)
        wl_loss = wirelength_attraction_loss(cell_features, pin_features, edge_list)
        total_wirelength = wl_loss.item() * edge_list.shape[0]  # Undo normalization

        # Calculate total area
        total_area = cell_features[:, 0].sum().item()

        num_nets = edge_list.shape[0]

        # Normalize: (wirelength / net) / sqrt(area)
        # This gives a dimensionless quality metric independent of design size
        normalized_wl = (total_wirelength / num_nets) / (total_area ** 0.5) if total_area > 0 else 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "normalized_wl": normalized_wl,
        "num_cells_with_overlaps": num_cells_with_overlaps,
        "total_cells": N,
        "num_nets": num_nets,
    }


def plot_placement(
    initial_cell_features,
    final_cell_features,
    pin_features,
    edge_list,
    filename="placement_result.png",
):
    """Create side-by-side visualization of initial vs final placement.

    Args:
        initial_cell_features: Initial cell positions and properties
        final_cell_features: Optimized cell positions and properties
        pin_features: Pin information
        edge_list: Edge connectivity
        filename: Output filename for the plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot both initial and final placements
        for ax, cell_features, title in [
            (ax1, initial_cell_features, "Initial Placement"),
            (ax2, final_cell_features, "Final Placement"),
        ]:
            N = cell_features.shape[0]
            positions = cell_features[:, 2:4].detach().numpy()
            widths = cell_features[:, 4].detach().numpy()
            heights = cell_features[:, 5].detach().numpy()

            # Draw cells
            for i in range(N):
                x = positions[i, 0] - widths[i] / 2
                y = positions[i, 1] - heights[i] / 2
                rect = Rectangle(
                    (x, y),
                    widths[i],
                    heights[i],
                    fill=True,
                    facecolor="lightblue",
                    edgecolor="darkblue",
                    linewidth=0.5,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            # Calculate and display overlap metrics
            metrics = calculate_overlap_metrics(cell_features)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{title}\n"
                f"Overlaps: {metrics['overlap_count']}, "
                f"Total Overlap Area: {metrics['total_overlap_area']:.2f}",
                fontsize=12,
            )

            # Set axis limits with margin
            all_x = positions[:, 0]
            all_y = positions[:, 1]
            margin = 10
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to enable visualization: pip install matplotlib")

# ======= MAIN FUNCTION =======

def main():
    """Main function demonstrating the placement optimization challenge."""
    print("=" * 70)
    print("VLSI CELL PLACEMENT OPTIMIZATION CHALLENGE")
    print("=" * 70)
    print("\nObjective: Implement overlap_repulsion_loss() to eliminate cell overlaps")
    print("while minimizing wirelength.\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate placement problem
    num_macros = 3
    num_std_cells = 50

    print(f"Generating placement problem:")
    print(f"  - {num_macros} macros")
    print(f"  - {num_std_cells} standard cells")

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Initialize positions with random spread to reduce initial overlaps
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius

    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Calculate initial metrics
    print("\n" + "=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    initial_metrics = calculate_overlap_metrics(cell_features)
    print(f"Overlap count: {initial_metrics['overlap_count']}")
    print(f"Total overlap area: {initial_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {initial_metrics['max_overlap_area']:.2f}")
    print(f"Overlap percentage: {initial_metrics['overlap_percentage']:.2f}%")

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)

    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        verbose=True,
        log_interval=200,
    )

    # Calculate final metrics (both detailed and normalized)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_cell_features = result["final_cell_features"]

    # Detailed metrics
    final_metrics = calculate_overlap_metrics(final_cell_features)
    print(f"Overlap count (pairs): {final_metrics['overlap_count']}")
    print(f"Total overlap area: {final_metrics['total_overlap_area']:.2f}")
    print(f"Max overlap area: {final_metrics['max_overlap_area']:.2f}")

    # Normalized metrics (matching test suite)
    print("\n" + "-" * 70)
    print("TEST SUITE METRICS (for leaderboard)")
    print("-" * 70)
    normalized_metrics = calculate_normalized_metrics(
        final_cell_features, pin_features, edge_list
    )
    print(f"Overlap Ratio: {normalized_metrics['overlap_ratio']:.4f} "
          f"({normalized_metrics['num_cells_with_overlaps']}/{normalized_metrics['total_cells']} cells)")
    print(f"Normalized Wirelength: {normalized_metrics['normalized_wl']:.4f}")

    # Success check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)
    if normalized_metrics["num_cells_with_overlaps"] == 0:
        print("✓ PASS: No overlapping cells!")
        print("✓ PASS: Overlap ratio is 0.0")
        print("\nCongratulations! Your implementation successfully eliminated all overlaps.")
        print(f"Your normalized wirelength: {normalized_metrics['normalized_wl']:.4f}")
    else:
        print("✗ FAIL: Overlaps still exist")
        print(f"  Need to eliminate overlaps in {normalized_metrics['num_cells_with_overlaps']} cells")
        print("\nSuggestions:")
        print("  1. Check your overlap_repulsion_loss() implementation")
        print("  2. Change lambdas (try increasing lambda_overlap)")
        print("  3. Change learning rate or number of epochs")

    # Generate visualization
    plot_placement(
        result["initial_cell_features"],
        result["final_cell_features"],
        pin_features,
        edge_list,
        filename="placement_result.png",
    )

if __name__ == "__main__":
    main()
