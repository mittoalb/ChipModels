#!/usr/bin/env python3
import os
import sys
import shutil
import json
import numpy as np
import z5py
from typing import Callable, Optional

current_dir = os.path.dirname(__file__)
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from tomomodels.logger import setup_custom_logger, log_exception
from tomomodels.utils import save_mzarr

logger = setup_custom_logger("chipgen", lfname="chipgen.log")


# Load material codes from external JSON
with open("chip_codes.json", "r") as f:
    LABELS = json.load(f)

# Create shortcut mapping: name -> label
CODES = {k: v["label"] for k, v in LABELS.items()}

#--------------------------------------------------
# 1) Substrate & Oxides
#--------------------------------------------------
def add_substrate(shape, layers):
    """
    Returns list of (name, thickness) including substrate to fill shape[0].
    """
    used = sum(th for _, th in layers)
    substrate_th = shape[0] - used
    return layers + [('Si_substrate', substrate_th)]


def add_well_taps(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    well_type: str = 'nwell',
    tap_code: str = 'pad_contact',
    plug_code: str = 'W_plug',
    diameter: int = 6,
    pitch: int = 64
) -> np.ndarray:
    """
    Add regular tap contacts in the specified well region (pwell or nwell).

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
        3D chip volume.
    layers : list of (name, thickness)
        Layer stack.
    well_type : str
        Either 'nwell' or 'pwell'.
    tap_code : str
        Label for metal pad to fill the top.
    plug_code : str
        Label for plug metal to fill the lower part.
    diameter : int
        Diameter of each tap contact.
    pitch : int
        Spacing between taps in Y and X.
    """
    nz, ny, nx = vol.shape
    ys, xs = np.ogrid[:ny, :nx]
    r = diameter // 2

    out = vol.copy()

    # --- Locate well z-range ---
    z_acc = 0
    z_well_top = z_well_bot = None
    for name, th in layers:
        if name == 'Si_substrate':
            z_well_top = z_acc
            z_well_bot = min(z_well_top + 20, nz)
            break
        z_acc += th

    if z_well_top is None:
        raise ValueError("Si_substrate not found in layers list.")

    # --- Locate pad_contact and passivation range ---
    z_pad_s = z_pass_s = None
    z_acc = 0
    for name, th in layers:
        if name == 'SiN_passivation':
            z_pass_s = z_acc
        if name == 'pad_contact':
            z_pad_s = z_acc
        z_acc += th

    if z_pass_s is None or z_pad_s is None:
        raise ValueError("Missing SiN_passivation or pad_contact layer.")

    # --- Place taps at regular intervals ---
    for cy in range(r, ny, pitch):
        for cx in range(r, nx, pitch):
            # check if we're over the correct well
            if vol[z_well_top, cy, cx] != CODES[well_type]:
                continue
            mask = (ys - cy)**2 + (xs - cx)**2 <= r*r
            # fill passivation
            for zz in range(z_pass_s, z_pass_s + 5):
                out[zz][mask] = CODES[tap_code]
            # fill pad_contact
            for zz in range(z_pad_s, z_pad_s + 4):
                out[zz][mask] = CODES[plug_code]

    return out

def add_io_periphery_routing(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    metal_layer: str = 'Cu5',
    trace_width: int = 6,
    trace_thickness: int = 2,
    ring_offset: int = 10
) -> np.ndarray:
    """
    Add thick I/O routing traces near the die edge (top metal layer).
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # Find top metal z-range
    z0 = 0
    for name, th in layers:
        if name == metal_layer:
            z_top = z0 + layers[-1][1] - trace_thickness
            break
        z0 += th

    metal_val = CODES[metal_layer]

    # Create horizontal and vertical periphery lines
    x0, x1 = ring_offset, nx - ring_offset
    y0, y1 = ring_offset, ny - ring_offset

    for z in range(z_top, z_top + trace_thickness):
        # Top and bottom lines
        out[z, y0:y0+trace_width, x0:x1] = metal_val
        out[z, y1-trace_width:y1, x0:x1] = metal_val
        # Left and right lines
        out[z, y0:y1, x0:x0+trace_width] = metal_val
        out[z, y0:y1, x1-trace_width:x1] = metal_val

    return out

def add_sram_block(vol, layers, region):
    """
    Add SRAM macro block as a dense rectangular fill of pattern.
    """
    z_range, y_range, x_range = region
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    out = vol.copy()
    metal_val = CODES['Cu2']

    # Fill as repeating grid pattern
    for z in range(z0 + 20, z0 + 24):  # pick mid metal layer
        for y in range(y0, y1, 8):
            for x in range(x0, x1, 8):
                out[z, y:y+2, x:x+6] = metal_val

    return out

def add_analog_block(vol, layers, region):
    """
    Add sparse analog transistors in a corner zone.
    """
    z_range, y_range, x_range = region
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    out = vol.copy()
    ys, xs = np.ogrid[y0:y1, x0:x1]

    cy, cx = (y0 + y1)//2, (x0 + x1)//2
    for z in range(z0 + 10, z0 + 14):
        mask = ((ys - cy)**2 + (xs - cx)**2) < 1000
        out[z, y0:y1, x0:x1][mask] = CODES['Cu1']

    return out


def add_dummy_fills(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    layer_name: str = 'Cu2',
    code_key: str = 'Cu2',
    size: int = 6,
    spacing: int = 24,
    coverage: float = 0.2,
    seed: int = 123
) -> np.ndarray:
    """
    Add dummy metal fill blocks to a metal layer for density balancing.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
    layer_name : str
        Which metal layer to sprinkle fills into.
    code_key : str
        Material code (e.g., Cu2).
    size : int
        Dummy square side length in voxels.
    spacing : int
        Center-to-center spacing of dummy fills.
    coverage : float
        Probability to actually place a dummy at each grid point.
    seed : int
        RNG seed for reproducibility.
    """
    out = vol.copy()
    nz, ny, nx = vol.shape
    rng = np.random.default_rng(seed)

    # Find layer z-range
    z_acc = 0
    z0 = z1 = None
    for name, th in layers:
        if name == layer_name:
            z0 = z_acc
            z1 = z0 + th
            break
        z_acc += th
    if z0 is None:
        raise ValueError(f"Layer {layer_name} not found.")

    # Define fill region
    code = CODES[code_key]
    half = size // 2
    for y in range(spacing, ny - spacing, spacing):
        for x in range(spacing, nx - spacing, spacing):
            if rng.random() > coverage:
                continue
            ys = slice(max(0, y - half), min(ny, y + half + 1))
            xs = slice(max(0, x - half), min(nx, x + half + 1))
            for z in range(z0, z1):
                out[z, ys, xs] = code
    return out



def add_active_region_isolation(vol, layers, y_positions, width=4, code_key='STI'):
    """Insert vertical STI strips to isolate transistor rows (CMOS active region separation)."""
    nz, ny, nx = vol.shape
    out = vol.copy()

    # find top of substrate
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_sub = z_acc
            break
        z_acc += th

    z_depth = 10  # how deep into substrate to go
    for y in y_positions:
        y0 = max(0, y - width//2)
        y1 = min(ny, y + width//2)
        out[z_sub:z_sub+z_depth, y0:y1, :] = CODES[code_key]

    return out


def add_sram_macro_blocks(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    layer_name: str = 'Cu3',
    block_size: tuple[int, int] = (64, 64),
    pattern_pitch: int = 4,
    pattern_type: str = 'checker',
    positions: list[tuple[int, int]] = None,
    code_key: str = 'Cu3'
) -> np.ndarray:
    """
    Add SRAM-like macro blocks with internal bitcell pattern to a metal layer.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
    layer_name : str
        Target layer for SRAM macro.
    block_size : (int, int)
        Size of each macro (height, width in voxels).
    pattern_pitch : int
        Pattern repetition size (e.g., checkerboard granularity).
    pattern_type : str
        One of: 'checker', 'lines', 'solid'.
    positions : list of (cy, cx)
        Center YX coordinates of each macro.
    code_key : str
        Material label from CODES.
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # Get z-range for layer
    z_acc = 0
    for name, th in layers:
        if name == layer_name:
            z_start = z_acc
            z_end   = z_acc + th
            break
        z_acc += th
    else:
        raise ValueError(f"Layer '{layer_name}' not found in layer stack.")

    code = CODES[code_key]
    by, bx = block_size
    ys, xs = np.ogrid[:ny, :nx]

    if positions is None:
        # Default: one macro per quadrant
        positions = [
            (ny//4,   nx//4),
            (ny//4,   3*nx//4),
            (3*ny//4, nx//4),
            (3*ny//4, 3*nx//4),
        ]

    for cy, cx in positions:
        y0 = max(0, cy - by//2)
        y1 = min(ny, cy + by//2)
        x0 = max(0, cx - bx//2)
        x1 = min(nx, cx + bx//2)

        # Define pattern inside block
        region_ys, region_xs = np.ogrid[y0:y1, x0:x1]
        if pattern_type == 'checker':
            pattern = ((region_ys // pattern_pitch) + (region_xs // pattern_pitch)) % 2 == 0
        elif pattern_type == 'lines':
            pattern = (region_ys // pattern_pitch) % 2 == 0
        else:
            pattern = np.ones((y1 - y0, x1 - x0), dtype=bool)

        # Paint pattern across all z-slices
        for z in range(z_start, z_end):
            out[z, y0:y1, x0:x1][pattern] = code

    return out


def route_local_interconnects(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    gate_layer: str = 'Cu1',
    via_code: str = 'Cu1',
    wire_pitch: int = 20,
    wire_width: int = 4,
    n_wires: int = 8,
    z_offset: int = 1
) -> np.ndarray:
    """
    Draw horizontal Cu1 wires across rows and connect to vertical Cu2 vias.

    Parameters
    ----------
    vol : ndarray
        Volume to modify.
    layers : list of (name, thickness)
    gate_layer : str
        Which layer to draw wires on.
    via_code : str
        Label for vias (usually same metal as lower layer).
    wire_pitch : int
        Spacing between wires.
    wire_width : int
        Width of wire stripes.
    n_wires : int
        Number of horizontal Cu1 wires.
    z_offset : int
        How deep into gate_layer to draw.
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # Find Cu1 z index
    z_acc = 0
    for name, th in layers:
        if name == gate_layer:
            z_cu1 = z_acc + z_offset
            break
        z_acc += th

    y0 = ny // 2 - (n_wires // 2) * wire_pitch
    for i in range(n_wires):
        y = y0 + i * wire_pitch
        rows = slice(max(0, y - wire_width // 2), min(ny, y + wire_width // 2))
        out[z_cu1, rows, :] = CODES[gate_layer]

    # Add vias to Cu2
    for name, th in layers:
        if name == 'Cu2':
            z_cu2 = z_acc
            break
        z_acc += th

    ys, xs = np.ogrid[:ny, :nx]
    r = wire_width // 2
    for i in range(n_wires):
        y = y0 + i * wire_pitch
        for x in range(r, nx, 3 * wire_pitch):
            mask = (ys - y) ** 2 + (xs - x) ** 2 <= r * r
            for z in range(z_cu1 + 1, z_cu2):
                out[z][mask] = CODES[via_code]

    return out


def add_power_grid(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    layer_name: str = 'Cu5',
    stripe_width: int = 8,
    spacing: int = 64,
    code_key: str = 'Cu5'
) -> np.ndarray:
    """
    Add a horizontal + vertical power stripe grid in a top metal layer.
    
    Parameters
    ----------
    layer_name : str
        Top metal layer to apply grid (e.g. 'Cu5')
    stripe_width : int
        Width of each power stripe
    spacing : int
        Distance between adjacent stripes
    code_key : str
        Label code from CODES (typically same as metal)
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # Locate layer z-range
    z_acc = 0
    for name, th in layers:
        if name == layer_name:
            z_start = z_acc
            z_end   = z_acc + th
            break
        z_acc += th
    else:
        raise ValueError(f"Layer '{layer_name}' not found.")

    fill_val = CODES[code_key]

    # Stripe grid mask
    mask = np.zeros((ny, nx), dtype=bool)
    for y in range(0, ny, spacing):
        y0 = max(0, y - stripe_width//2)
        y1 = min(ny, y + stripe_width//2)
        mask[y0:y1, :] = True
    for x in range(0, nx, spacing):
        x0 = max(0, x - stripe_width//2)
        x1 = min(nx, x + stripe_width//2)
        mask[:, x0:x1] = True

    # Apply to z-slices
    for z in range(z_start, z_end):
        out[z][mask] = fill_val

    return out




def add_metal_metal_decaps(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    top: str = 'Cu4',
    bottom: str = 'Cu3',
    coverage: float = 0.1,
    patch_size: int = 8,
    pitch: int = 32
) -> np.ndarray:
    """
    Add periodic parallel-plate decoupling capacitors between two metal layers.

    Parameters
    ----------
    top : str
        Top metal layer name (e.g., 'Cu4')
    bottom : str
        Bottom metal layer name (e.g., 'Cu3')
    coverage : float
        Approximate percentage of area covered with decaps
    patch_size : int
        Size of square metal plates
    pitch : int
        Distance between decaps
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # Find z indices for bottom and top metal layers
    z_acc = 0
    z_bot, z_top = None, None
    for name, th in layers:
        if name == bottom:
            z_bot = z_acc
        if name == top:
            z_top = z_acc
        z_acc += th
    if z_bot is None or z_top is None:
        raise ValueError(f"Could not locate metal layers '{bottom}' or '{top}'")

    y_centers = np.arange(patch_size//2, ny, pitch)
    x_centers = np.arange(patch_size//2, nx, pitch)

    for y in y_centers:
        for x in x_centers:
            if np.random.rand() > coverage:
                continue
            y0 = max(0, y - patch_size//2)
            y1 = min(ny, y + patch_size//2)
            x0 = max(0, x - patch_size//2)
            x1 = min(nx, x + patch_size//2)
            out[z_bot, y0:y1, x0:x1] = CODES[bottom]
            out[z_top, y0:y1, x0:x1] = CODES[top]

    return out



def generate_transistor_array(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    n_rows: int = 8,
    n_cols: int = 16,
    pitch_y: int = 24,
    pitch_x: int = 20,
    gate_w: int = 8,
    gate_l: int = 4,
    gate_th: int = 4,
    oxide_th: int = 2,
    diff_th: int = 3,
    silicide_th: int = 1,
    well_margin: int = 10
) -> np.ndarray:
    """
    Generate a grid of MOSFETs (NMOS/PMOS alternating in Y) with:
    - shared diffusion
    - alternating n-well and p-well
    - gate, oxide, silicide
    - STI isolation between rows
    """
    out = vol.copy()
    nz, ny, nx = vol.shape
    cy = ny // 2
    cx = nx // 2

    ys, xs = np.ogrid[:ny, :nx]

    # Find top of substrate
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_sub = z_acc
            break
        z_acc += th

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = cy - (n_rows//2)*pitch_y + i*pitch_y
            x0 = cx - (n_cols//2)*pitch_x + j*pitch_x

            is_nmos = (i % 2 == 0)

            half_w = gate_w // 2
            gate_mask = (ys >= y0 - half_w) & (ys <= y0 + half_w) & \
                        (xs >= x0 - half_w) & (xs <= x0 + half_w)

            # Gate oxide
            for z in range(z_sub, min(z_sub + oxide_th, nz)):
                out[z][gate_mask] = CODES['Buried_SiO2']

            # Poly gate
            for z in range(z_sub + oxide_th, min(z_sub + oxide_th + gate_th, nz)):
                out[z][gate_mask] = CODES.get('Cu1', CODES['pad_contact'])

            # Diffusion left/right
            diff_mask_L = (ys >= y0 - half_w) & (ys <= y0 + half_w) & (xs < x0 - half_w)
            diff_mask_R = (ys >= y0 - half_w) & (ys <= y0 + half_w) & (xs > x0 + half_w)
            for z in range(z_sub + oxide_th + gate_th,
                           min(z_sub + oxide_th + gate_th + diff_th, nz)):
                out[z][diff_mask_L] = CODES['SOD']
                out[z][diff_mask_R] = CODES['SOD']

            # Silicide
            for z in range(z_sub + oxide_th + gate_th + diff_th,
                           min(z_sub + oxide_th + gate_th + diff_th + silicide_th, nz)):
                out[z][diff_mask_L] = CODES['Barrier']
                out[z][diff_mask_R] = CODES['Barrier']

            # Well region (entire row stripe)
            well_y0 = max(0, y0 - pitch_y//2 + well_margin)
            well_y1 = min(ny, y0 + pitch_y//2 - well_margin)
            well_mask = np.zeros((ny, nx), dtype=bool)
            well_mask[well_y0:well_y1, :] = True

            well_code = CODES['nwell'] if is_nmos else CODES['pwell']
            for z in range(z_sub, min(z_sub + 20, nz)):
                out[z][well_mask] = well_code

    # Add STI trenches between rows
    y_gaps = [cy - (n_rows // 2) * pitch_y + i * pitch_y for i in range(1, n_rows)]
    out = add_active_region_isolation(out, layers, y_positions=y_gaps, width=4)

    return out


#--------------------------------------------------
# 2) Planar layer generator
#--------------------------------------------------
def generate_planar_stack(shape, layers):
    """
    Stack planar layers into a 3D volume.
    """
    nz, ny, nx = shape
    vol = np.zeros((nz, ny, nx), dtype=np.uint8)
    z = 0
    for name, th in layers:
        code = CODES.get(name, 0)
        vol[z:z+th, :, :] = code
        z += th
    return vol

#--------------------------------------------------
# 3) Feature additions
#--------------------------------------------------

def add_STI_trenches(vol, layers, trench_width=10, trench_depth=6, code='STI'):
    """
    Carve parallel STI trenches into the Si substrate and fill them with oxide.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
        Must include 'Si_substrate' as the last entry.
    trench_width : int
        Width of each trench in voxels.
    trench_depth : int
        How many voxels deep into the substrate.
    code : str
        Material code key, default 'STI'.
    """
    nz, ny, nx = vol.shape

    # 1) Find start of substrate
    z_acc = 0
    for name, th in layers:
        z_acc += th
        if name == 'Si_substrate':
            z_sub_start = z_acc - th
            break

    # 2) Prepare 1D x‐mask for trenches (runs full y)
    half = trench_width // 2
    xs = np.arange(nx)
    # pick 3 trench centers evenly spaced
    centers = np.linspace(half, nx-half-1, 3, dtype=int)
    sticode = CODES[code]

    new_vol = vol.copy()
    for cx in centers:
        # boolean mask over x‐axis
        mask_x = (xs >= cx - half) & (xs <= cx + half)
        # carve + fill each z‐slice in trench depth
        for dz in range(trench_depth):
            z0 = z_sub_start + dz
            # fill entire column at these x's
            new_vol[z0, :, mask_x] = sticode

    return new_vol

def add_dielectric_layer(vol, layers, layer_name='PSG', code='PSG'):
    """
    Ensure the entire named dielectric layer is uniformly filled.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
        Your current 3D volume.
    layers : list of (name, thickness)
        Must include an entry named `layer_name`.
    layer_name : str
        The layer to fill (e.g. 'PSG' or 'SOD').
    code : str
        Material code key (should match CODES[code]).
    """
    nz, ny, nx = vol.shape

    # 1) Locate the z‐range for layer_name
    z0 = 0
    z_start = z_end = None
    for name, th in layers:
        if name == layer_name:
            z_start = z0
            z_end   = z0 + th
            break
        z0 += th

    if z_start is None:
        raise ValueError(f"Layer '{layer_name}' not found in stack.")

    # 2) Fill that slab
    psg_code = CODES[code]
    vol[z_start:z_end, :, :] = psg_code

    return vol

def add_metal_layer(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    layer_name: str,
    code_key: str,
    pattern: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Fill (or pattern) a metal layer.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
        Current 3D volume.
    layers : list of (name, thickness)
        Your stack (including substrate).
    layer_name : str
        Which layer to paint (e.g. 'Cu3', 'Cu4', 'Cu5').
    code_key : str
        Key in CODES for this metal (e.g. 'Cu3').
    pattern : callable or None
        If None, fill the entire slab.
        Otherwise, a function f(ys, xs) -> boolean mask of shape (ny, nx),
        indicating where interconnect metal should go.
    """
    nz, ny, nx = vol.shape

    # 1) Locate z‐start/z‐end for layer_name
    z0 = 0
    z_start = z_end = None
    for name, th in layers:
        if name == layer_name:
            z_start = z0
            z_end   = z0 + th
            break
        z0 += th
    if z_start is None:
        raise ValueError(f"Layer '{layer_name}' not found in layer stack.")

    metal_code = CODES[code_key]

    # 2) Build 2D mask
    ys, xs = np.mgrid[:ny, :nx]
    if pattern is None:
        mask2d = np.ones((ny, nx), dtype=bool)
    else:
        mask2d = pattern(ys, xs)

    # 3) Paint in each slice of that slab
    out = vol.copy()
    for z in range(z_start, z_end):
        slice2d = out[z]
        slice2d[mask2d] = metal_code
        out[z] = slice2d

    return out

def grid_pattern(ys, xs, pitch=32, width=4):
    """
    Returns True in narrow horizontal & vertical bars:
      - horizontal lines every `pitch` pixels
      - vertical lines every `pitch` pixels
    """
    hx = (ys % pitch) < width
    vx = (xs % pitch) < width
    return hx | vx

def add_barrier_layer(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    metal_layer: str,
    barrier_thickness: int = 2,
    code_key: str = 'Barrier'
) -> np.ndarray:
    """
    Insert a thin barrier layer immediately below `metal_layer`,
    matching the metal’s in-plane footprint exactly.
    """
    nz, ny, nx = vol.shape

    # 1) Find z-range for the named metal layer
    z_acc = 0
    z_mstart = None
    for name, th in layers:
        if name == metal_layer:
            z_mstart = z_acc
            z_mend = z_acc + th
            break
        z_acc += th
    if z_mstart is None:
        raise ValueError(f"Metal layer '{metal_layer}' not in layers list.")

    # 2) Barrier slab sits immediately below the metal
    z_bstart = max(0, z_mstart - barrier_thickness)
    z_bend = z_mstart

    # 3) Compute 2D mask of where that metal lives
    metal_code = CODES.get(metal_layer)
    if metal_code is None:
        raise KeyError(f"No code for metal layer '{metal_layer}' in CODES.")
    footprint = (vol[z_mstart] == metal_code)

    # 4) Paint the barrier
    out = vol.copy()
    barrier_val = CODES[code_key]
    for zz in range(z_bstart, z_bend):
        out[zz][footprint] = barrier_val

    return out

def add_passivation_layer(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    layer_name: str = 'SiN_passivation',
    code_key: str = 'SiN_passivation'
) -> np.ndarray:
    """
    Fill the entire passivation slab with SiN.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
        Must include an entry named `layer_name`.
    layer_name : str
        Name of the passivation layer in your stack.
    code_key : str
        Material code key in CODES.
    """
    # 1) Locate the z-range for the passivation layer
    z0 = 0
    z_start = z_end = None
    for name, th in layers:
        if name == layer_name:
            z_start = z0
            z_end   = z0 + th
            break
        z0 += th

    if z_start is None:
        raise ValueError(f"Passivation layer '{layer_name}' not found in stack.")

    # 2) Fill that entire slab
    vol[z_start:z_end, :, :] = CODES[code_key]
    return vol

def add_solder_pad_array(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    pad_diameter: int = 30,
    array_shape: tuple[int, int] = (4, 4),
    spacing: int = 64,
    pad_code: str = 'pad_contact',
    plug_code: str = 'W_plug',
    bump_code: str = 'solder_bump'
) -> np.ndarray:
    """
    Add a grid of soldering pads and bumps with plug vias beneath.
    """
    nz, ny, nx = vol.shape
    cy, cx = ny // 2, nx // 2
    out = vol.copy()

    # locate z-ranges
    z_pass, z_pad = 0, 0
    for name, th in layers:
        if name == 'SiN_passivation':
            z_pass = z_pad
        elif name == 'pad_contact':
            z_pad = z_pass + th
            break
        z_pass += th

    # bump height
    R = pad_diameter // 2
    bump_h = R

    # Extend volume for solder bumps
    extended = np.zeros((nz + bump_h, ny, nx), dtype=vol.dtype)
    extended[bump_h:] = out

    # Grid placement
    ys, xs = np.ogrid[:ny, :nx]
    offset_y = cy - (array_shape[0] // 2) * spacing + spacing // 2
    offset_x = cx - (array_shape[1] // 2) * spacing + spacing // 2

    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            py = offset_y + i * spacing
            px = offset_x + j * spacing

            mask = (ys - py)**2 + (xs - px)**2 <= R**2

            # Passivation cut + pad metal
            for z in range(z_pass, z_pass + 5):
                extended[bump_h + z][mask] = CODES[pad_code]
            # Tungsten plug
            for z in range(z_pad, z_pad + 4):
                extended[bump_h + z][mask] = CODES[plug_code]

            # Bump (hemisphere)
            for dz in range(bump_h):
                r = int(np.sqrt(R * R - dz * dz))
                if r <= 0: continue
                bump_mask = (ys - py)**2 + (xs - px)**2 <= r**2
                extended[bump_h - dz - 1][bump_mask] = CODES[bump_code]

    return extended

def add_underfill_and_rdl(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    underfill_th: int = 3,
    rdl_th: int = 2,
    underfill_code: str = 'underfill',
    rdl_code: str = 'Cu5'
) -> np.ndarray:
    """
    Adds a polymer underfill layer and an RDL metal layer just above solder bumps.
    - underfill fills the space between bumps
    - RDL is a thin metal routing layer over underfill
    """
    nz, ny, nx = vol.shape
    out = np.zeros((nz + underfill_th + rdl_th, ny, nx), dtype=vol.dtype)
    out[underfill_th + rdl_th:] = vol

    ys, xs = np.ogrid[:ny, :nx]

    # Identify bump positions
    bump_code = CODES['solder_bump']
    bump_mask = np.any(vol == bump_code, axis=0)

    # Fill underfill: all space not occupied by bumps
    for z in range(underfill_th, underfill_th + rdl_th):
        mask = ~bump_mask
        out[z][mask] = CODES[underfill_code]

    # RDL layer: fill everything (or add pattern later)
    out[:rdl_th] = CODES[rdl_code]

    return out



def add_via(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    via_diameter: int,
    center: tuple[int,int],
    start_layer: str,
    end_layer: str,
    via_code: str = 'Cu1'
) -> np.ndarray:
    """
    Drill and fill a cylindrical via between two named layers,
    regardless of which comes first in the layers list.
    """
    nz, ny, nx = vol.shape
    cy, cx = center

    # build 2D mask
    ys, xs = np.ogrid[:ny, :nx]
    r = via_diameter // 2
    mask = (ys - cy)**2 + (xs - cx)**2 <= r*r

    # first pass: find absolute z‐positions of both layers
    z_acc = 0
    z_start = z_end = None
    for name, th in layers:
        if name == start_layer:
            z_start = z_acc + th  # just above start_layer
        if name == end_layer:
            z_end = z_acc + th    # bottom of end_layer
        z_acc += th

    if z_start is None or z_end is None:
        raise ValueError(f"Via layers '{start_layer}' or '{end_layer}' not found in layers list.")

    # ensure z_start < z_end
    z_low, z_high = sorted((z_start, z_end))

    # carve + fill
    out = vol.copy()
    via_val = CODES[via_code]
    for z in range(z_low, z_high):
        out[z][mask] = via_val

    return out

def add_interconnects(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    start_layer: str,
    end_layer: str,
    grid_shape: tuple[int,int] = (2,2),
    pitch: tuple[int,int] = None,
    via_diameter: int = 8,
    via_code: str = 'Cu1'
) -> np.ndarray:
    """
    Place a regular grid of metal-filled vias connecting start_layer → end_layer.
    """
    nz, ny, nx = vol.shape
    # default pitch = evenly spread
    if pitch is None:
        pitch = (ny // (grid_shape[0] + 1), nx // (grid_shape[1] + 1))

    out = vol.copy()
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cy = pitch[0] * (i+1)
            cx = pitch[1] * (j+1)
            # *** call add_via positionally to avoid keyword clashes ***
            out = add_via(
                out,
                layers,
                via_diameter,   # third positional arg
                (cy, cx),       # fourth
                start_layer,    # fifth
                end_layer,      # sixth
                via_code        # seventh
            )
    return out

def add_FEOL_transistor_pair(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    channel_width: int = 20,
    channel_length: int = 4,
    poly_th: int = 4,
    gate_oxide_th: int = 2,
    spacing: int = 40,
    center_y: int | None = None,
    center_x: int | None = None
) -> np.ndarray:
    """
    Place two identical MOSFETs (inverter pair) separated by `spacing`.
    Each transistor has a channel stripe of width `channel_width`,
    gate oxide thickness `gate_oxide_th` and poly thickness `poly_th`.
    """
    nz, ny, nx = vol.shape
    cy = (ny//2) if center_y is None else center_y
    cx = (nx//2) if center_x is None else center_x

    out = vol.copy()

    # compute channel‐top z
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_ch = z_acc
            break
        z_acc += th

    # for each of two transistors
    for sign in (-1, +1):
        x_center = cx + sign*(spacing//2)
        half = channel_width//2

        # deposit gate oxide
        out[z_ch : z_ch + gate_oxide_th,
            :, :] = CODES['Buried_SiO2']

        # deposit poly gate stripe
        ys, xs = np.ogrid[:ny, :nx]
        mask = (ys >= cy-half) & (ys <= cy+half) & \
               (xs >= x_center-half) & (xs <= x_center+half)
        for z in range(z_ch + gate_oxide_th,
                       z_ch + gate_oxide_th + poly_th):
            out[z][mask] = CODES.get('Cu1', CODES['pad_contact'])

    return out

def add_FEOL_local_interconnect(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    layer_name: str = 'Cu1',
    code_key: str = 'Cu1',
    width: int = 4,
    y_positions: tuple[int,int] | None = None,
    z_layer_offset: int = 1
) -> np.ndarray:
    """
    Draw horizontal Cu1 wires connecting S/D contacts.

    - width: wire thickness in voxels
    - y_positions: y‐coordinates of the two traces
    - z_layer_offset: how many voxels into the Cu1 slab to draw
    """
    nz, ny, nx = vol.shape

    # 1) find the z‐index inside the Cu1 slab
    z_acc = 0
    z_wire = None
    for name, th in layers:
        if name == layer_name:
            z_wire = z_acc + z_layer_offset
            break
        z_acc += th
    if z_wire is None:
        raise ValueError(f"Layer '{layer_name}' not found.")

    # 2) default positions if none given
    ys = np.arange(ny)
    cy = ny // 2
    if y_positions is None:
        y_positions = (cy - (width+2), cy + (width+2))

    out = vol.copy()
    wire_val = CODES[code_key]

    # 3) paint each horizontal wire
    for y0 in y_positions:
        half = width // 2
        # compute row indices
        rows = np.arange(max(0, y0-half), min(ny, y0+half+1))
        for row in rows:
            out[z_wire, row, :] = wire_val

    return out

def add_FEOL_wellties_and_guard(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    ring_offset: int = 10,
    ring_width: int = 4,
    contact_diameter: int = 6,
    diffusion_code: str = 'SOD',
    plug_code: str = 'W_plug'
) -> np.ndarray:
    """
    Create a rectangular ring of source/drain diffusion around the cell edge,
    with inserted via‐contacts (W_plug) every `contact_diameter` pixels.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
    ring_offset : int
        Distance from cell boundary to inner edge of the ring.
    ring_width : int
        Thickness of the diffusion ring.
    contact_diameter : int
        Spacing/size of each contact in the ring.
    diffusion_code : str
        CODES key for diffusion (e.g. 'SOD').
    plug_code : str
        CODES key for tungsten plugs.
    """
    nz, ny, nx = vol.shape
    out = vol.copy()

    # 1) Find diffusion z-range just like S/D
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_ch = z_acc
        if name == 'SOD':
            # assume first SOD below Cu1 is diffusion layer
            z_diff_s, z_diff_e = z_acc, z_acc + th
            break
        z_acc += th

    # 2) Draw diffusion ring in that z‐range
    y0, y1 = ring_offset, ny - ring_offset
    x0, x1 = ring_offset, nx - ring_offset
    mask = np.zeros((ny, nx), bool)
    # top & bottom
    mask[y0 : y0+ring_width, x0:x1] = True
    mask[y1-ring_width:y1,   x0:x1] = True
    # left & right
    mask[y0:y1, x0 : x0+ring_width] = True
    mask[y0:y1, x1-ring_width:x1] = True

    for z in range(z_diff_s, z_diff_e):
        out[z][mask] = CODES[diffusion_code]

    # 3) Place via‐contacts along the ring every contact_diameter
    ys, xs = np.ogrid[:ny, :nx]
    r = contact_diameter // 2
    coords = []
    # sample points on the ring edges
    for y in range(y0 + ring_width//2, y1, contact_diameter):
        coords.append((y, x0 + ring_width//2))
        coords.append((y, x1 - ring_width//2))
    for x in range(x0 + ring_width//2, x1, contact_diameter):
        coords.append((y0 + ring_width//2, x))
        coords.append((y1 - ring_width//2, x))

    # find pad_contact & W_plug z‐ranges
    z_acc = 0
    for name, th in layers:
        if name == 'SiN_passivation':
            z_pass_s, z_pass_e = z_acc, z_acc + th
        if name == 'pad_contact':
            z_pad_s, z_pad_e = z_acc, z_acc + th
            break
        z_acc += th

    # carve & fill each contact
    for cy, cx in coords:
        cmask = (ys - cy)**2 + (xs - cx)**2 <= r*r
        for z in range(z_pass_s, z_pass_e):
            out[z][cmask] = CODES['pad_contact']
        for z in range(z_pad_s, z_pad_e):
            out[z][cmask] = CODES[plug_code]

    return out


#--------------------------------------------------
# FEOL: substrate‐level features
#--------------------------------------------------

def add_FEOL_USG_and_wells(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    usg_thickness: int = 4,
    well_depth: int = 20,
    pwell_region: tuple[slice,slice] | None = None,
    nwell_region: tuple[slice,slice] | None = None,
    pwell_code: str = 'pwell',
    nwell_code: str = 'nwell'
) -> np.ndarray:
    """
    1) Blanket‐deposit USG oxide at the bottom of the stack.
    2) Implant shallow wells (to well_depth) within the substrate.
    """
    nz, ny, nx = vol.shape

    # 1) USG at the very bottom
    vol[0:usg_thickness, :, :] = CODES['Buried_SiO2']

    # 2) Locate top of substrate
    z_acc = 0
    z_sub = None
    for name, th in layers:
        if name == 'Si_substrate':
            z_sub = z_acc
            break
        z_acc += th
    if z_sub is None:
        raise ValueError("Si_substrate layer not found.")

    ys, xs = np.ogrid[:ny, :nx]
    z_stop = min(z_sub + well_depth, nz)

    # p-well
    if pwell_region:
        y0,y1 = pwell_region[0].start, pwell_region[0].stop
        x0,x1 = pwell_region[1].start, pwell_region[1].stop
        mask = (ys>=y0)&(ys<y1)&(xs>=x0)&(xs<x1)
        for z in range(z_sub, z_stop):
            vol[z][mask] = CODES[pwell_code]

    # n-well
    if nwell_region:
        y0,y1 = nwell_region[0].start, nwell_region[0].stop
        x0,x1 = nwell_region[1].start, nwell_region[1].stop
        mask = (ys>=y0)&(ys<y1)&(xs>=x0)&(xs<x1)
        for z in range(z_sub, z_stop):
            vol[z][mask] = CODES[nwell_code]

    return vol

def add_FEOL_STI(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    trench_width: int = 8,
    trench_depth: int = 6,
    code_key: str = 'STI'
) -> np.ndarray:
    """
    Carve narrow trenches into the USG+substrate stack and fill with STI oxide.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
        Must include 'Si_substrate'.
    trench_width : int
        Width of each trench in voxels.
    trench_depth : int
        Depth from the USG/substrate interface down into substrate.
    code_key : str
        CODES key for the STI fill material.
    """
    nz, ny, nx = vol.shape
    half = trench_width // 2

    # 1) Find the USG/substrate interface z-index
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_iface = z_acc  # interface sits here
            break
        z_acc += th

    # 2) Choose trench centers (e.g. 3 trenches evenly spaced in x)
    xs = np.arange(nx)
    centers = np.linspace(half, nx-half-1, 3, dtype=int)

    sticode = CODES[code_key]
    out = vol.copy()

    # 3) For each center, carve & fill trench_depth voxels down
    for cx in centers:
        mask_x = (xs >= cx-half) & (xs <= cx+half)
        for dz in range(trench_depth):
            z0 = z_iface + dz
            out[z0, :, mask_x] = sticode

    return out

def add_FEOL_gate_stack(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    gate_oxide_th: int = 2,
    poly_th: int       = 4,
    gate_code: str     = 'Cu1',       # or your 'PolySi' alias
    channel_width: int = 20,
    channel_center: tuple[int,int] | None = None
) -> np.ndarray:
    """
    1) Deposit a thin gate-oxide slab on top of the channel region.
    2) Deposit a polysilicon (or metal) gate of specified width on top.
    """
    nz, ny, nx = vol.shape
    cy, cx = (ny//2, nx//2) if channel_center is None else channel_center

    # find the *start* of the substrate (top of Si_substrate)
    z_acc = 0
    z_chan_top = None
    for name, th in layers:
        if name == 'Si_substrate':
            z_chan_top = z_acc     # <-- correct top index
            break
        z_acc += th
    if z_chan_top is None:
        raise ValueError("Si_substrate layer not found.")

    out = vol.copy()

    # 1) Gate oxide deposition right above the channel
    out[z_chan_top : z_chan_top + gate_oxide_th, :, :] = CODES['Buried_SiO2']

    # 2) Polysilicon (gate) deposition: carve a stripe
    half = channel_width // 2
    ys, xs = np.ogrid[:ny, :nx]
    mask = (ys >= cy-half) & (ys <= cy+half) & (xs >= cx-half) & (xs <= cx+half)
    for z in range(z_chan_top + gate_oxide_th,
                   z_chan_top + gate_oxide_th + poly_th):
        out[z][mask] = CODES.get(gate_code, CODES['pad_contact'])

    return out

def add_FEOL_spacers(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    spacer_th: int = 2,
    gate_oxide_th: int = 2,
    poly_th: int = 4,
    channel_width: int = 20,
    channel_center: tuple[int,int] | None = None,
    spacer_code: str = 'Barrier'
) -> np.ndarray:
    """
    Grow lateral spacers on the sides of the polysilicon gate.

    - spacer_th: width of the spacer in voxels (on each side)
    - gate_oxide_th, poly_th: must match your gate‐stack call
    - channel_width, channel_center: same as gate‐stack
    """
    nz, ny, nx = vol.shape
    cy, cx = (ny//2, nx//2) if channel_center is None else channel_center

    # Locate top of substrate
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_chan_top = z_acc
            break
        z_acc += th

    # Define the vertical extent of the poly gate
    z_poly_start = z_chan_top + gate_oxide_th
    z_poly_end   = z_poly_start + poly_th

    # Create a 2D mask of the gate region
    ys, xs = np.ogrid[:ny, :nx]
    half = channel_width // 2
    mask_gate = (ys >= cy-half) & (ys <= cy+half) & (xs >= cx-half) & (xs <= cx+half)

    # Expand that mask outward by spacer_th to get the spacer region
    # (i.e. all pixels within channel_width/2 + spacer_th, minus the gate itself)
    outer = (ys >= cy-half-spacer_th) & (ys <= cy+half+spacer_th) \
          & (xs >= cx-half-spacer_th) & (xs <= cx+half+spacer_th)
    mask_spacer = outer & ~mask_gate

    # Paint spacer in every slice that the poly occupies
    out = vol.copy()
    for z in range(z_poly_start, z_poly_end):
        out[z][mask_spacer] = CODES[spacer_code]

    return out

def add_FEOL_source_drain_and_silicide(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    gate_oxide_th: int = 2,
    poly_th: int = 4,
    channel_width: int = 20,
    diffusion_th: int = 3,
    silicide_th: int = 1,
    channel_center: tuple[int,int] | None = None,
    diffusion_code: str = 'SOD',
    silicide_code: str = 'Barrier'
) -> np.ndarray:
    """
    1) Implant source/drain diffusions on either side of the gate.
    2) Cap the topmost diffusion voxels with silicide.
    """
    nz, ny, nx = vol.shape
    cy, cx = channel_center or (ny//2, nx//2)

    # 1) Find top-of-substrate index
    z_acc = 0
    for name, th in layers:
        if name == 'Si_substrate':
            z_ch = z_acc
            break
        z_acc += th

    # 2) Compute z ranges
    z_diff_start = z_ch + gate_oxide_th + poly_th
    z_diff_end   = z_diff_start + diffusion_th
    z_sil_start  = z_diff_end
    z_sil_end    = z_sil_start + silicide_th

    # 3) Build 2D masks for left & right S/D regions
    ys, xs = np.ogrid[:ny, :nx]
    half = channel_width // 2
    maskL = (ys >= cy - half) & (ys <= cy + half) & (xs < cx - half)
    maskR = (ys >= cy - half) & (ys <= cy + half) & (xs > cx + half)

    out = vol.copy()
    diff_val    = CODES[diffusion_code]
    silicide_val= CODES[silicide_code]

    # 4) Fill diffusion
    for z in range(z_diff_start, z_diff_end):
        out[z][maskL] = diff_val
        out[z][maskR] = diff_val

    # 5) Cap with silicide
    for z in range(z_sil_start, z_sil_end):
        out[z][maskL] = silicide_val
        out[z][maskR] = silicide_val

    return out

def add_FEOL_contact_cut_and_plug(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    centers: list[tuple[int,int]],
    cut_diameter: int = 6,
    pad_code: str = 'pad_contact',
    plug_code: str = 'W_plug'
) -> np.ndarray:
    """
    Carve contact holes through the SiN passivation and fill:
      - the passivation slab with pad metal (pad_code)
      - the pad_contact slab with tungsten (plug_code)

    centers: list of (y,x) positions for each contact.
    """
    nz, ny, nx = vol.shape
    ys, xs = np.ogrid[:ny, :nx]
    r = cut_diameter // 2
    out = vol.copy()

    # locate passivation z-range
    z = 0
    for name, th in layers:
        if name == 'SiN_passivation':
            z_pass_s, z_pass_e = z, z + th
            break
        z += th

    # locate pad_contact z-range
    z = 0
    for name, th in layers:
        if name == 'pad_contact':
            z_pad_s, z_pad_e = z, z + th
            break
        z += th

    # carve and fill each contact
    for cy, cx in centers:
        mask = (ys - cy)**2 + (xs - cx)**2 <= r*r
        # fill passivation slab with pad metal
        for zz in range(z_pass_s, z_pass_e):
            out[zz][mask] = CODES[pad_code]
        # fill pad_contact slab with W_plug
        for zz in range(z_pad_s, z_pad_e):
            out[zz][mask] = CODES[plug_code]

    return out

def tile_unit_cell(cell: np.ndarray, grid: tuple[int,int] = (2,2)) -> np.ndarray:
    """
    Tile a single unit‐cell volume into a larger array.
    grid = (n_y, n_x) number of repeats in each in‐plane direction.
    """
    # only tile in Y and X; keep Z-depth the same
    return np.tile(cell, (1, grid[0], grid[1]))

def sample_FEOL_variations(seed: int = 42) -> dict:
    """
    Randomize key FEOL dimensions within realistic bounds.
    Returns a dict you can unpack into your add_FEOL_* calls.
    """
    rng = np.random.default_rng(seed)
    return {
        # trench depth varies ±2 voxels around 6
        'trench_depth': int(rng.integers(4, 9)),
        # gate oxide thickness 1–3
        'gate_oxide_th': int(rng.integers(1, 4)),
        # poly thickness 3–6
        'poly_th':       int(rng.integers(3, 7)),
        # well implant depth 15–25
        'well_depth':    int(rng.integers(15, 26)),
    }

def add_FEOL_thermal_vias_and_spreader(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    via_diameter: int = 12,
    grid_shape: tuple[int,int] = (3,3),
    plate_thickness: int = 2,
    via_code: str = 'Cu5',
    plate_code: str = 'Cu5',
    end_layer: str = 'Cu1'            # drill only down to this layer’s bottom
) -> np.ndarray:
    """
    1) Add a heat‐spreader plate beneath the bump.
    2) Drill a grid of thermal vias from plate down to just below `end_layer`.
    """
    nz, ny, nx = vol.shape
    cy, cx = ny//2, nx//2
    ys, xs = np.ogrid[:ny, :nx]
    r = via_diameter // 2

    # find the first bump slice
    z_base = next(z for z in range(nz) if np.any(vol[z]==CODES['solder_bump']))

    # find bottom of end_layer
    z_acc = 0
    z_end = None
    for name, th in layers:
        z_acc += th
        if name == end_layer:
            z_end = z_acc
            break
    if z_end is None:
        raise ValueError(f"End layer '{end_layer}' not found.")

    out = vol.copy()

    # heat‐spreader plate
    for dz in range(plate_thickness):
        z = z_base - dz - 1
        if z < 0:
            break
        out[z, :, :] = CODES[plate_code]

    # via grid under bump
    centers_y = np.linspace(cy - r, cy + r, grid_shape[0], dtype=int)
    centers_x = np.linspace(cx - r, cx + r, grid_shape[1], dtype=int)
    for cy_v in centers_y:
        for cx_v in centers_x:
            mask = (ys - cy_v)**2 + (xs - cx_v)**2 <= r*r
            # drill from plate bottom (z_base-plate_thickness) down to z_end
            for z in range(max(0, z_base-plate_thickness), z_end):
                out[z][mask] = CODES[via_code]

    return out


def apply_floorplan_regions(
    vol: np.ndarray,
    floorplan: dict[str, tuple[tuple[int,int], tuple[int,int], tuple[int,int]]],
    label_map: dict[str, str]
) -> np.ndarray:
    """
    Apply unique material codes to different chip regions based on floorplan.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    floorplan : dict of region -> ((z0,z1), (y0,y1), (x0,x1))
    label_map : dict of region -> material name (must exist in CODES)

    Returns
    -------
    Modified volume with regions filled with corresponding material.
    """
    out = vol.copy()
    for region, bounds in floorplan.items():
        if region not in label_map:
            continue
        z_rng, y_rng, x_rng = bounds
        code = CODES[label_map[region]]
        out[z_rng[0]:z_rng[1], y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]] = code
    return out


def populate_region_features(vol, layers, floorplan):
    """
    Apply region-specific micro-features to chip floorplan areas.
    
    Parameters:
    -----------
    vol : np.ndarray
        The full chip volume (z, y, x).
    layers : list of (name, thickness)
        Layer stack.
    floorplan : dict
        Dictionary of region name -> ((z0,z1),(y0,y1),(x0,x1)) bounds.

    Returns:
    --------
    Modified volume with regional features applied.
    """
    for name, bounds in floorplan.items():
        zrange, yrange, xrange = bounds
        region = vol[zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]]

        if name == 'core':
            region = generate_transistor_array(
                region, layers,
                n_rows=8, n_cols=16,
                pitch_y=24, pitch_x=20,
                gate_w=8, gate_l=4,
                gate_th=4, oxide_th=2,
                diff_th=3, silicide_th=1
            )

        elif name == 'sram':
            region = add_sram_macro_blocks(
                region, layers,
                layer_name='Cu3',
                block_size=(64, 64),
                pattern_pitch=4,
                pattern_type='checker',
                positions=None,
                code_key='Cu3'
            )

        elif name == 'periphery':
            region = add_io_periphery_routing(region, layers)

        elif name == 'analog':
            region = add_analog_block(
                region, layers,
                region=((0, region.shape[0]), (0, region.shape[1]), (0, region.shape[2]))
            )
        vol[zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]] = region

    return vol


#--------------------------------------------------
# # Defects and other features
#--------------------------------------------------


def add_structural_defects(
    vol: np.ndarray,
    layers: list[tuple[str, int]],
    codes: dict,
    defect_types=('void', 'open', 'short', 'particle'),
    rate=1e-5,
    affected_labels: list[int] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Realistic chip defects into specific material layers.

    Parameters
    ----------
    vol : ndarray
        The 3D volume of the chip.
    layers : list of (name, thickness)
        Layer definitions to identify z-ranges.
    codes : dict
        Material name to label code mapping.
    defect_types : tuple of str
        Types of defects: 'void', 'open', 'short', 'particle'.
    rate : float
        Fraction of voxels to be affected.
    affected_labels : list[int], optional
        Only apply defects to these material labels (e.g., [Cu1, Cu2, ...]).
        If None, affects all non-zero voxels.
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = vol.shape
    total_voxels = nz * ny * nx

    vol_out = vol.copy()

    # Mask where defects can be applied
    if affected_labels:
        mask = np.isin(vol, affected_labels)
    else:
        mask = vol > 0  # apply to any non-zero material

    indices = np.argwhere(mask)
    n_defects = int(len(indices) * rate)

    if n_defects == 0:
        return vol_out

    defect_inds = indices[rng.choice(len(indices), n_defects, replace=False)]

    for z, y, x in defect_inds:
        typ = rng.choice(defect_types)
        if typ == 'void':
            vol_out[z, y, x] = 0
        elif typ == 'open':
            # Remove local wire patch
            vol_out[z, max(0,y-1):y+2, max(0,x-1):x+2] = 0
        elif typ == 'short':
            # Bridge to a neighboring layer
            if z < nz - 1:
                vol_out[z+1, y, x] = vol[z, y, x]
        elif typ == 'particle':
            # Add random speckle (foreign contaminant)
            vol_out[z, y, x] = rng.integers(1, 255)

    return vol_out


#--------------------------------------------------
# 4) Main generator
#--------------------------------------------------

def generate_chip(
    shape=(300,256,256),
    base_diameter=200,
    via_specs=None
):
    # define initial layers stack:
    layers = [
        ('Buried_SiO2',     10),
        ('STI',              6),
        ('Barrier',          2),
        # Metal 1 (Cu1) with its dielectric
        ('Cu1',              8),
        ('SOD',              6),
        # Metal 2 (Cu2)
        ('Cu2',              8),
        ('SOD',              6),
        # Metal 3 (Cu3)
        ('Cu3',              8),
        ('SOD',              6),
        # Metal 4 (Cu4)
        ('Cu4',              8),
        ('SOD',              6),
        # Metal 5 (Cu5)
        ('Cu5',              8),
        # Passivation & contact
        ('PSG',              6),
        ('SiN_passivation',  5),
        ('pad_contact',      4)
    ]
    layers = add_substrate(shape, layers)
    # planar stack
    vol = generate_planar_stack(shape, layers)


    # 1. Unpack shape first
    nz, ny, nx = shape

    channel_width = 20

    rng_vars = sample_FEOL_variations(seed=42)

    # 1) USG + wells with varied well_depth
    vol = add_FEOL_USG_and_wells(
        vol, layers,
        usg_thickness=4,
        well_depth=rng_vars['well_depth'],
        pwell_region=(slice(60,140), slice(40,120)),
        nwell_region=(slice(160,240), slice(80,160)),
        pwell_code='pwell',
        nwell_code='nwell'
    )

    # 2. STI trenches & fill
    # FEOL: STI trenches
    vol = add_FEOL_STI(
        vol,
        layers,
        trench_width=8,
        trench_depth=rng_vars['trench_depth'],
        code_key='STI'
    )

    vol = generate_transistor_array(
        vol, layers,
        n_rows=8,
        n_cols=16,
        pitch_y=24,
        pitch_x=20,
        gate_w=8,
        gate_l=4,
        gate_th=4,
        oxide_th=2,
        diff_th=3,
        silicide_th=1
    )


    vol = add_well_taps(vol, layers, well_type='nwell', pitch=64)
    vol = add_well_taps(vol, layers, well_type='pwell', pitch=64)

    vol = route_local_interconnects(
        vol, layers,
        gate_layer='Cu1',
        via_code='Cu1',
        wire_pitch=20,
        wire_width=4,
        n_wires=8
    )
   
    # 3. Gate stack (oxide + polysilicon)
    # FEOL: gate stack
    vol = add_FEOL_gate_stack(
        vol,
        layers,
        gate_oxide_th=2,
        poly_th=rng_vars['poly_th'],
        gate_code='Cu1',            # or 'PolySi' if you define that
        channel_width=20,
        channel_center=(ny//2, nx//2)
    )

# next: spacers, source/drain, silicide, contacts…

    # 4. Spacers
    # FEOL: sidewall spacers
    vol = add_FEOL_spacers(
        vol,
        layers,
        spacer_th=2,
        gate_oxide_th=2,   # same as your gate‐oxide thickness
        poly_th=4,         # same as your poly thickness
        channel_width=20,
        channel_center=(ny//2, nx//2),
        spacer_code='Barrier'  # or define a SiC/SiN spacer code
    )

    # 5. Source/Drain & silicide
    # FEOL: source/drain + silicide
    vol = add_FEOL_source_drain_and_silicide(
        vol, layers,
        gate_oxide_th=2,
        poly_th=4,
        channel_width=20,
        diffusion_th=3,
        silicide_th=1,
        channel_center=(ny//2, nx//2),
        diffusion_code='SOD',
        silicide_code='Barrier'
    )

    # FEOL: local Cu1 interconnects
    vol = add_FEOL_local_interconnect(vol, layers,
                                    layer_name='Cu1',
                                    code_key='Cu1',
                                    width=4)


    # FEOL: well‐tie diffusion ring + guard‐ring contacts
    vol = add_FEOL_wellties_and_guard(
        vol, layers,
        ring_offset=10,
        ring_width=4,
        contact_diameter=8,
        diffusion_code='SOD',
        plug_code='W_plug'
    )


    # 6. Contact cuts & W-plug
    # FEOL: contact cuts & tungsten plugs at S/D
    sd_contacts = [
        (ny//2, nx//2 - channel_width//2),  # left S/D
        (ny//2, nx//2 + channel_width//2)   # right S/D
    ]
    vol = add_FEOL_contact_cut_and_plug(
        vol, layers,
        centers=sd_contacts,
        cut_diameter=6,
        pad_code='pad_contact',
        plug_code='W_plug'
    )


    #START BEOL
    # now add features in sequence:
    # after building vol = generate_planar_stack(...)
    vol = add_STI_trenches(vol, layers,
                        trench_width=10,
                        trench_depth=6,
                        code='STI')


    #A  dd dielectric layers
    vol = add_dielectric_layer(vol, layers, layer_name='PSG', code='PSG')
    vol = add_dielectric_layer(vol, layers, layer_name='SOD', code='SOD')
    vol = add_dielectric_layer(vol, layers, layer_name='SOD', code='SOD')



    vol = add_metal_layer(vol, layers, layer_name='Cu1', code_key='Cu1')
    vol = add_barrier_layer(vol, layers, metal_layer='Cu1', barrier_thickness=1)
    vol = add_metal_layer(vol, layers, layer_name='Cu2', code_key='Cu2')
    vol = add_barrier_layer(vol, layers, metal_layer='Cu2', barrier_thickness=1)
    vol = add_metal_layer(vol, layers, layer_name='Cu3', code_key='Cu3')
    vol = add_metal_layer(vol, layers, layer_name='Cu4', code_key='Cu4', 
                        pattern=lambda ys,xs: grid_pattern(ys, xs, pitch=32, width=4))
    vol = add_metal_layer(vol, layers, layer_name='Cu5', code_key='Cu5')


    vol = add_io_periphery_routing(vol, layers)

    vol = add_sram_macro_blocks(
        vol, layers,
        layer_name='Cu3',
        block_size=(64, 64),
        pattern_pitch=4,
        pattern_type='checker',  # or 'lines' / 'solid'
        positions=None,          # or manually set [(cy, cx), ...]
        code_key='Cu3'
    )

    vol = add_barrier_layer(vol, layers, metal_layer='Cu3', barrier_thickness=2)
    vol = add_barrier_layer(vol, layers, metal_layer='Cu4', barrier_thickness=2)
    vol = add_barrier_layer(vol, layers, metal_layer='Cu5', barrier_thickness=2)

    # Cu1→Cu2
    vol = add_interconnects(
        vol, layers,
        start_layer='Cu1',
        end_layer='Cu2',
        grid_shape=(3,3),
        via_diameter=8,
        via_code='Cu1'
    )

    # Cu2→Cu3
    vol = add_interconnects(
        vol, layers,
        start_layer='Cu2',
        end_layer='Cu3',
        grid_shape=(3,3),
        via_diameter=8,
        via_code='Cu2'
    )

    # Cu3→Cu4
    vol = add_interconnects(
        vol, layers,
        start_layer='Cu3',
        end_layer='Cu4',
        grid_shape=(3,3),
        via_diameter=8,
        via_code='Cu3'
    )

    # Cu4→Cu5
    vol = add_interconnects(
        vol, layers,
        start_layer='Cu4',
        end_layer='Cu5',
        grid_shape=(3,3),
        via_diameter=8,
        via_code='Cu4'
    )

    # Add power grid to Cu5
    vol = add_power_grid(
        vol, layers,
        layer_name='Cu5',
        stripe_width=8,
        spacing=64,
        code_key='Cu5'
    )

    vol = add_dummy_fills(vol, layers, layer_name='Cu2', code_key='Cu2')
    vol = add_dummy_fills(vol, layers, layer_name='Cu3', code_key='Cu3')
    vol = add_dummy_fills(vol, layers, layer_name='Cu4', code_key='Cu4')


    # Decoupling capacitors between Cu3 and Cu4
    vol = add_metal_metal_decaps(vol, layers,
        top='Cu4', bottom='Cu3',
        coverage=0.2, patch_size=6, pitch=32
    )


    vol = add_passivation_layer(vol, layers,
                                layer_name='SiN_passivation',
                                code_key='SiN_passivation')


    vol = add_solder_pad_array(
        vol, layers,
        pad_diameter=30,
        array_shape=(4, 4),
        spacing=64,
        pad_code='pad_contact',
        plug_code='W_plug',
        bump_code='solder_bump'
    )

    vol = add_underfill_and_rdl(
        vol, layers,
        underfill_th=3,
        rdl_th=2,
        underfill_code='underfill',
        rdl_code='Cu5'  # or custom 'RDL' label
    )


    if via_specs:
        for spec in via_specs:
            # expand spec dict to positional args for add_via
            vol = add_via(
                vol,
                layers,
                spec['via_diameter'],
                spec['center'],
                spec['start_layer'],
                spec['end_layer'],
                spec.get('via_code', 'Cu1')
            )


    vol = add_FEOL_thermal_vias_and_spreader(
        vol, layers,
        via_diameter=12,
        grid_shape=(3,3),
        plate_thickness=2,
        via_code='Cu5',
        plate_code='Cu5',
        end_layer='Cu1'
    )
    return vol, layers

if __name__ == '__main__':

    #CREATE UNIT CELL
    shape=(300,256,256)
    via_specs = [
        {'via_diameter':40, 'center':(shape[1]//2,shape[2]//2), 'start_layer':'PSG', 'end_layer':'pad_contact', 'via_code':'Cu3'}
    ]

    model, layers = generate_chip(
        shape=shape,
        base_diameter=200,
        via_specs = via_specs
    )
    logger.info(f"Generated model shape={model.shape}")


    # tile it 2×2 in‐plane
    tiled = tile_unit_cell(model, grid=(8,8))

    nz, ny, nx = tiled.shape

    floorplan = {
        'core':     ((0, nz),   (0, 1024),   (0, 1024)),
        'sram':     ((0, nz),   (0, 1024),   (1024, 2048)),
        'periphery':((0, nz),   (1024, 2048), (0, 1024)),
        'analog':   ((0, nz),   (1024, 2048), (1024, 2048)),
    }

    region_labels = {
        'core':     'Cu3',
        'sram':     'Cu2',
        'periphery':'Cu1',
        'analog':   'Barrier'
    }
    tiled = populate_region_features(tiled, layers, floorplan)

    tiled = add_structural_defects(
        tiled, layers, codes=CODES,
        defect_types=('void', 'short', 'particle'),
        rate=2e-5,
        affected_labels=[CODES['Cu1'], CODES['Cu2'], CODES['Cu3']],
        seed=123
    )

    logger.info(f"Tiled model shape={tiled.shape}")

    #SAVE DATA
    save_mzarr(
        data=tiled,
        codes=CODES,
        out_dir="data/chip_model.zarr",
        voxel_size=(0.25, 0.25, 0.1),
        n_scales=3,
        base_chunk=(64, 64, 64),
        logger=logger
    )
