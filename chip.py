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

from logger import setup_custom_logger, log_exception
#from utils import save_mzarr


logger = setup_custom_logger("chipgen", lfname="chipgen.log")


def save_mzarr(data, codes, out_dir,
               voxel_size, n_scales, base_chunk, logger):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data
    datasets = []
    for lvl in range(n_scales):
        path   = str(lvl)
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(path, data=curr, chunks=chunks, compression="raw")
        scale = 2**lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale",       "scale": [scale]*3},
                {"type": "translation", "translation": [scale/2 - 0.5]*3},
            ]
        })
        curr = curr[::2, ::2, ::2]

    # Save original codes as-is for internal use
    f.attrs["lookup"]     = codes
    f.attrs["voxel_size"] = voxel_size

    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {
            "voxel_size": voxel_size
        }
    }

    # Add brain-style "materials" layout to the JSON
    materials = {
        str(v): {"alias": k} for k, v in codes.items()
    }

    with open(os.path.join(out_dir, "multiscale.json"), "w") as fj:
        json.dump({
            "multiscale": multiscale_meta,
            "materials": materials
        }, fj, indent=2)

    if logger:
        logger.info(f"Saved multiscale Zarr → {out_dir}")




#--------------------------------------------------
# 0) Load parameters
#--------------------------------------------------
# Load material codes from external JSON
with open("chip_codes.json", "r") as f:
    CODES = json.load(f)

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
    ys, xs = np.ogrid[:ny, :nx]
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

def add_pad_and_plug(
    vol: np.ndarray,
    layers: list[tuple[str,int]],
    pad_diameter: int,
    center: tuple[int,int] | None = None,
    pad_code: str = 'pad_contact',
    plug_code: str = 'W_plug'
) -> np.ndarray:
    """
    Create a circular contact pad & tungsten plug.

    - Carve through the passivation slab (SiN_passivation) and fill with pad metal.
    - In the pad_contact slab below, fill the same footprint with tungsten plug.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    layers : list of (name, thickness)
        Must include 'SiN_passivation' and 'pad_contact'.
    pad_diameter : int
        Diameter of the pad/plug in voxels.
    center : (y,x) or None
        Center position. Defaults to image center.
    pad_code : str
        Key in CODES for pad metal (e.g. 'pad_contact').
    plug_code : str
        Key in CODES for plug metal (e.g. 'W_plug').
    """
    nz, ny, nx = vol.shape
    cy, cx = (ny//2, nx//2) if center is None else center

    # build 2D mask
    ys, xs = np.ogrid[:ny, :nx]
    r = pad_diameter // 2
    mask = (ys - cy)**2 + (xs - cx)**2 <= r*r

    out = vol.copy()

    # 1) Find passivation z-range
    z0 = 0
    for name, th in layers:
        if name == 'SiN_passivation':
            z_pass_s = z0
            z_pass_e = z0 + th
            break
        z0 += th

    # carve pad through passivation
    pad_val = CODES[pad_code]
    out[z_pass_s:z_pass_e, :, :][ :, mask ] = pad_val

    # 2) Find pad_contact z-range
    z0 = 0
    for name, th in layers:
        if name == 'pad_contact':
            z_pad_s = z0
            z_pad_e = z0 + th
            break
        z0 += th

    # fill tungsten plug in pad_contact slab
    plug_val = CODES[plug_code]
    out[z_pad_s:z_pad_e, :, :][:, mask] = plug_val

    return out

def add_solder_bump(vol, base_diameter, code='solder_bump'):
    """
    Append a hemispherical solder bump on top of passivation.
    """
    # existing bump logic
    nz, ny, nx = vol.shape
    R = base_diameter//2
    H = R
    new_vol = np.zeros((nz+H, ny, nx), dtype=vol.dtype)
    new_vol[H:] = vol
    cy, cx = ny//2, nx//2
    ys, xs = np.ogrid[:ny, :nx]
    for dz in range(H):
        r = int(np.sqrt(R*R - dz*dz))
        z0 = H - dz - 1
        if r>0:
            mask = (ys-cy)**2 + (xs-cx)**2 <= r*r
            new_vol[z0][mask] = CODES[code]
        else:
            new_vol[z0, cy, cx] = CODES[code]
    return new_vol

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
      – the passivation slab with pad metal (pad_code)
      – the pad_contact slab with tungsten (plug_code)

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


#--------------------------------------------------
# # Defects and other features
#--------------------------------------------------

def add_random_defects(
    vol: np.ndarray,
    defect_rate: float = 1e-4,
    defect_code: int | None = 0,
    seed: int = 42
) -> np.ndarray:
    """
    Randomly introduce defects into the volume.

    Parameters
    ----------
    vol : ndarray (nz, ny, nx)
    defect_rate : float
        Fraction of voxels to corrupt (e.g. 1e-4 = 0.01%).
    defect_code : int or None
        Value to write into defected voxels (0 = void, or any CODES value).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    new_vol : ndarray
        Volume with random defects applied.
    """
    nz, ny, nx = vol.shape
    total = nz * ny * nx
    n_defects = int(total * defect_rate)
    rng = np.random.default_rng(seed)

    # randomly choose linear indices to defect
    inds = rng.choice(total, size=n_defects, replace=False)
    new_vol = vol.copy().ravel()
    fill_val = defect_code if defect_code is not None else 0
    new_vol[inds] = fill_val

    return new_vol.reshape((nz, ny, nx))


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

    nz, ny, nx = shape               # ← unpack here
    channel_width = 20



    layers = add_substrate(shape, layers)
    # planar stack
    vol = generate_planar_stack(shape, layers)


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

    vol = add_FEOL_transistor_pair(
        vol, layers,
        channel_width=20,
        channel_length=4,
        gate_oxide_th=2,
        poly_th=4,
        spacing=40
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

    vol = add_passivation_layer(vol, layers,
                                layer_name='SiN_passivation',
                                code_key='SiN_passivation')


    # now add contact pad & plug:
    vol = add_pad_and_plug(
        vol, layers,
        pad_diameter=50,
        center=(ny//2, nx//2),
        pad_code='pad_contact',
        plug_code='W_plug'
    )


    vol = add_solder_bump(vol, base_diameter)

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
    seed_defects = 44
    vol = add_random_defects(
        vol,
        defect_rate=5e-5,    # ~0.005% of voxels
        defect_code=0,        # carve voids
        seed=seed_defects             # use same seed for reproducibility
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

    # update shape and (if needed) layers metadata
    # shape_z unchanged, but y,x doubled
    nz, ny, nx = tiled.shape
    logger.info(f"Tiled model shape={tiled.shape}")

    #SAVE DATA
    save_mzarr(
        data=tiled,
        codes=CODES,
        out_dir="chip_model.zarr",
        voxel_size=(0.25, 0.25, 0.1),
        n_scales=3,
        base_chunk=(64, 64, 64),
        logger=logger
    )
