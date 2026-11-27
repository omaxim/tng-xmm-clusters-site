"""EPIC active CCDs: image each camera on the CCDs its donor actually had.

Under ``instrument.active_ccds = 'donor'`` every camera images only the CCDs
that have exposure in the donor's own PPS exposure map (table
``$TNGXMM_DATA/tables/donor_active_ccds.csv``, built once per donor list by
:mod:`tngxmm.donors.build_active_ccds`), instead of the packaged detector masks
of obsid 0827340901 (2018), whose MOS1 lacks CCD6 (lost 2005-03-09) and CCD3
(lost 2012-12-11) whatever the donor's epoch.  The CCD footprints are the per-CCD
label maps of ``$TNGXMM_DATA/tables/epic_ccd_maps.fits`` on the 900 x 900
detector grid of :class:`~tngxmm.instrument.load_responses.CameraResponse`
(2.5" pixels): every active pixel of a packaged mask carries its CCD number,
and MOS1 CCD3 and CCD6, absent from the packaged MOS1 mask, carry the silicon
measured on the PPS exposure maps of the donors observed before 2005-03-09, as
does the outer stripe of MOS1 CCD4 (segment 14, :data:`MOS1_CCD4_STRIPE`), whose
columns the PPS exposure maps no longer expose since the 2012-12-11 event, absent from the
packaged mask too.  The
response of a donor (:func:`ccd_masked_response`) is a copy whose mask holds the
donor's CCDs, with its active pixels and a QPB normalisation scaled to the
imaged area (the readout windows of :mod:`tngxmm.instrument.readout_windows` are
applied on top).  Every mask lookup of the instrument stage and both exposure-map
constructions read ``resp.mask``, so events and EXPMAPs share one footprint.  A
camera whose CCD set equals the packaged mask keeps the packaged response
object, so a post-2012 donor gives the same products as ``'mask'`` (the packaged
mask).  A donor without a PPS exposure map falls back to the epoch rule
(:func:`date_rule_ccds`), recorded as such.

Geometry (:func:`ccd_label_map`): the CCDs are squares of 600 x 1.1" (MOS) in
the CCD-aligned frame of the readout windows (rotation ``pn_readout_tilt_deg``
of the mask axes), centred on the CCD centres of EMOS1/2_LINCOORD_0019 (mm in the
focal plane, 27.5" per mm) about the measured centre of the central CCD
(``readout_windows.MOS_CENTRAL_CCD_CENTRE_ARCMIN``), with the focal-plane Y axis
mirrored and rotated by 270 deg (MOS1) or 0 deg (MOS2) into the mask frame: the
orientation whose squares hold every packaged-mask pixel (1.000 vs 0.990 for the
unmirrored one) and whose MOS1 CCD3 and CCD6 squares are the two the packaged
mask lacks, CCD3 the one the 2005-2012 PPS maps still expose.  Every mask pixel
takes the CCD of the nearest square (Chebyshev distance).  The pn is twelve
cells of 64 columns x one CCD row (two rows split at the measured row boundary,
the column edges of CCD 4 from the small-window geometry); no donor lacks a pn
CCD (the donor table shows it), so the pn numbering is a convention and never
changes a product.

References
----------
XMM-Newton Users Handbook section 3.3 (EPIC CCD layout); XMM-Newton CCF
EMOS1_LINCOORD_0019, EMOS2_LINCOORD_0019 (CCD positions); ESA XMM-Newton news
"EPIC MOS1 CCD6" (2005-03-09, revolution 961) and "EPIC MOS1 event of revolution
2382" (CCD3, 2012-12-11); Turner et al. 2001 (EPIC-MOS); Strueder et al. 2001
(EPIC-pn).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: rules of ``instrument.active_ccds``
ACTIVE_CCD_RULES = ("mask", "donor")
CAMERAS = ("MOS1", "MOS2", "PN")
#: segment 14 of MOS1: the outer columns of CCD4 (a 6' x 2.6' stripe along its outer
#: edge), exposed on every PPS map before the 2012-12-11 event and on none after (the
#: Users Handbook records the raised noise of CCD4 after the event; the stripe is what the
#: PPS exposure maps show); part of CCD4, its own segment so that a donor images it
#: exactly when its own map exposes it
MOS1_CCD4_STRIPE = 14
#: imaging segments per camera: the CCD numbers (and the MOS1 CCD4 stripe)
CCDS = {"MOS1": (*range(1, 8), MOS1_CCD4_STRIPE), "MOS2": tuple(range(1, 8)),
        "PN": tuple(range(1, 13))}
#: segments the packaged masks (obsid 0827340901, 2018) image
MASK_CCDS = {"MOS1": (1, 2, 4, 5, 7), "MOS2": CCDS["MOS2"], "PN": CCDS["PN"]}
#: permanent losses (UTC): MOS1 CCD6 at revolution 961; CCD3 and the CCD4 stripe at
#: revolution 2382
CCD_LOSSES = {"MOS1": (("2005-03-09T01:30:00", 6), ("2012-12-11T06:51:00", 3),
                       ("2012-12-11T06:51:00", MOS1_CCD4_STRIPE))}

#: CCD centres [mm] in the focal plane relative to CCD1 (EMOS*_LINCOORD_0019, mean of
#: the two readout nodes)
MOS_CCD_CENTRES_MM = {
    "MOS1": {1: (0.0, 0.0), 2: (11.91, -24.65), 3: (24.13, -0.54), 4: (11.94, 23.73),
             5: (-11.73, 23.93), 6: (-24.04, -0.18), 7: (-11.87, -24.47)},
    "MOS2": {1: (0.0, 0.0), 2: (12.09, -24.49), 3: (24.37, -0.37), 4: (12.13, 23.87),
             5: (-12.03, 23.97), 6: (-24.14, -0.11), 7: (-12.22, -24.40)},
}
#: focal plane -> mask frame: Y mirrored, then rotated by this angle [deg] in the
#: CCD-aligned (q, s) frame
MOS_FP_ROTATION_DEG = {"MOS1": 270.0, "MOS2": 0.0}
MOS_ARCSEC_PER_MM = 27.5
MOS_CCD_SIDE_ARCMIN = 600 * 1.1 / 60.0
#: pixels farther than this from every CCD square [arcmin] carry no label
LABEL_TOLERANCE_ARCMIN = 1.0
#: pn: 64 columns of 4.1" per CCD; edges of CCD 4 (readout_windows.PN_SW_Q_RANGE_ARCMIN)
PN_CCD_WIDTH_ARCMIN = 64 * 4.1 / 60.0
#: pn cell numbers by column (increasing q) of the upper and lower CCD row
PN_ROW_NUMBERS = ((6, 5, 4, 1, 2, 3), (9, 8, 7, 10, 11, 12))

CCD_MAPS_NAME = "epic_ccd_maps.fits"
TABLE_NAME = "donor_active_ccds.csv"

_MAPS: dict = {}
_MASKED: dict = {}


def format_ccds(ccds: Any) -> str:
    """``'1,2,4,5,7'`` (sorted, comma-separated): the table and CCDLIST form."""
    return ",".join(str(int(c)) for c in sorted(set(int(c) for c in ccds)))


def parse_ccds(text: Any) -> tuple[int, ...]:
    """Inverse of :func:`format_ccds` (also accepts ';' and blanks)."""
    s = str(text).replace(";", ",").replace(" ", "")
    return tuple(sorted({int(x) for x in s.split(",") if x}))


@cites("xmm_mission")
def date_rule_ccds(inst: str, date_obs: str | None) -> tuple[int, ...]:
    """CCDs of ``inst`` working at ``date_obs`` (ISO UTC) by the permanent losses
    (:data:`CCD_LOSSES`); raises ``ValueError`` without a date (no epoch is assumed)."""
    if not date_obs or str(date_obs) in ("nan", "None"):
        raise ValueError(f"no observation date for the {inst} epoch rule")
    lost = {ccd for when, ccd in CCD_LOSSES.get(inst, ()) if str(date_obs) >= when}
    return tuple(c for c in CCDS[inst] if c not in lost)


def ccd_frame_arcmin(shape: tuple[int, int], cx: float, cy: float, pixsize: float,
                     tilt_deg: float):
    """(q, s) [arcmin from the detector centre] of every pixel of the mask grid in the
    CCD-aligned frame (``readout_windows.ccd_frame`` in arcmin)."""
    import numpy as np

    a = np.deg2rad(tilt_deg)
    yy, xx = np.indices(shape)
    dx, dy = (xx - cx) * pixsize, (yy - cy) * pixsize
    return dx * np.cos(a) - dy * np.sin(a), dx * np.sin(a) + dy * np.cos(a)


def mos_focal_to_frame_arcmin(inst: str, x_mm: float, y_mm: float,
                              arcsec_per_mm: float = MOS_ARCSEC_PER_MM) -> tuple[float, float]:
    """(q, s) [arcmin, CCD-aligned frame of the mask] of the focal-plane point (``x_mm``, ``y_mm``)
    measured from the centre of the central CCD (CCD1) of MOS camera ``inst``: the focal plane is
    mirrored in Y, rotated by :data:`MOS_FP_ROTATION_DEG` and set on the measured CCD1 centre."""
    import math

    from tngxmm.instrument.readout_windows import MOS_CENTRAL_CCD_CENTRE_ARCMIN

    qc, sc = MOS_CENTRAL_CCD_CENTRE_ARCMIN[inst]
    r = math.radians(MOS_FP_ROTATION_DEG[inst])
    k = arcsec_per_mm / 60.0
    y_mm = -y_mm                                         # the focal plane is mirrored
    return (qc + k * (x_mm * math.cos(r) - y_mm * math.sin(r)),
            sc + k * (x_mm * math.sin(r) + y_mm * math.cos(r)))


def mos_ccd_centres_arcmin(inst: str) -> dict[int, tuple[float, float]]:
    """CCD centres [arcmin, (q, s)] of a MOS camera in the mask frame."""
    return {c: mos_focal_to_frame_arcmin(inst, x, y) for c, (x, y) in MOS_CCD_CENTRES_MM[inst].items()}


@cites("xmm_uhb", "xmm_ccf_sas", "epic_mos", "epic_pn")
def ccd_label_map(inst: str, shape: tuple[int, int], cx: float, cy: float, pixsize: float,
                  tilt_deg: float, pn_half_s_px: float = 0.0, ccds: Any = None):
    """uint8 CCD number of every pixel of the mask grid (0: no CCD), from the
    geometry of the module docstring (no mask applied); MOS: ``ccds`` restricts the
    nearest-square assignment to those CCDs (the packaged MOS1 mask: no pixel of it
    goes to CCD3 or CCD6)."""
    import numpy as np

    q, s = ccd_frame_arcmin(shape, cx, cy, pixsize, tilt_deg)
    lab = np.zeros(shape, np.uint8)
    if inst in ("MOS1", "MOS2"):
        half = 0.5 * MOS_CCD_SIDE_ARCMIN
        cen = mos_ccd_centres_arcmin(inst)
        ids = np.array(sorted(cen if ccds is None else set(int(c) for c in ccds)))
        d = np.stack([np.maximum(np.abs(q - cen[c][0]), np.abs(s - cen[c][1])) for c in ids])
        best = np.argmin(d, axis=0)
        ok = np.min(d, axis=0) <= half + LABEL_TOLERANCE_ARCMIN
        lab[ok] = ids[best[ok]]
        return lab
    if inst == "PN":
        from tngxmm.instrument.readout_windows import PN_SW_Q_RANGE_ARCMIN

        q0 = PN_SW_Q_RANGE_ARCMIN[0] - 2 * PN_CCD_WIDTH_ARCMIN       # left edge of column 0
        x = (q - q0) / PN_CCD_WIDTH_ARCMIN
        tol = LABEL_TOLERANCE_ARCMIN / PN_CCD_WIDTH_ARCMIN
        col = np.clip(np.floor(x), 0, 5).astype(int)
        upper = s >= pn_half_s_px * pixsize
        inside = (x >= -tol) & (x < 6 + tol)
        for row, nums in ((True, PN_ROW_NUMBERS[0]), (False, PN_ROW_NUMBERS[1])):
            for k, n in enumerate(nums):
                lab[inside & (upper == row) & (col == k)] = n
        return lab
    raise ValueError(f"unknown camera {inst!r}")


def default_maps_path() -> Path:
    """``$TNGXMM_DATA/tables/epic_ccd_maps.fits`` (data package)."""
    from tngxmm.core.paths import data_asset

    return data_asset("tables") / CCD_MAPS_NAME


def default_table_path() -> Path:
    """``$TNGXMM_DATA/tables/donor_active_ccds.csv`` (data package)."""
    from tngxmm.core.paths import data_asset

    return data_asset("tables") / TABLE_NAME


@cites("xmm_ccf_sas")
def _check_package_copy(name: str, path: str) -> None:
    """:func:`tngxmm.instrument.donor_tables.check_table` of ``path`` when it is the data package's copy of ``name``
    (a file named explicitly is the caller's); the check remembers a pass until a file changes."""
    from tngxmm.instrument import donor_tables as DT

    if DT.is_package_copy(name, path):
        DT.check_table(name, path)


def read_ccd_maps(path: str | Path | None = None) -> dict[str, Any]:
    """``{camera: uint8 (900, 900) label map}`` of the CCD-map product (cached).  The package copy is read only when
    its provenance says it is current (:func:`tngxmm.instrument.donor_tables.check_table`)."""
    import numpy as np
    from astropy.io import fits

    p = str(path or default_maps_path())
    _check_package_copy("ccd_maps", p)
    if p not in _MAPS:
        with fits.open(p) as f:
            maps = {}
            for c in CAMERAS:
                a = np.asarray(f[c].data, np.uint8)
                a.setflags(write=False)
                maps[c] = a
        _MAPS[p] = maps
    return _MAPS[p]


_TABLES: dict = {}


def read_active_ccds(path: str | Path | None = None) -> Any:
    """The donor active-CCD table (``obsid`` as 10-digit strings; cached per path).  The package copy is read only
    when its provenance says it is current (:func:`tngxmm.instrument.donor_tables.check_table`)."""
    import pandas as pd

    p = str(path or default_table_path())
    _check_package_copy("active_ccds", p)
    if p not in _TABLES:
        _TABLES[p] = pd.read_csv(p, dtype={"obsid": str, **{f"ccds_{c.lower()}": str
                                                            for c in CAMERAS}})
    return _TABLES[p]


def package_camera_ccds(package: dict[str, Any], table: Any = None) -> dict[str, dict[str, Any]]:
    """The active CCDs of a donor package (``obsid`` / ``donor_id``) from the donor table
    (:func:`read_active_ccds` when ``table`` is None); a donor absent from the table
    gets the epoch rule of its ``date_obs`` (source ``date-rule``), and without a date
    the call fails."""
    obsid = str(package.get("obsid") or package.get("donor_id") or "").zfill(10)
    t = read_active_ccds() if table is None else table
    try:
        return donor_camera_ccds(t, obsid)
    except KeyError:
        date = package.get("date_obs")
        if not date:
            raise KeyError(f"donor {obsid}: not in the active-CCD table and no date_obs") from None
        return {c: {"ccds": date_rule_ccds(c, str(date)), "source": "date-rule"} for c in CAMERAS}


def donor_camera_ccds(table: Any, obsid: str) -> dict[str, dict[str, Any]]:
    """``{camera: {'ccds': (...), 'source': 'pps-expmap' | 'date-rule'}}`` of one donor
    from the table; raises ``KeyError`` when the donor is absent."""
    key = str(obsid).zfill(10)
    hit = table[table.obsid.astype(str).str.zfill(10) == key]
    if not len(hit):
        raise KeyError(f"donor {key} is not in the active-CCD table")
    r = hit.iloc[0]
    out = {}
    for c in CAMERAS:
        lc = c.lower()
        if str(r.get(f"ccds_{lc}", "")) in ("", "nan"):
            continue
        out[c] = {"ccds": parse_ccds(r[f"ccds_{lc}"]), "source": str(r[f"source_{lc}"])}
    return out


@cites("xmm_uhb", "xmm_simulator")
def ccd_masked_response(resp: Any, ccds: Any, labels: Any) -> Any:
    """``resp`` imaged on the CCDs ``ccds`` only (``labels``: the camera's label map):
    mask = label in ccds, its active pixels, the QPB spectrum scaled to the imaged
    area and ``ccds`` = the CCD list.  Returns ``resp`` itself when the mask equals
    the packaged one.  Cached (the object identity keys the downstream caches)."""
    import numpy as np

    ccds = tuple(sorted(set(int(c) for c in ccds)))
    mask = np.isin(labels, np.asarray(ccds, np.uint8)).astype(np.uint8)
    if np.array_equal(mask, resp.mask):
        return resp
    key = (id(resp), id(labels), ccds)
    if key in _MASKED:
        return _MASKED[key][1]
    if not mask.any():
        raise ValueError(f"{resp.inst}: no imaging pixel on CCDs {ccds}")
    ay, ax = np.nonzero(mask)
    frac = float(mask.sum()) / float(resp.mask.sum())
    out = dataclasses.replace(resp, mask=mask, active_x=ax, active_y=ay,
                              fwc_spec=np.asarray(resp.fwc_spec) * frac, ccds=ccds)
    _MASKED[key] = (resp, out)            # keep resp alive: its id is in the key
    return out


@cites("xmm_uhb")
def apply_active_ccds(resps: dict[str, Any], camera_ccds: dict[str, Any] | None,
                      rule: str = "donor", maps: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-camera responses on the donor's CCDs (``rule='donor'``; ``camera_ccds``:
    camera -> CCD list or ``{'ccds': ...}``) or unchanged (``'mask'``)."""
    if rule not in ACTIVE_CCD_RULES:
        raise ValueError(f"unknown active_ccds rule {rule!r}; known: {ACTIVE_CCD_RULES}")
    if rule == "mask":
        return dict(resps)
    if not camera_ccds:
        raise ValueError("active_ccds 'donor' needs the donor's CCD lists "
                         "(tngxmm donors active-ccds; donor_active_ccds.csv)")
    maps = read_ccd_maps() if maps is None else maps
    out = {}
    for c, r in resps.items():
        if c not in camera_ccds:
            raise ValueError(f"active_ccds 'donor': no CCD list for {c}")
        v = camera_ccds[c]
        out[c] = ccd_masked_response(r, v["ccds"] if isinstance(v, dict) else v, maps[c])
    return out


__cites__ = collect_cites(__name__)
