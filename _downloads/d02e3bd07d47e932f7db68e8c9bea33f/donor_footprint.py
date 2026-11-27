"""EPIC detector footprint of a view: the packaged mask, or the donor's own PPS exposure maps.

Under ``instrument.detector_footprint = 'mask'`` every camera images the 0/1
detector mask of xmm_simulator (the eexpmap map of obsid 0827340901, 2020, thresholded on
2.5" pixels), restricted to the donor's CCDs and readout window.  Measured against the donors
(195 donor / replica pairs, 0.5-1 keV EXPMAP, per camera): the thresholded mask carries that
one observation's bad columns at full depth and 2.5-5" width in every view, its chip gaps
without attitude smoothing, and it misses the silicon the donor's attitude track exposes
along every edge, so the mock footprint is 5-9 % smaller than the donor's for every view
and camera (and the merged ``exposure > 0.3 max`` area 7 % smaller).

Under ``'donor-pps'`` every camera images the efficiency map of its donor: the donor's PPS
exposure map (summed over the camera's exposures, 0.5-1 keV band ``2000``), moved into the
mask frame through the donor pointing and roll (the transform of
:func:`tngxmm.donors.build_active_ccds.expmap_in_mask_frame`, the inverse of the core-offset
transform), divided by the stage's own vignetting at the band centre (the energy SAS
``eexpmap`` evaluates the band at) and by its clean on-chip level:

    eff(p) = min(1, EXPMAP_donor(p) / (T_ref vig(theta_p, 0.75 keV)))

``eff`` is the fraction of the exposure time a sky position spent on good silicon: SAS
``eexpmap`` excludes the bad pixels of the event file's bad-pixel extension and accumulates
the instrument map over the attitude histogram (attitude bins of ``attrebin`` = 4"), so the
donor's own bad columns, CCDs, windows, chip gaps and their smearing by the attitude track
are in it.  The response keeps the stage's vignetting; ``eff`` multiplies it in the EXPMAPs
and thins every event (cluster, sky background, AGN, out-of-time, particle background) at its
detector pixel, so events and EXPMAPs share one footprint as before.  The mask becomes
``eff > 0``; the particle-background normalisation is the packaged per-pixel rate times the
pixels of that mask, thinned by ``eff``.  The CCD list and readout-window rules stay in force
for the headers; the footprint is the donor's map whatever they say.

The cluster sits at the donor's core offset in the same frame, so the gaps and bad columns
fall where they fell relative to the donor's cluster.  Registration of the donor maps on the
packaged masks (phase correlation of the exposed pixels, 60 donors): 0-2 mask pixels
(2.5-5"), 90th percentile 1-2 px.

References
----------
XMM-Newton SAS ``eexpmap`` documentation (bad pixels, attitude histogram, ``attrebin``,
band-centre energy); XMM-Newton Users Handbook section 3.3.1 (EPIC chip geometry); Rosen et
al. 2016 (3XMM: image data within the attitude GTI, frame shifts of the attitude solution).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: rules of ``instrument.detector_footprint``
FOOTPRINT_RULES = ("mask", "donor-pps")
#: the PPS band whose exposure map gives the footprint, and its energy range [keV]
FOOTPRINT_BAND = "2000"
FOOTPRINT_BAND_KEV = (0.5, 1.0)
#: the clean on-chip level is the median of EXPMAP / vig over pixels within this off-axis
#: angle [arcmin] that exceed :data:`LEVEL_FLOOR` of the maximum ratio
LEVEL_RADIUS_ARCMIN = 8.0
LEVEL_FLOOR = 0.5
CAMERA_CODES = {"MOS1": "M1", "MOS2": "M2", "PN": "PN"}

_EFF: dict = {}
_FOOT: dict = {}


def find_expmaps(obsid: str, inst: str, image_root: str | Path, band: str = FOOTPRINT_BAND) -> list[Path]:
    """Every PPS exposure map of one camera of one observation (all exposures, name order)."""
    return sorted(Path(image_root).glob(f"*/*/P{str(obsid).zfill(10)}{CAMERA_CODES[inst]}*EXPMAP{band}.FTZ"))


@cites("xmm_sas_eexpmap", "xmm_ccf_sas")
def efficiency_from_expmap(expmap_mask_frame: Any, vig: Any, theta_arcmin: Any,
                           level_radius_arcmin: float = LEVEL_RADIUS_ARCMIN,
                           level_floor: float = LEVEL_FLOOR) -> tuple[Any, float]:
    """(eff, T_ref): the donor's good-silicon fraction on the mask grid from its exposure map
    in the mask frame (``expmap_mask_frame`` [s]) and the stage's vignetting ``vig`` (1 on
    axis) and off-axis angle ``theta_arcmin`` on the same grid.

    T_ref is the median of EXPMAP / vig over the pixels within ``level_radius_arcmin`` whose
    ratio exceeds ``level_floor`` of the maximum (clean silicon; bad columns, gaps and the
    smeared edges are a minority there); eff = clip(EXPMAP / (T_ref vig), 0, 1).  Exposed
    pixels where the stage has no effective area (vig = 0, beyond the 15' vignetting grid)
    get eff = 1 (silicon the particle background reaches; the out-of-FOV rule drops their
    events as before)."""
    import numpy as np

    e = np.asarray(expmap_mask_frame, dtype=np.float64)
    v = np.asarray(vig, dtype=np.float64)
    ratio = np.where(v > 0, e / np.maximum(v, 1e-30), 0.0)
    inner = (v > 0) & (np.asarray(theta_arcmin) < level_radius_arcmin) & (ratio > 0)
    if not inner.any():
        raise ValueError("the donor exposure map has no exposure near the optical axis")
    top = float(ratio[inner].max())
    t_ref = float(np.median(ratio[inner & (ratio > level_floor * top)]))
    eff = np.clip(ratio / t_ref, 0.0, 1.0)
    eff[(v <= 0) & (e > 0)] = 1.0
    return eff.astype(np.float32), t_ref


@cites("xmm_sas_eexpmap", "xmm_ccf_sas")
def donor_efficiency(resp: Any, paths: list[Path], ra_pnt: float, dec_pnt: float, pa_pnt: float) -> Any:
    """The donor's efficiency map (float32, mask grid) of one camera from its PPS maps
    ``paths`` (summed) and pointing; cached per (camera, files, pointing)."""
    import numpy as np

    from tngxmm.donors.build_active_ccds import expmap_in_mask_frame
    from tngxmm.instrument.make_exposure_maps import vig_on_mask

    key = (resp.inst, tuple(str(p) for p in paths), float(ra_pnt), float(dec_pnt), float(pa_pnt),
           id(resp.arf_theta), id(resp.theta_idx_pix), getattr(resp, "rga", None) is not None)
    if key in _EFF:
        return _EFF[key]
    if not paths:
        raise FileNotFoundError(f"{resp.inst}: no donor PPS exposure map")
    e = sum(expmap_in_mask_frame(p, ra_pnt, dec_pnt, pa_pnt, resp.mask.shape, resp.cx, resp.cy,
                                 resp.pixsize) for p in paths)
    vig = vig_on_mask(resp, 0.5 * sum(FOOTPRINT_BAND_KEV))
    yy, xx = np.indices(resp.mask.shape)
    theta = np.hypot(xx - resp.cx, yy - resp.cy) * resp.pixsize
    eff, _ = efficiency_from_expmap(e, vig, theta)
    eff.setflags(write=False)
    _EFF[key] = eff
    return eff


@cites("xmm_sas_eexpmap", "xmm_simulator")
def footprint_response(resp: Any, base: Any, eff: Any, source: str = "donor-pps") -> Any:
    """``resp`` imaged on the efficiency map ``eff``: mask = eff > 0, its active pixels, the
    particle background of ``base`` (the packaged response) per packaged pixel times the new
    active pixels, and ``efficiency`` = eff.  Cached on the object identities."""
    import numpy as np

    key = (id(resp), id(base), id(eff))
    if key in _FOOT:
        return _FOOT[key][-1]
    mask = (np.asarray(eff) > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError(f"{resp.inst}: the donor footprint has no exposed pixel")
    ay, ax = np.nonzero(mask)
    frac = float(mask.sum()) / float(base.mask.sum())
    out = dataclasses.replace(resp, mask=mask, active_x=ax, active_y=ay,
                              fwc_spec=np.asarray(base.fwc_spec) * frac,
                              efficiency=eff, footprint=source)
    _FOOT[key] = (resp, base, eff, out)          # keep the keyed objects alive
    return out


@cites("xmm_sas_eexpmap", "xmm_uhb")
def apply_detector_footprint(resps: dict[str, Any], base: dict[str, Any], rule: str = "mask",
                             donor_obsid: str | None = None, pointing: Any = None,
                             image_root: str | Path | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """``(responses, {camera: footprint source})``: unchanged under ``'mask'``; under
    ``'donor-pps'`` every camera on its donor's efficiency map (module docstring).
    ``base``: the packaged responses (particle-background normalisation); ``pointing``:
    (ra_pnt, dec_pnt, pa_pnt) [deg] of the donor, default its row of the core-offset table;
    ``image_root``: the PPS image tree (default the site's)."""
    if rule not in FOOTPRINT_RULES:
        raise ValueError(f"unknown detector_footprint rule {rule!r}; known: {FOOTPRINT_RULES}")
    if rule == "mask":
        return dict(resps), {}
    if not donor_obsid:
        raise ValueError("detector_footprint 'donor-pps' needs the donor obsid")
    obsid = str(donor_obsid).zfill(10)
    if pointing is None:
        pointing = donor_pointing(obsid)
    ra, dec, pa = (float(x) for x in pointing)
    if not all(math.isfinite(x) for x in (ra, dec, pa)):
        raise ValueError(f"donor {obsid}: no pointing (RA_PNT, DEC_PNT, PA_PNT)")
    if image_root is None:
        from tngxmm.donors.build_core_offsets import DEFAULT_IMAGE_ROOT
        image_root = DEFAULT_IMAGE_ROOT
    out, src = {}, {}
    for c, r in resps.items():
        paths = find_expmaps(obsid, c, image_root)
        eff = donor_efficiency(base[c], paths, ra, dec, pa)
        out[c] = footprint_response(r, base[c], eff)
        src[c] = ",".join(p.name for p in paths)
    return out, src


def donor_pointing(obsid: str, table: Any = None) -> tuple[float, float, float]:
    """(RA_PNT, DEC_PNT, PA_PNT) [deg] of a donor from the core-offset table of the data
    package (``donor_core_offsets.csv``); raises ``KeyError`` when absent."""
    from tngxmm.donors.build_core_offsets import read_core_offsets

    t = read_core_offsets() if table is None else table
    hit = t[t.obsid.astype(str).str.zfill(10) == str(obsid).zfill(10)]
    if not len(hit):
        raise KeyError(f"donor {obsid} is not in the core-offset table (no pointing)")
    r = hit.iloc[0]
    return float(r.ra_pnt_deg), float(r.dec_pnt_deg), float(r.pa_pnt_deg)


__cites__ = collect_cites(__name__)
