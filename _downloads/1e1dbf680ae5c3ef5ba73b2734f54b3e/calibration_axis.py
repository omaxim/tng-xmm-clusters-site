"""Telescope axis of the EPIC vignetting from the XMM calibration files.

SAS evaluates the mirror vignetting of a camera about its telescope (optical) axis, not about the pointing
direction.  The axis is calibrated, per camera, as the pixel where it crosses the focal plane: the ``OPTICS_X``,
``OPTICS_Y``, ``OPTICS_CCD`` and ``OPTICS_NODE`` parameters of the ``XMM_MISCDATA`` CCF (XMM-SOC-CAL-TN-0018 Sect. 4.5:
"the latest calibration of the position of the telescope axis (embedded in XMM MISCDATA 0020)").  The pointing is
the origin of the focal-plane (FOC) system of the ``E*_LINCOORD`` CCFs, which is not the telescope axis, so the
vignetting centre sits a fixed distance from the pointing in the detector frame.

:func:`telescope_axis_arcmin` turns that pixel into the offset of the vignetting centre from the mask centre, in the
mask frame and arcmin, which is what :func:`tngxmm.instrument.simulate_instrument.with_vignetting_axis` takes:

1. the raw pixel (``OPTICS_X``, ``OPTICS_Y``) of the axis CCD is placed in the focal plane [mm] with the position and
   Euler rotation of its readout node (``LINCOORD`` ``X0``, ``Y0``, ``EULER_*``, ``RAWX0``, ``RAWY0``) and the pixel
   size of ``MISCDATA`` (``MM_PER_PIXEL_X/Y``), and referred to the centre of that CCD;
2. the focal-plane offset [mm] is converted to arcmin with the ``MISCDATA`` plate scale (``PLATE_SCALE_X`` over
   ``MM_PER_PIXEL_X``) and set on the position of the CCD in the CCD-aligned frame (q across the pn columns, s along
   them) of the mask, the same registration that labels the CCDs
   (:func:`tngxmm.instrument.active_ccds.mos_focal_to_frame_arcmin` for the MOS; for the pn the focal plane is mirrored
   in X, CCD 4 spans the column range of :data:`tngxmm.instrument.readout_windows.PN_SW_Q_RANGE_ARCMIN` and the
   boundary of the two CCD rows, at focal Y = 0, is the one the response measured on the mask);
3. (q, s) is rotated by the readout tilt of the mask axes to (x, y).

Only the offset of the axis from the CCD centre comes from the CCF; the position of the CCD in the mask frame is the
frame registration that the CCD labels and readout windows already use (it does not involve any exposure-map
vignetting).

RGA: the two MOS telescopes carry a Reflection Grating Assembly that obscures part of the beam, so their vignetting
depends on azimuth (Users Handbook Sect. 3.2.2.2).  ``XRTn_XAREAEF`` ``VIGNETTING`` holds the factor against energy and
off-axis angle only and leaves the RGA out; the SAS calibration layer adds the RGA obscuration separately
(``CAL_getRGAObscure``, from ``RGSn_QUANTUMEF`` ``RGA_OBSCURATE``).  :mod:`tngxmm.instrument.rga_obscuration` computes
that term (switch ``mos_rga_vignetting = 'ccf'``; off by default).  Until release 1.1.3 the stage applied a dipole
fitted on the PPS exposure maps of 60 donors instead; it was removed at the integration of 2026-10-30 under the rule
against measured factors.  :func:`ccf_has_azimuthal_vignetting` tells whether an area file with an azimuthal axis
has appeared.

References
----------
XMM-SOC-CAL-TN-0018 Issue 3.14 Sect. 4.5; XMM-Newton CCF XMM_MISCDATA_0022, E*_LINCOORD, XRTn_XAREAEF; Users Handbook
Sects. 3.2.2.2 and 3.3.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

from tngxmm.cites import cites, collect_cites

#: camera -> instrument identifier of the CCF tables
CCF_INSTRUMENT = {"MOS1": "EMOS1", "MOS2": "EMOS2", "PN": "EPN"}
#: camera -> telescope whose area file holds its vignetting
CCF_TELESCOPE = {"MOS1": "XRT1", "MOS2": "XRT2", "PN": "XRT3"}
#: pn CCD that carries the telescope axis (MISCDATA OPTICS_CCD); the frame registration below is that of CCD 4
PN_AXIS_CCD = 4


def latest_ccf(ccf_dir: str | Path, stem: str) -> Path:
    """The highest-numbered ``<stem>_NNNN.CCF`` of ``ccf_dir`` (the release a fresh CIF would pick for the
    current epoch of these time-independent constituents); ``FileNotFoundError`` when there is none."""
    best = None
    for p in Path(ccf_dir).glob(f"{stem}_[0-9][0-9][0-9][0-9].CCF"):
        m = re.fullmatch(re.escape(stem) + r"_(\d{4})\.CCF", p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        raise FileNotFoundError(f"no {stem}_NNNN.CCF in {ccf_dir}")
    return best[1]


@functools.lru_cache(maxsize=None)
def _miscdata(path: str) -> dict:
    """MISCDATA table of ``path`` as {(INSTRUMENT_ID, PARM_ID): PARM_VAL}."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        d = hdul["MISCDATA"].data
        return {(str(i).strip(), str(p).strip()): float(v) for i, p, v in zip(d["INSTRUMENT_ID"], d["PARM_ID"],
                                                                            d["PARM_VAL"])}


@functools.lru_cache(maxsize=None)
def _lincoord(path: str) -> dict:
    """LINCOORD table of ``path`` as {(CCD_ID, NODE_ID): row dict}."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        d = hdul["LINCOORD"].data
        names = d.columns.names
        return {(int(r["CCD_ID"]), int(r["NODE_ID"])): {n: float(r[n]) for n in names} for r in d}


def _euler_matrix(phi: float, theta: float, psi: float):
    """Rotation Rz(phi) Rx(theta) Rz(psi) (radians) of the LINCOORD Euler angles."""
    import numpy as np

    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])
    return rz(phi) @ rx @ rz(psi)


@cites("xmm_cal_tn0018", "xmm_ccf_sas")
def optical_axis_focal_mm(camera: str, ccf_dir: str | Path) -> dict:
    """Where the telescope axis of ``camera`` crosses the focal plane, from the CCF in ``ccf_dir``.

    Returns ``{'ccd': axis CCD, 'x_mm', 'y_mm': focal-plane offset [mm] from the centre of that CCD (the mean of its
    readout nodes' RAW origins), 'arcsec_per_mm': plate scale, 'x_ccd_mm', 'y_ccd_mm': focal position of the CCD
    centre [mm], 'files': (MISCDATA, LINCOORD)}``."""
    import numpy as np

    inst = CCF_INSTRUMENT[camera]
    misc_path = latest_ccf(ccf_dir, "XMM_MISCDATA")
    lin_path = latest_ccf(ccf_dir, f"{inst}_LINCOORD")
    misc, lin = _miscdata(str(misc_path)), _lincoord(str(lin_path))

    def parm(name: str) -> float:
        return misc[(inst, name)]

    ccd = int(parm("OPTICS_CCD"))
    node = int(misc.get((inst, "OPTICS_NODE"), 0.0))
    row = lin[(ccd, node)]
    # pixel of the axis relative to the node's RAW origin, in focal-plane mm along the node's RAW axes
    raw = np.array([(parm("OPTICS_X") - row["RAWX0"]) * parm("MM_PER_PIXEL_X"),
                    (parm("OPTICS_Y") - row["RAWY0"]) * parm("MM_PER_PIXEL_Y"), 0.0])
    rot = _euler_matrix(row["EULER_PHI"], row["EULER_THETA"], row["EULER_PSI"])
    axis = np.array([row["X0"], row["Y0"]]) + (rot @ raw)[:2]
    nodes = [r for (c, _), r in lin.items() if c == ccd]
    centre = np.array([np.mean([r["X0"] for r in nodes]), np.mean([r["Y0"] for r in nodes])])
    return {"ccd": ccd, "x_mm": float(axis[0] - centre[0]), "y_mm": float(axis[1] - centre[1]),
            "arcsec_per_mm": parm("PLATE_SCALE_X") / parm("MM_PER_PIXEL_X"),
            "x_ccd_mm": float(centre[0]), "y_ccd_mm": float(centre[1]), "files": (misc_path.name, lin_path.name)}


_AXIS: dict = {}


@cites("xmm_cal_tn0018", "xmm_ccf_sas")
def telescope_axis_arcmin(camera: str, ccf_dir: str | Path, tilt_deg: float,
                          pn_row_boundary_s_arcmin: float | None = None) -> tuple[float, float]:
    """Offset (x, y) [arcmin, mask frame] of the vignetting centre of ``camera`` from the mask centre (the pointing).

    ``tilt_deg``: angle of the CCD-aligned frame to the mask axes (``InstrumentOptions.pn_readout_tilt_deg``);
    ``pn_row_boundary_s_arcmin``: s of the boundary between the two pn CCD rows (``resp.pn_half_s * resp.pixsize``),
    required for the pn.  Cached by (CCF files, camera, tilt, boundary)."""
    if camera not in CCF_INSTRUMENT:
        raise ValueError(f"unknown camera {camera!r}; known: {tuple(CCF_INSTRUMENT)}")
    key = (str(latest_ccf(ccf_dir, "XMM_MISCDATA")), str(latest_ccf(ccf_dir, f"{CCF_INSTRUMENT[camera]}_LINCOORD")),
           camera, float(tilt_deg), pn_row_boundary_s_arcmin)
    if key not in _AXIS:
        ax = optical_axis_focal_mm(camera, ccf_dir)
        k = ax["arcsec_per_mm"] / 60.0                                      # arcmin per mm
        if camera == "PN":
            from tngxmm.instrument.readout_windows import PN_SW_Q_RANGE_ARCMIN

            if ax["ccd"] != PN_AXIS_CCD:
                raise ValueError(f"the pn frame registration is that of CCD {PN_AXIS_CCD}, "
                                 f"the CCF axis is on {ax['ccd']}")
            if pn_row_boundary_s_arcmin is None:
                raise ValueError("pn_row_boundary_s_arcmin is required for the pn")
            q_ccd = 0.5 * (PN_SW_Q_RANGE_ARCMIN[0] + PN_SW_Q_RANGE_ARCMIN[1])    # q of the CCD 4 centre
            q = q_ccd - k * ax["x_mm"]                                      # the pn focal X runs against q
            s = pn_row_boundary_s_arcmin + k * (ax["y_ccd_mm"] + ax["y_mm"])  # focal Y = 0: the row boundary
        else:
            from tngxmm.instrument.active_ccds import mos_focal_to_frame_arcmin

            if ax["ccd"] != 1:
                raise ValueError(f"the MOS frame registration is that of CCD 1, the CCF axis is on {ax['ccd']}")
            q, s = mos_focal_to_frame_arcmin(camera, ax["x_mm"], ax["y_mm"], ax["arcsec_per_mm"])
        a = math.radians(tilt_deg)                                          # (q, s) -> mask (x, y)
        _AXIS[key] = (q * math.cos(a) + s * math.sin(a), -q * math.sin(a) + s * math.cos(a))
    return _AXIS[key]


def ccf_has_azimuthal_vignetting(ccf_dir: str | Path, camera: str) -> bool:
    """Whether the vignetting table of the camera's telescope depends on azimuth: ``VIGNETTING_FACTOR`` of the
    ``VIGNETTING`` extension of ``XRTn_XAREAEF`` is, per energy, a vector over off-axis angle only (False) or
    carries a further axis (True).  A camera whose area file is absent is refused."""
    from astropy.io import fits

    with fits.open(latest_ccf(ccf_dir, f"{CCF_TELESCOPE[camera]}_XAREAEF")) as hdul:
        return hdul["VIGNETTING"].data["VIGNETTING_FACTOR"].ndim > 2          # (energy, angle): 2-D


__cites__ = collect_cites(__name__)
