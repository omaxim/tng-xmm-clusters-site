"""Azimuthal modulation of the MOS vignetting by the Reflection Grating Assembly, as the SAS calibration layer computes it.

The two MOS telescopes (XRT1, XRT2) carry an RGA behind the mirror module.  The fraction of the beam it removes from the
EPIC focus is the CCF quantity ``RGA_OBSCURATE`` (extension of ``RGSn_QUANTUMEF``), tabulated against energy for an
on-axis source.  The SAS routine ``CAL_getRGAObscure`` scales it for an off-axis source in the way
``CAL_getRGAIntercept`` does (Calibration Access and Data Handbook, sections on both routines, M. Smith 2011):

    eta'(E, theta, phi) = eta(E) alpha / alpha0,    alpha = alpha0 - (F / L) theta cos(phi)

with alpha0 the on-axis grating incidence angle, F the focal length of the mirror module, L the distance between the
RGA and the RGS focal camera, theta the off-axis angle and phi its azimuth in TELCOORD, counted from +Z.  The MOS
throughput relative to its on-axis value is therefore

    (1 - eta') / (1 - eta) = 1 + a(E) theta cos(phi),    a(E) = eta / (1 - eta) * F / (L alpha0),

a dipole about the telescope axis whose amplitude follows eta(E).  Every number comes from a calibration file:

* ``eta(E)``: ``RGA_OBSCURATE`` of ``RGS1_QUANTUMEF`` (MOS1, telescope XRT1) and ``RGS2_QUANTUMEF`` (MOS2, XRT2);
* ``alpha0``: ``G_ALPHA`` of ``RGSn_LINCOORD`` (1.576191 deg; ``INCIDENCE_ANGLE`` of ``XMM_MISCDATA`` holds the same);
* ``L``: ``CC2_TX - G_TX`` of ``RGSn_LINCOORD``, the X distance [mm] from the grating point to the RGS camera origin
  (both measured along the telescope X axis; the telescope frame origin has ``MM_TX = 0``);
* ``F``: ``FOCAL_LENGTH`` of XRT1 / XRT2 in ``XMM_MISCDATA``.

Direction.  The RGS dispersion axis is the spacecraft -Z axis, and the position angle of an observation is the angle
of that axis from celestial North through East (XMM-Newton Users Handbook, Sect. 3.4 on the RGS and the definition of
the position angle).  TELCOORD +X points along the incoming photons, so a source displaced on the sky towards -Z sends
photons with a +Z component: its azimuth is phi = 0, the incidence angle on the gratings is smaller and the RGA
obscures less.  The stage's mask frame is the sky frame of the observation whose masks the package holds (PA_PNT =
:data:`tngxmm.donors.build_core_offsets.PA_REF_DEG`, x to the West, y to the North); a position angle PA points at the
mask-frame angle 90 deg + PA, so the direction of maximum throughput is phi0 = 90 deg + PA_REF in the mask frame, for
both MOS cameras.  The donors' PPS exposure maps, which SAS ``eexpmap`` makes with this calibration, show a dipole at
-157.0 and -156.8 deg (MOS1, MOS2; measured on 60 donors before release 1.2); the value here, -155.8 deg, is not fitted
to them.

References
----------
Calibration Access and Data Handbook (XMM-PS-GM-20), CAL_getRGAIntercept (Procedure, Eq. 28) and CAL_getRGAObscure;
XMM-Newton CCF RGS1_QUANTUMEF_0015, RGS2_QUANTUMEF_0016, RGS1_LINCOORD_0008, RGS2_LINCOORD_0008, XMM_MISCDATA_0022;
XMM-Newton Users Handbook Sects. 3.2.2.2 and 3.4.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: values of InstrumentOptions.mos_rga_vignetting
MOS_RGA_RULES = ("off", "ccf")
#: camera -> (RGS unit behind the same mirror module, telescope of XMM_MISCDATA)
RGA_UNIT = {"MOS1": ("RGS1", "XRT1"), "MOS2": ("RGS2", "XRT2")}
ARCMIN_RAD = math.pi / (180.0 * 60.0)


def ccf_file(ccf_dir: str | Path, stem: str) -> Path:
    """The highest-numbered ``<stem>_NNNN.CCF`` of ``ccf_dir``, else of the site directory ``TNGXMM_CCF_EXTRA``
    (CCF constituents the stage's CCF directory does not hold, such as the RGS files)."""
    from tngxmm.core import site
    from tngxmm.instrument.calibration_axis import latest_ccf

    tried = []
    for d in (ccf_dir, site.get("TNGXMM_CCF_EXTRA") or ""):
        if not d or not Path(d).is_dir():
            continue
        tried.append(str(d))
        try:
            return latest_ccf(d, stem)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no {stem}_NNNN.CCF in {tried or [str(ccf_dir)]} (CCF directory and $TNGXMM_CCF_EXTRA)")


@functools.lru_cache(maxsize=None)
def _read(qe_path: str, lin_path: str, misc_path: str, telescope: str) -> dict[str, Any]:
    import numpy as np
    from astropy.io import fits

    from tngxmm.instrument.calibration_axis import _miscdata

    with fits.open(qe_path) as h:
        d = h["RGA_OBSCURATE"].data
        e_kev = np.asarray(d["ENERGY"], float) / 1000.0
        eta = np.asarray(d["FRACTION"], float)
    keep = np.r_[True, np.diff(e_kev) > 0]                 # the table repeats its last row
    with fits.open(lin_path) as h:
        hd = h["LINCOORD"].header
        alpha0 = math.radians(float(hd["G_ALPHA"]))
        length = float(hd["CC2_TX"]) - float(hd["G_TX"])
    focal = _miscdata(misc_path)[(telescope, "FOCAL_LENGTH")]
    return {"energy_kev": e_kev[keep], "eta": eta[keep], "alpha0_rad": alpha0, "l_mm": length, "f_mm": focal,
            "files": (Path(qe_path).name, Path(lin_path).name, Path(misc_path).name)}


@cites("xmm_calhb", "xmm_ccf_sas")
def rga_parameters(ccf_dir: str | Path, camera: str) -> dict[str, Any]:
    """Calibration inputs of the RGA modulation of ``camera`` (MOS1 or MOS2): ``energy_kev`` and ``eta`` (the on-axis
    obscured fraction), ``alpha0_rad``, ``l_mm``, ``f_mm`` and the CCF ``files`` read."""
    if camera not in RGA_UNIT:
        raise ValueError(f"camera {camera!r} has no RGA; known: {tuple(RGA_UNIT)}")
    rgs, xrt = RGA_UNIT[camera]
    return _read(str(ccf_file(ccf_dir, f"{rgs}_QUANTUMEF")), str(ccf_file(ccf_dir, f"{rgs}_LINCOORD")),
                 str(ccf_file(ccf_dir, "XMM_MISCDATA")), xrt)


def dipole_amplitude(params: dict[str, Any], energy_kev: Any) -> Any:
    """a(E) [1 / arcmin] = eta / (1 - eta) F / (L alpha0) of :func:`rga_parameters` at ``energy_kev`` (eta
    interpolated linearly in energy, held at the end values of the table)."""
    import numpy as np

    eta = np.interp(np.asarray(energy_kev, float), params["energy_kev"], params["eta"])
    return eta / (1.0 - eta) * params["f_mm"] / (params["l_mm"] * params["alpha0_rad"]) * ARCMIN_RAD


def dispersion_angle_deg(pa_ref_deg: float | None = None) -> float:
    """Mask-frame angle [deg, counter-clockwise from +x] of the sky direction of the spacecraft -Z axis (the RGS
    dispersion axis), the direction of maximum MOS throughput: 90 + PA_REF, folded to (-180, 180]."""
    if pa_ref_deg is None:
        from tngxmm.donors.build_core_offsets import PA_REF_DEG

        pa_ref_deg = PA_REF_DEG
    a = (90.0 + float(pa_ref_deg)) % 360.0
    return a - 360.0 if a > 180.0 else a


@cites("xmm_calhb", "xmm_ccf_sas", "xmm_uhb")
def camera_dipole(ccf_dir: str | Path, camera: str, energy_kev: Any) -> tuple[Any, float]:
    """(a(E) [1 / arcmin] on ``energy_kev``, phi0 [deg, mask frame]) of a MOS camera."""
    return dipole_amplitude(rga_parameters(ccf_dir, camera), energy_kev), dispersion_angle_deg()


def factor(a: Any, phi0_deg: float, dx_arcmin: Any, dy_arcmin: Any) -> Any:
    """Throughput factor 1 + a (dx cos phi0 + dy sin phi0) at offsets (dx, dy) [arcmin, mask frame] from the telescope
    axis; ``a`` broadcasts against the offsets (a scalar, or one value per energy on a trailing axis)."""
    import numpy as np

    c, s = math.cos(math.radians(phi0_deg)), math.sin(math.radians(phi0_deg))
    return 1.0 + np.asarray(a, float) * (np.asarray(dx_arcmin, float) * c + np.asarray(dy_arcmin, float) * s)


__cites__ = collect_cites(__name__)
