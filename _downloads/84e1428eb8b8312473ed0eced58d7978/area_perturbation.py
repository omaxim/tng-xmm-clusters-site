"""Published effective-area differences of EPIC, applied to the generator only (closure arms of the calibration error).

A mock is generated and fitted with the same calibration, so its fit never meets an error of the EPIC effective area;
a real fit does.  A study arm multiplies the effective area of every camera (cluster, sky and AGN photons) by a
published energy-dependent difference, or its vignetting by a published off-axis accuracy, in one direction or the
other, and the fit keeps the nominal response and vignetting.  The
exposure maps and the background templates are those of the nominal response: they are SAS products of the
calibration, which knows nothing of the true area.

``instrument.area_perturbation`` names the model and the direction, ``"<model>:+1"`` or ``"<model>:-1"``; the factor
is f(E) for ``+1`` and 1 / f(E) for ``-1`` (for the vignetting model a function of the off-axis angle, g(theta) and
1 / g(theta)).  Models (:data:`AREA_MODELS`):

``acis-pn-s15``
    Schellenberger et al. (2015), Table A.3: the cubic spline of the stacked residuals ratio ACIS/PN (knots 0.54-6.55
    keV, value y and second derivative y'', normalised to unity at 1.1 keV), the energy-dependent difference of the
    Chandra ACIS and XMM-Newton EPIC-pn effective areas measured with the HIFLUGCS clusters.  The paper's rule: the
    effective area of one instrument multiplied by the spline gives the temperatures of the other, if that one is
    precisely calibrated.  The same factor is applied to the three EPIC cameras (the MOS cameras are tied to the pn by
    the MOS-to-pn correction CORRAREA of the current calibration).  Outside the knots the value of the end knot is
    held (1.01 below 0.54 keV, 1.16 above 6.55 keV; the paper finds the ratio flat above 3 keV).
``epic-nustar``
    The EPIC-to-NuSTAR effective-area correction of the XMM-Newton calibration (Fuerst 2022, XMM-CCF-REL-388; the
    ``ABSCORRAREA`` extension of ``XRT3_XAREAEF``, applied by SAS arfgen to the three cameras by default since SAS
    22.0, changes of 6-8 per cent between 3 and 12 keV, XMM-SOC-CAL-TN-0018 Sect. 4): read from the CCF of the stage,
    the highest issue of ``XRT3_XAREAEF`` that has the extension.
``vignetting-tn18``
    The vignetting, not the on-axis area: g(theta) = 1 + d theta / theta_d with d = 0.05 at theta_d = 10 arcmin
    (:data:`VIGNETTING_TN18`), the size the EPIC calibration status gives (XMM-SOC-CAL-TN-0018, Sect. 4.5, p. 13:
    with the current telescope-axis calibration "the differences in flux for offaxis sources for each camera are
    ~5%"), a residual flux difference of off-axis sources after the axis correction, read as a radially coherent
    error of that size at 10 arcmin (the conservative reading), the same factor at every energy and for the three
    cameras.  g is 1 on the axis, where the vignetting is 1 by definition (the on-axis area
    is the other models' term).  The document gives the size at one angle; linear in theta through it and continued
    beyond 10 arcmin, and the same at every energy, are declared choices (docs/assumptions/OPEN_ITEMS.md, section P).

References
----------
schellenberger2015 (Table A.3, Sect. 6.1 and 6.2; the spline is evaluated as the paper does, Press et al. 1992,
Chapter 3.3), xmm_cal_tn0018 (Sect. 2.2, 4 and 4.5), xmm_ccf_sas.
"""

from __future__ import annotations

import glob
import os
from typing import Any

from tngxmm.cites import cites, collect_cites

#: Schellenberger et al. (2015), Table A.3, column ACIS/PN: (energy [keV], y, y'') of the spline knots
ACIS_PN_S15 = (
    (0.54, 1.01, -14.69), (0.62, 1.04, -0.81), (0.71, 0.99, 0.02), (0.81, 0.96, 0.49), (0.94, 0.98, -0.06),
    (1.08, 1.00, 0.02), (1.24, 1.02, 0.07), (1.42, 1.06, -0.10), (1.63, 1.07, -0.08), (1.88, 1.08, 0.09),
    (2.15, 1.11, 0.03), (2.48, 1.15, -0.06), (2.84, 1.17, -0.09), (3.27, 1.15, 0.03), (3.76, 1.15, -0.01),
    (4.32, 1.15, 0.09), (4.96, 1.18, -0.17), (5.70, 1.13, 0.15), (6.55, 1.16, -0.17),
)

#: XMM-SOC-CAL-TN-0018 (Issue 3.14), Sect. 4.5, p. 13: the residual flux differences of off-axis sources for each
#: camera after the telescope-axis correction, ~5 per cent, taken at 10 arcmin: (d, theta_d [arcmin]) of the
#: vignetting model
VIGNETTING_TN18 = (0.05, 10.0)

#: the models of ``instrument.area_perturbation`` (see the module docstring)
AREA_MODELS = ("acis-pn-s15", "epic-nustar", "vignetting-tn18")
#: the models that change the vignetting (a factor in the off-axis angle) instead of the on-axis area
THETA_MODELS = ("vignetting-tn18",)


def parse(text: str | None) -> tuple[str, int] | None:
    """``(model, sign)`` of an ``area_perturbation`` value, or None for none (None or '')."""
    if text is None or str(text).strip() in ("", "none", "None"):
        return None
    model, sep, sign = str(text).strip().rpartition(":")
    if not sep or model not in AREA_MODELS or sign not in ("+1", "-1"):
        raise ValueError(f"area_perturbation {text!r}: '<model>:+1' or '<model>:-1' with model one of {AREA_MODELS}")
    return model, int(sign)


@cites("schellenberger2015")
def splint(knots: tuple, e_kev: Any) -> Any:
    """The cubic spline of ``knots`` ((x, y, y'') rows) at ``e_kev`` (the evaluation of Press et al. 1992, Sect. 3.3,
    with the tabulated second derivatives); outside the knots the end values are held."""
    import numpy as np

    x = np.array([k[0] for k in knots], dtype=float)
    y = np.array([k[1] for k in knots], dtype=float)
    y2 = np.array([k[2] for k in knots], dtype=float)
    e = np.asarray(e_kev, dtype=float)
    ec = np.clip(e, x[0], x[-1])
    hi = np.clip(np.searchsorted(x, ec, side="right"), 1, len(x) - 1)
    lo = hi - 1
    h = x[hi] - x[lo]
    a = (x[hi] - ec) / h
    b = (ec - x[lo]) / h
    return a * y[lo] + b * y[hi] + ((a ** 3 - a) * y2[lo] + (b ** 3 - b) * y2[hi]) * h ** 2 / 6.0


@cites("xmm_ccf_sas", "xmm_cal_tn0018")
def abscorrarea(ccf_dir: str, e_kev: Any) -> Any:
    """The EPIC-to-NuSTAR correction factor (``ABSCORRAREA`` of the highest ``XRT3_XAREAEF`` issue of ``ccf_dir``
    that has the extension) at ``e_kev``; outside its table the end values are held."""
    import numpy as np
    from astropy.io import fits

    files = sorted(glob.glob(os.path.join(str(ccf_dir), "XRT3_XAREAEF_*.CCF")),
                   key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]), reverse=True)
    for p in files:
        with fits.open(p) as f:
            if "ABSCORRAREA" in f:
                d = f["ABSCORRAREA"].data
                return np.interp(np.asarray(e_kev, dtype=float), np.asarray(d["ENERGY"], float) / 1e3,
                                 np.asarray(d["FACTOR"], float))
    raise FileNotFoundError(f"no XRT3_XAREAEF with an ABSCORRAREA extension in {ccf_dir}")


@cites("schellenberger2015", "xmm_cal_tn0018")
def factor(text: str | None, e_kev: Any, ccf_dir: str = "") -> Any:
    """The multiplicative factor of the effective area at ``e_kev`` for the ``area_perturbation`` value ``text``
    (ones for none)."""
    import numpy as np

    p = parse(text)
    e = np.asarray(e_kev, dtype=float)
    if p is None or p[0] in THETA_MODELS:
        return np.ones_like(e)
    model, sign = p
    f = splint(ACIS_PN_S15, e) if model == "acis-pn-s15" else abscorrarea(ccf_dir, e)
    return f if sign > 0 else 1.0 / f


@cites("xmm_cal_tn0018")
def theta_factor(text: str | None, theta_arcmin: Any) -> Any:
    """The multiplicative factor of the vignetting at off-axis angles ``theta_arcmin`` for the ``area_perturbation``
    value ``text``: g(theta) = 1 + d theta / theta_d (:data:`VIGNETTING_TN18`) for ``vignetting-tn18:+1``, 1 / g for
    ``-1``; ones for none and for the energy models."""
    import numpy as np

    p = parse(text)
    th = np.asarray(theta_arcmin, dtype=float)
    if p is None or p[0] not in THETA_MODELS:
        return np.ones_like(th)
    d, theta_d = VIGNETTING_TN18
    g = 1.0 + d * th / theta_d
    return g if p[1] > 0 else 1.0 / g


_PERTURBED: dict = {}


def perturbed_response(resp: Any, text: str | None, ccf_dir: str = "") -> Any:
    """``resp`` (a :class:`~tngxmm.instrument.load_responses.CameraResponse`) whose effective area ``arf_theta`` is
    multiplied by :func:`factor` on its energy grid (every off-axis angle alike) and by :func:`theta_factor` on its
    off-axis grid ``thetas`` (every energy alike); the response itself for none.  Cached by the identity of the
    response (object identities key downstream caches)."""
    import dataclasses

    import numpy as np

    if parse(text) is None:
        return resp
    key = (id(resp), str(text), str(ccf_dir))
    if key not in _PERTURBED:
        f = factor(text, np.asarray(resp.box_ene_mean), ccf_dir)
        g = theta_factor(text, np.asarray(resp.thetas))
        if len(_PERTURBED) >= 12:                       # a few views' cameras; the oldest goes first
            _PERTURBED.pop(next(iter(_PERTURBED)))
        _PERTURBED[key] = (resp, dataclasses.replace(resp, arf_theta=np.asarray(resp.arf_theta) * g[:, None]
                                                     * f[None, :]))
    return _PERTURBED[key][1]


__cites__ = collect_cites(__name__)
