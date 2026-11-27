"""EPIC effective area off axis, one model for the instrument stage and the fit.

The effective area of camera ``c`` at off-axis angle theta and energy E is

    A(theta, E) = A_0(E) V(theta, E)

with ``A_0`` the on-axis ARF of the fit (:func:`tngxmm.massfit.build_responses.onaxis_arf`: mirror area x
CORRAREA x filter x QE, x mos_areacorr for MOS, every curve of the CCF interpolated linearly) and ``V`` the
vignetting of the mirror module's ``XRT*_XAREAEF`` CCF (extension VIGNETTING, VIGNETTING_FACTOR tabulated on an
energy grid and on off-axis angles 0, D_THETA, 2 D_THETA, ...), interpolated linearly in energy and in angle.  The
instrument stage builds its off-axis ARF grid from :func:`effective_area_theta` (``instrument.arf_model =
'fit-linear'``), the fit vignettes its model with :func:`vignetting` (``massfit.vignetting_model = 'sub-band'``),
so the generator and the fitter fold the same calibration.  The stage's ARF grid is the RMF's energy bins.  The
upstream simulator (``xmm_simulator.utils.calc_arf``) samples the same curves at the centres of 20 eV bins: the bin
0.52-0.54 keV takes the area below the O-K edge (0.532 keV, filter and CCD) for all of it, 33 per cent too much for
MOS, so its MOS 0.5-1 keV area was 1.1-1.2 per cent high (pn 0.2 per cent).  Evaluated on the RMF's bins, calc_arf
equals the fit's on-axis ARF to 2e-4 in every band: the grid, not the interpolation, made the difference.

The three mirror modules share one vignetting table in the CCF used here (XRT1/2/3_XAREAEF, identical
VIGNETTING_FACTOR); each camera still reads its own file.

References
----------
xmm_ccf_sas (XRT*_XAREAEF, E*_QUANTUMEF, E*_FILTERTRANSX), xmm_uhb (XMM-Newton Users Handbook: the vignetting and
its energy dependence), xmm_simulator (the upstream ARF model this replaces).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: mirror module of each camera (the CCF prefix of its effective-area file)
XRT = {"PN": "XRT3", "MOS1": "XRT1", "MOS2": "XRT2"}
#: rules for the area beyond the stage's 15 arcmin (``instrument.vignetting_beyond``; load_responses.VIGNETTING_BEYOND)
BEYOND_RULES = ("zero", "ccf-edge")
#: off-axis angle [arcmin] beyond which ``'zero'`` gives no area (the upstream ARF_Box grid)
ZERO_BEYOND_ARCMIN = 15.0

_TABLES: dict = {}


def _newest(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"no CCF matching {pattern}")
    return sorted(files, key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]


@cites("xmm_ccf_sas")
def vignetting_table(camera: str, ccf_dir: str | Path, area_file: str | None = None) -> tuple[Any, Any, Any, str]:
    """(theta [arcmin], energy [keV], factor [n_energy, n_theta], file name) of the camera's newest XAREAEF CCF.
    Cached per (camera, directory)."""
    import numpy as np
    from astropy.io import fits

    key = (camera, str(ccf_dir), area_file)
    if key not in _TABLES:
        path = (os.path.join(str(ccf_dir), area_file) if area_file
                else _newest(os.path.join(str(ccf_dir), f"{XRT[camera]}_XAREAEF_*.CCF")))
        with fits.open(path) as f:
            v = f["VIGNETTING"].data
            ene = np.asarray(v["ENERGY"], dtype=np.float64).ravel() / 1e3
            fac = np.asarray(v["VIGNETTING_FACTOR"], dtype=np.float64)
            dth = float(f["VIGNETTING"].header["D_THETA"]) * 60.0
        fac = fac.reshape(len(ene), -1)
        order = np.argsort(ene)
        theta = np.arange(fac.shape[1]) * dth
        _TABLES[key] = (theta, ene[order], fac[order], os.path.basename(path))
    return _TABLES[key]


@cites("xmm_ccf_sas")
def vignetting(camera: str, theta_arcmin: Any, energy_kev: Any, ccf_dir: str | Path,
               beyond: str = "ccf-edge", area_file: str | None = None) -> Any:
    """V(theta, E): the CCF vignetting interpolated linearly in energy and off-axis angle (broadcast over the two
    arguments).  Beyond the last tabulated angle: ``'ccf-edge'`` holds it, ``'zero'`` gives 0 beyond
    :data:`ZERO_BEYOND_ARCMIN`.  Energies outside the table take its end rows."""
    import numpy as np

    if beyond not in BEYOND_RULES:
        raise ValueError(f"unknown rule beyond the vignetting table {beyond!r}; known: {BEYOND_RULES}")
    theta, ene, fac, _ = vignetting_table(camera, ccf_dir, area_file)
    th = np.asarray(theta_arcmin, dtype=np.float64)
    e = np.asarray(energy_kev, dtype=np.float64)
    th_b, e_b = np.broadcast_arrays(th, e)
    # linear in energy: bracketing rows and weights
    j = np.clip(np.searchsorted(ene, e_b, side="right") - 1, 0, len(ene) - 2)
    w = np.clip((e_b - ene[j]) / (ene[j + 1] - ene[j]), 0.0, 1.0)
    # linear in angle on the uniform grid
    thc = np.clip(th_b, 0.0, theta[-1])
    dth = theta[1] - theta[0]
    k = np.clip(np.floor(thc / dth).astype(np.int64), 0, len(theta) - 2)
    u = np.clip((thc - theta[k]) / dth, 0.0, 1.0)
    v_lo = fac[j, k] * (1.0 - u) + fac[j, k + 1] * u
    v_hi = fac[j + 1, k] * (1.0 - u) + fac[j + 1, k + 1] * u
    out = v_lo * (1.0 - w) + v_hi * w
    if beyond == "zero":
        out = np.where(th_b > ZERO_BEYOND_ARCMIN, 0.0, out)
    return out


@cites("xmm_simulator")
def rmf_energy_edges(camera: str, emin_kev: float = 0.1, emax_kev: float = 100.0,
                     rmf_src_dir: str | Path | None = None) -> Any:
    """Energy bin edges [keV] of the camera RMF the fit uses (``massfit.build_responses.rmf_grid``) from the first bin
    edge at or above ``emin_kev`` to the last, then one bin up to ``emax_kev`` (the stage gives it no area: XSPEC folds
    nothing beyond the RMF).  The stage's ARF grid under ``instrument.arf_model = 'fit-linear'``: a photon then reads
    the area of the RMF bin it falls in, as XSPEC does (the upstream 20 eV grid read the area at its bin centre, which
    put the O-K edge 10 eV off and made the MOS 0.5-1 keV area 1.1-1.2 per cent high)."""
    import numpy as np

    from tngxmm.massfit.build_responses import LEGACY_RMF_SRC, rmf_grid

    lo, hi = rmf_grid(camera, rmf_src_dir if rmf_src_dir is not None else LEGACY_RMF_SRC)
    edges = np.concatenate([lo, hi[-1:]])
    edges = edges[edges >= float(emin_kev) - 1e-9]
    if float(emax_kev) > edges[-1]:
        edges = np.concatenate([edges, [float(emax_kev)]])
    return edges


@cites("xmm_ccf_sas", "xmm_simulator")
def effective_area_theta(camera: str, filt: str, ene_lo: Any, ene_hi: Any, thetas: Any, ccf_dir: str | Path,
                         rmf_src_dir: str | Path | None = None, beyond: str = "ccf-edge", date: str | None = None,
                         rule: str = "highest-issue", area_file: str | None = None) -> Any:
    """A(theta, E) [cm^2] on the energy bins (``ene_lo``, ``ene_hi``) [keV] and the off-axis angles ``thetas``
    [arcmin], shape (n_theta, n_energy): the fit's on-axis ARF model at the bin centres times :func:`vignetting`.
    The one function behind the stage's ARF grid (``instrument.load_responses.get_response``, ``arf_model =
    'fit-linear'``) for the shared and for per-observation responses."""
    import numpy as np

    from tngxmm.massfit.build_responses import FILTERS, LEGACY_RMF_SRC, onaxis_arf

    if filt not in FILTERS:
        raise ValueError(f"filter {filt!r}: the fit's responses (massfit.build_responses.FILTERS) have no such filter, "
                         "so its effective area cannot be folded")
    lo = np.asarray(ene_lo, dtype=np.float64)
    hi = np.asarray(ene_hi, dtype=np.float64)
    a0 = onaxis_arf(camera, filt, lo, hi, ccf_dir, rmf_src_dir if rmf_src_dir is not None else LEGACY_RMF_SRC,
                    date, rule)
    emid = 0.5 * (lo + hi)
    th = np.asarray(thetas, dtype=np.float64)
    return a0[None, :] * vignetting(camera, th[:, None], emid[None, :], ccf_dir, beyond, area_file)


__cites__ = collect_cites(__name__)
