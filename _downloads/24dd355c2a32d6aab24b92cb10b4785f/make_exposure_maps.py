"""Vignetted exposure maps and the merged coverage map on the 4"/px product grid.

Construction (``inverse-map``): every output pixel is sampled with ``ss x ss``
sub-pixels, each mapped back through the exact inverse of the event transform
used for the images (bin edges, rotation about the detector centre, boresight
offset, rounding to the 2.5" detector pixel with the same interior test as the
events), and the mask x vignetting is averaged.  Image and map are registered
by construction.

References
----------
XMM-Newton SOC, Current Calibration Files (XRT*_XAREAEF vignetting) through
xmm_simulator ``calc_arf`` (Eckert, xmm_simulator 0.2.0).
"""

from __future__ import annotations

from typing import Any

from tngxmm.cites import cites, collect_cites

EXPMAP_METHODS = ("inverse-map",)
#: rules for the energy at which a band's EXPMAP is vignetted
VIG_ENERGY_RULES = ("band-centre", "geometric-mean", "count-weighted")
#: EXPFIX header value per construction
EXPFIX_TAG = {"inverse-map": "v4-inverse-map-ss4"}


@cites("xmm_ccf_sas")
def vignetting_energy(elo: float, ehi: float, rule: str = "band-centre") -> float:
    """Energy [keV] at which the EXPMAP of band [elo, ehi) is vignetted, for
    the rules that do not depend on the view's spectrum.

    ``band-centre``: (elo + ehi)/2, i.e.
    8.25 keV for 4.5-12 keV.  This is also the documented rule of SAS
    ``eexpmap`` ("quantum efficiency, filter transmission and vignetting are
    evaluated assuming an event energy which corresponds to the mean of the PI
    channel boundaries pimin and pimax").  ``geometric-mean``: sqrt(elo ehi)
    (7.35 keV for 4.5-12 keV).  The PPS EXPMAP5000 of the two fixture obsids
    (0551830101, 0551780401; all three cameras), divided by their EXPMAP2000
    (which removes the masks and the attitude smearing common to both), follow
    V(theta, E) / V(theta, 0.75 keV) of our CCF set best at E = 8.26-8.29 keV
    (rms 0.0006-0.0028 to 13'): the band centre, the documented rule.  ``count-weighted`` needs the view's spectrum:
    :func:`count_weighted_energy`."""
    if rule == "band-centre":
        return 0.5 * (elo + ehi)
    if rule == "geometric-mean":
        return float((elo * ehi) ** 0.5)
    raise ValueError(f"vignetting-energy rule {rule!r} is not spectrum-free; known: "
                     f"{VIG_ENERGY_RULES} (count-weighted: count_weighted_energy)")


@cites("xmm_ccf_sas")
def count_weighted_energy(resp: Any, elo: float, ehi: float, photon_counts) -> float | None:
    """Mean energy [keV] of the on-axis COUNTS in band [elo, ehi):
    sum_k n_k A0_k E_k / sum_k n_k A0_k, with n_k the incident photons per
    coarse energy bin k (``photon_counts``, on ``resp.box_ene``; e.g. the
    view's parent photon list) and A0 the on-axis ARF of this camera.

    This is the energy at which a single-energy vignetting best represents
    the counts of a hard band (about 6 keV for 4.5-12 keV and a cluster
    spectrum).  ``None`` when the band holds no counts."""
    import numpy as np

    n = np.asarray(photon_counts, dtype=np.float64)
    e = resp.box_ene_mean
    sel = (e >= elo) & (e < ehi)
    w = n[sel] * resp.arf_theta[0, sel]
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        return None
    return float((w * e[sel]).sum() / tot)


@cites("xmm_ccf_sas")
def band_vignetting_energies(resp: Any, bands, rule: str = "band-centre",
                             photon_counts=None) -> dict[str, tuple[float, str]]:
    """{band code: (energy [keV], rule actually used)} for one camera.

    ``count-weighted`` falls back to ``geometric-mean`` for a band without
    counts (recorded as ``'geometric-mean (no counts)'``)."""
    out: dict[str, tuple[float, str]] = {}
    for elo, ehi, code in bands:
        if rule == "count-weighted":
            e = (None if photon_counts is None
                 else count_weighted_energy(resp, elo, ehi, photon_counts))
            out[code] = ((e, rule) if e is not None
                         else (vignetting_energy(elo, ehi, "geometric-mean"),
                               "geometric-mean (no counts)"))
        else:
            out[code] = (vignetting_energy(elo, ehi, rule), rule)
    return out


@cites("xmm_ccf_sas")
def photon_spectrum_on_grid(resp: Any, energies_kev):
    """Histogram of photon energies on the response's coarse ARF grid
    (photons beyond the grid go to the end bins, as in the camera partition)."""
    import numpy as np

    k = np.clip(np.searchsorted(resp.box_ene, energies_kev, side='right') - 1,
                0, len(resp.box_ene_mean) - 1)
    return np.bincount(k, minlength=len(resp.box_ene_mean)).astype(np.float64)


@cites("xmm_ccf_sas")
def vig_on_mask(resp: Any, e_mean_keV: float):
    """Vignetting factor vig(theta,E)=ARF(theta)/ARF(0) on the (900,900) mask grid, times the MOS RGA modulation
    at the same energy when the response carries one (``instrument.mos_rga_vignetting = 'ccf'``)."""
    import numpy as np

    k = int(np.clip(np.searchsorted(resp.box_ene, e_mean_keV) - 1,
                    0, len(resp.box_ene_mean) - 1))
    onax = resp.arf_theta[0, k]
    vig = (resp.arf_theta[:, k] / max(onax, 1e-30))[resp.theta_idx_pix]
    if getattr(resp, "rga", None) is not None:
        from tngxmm.instrument.simulate_instrument import rga_factor, rga_offset_map

        dx, dy = rga_offset_map(resp)
        vig = vig * rga_factor(resp, dx, dy, energy_index=np.full(dx.shape, k))
    return vig


@cites("xmm_ccf_sas")
def inverse_map_samples(resp: Any, off_px, ca, sa, wpix, n4, do_rot, ss: int = 4):
    """Detector pixel (yd, xd) and validity of ``ss x ss`` sub-samples of every
    output pixel, through the exact inverse of the event transform.

    Events (``simulate_instrument``) go sky (mask px, X, Y) -> rotated
    ``xr = cx + (X-cx) ca - (Y-cy) sa``, ``yr = cy + (X-cx) sa + (Y-cy) ca`` ->
    bin ``k`` with edges ``(k - n4/2) wpix + cx``; detector pixel
    ``round(X - ox)``, valid if ``0 < xd < npix-1`` and on the mask.
    Returns (yd, xd, ok) as (n4*ss, n4*ss) arrays (row = y)."""
    import numpy as np

    cx, cy = resp.cx, resp.cy
    sub = (np.arange(ss) + 0.5) / ss
    k = np.arange(n4)
    u = (k[:, None] + sub[None, :]).ravel()          # continuous bin coordinate
    xr = (u - n4 / 2.0) * wpix + cx
    yr = (u - n4 / 2.0) * wpix + cy
    XR, YR = np.meshgrid(xr, yr)                      # [y, x]
    if do_rot:
        dx, dy = XR - cx, YR - cy
        xs = cx + dx * ca + dy * sa
        ys = cy - dx * sa + dy * ca
    else:
        xs, ys = XR, YR
    xd = np.round(xs - off_px[0]).astype(np.int64)
    yd = np.round(ys - off_px[1]).astype(np.int64)
    ok = (xd > 0) & (xd < resp.npix - 1) & (yd > 0) & (yd < resp.npix - 1)
    xd = np.where(ok, xd, 0)
    yd = np.where(ok, yd, 0)
    ok &= resp.mask[yd, xd] == 1
    return yd, xd, ok


@cites("xmm_ccf_sas")
def inverse_map_average(values_on_mask, samples, n4: int, ss: int = 4):
    """Average a mask-grid map over the sub-samples of each output pixel."""
    import numpy as np

    yd, xd, ok = samples
    # one flat gather instead of a two-index fancy lookup (same values, about twice as fast)
    v = np.asarray(values_on_mask)
    val = np.where(ok, np.take(v.ravel(), yd * v.shape[1] + xd), 0.0)
    return val.reshape(n4, ss, n4, ss).mean(axis=(1, 3))


@cites("xmm_ccf_sas")
def make_exposure_maps(resp: Any, t_live: float, bands, off_px, posangle_deg: float,
                       n4: int, wpix: float, method: str = "inverse-map",
                       ss: int = 4, vig_rule: str = "band-centre",
                       photon_counts=None,
                       vig_energies: dict[str, tuple[float, str]] | None = None,
                       ) -> tuple[dict[str, Any], Any]:
    """Per-band vignetted EXPMAPs [s] and the unvignetted coverage (mask) map
    of one camera on the n4 product grid.

    ``bands`` = [(elo, ehi, code), ...]; returns ({code: float64 map}, coverage),
    coverage NOT multiplied by ``t_live``.  A response with an ``efficiency`` map
    (:mod:`tngxmm.instrument.donor_footprint`) has it multiplied into both.  ``method`` is ``'inverse-map'``, the
    only construction.  ``vig_rule`` picks the vignetting energy per band (see
    :func:`band_vignetting_energies`; ``count-weighted`` needs
    ``photon_counts``); ``vig_energies`` overrides it with precomputed
    energies."""
    import numpy as np

    ang = np.deg2rad(posangle_deg)
    ca, sa = np.cos(ang), np.sin(ang)
    do_rot = abs(posangle_deg % 360.0) > 0.1
    out: dict[str, Any] = {}
    if vig_energies is None:
        vig_energies = band_vignetting_energies(resp, bands, vig_rule, photon_counts)
    if method != "inverse-map":
        raise ValueError(f"unknown EXPMAP method {method!r}; known: {EXPMAP_METHODS}")
    samples = inverse_map_samples(resp, off_px, ca, sa, wpix, n4, do_rot, ss=ss)
    # the detector footprint's good-silicon fraction (donor_footprint), 1 on the 0/1 mask
    eff = getattr(resp, 'efficiency', None)
    ones = np.ones(resp.mask.shape) if eff is None else np.asarray(eff, dtype=np.float64)
    for elo, ehi, code in bands:
        vig = vig_on_mask(resp, vig_energies[code][0])
        out[code] = inverse_map_average(vig if eff is None else vig * ones,
                                        samples, n4, ss) * t_live
    cov = inverse_map_average(ones, samples, n4, ss)
    return out, cov


__cites__ = collect_cites(__name__)
