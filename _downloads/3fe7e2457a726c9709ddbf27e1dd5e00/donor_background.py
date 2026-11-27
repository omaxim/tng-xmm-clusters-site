"""The background of a donor observation, measured from its PPS images, and the donor-conditioned
background model of the instrument stage.

What a PPS image holds outside the cluster is the sum of components with different spatial forms
(Snowden et al. 2008, Sect. 3; Leccardi & Molendi 2008, App. B):

* photons (sky foreground, unresolved CXB, faint sources): folded through the effective area, so
  they follow the X-ray vignetting of the band, i.e. the exposure map ``E``;
* the quiescent particle background (QPB) with its fluorescence lines: not focused, flat over the
  detector to first order, i.e. proportional to the UNVIGNETTED exposure ``F = E / V`` (``V`` the
  vignetting of the band on the camera's optical axis frame);
* residual soft protons (SP): focused by the mirrors but not X-rays; their spatial distribution is
  less centrally concentrated than the X-ray vignetting and their spectrum is a power law not folded
  through the effective area.

:func:`decompose_background` fits, per camera and band, the counts of the background region as a
Poisson sum ``a E + b F`` (the vignetted and the flat parts) with compact sources clipped against the
Poisson expectation.  ``b`` in the 4.5-12 keV band is the particle level of the observation
(QPB plus the flat share of SP); :func:`measure_observation_background` runs it on every camera and
band of a set of PPS-like products.  The same function runs on mock products, so a donor and its
replica are measured in one code path.

The instrument stage uses the measurement through the donor background table (``InstrumentOptions.
background = 'donor'``; :func:`donor_background_record` builds a row): the camera's FWC spectrum is scaled to the
donor's flat 4.5-12 keV level (the per-observation FWC normalisation of Snowden et al. 2008, Sect. 3.1.1, done here
on the flat in-FOV component because the PPS images drop the out-of-FOV corners).  The row's ``sp_rate_*`` is 0.0:
the soft-proton rate comes from the IN/OUT excess table of the EPIC cameras (``donor_inout_sp``) through :func:`soft_proton_events`
(broken power law in counts, Leccardi & Molendi 2008, App. B).  Cluster photons are not touched.

References
----------
Snowden, S. L., Mushotzky, R. F., Kuntz, K. D. & Davis, D. S. 2008, A&A 478, 615.
Leccardi, A. & Molendi, S. 2008, A&A 486, 359.
Katayama, H. et al. 2004, A&A 414, 767 (EPIC-pn background; renormalising to the particle rate).
Gastaldello, F. et al. 2017, Exp. Astron. 44, 321 (the particle level follows the solar cycle).
De Luca, A. & Molendi, S. 2004, A&A 419, 837 (the IN over OUT diagnostic).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tngxmm.cites import cites, collect_cites

#: PPS band codes and edges [keV] (write_camera_products.PRODUCT_BANDS 'pps-1-5')
PPS_BANDS = {"1000": (0.2, 0.5), "2000": (0.5, 1.0), "3000": (1.0, 2.0), "4000": (2.0, 4.5), "5000": (4.5, 12.0)}
#: the band whose flat component sets the particle level
QPB_BAND = "5000"
#: the band whose on-axis exposure-map level is the camera's live time (least vignetted on axis)
LIVE_BAND = "2000"
#: soft-proton broken power law in counts space (Leccardi & Molendi 2008, App. B, 0.7-10 keV fit):
#: photon index below / above the break, break energy [keV]
SP_INDEX_LO, SP_INDEX_HI, SP_BREAK_KEV = 0.4, 0.8, 5.0
#: radial bins of the empirical vignetting [arcmin] and the on-axis reference radius [arcmin]
VIG_BIN_ARCMIN = 0.5
VIG_REF_ARCMIN = 1.5
#: floor of the vignetting when forming the unvignetted exposure
VIG_FLOOR = 0.05


# ── measurement ──────────────────────────────────────────────────────────────

def radial_vignetting(expmap: Any, axis: tuple[float, float] | None = None, pix_arcsec: float = 4.0,
                      bin_arcmin: float = VIG_BIN_ARCMIN, ref_arcmin: float = VIG_REF_ARCMIN,
                      q: float = 95.0) -> Any:
    """Vignetting map ``V`` (1 on axis) of an exposure map, from the exposure map alone.

    The optical axis is the maximum of the smoothed map (``axis`` overrides it, 0-indexed (x, y)
    pixels).  In each ring of ``bin_arcmin`` the ``q``-th percentile of the exposure, divided by
    the ``q``-th percentile within ``ref_arcmin`` of the axis, is the vignetting of the ring: an
    upper envelope, so chip gaps, bad columns and the CCD edges (which lower the exposure of a
    pixel, not its vignetting) do not enter.  Returns V per pixel (NaN where the exposure is 0)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    e = np.nan_to_num(np.asarray(expmap, dtype=float))
    if axis is None:
        s = gaussian_filter(e, 10.0)
        iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
        axis = (float(ix), float(iy))
    yy, xx = np.indices(e.shape)
    r = np.hypot(xx - axis[0], yy - axis[1]) * pix_arcsec / 60.0
    on = e > 0
    ref_sel = on & (r < ref_arcmin)
    if not ref_sel.any():
        ref_sel = on
    if not ref_sel.any():
        return np.full(e.shape, np.nan)
    ref = float(np.percentile(e[ref_sel], q))
    ib = np.floor(r / bin_arcmin).astype(int)
    nb = int(ib[on].max()) + 1
    env = np.full(nb, np.nan)
    for k in np.unique(ib[on]):
        sel = on & (ib == k)
        env[k] = np.percentile(e[sel], q) / ref
    # monotone non-increasing outward (vignetting falls with off-axis angle) and filled
    good = np.isfinite(env)
    env = np.interp(np.arange(nb), np.flatnonzero(good), env[good])
    env = np.minimum.accumulate(np.minimum(env, 1.0))
    v = np.where(on, env[np.clip(ib, 0, nb - 1)], np.nan)
    return v


def fit_two_poisson(c: Any, x1: Any, x2: Any, n_newton: int = 50) -> tuple[float, float]:
    """Maximum-likelihood (a, b) >= 0 of counts ``c`` ~ Poisson(a x1 + b x2).

    The log-likelihood is concave; Newton steps from the one-component start, halved until the
    likelihood rises and both parameters stay positive; the two boundary solutions (a = 0 or
    b = 0, each closed form) are compared with the interior one."""
    import numpy as np

    c, x1, x2 = (np.asarray(v, dtype=float) for v in (c, x1, x2))
    s1, s2, n = float(x1.sum()), float(x2.sum()), float(c.sum())
    if n <= 0 or s1 <= 0 or s2 <= 0:
        return 0.0, 0.0
    pos = c > 0
    cp, y1, y2 = c[pos], x1[pos], x2[pos]

    def loglike(a: float, b: float) -> float:
        mu = a * y1 + b * y2
        if np.any(mu <= 0):
            return -np.inf
        return float((cp * np.log(mu)).sum() - a * s1 - b * s2)

    cands = [(n / s1, 0.0), (0.0, n / s2)]
    a, b = 0.5 * n / s1, 0.5 * n / s2
    ll = loglike(a, b)
    for _ in range(n_newton):
        mu = a * y1 + b * y2
        w = cp / mu
        g = np.array([(w * y1).sum() - s1, (w * y2).sum() - s2])
        w2 = cp / mu ** 2
        h = -np.array([[(w2 * y1 * y1).sum(), (w2 * y1 * y2).sum()],
                       [(w2 * y1 * y2).sum(), (w2 * y2 * y2).sum()]])
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6:
            an, bn = a + t * step[0], b + t * step[1]
            if an > 0 and bn > 0:
                lln = loglike(an, bn)
                if lln >= ll:
                    break
            t *= 0.5
        else:
            break
        conv = abs(an - a) <= 1e-9 * (a + b) and abs(bn - b) <= 1e-9 * (a + b)
        a, b, ll = an, bn, lln
        if conv:
            break
    cands.append((a, b))
    best = max(cands, key=lambda ab: loglike(*ab) if min(ab) >= 0 else -np.inf)
    return float(best[0]), float(best[1])


def decompose_background(counts: Any, expmap: Any, region: Any, vig: Any, nsig: float = 5.0,
                         n_iter: int = 8) -> dict[str, float]:
    """Fit the counts of ``region`` as Poisson(a E + b F), ``F = E / max(V, VIG_FLOOR)``.

    ``a`` [counts / s / px, per unit vignetted exposure]: the vignetted (photon-like) part;
    ``b`` [counts / s / px, per unit unvignetted exposure]: the flat (particle-like) part.
    Both are non-negative (:func:`fit_two_poisson`); ``a_err``, ``b_err`` are their Fisher errors.  Pixels above
    ``mu + nsig sqrt(mu) + 3`` (mu the current expectation) are left out, iterated (compact
    sources), as :func:`tngxmm.validation.realism.background_rate` does.  Returns a, b, the total
    rate per vignetted exposure ``rate`` (= sum c / sum E over kept pixels), the flat share
    ``flat_frac`` (b sum F / (a sum E + b sum F)) and the pixel count ``n_pix``."""
    import numpy as np

    c = np.asarray(counts, dtype=float)
    e = np.asarray(expmap, dtype=float)
    v = np.asarray(vig, dtype=float)
    sel = np.asarray(region, bool) & (e > 0) & np.isfinite(v)
    out = {"a": float("nan"), "b": float("nan"), "a_err": float("nan"), "b_err": float("nan"),
           "rate": float("nan"), "flat_frac": float("nan"), "n_pix": 0}
    if sel.sum() < 10:
        return out
    c, e = c[sel], e[sel]
    f = e / np.maximum(v[sel], VIG_FLOOR)
    keep = np.ones(c.size, bool)
    a = b = 0.0
    for _ in range(n_iter):
        a, b = fit_two_poisson(c[keep], e[keep], f[keep])
        mu = a * e + b * f
        new = c <= mu + nsig * np.sqrt(mu) + 3.0
        if np.array_equal(new, keep):
            break
        keep = new
    ek, fk = e[keep], f[keep]
    tot = a * ek.sum() + b * fk.sum()
    mu = a * ek + b * fk
    ok = mu > 0
    w2 = np.where(ok, 1.0 / np.where(ok, mu, 1.0), 0.0)          # Fisher information of Poisson(mu)
    fish = np.array([[(w2 * ek * ek).sum(), (w2 * ek * fk).sum()], [(w2 * ek * fk).sum(), (w2 * fk * fk).sum()]])
    try:
        cov = np.linalg.inv(fish)
        out.update(a_err=float(math.sqrt(max(cov[0, 0], 0.0))), b_err=float(math.sqrt(max(cov[1, 1], 0.0))))
    except np.linalg.LinAlgError:
        pass
    out.update(a=float(a), b=float(b), rate=float(c[keep].sum() / ek.sum()),
               flat_frac=float(b * fk.sum() / tot) if tot > 0 else float("nan"), n_pix=int(keep.sum()))
    return out


@cites("snowden2008", "leccardi2008")
def measure_observation_background(products: Mapping[str, Mapping[str, tuple[Any, Any]]], region: Any,
                                   pix_arcsec: float = 4.0) -> dict[str, dict[str, dict[str, float]]]:
    """Background components of one observation, per camera and band.

    ``products``: camera -> band code -> (counts image, exposure map), all on one pixel grid
    (several exposures of a camera summed beforehand); ``region``: the background region (boolean,
    the cluster and its outskirts excluded).  The vignetting of each band comes from that band's
    exposure map about the camera's own optical axis (:func:`radial_vignetting`).  Returns
    camera -> band -> :func:`decompose_background` output plus ``live`` (the on-axis exposure,
    the ``VIG_REF_ARCMIN`` percentile of the map, [s])."""
    import numpy as np

    out: dict[str, dict[str, dict[str, float]]] = {}
    for cam, bands in products.items():
        out[cam] = {}
        axis = None
        for code, (cimg, emap) in bands.items():
            e = np.nan_to_num(np.asarray(emap, dtype=float))
            if axis is None and e.max() > 0:
                from scipy.ndimage import gaussian_filter

                s = gaussian_filter(e, 10.0)
                iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
                axis = (float(ix), float(iy))
            v = radial_vignetting(e, axis=axis, pix_arcsec=pix_arcsec)
            res = decompose_background(np.nan_to_num(np.asarray(cimg, dtype=float)), e, region, v)
            yy, xx = np.indices(e.shape)
            near = (np.hypot(xx - axis[0], yy - axis[1]) * pix_arcsec / 60.0 < VIG_REF_ARCMIN) & (e > 0) \
                if axis is not None else np.zeros(e.shape, bool)
            res["live"] = float(np.percentile(e[near], 95)) if near.any() else float("nan")
            out[cam][code] = res
    return out


# ── the donor-conditioned model of the instrument stage ─────────────────────

#: cameras and their table column suffixes
CAMERAS = {"MOS1": "mos1", "MOS2": "mos2", "PN": "pn"}


def nominal_qpb_band_rate(resp: Any, elo: float, ehi: float, pix_arcsec: float = 4.0) -> float:
    """QPB rate [counts / s] per ``pix_arcsec`` pixel of unvignetted exposure in [elo, ehi) keV that the
    stage draws at the nominal level: the camera's FWC spectrum (``resp.fwc_spec``, scaled to the active
    area, times the channel width) per active area.  This is the ``b`` of :func:`decompose_background`
    that a nominal mock carries."""
    import numpy as np

    bw = np.asarray(resp.eb_max, dtype=float) - np.asarray(resp.eb_min, dtype=float)
    e = 0.5 * (np.asarray(resp.eb_max, dtype=float) + np.asarray(resp.eb_min, dtype=float))
    sel = (e >= elo) & (e < ehi)
    area = float(np.asarray(resp.mask).sum()) * float(resp.pixsize) ** 2          # arcmin^2
    rate = float((np.asarray(resp.fwc_spec, dtype=float)[sel] * bw[sel]).sum()) / area
    return rate * (pix_arcsec / 60.0) ** 2


def donor_background_parameters(table: Any, obsid: str) -> dict[str, dict[str, float]]:
    """``{camera: {'qpb_scale': .., 'sp_rate': .., 'live_s': .., 'qpb_band': {band: factor}}}`` of one
    donor from the table (columns ``qpb_scale_<cam>``, ``sp_rate_<cam>``, ``live_s_<cam>``,
    ``qpb_<band>_<cam>``; ``qpb_band`` only when the table has band factors); raises ``KeyError`` when
    the donor is absent."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = donor_rows(table, obsid, "donor background").iloc[0]
    out = {}
    for cam, suf in CAMERAS.items():
        q = r.get(f"qpb_scale_{suf}", 1.0)
        sp = r.get(f"sp_rate_{suf}", 0.0)
        q = 1.0 if q is None or not math.isfinite(float(q)) else float(q)
        sp = 0.0 if sp is None or not math.isfinite(float(sp)) else max(float(sp), 0.0)
        lv = r.get(f"live_s_{suf}", float("nan"))
        lv = float("nan") if lv is None else float(lv)
        out[cam] = {"qpb_scale": q, "sp_rate": sp, "live_s": lv}
        bands = {k: float(r[f"qpb_{k}_{suf}"]) for k in PPS_BANDS if f"qpb_{k}_{suf}" in r.index
                 and r[f"qpb_{k}_{suf}"] is not None and math.isfinite(float(r[f"qpb_{k}_{suf}"]))}
        if bands:
            out[cam]["qpb_band"] = bands
    return out


@cites("xmm_uhb", "epic_pn")
def donor_live_times(cameras_s: Mapping[str, float], params: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Per-camera live times of a view for ``live_time = 'donor-expmap'``: the donor's on-axis PPS
    exposure-map level (``live_s``, :func:`measure_observation_background`; all exposures of the camera
    summed), which carries the dead time and the frame losses of the camera's mode, in place of the
    package's ``cameras_s`` (the EXPOSURE keyword); a camera without a positive finite ``live_s`` keeps
    its package value."""
    out = {}
    for cam, t in cameras_s.items():
        lv = (params.get(cam) or {}).get("live_s", float("nan"))
        out[cam] = float(lv) if lv is not None and math.isfinite(float(lv)) and float(lv) > 0 else float(t)
    return out


@cites("leccardi2008", "deluca2004", "snowden2008")
def sp_counts_spectrum(e_kev: Any) -> Any:
    """Residual soft-proton spectrum in COUNTS (not folded through the effective area): the broken
    power law of Leccardi & Molendi 2008 (App. B), index 0.4 below and 0.8 above 5 keV, continuous
    at the break; relative units."""
    import numpy as np

    e = np.asarray(e_kev, dtype=float)
    lo = (e / SP_BREAK_KEV) ** (-SP_INDEX_LO)
    hi = (e / SP_BREAK_KEV) ** (-SP_INDEX_HI)
    return np.where(e < SP_BREAK_KEV, lo, hi)


def soft_proton_events(resp: Any, t_live: float, params: Mapping[str, float], rng: Any,
                       off_px: tuple = (0.0, 0.0), emin: float = 0.2, emax: float = 12.0, sp_map: Any = None) -> tuple:
    """Residual soft-proton events of one camera (X, Y [sky mask px], channel energy [keV]).

    ``params['sp_rate']`` [counts / s / arcmin^2, 0.5-12 keV, unvignetted]: the level; the events
    are spread uniformly over the active pixels inside the vignetted field of view (soft protons are
    focused by the mirrors, so they reach only the field of view; De Luca & Molendi 2004, App. B.1),
    their energies drawn from :func:`sp_counts_spectrum` with no redistribution (a particle
    spectrum, not convolved with the effective area or the response).  ``sp_map`` (the ESAS soft-proton detector map
    on the mask grid, :func:`spdetmap_stage`): the events follow it instead of a flat distribution, with the same
    total (``instrument.soft_proton_map = 'spdetmap'``).  A dict ``{(lo, hi) [keV]: map}`` (:func:`flare_maps_stage`,
    ``soft_proton_map = 'flare'``) is drawn band by band: :func:`flare_band_events`."""
    import numpy as np

    from tngxmm.instrument.simulate_instrument import fov_map

    rate = float(params.get("sp_rate", 0.0))
    fov = fov_map(resp)
    ay, ax = np.nonzero(fov)
    z = np.zeros(0)
    if rate <= 0 or len(ax) == 0:
        return z, z, z
    if isinstance(sp_map, Mapping):
        return flare_band_events(resp, t_live, rate, rng, sp_map, off_px=off_px, emin=emin, emax=emax)
    grid = np.linspace(emin, emax, 2401)
    mid = 0.5 * (grid[1:] + grid[:-1])
    w = sp_counts_spectrum(mid)
    band = (mid >= 0.5) & (mid < 12.0)
    frac_all = float(w.sum() / w[band].sum())
    area = len(ax) * float(resp.pixsize) ** 2
    n = int(rng.poisson(rate * frac_all * area * float(t_live)))
    if n == 0:
        return z, z, z
    cdf = np.cumsum(w) / w.sum()
    k = np.searchsorted(cdf, rng.random(n), side="right")
    k = np.clip(k, 0, len(mid) - 1)
    chan = grid[k] + rng.random(n) * (grid[k + 1] - grid[k])
    if sp_map is None:
        sel = rng.integers(0, len(ax), n)
    else:
        w = np.nan_to_num(np.asarray(sp_map)[ay, ax], nan=0.0)
        w = np.where(w > 0, w, 0.0)
        cdf = np.cumsum(w)
        sel = np.minimum(np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right"), len(ax) - 1)
    X = ax[sel] + rng.uniform(-0.5, 0.5, n) + off_px[0]
    Y = ay[sel] + rng.uniform(-0.5, 0.5, n) + off_px[1]
    return X, Y, chan


def nominal_qpb_rates(ccf_dir: str, pix_arcsec: float = 4.0) -> dict[str, dict[str, float]]:
    """:func:`nominal_qpb_band_rate` of each camera and PPS band (full-frame packaged masks; the FWC
    spectrum does not depend on the filter, the per-area rate not on the mask)."""
    from tngxmm.instrument.load_responses import get_response

    out = {}
    for cam in CAMERAS:
        r = get_response(cam, ccf_dir, filter="Medium")
        out[cam] = {k: nominal_qpb_band_rate(r, lo, hi, pix_arcsec=pix_arcsec) for k, (lo, hi) in PPS_BANDS.items()}
    return out


def qpb_channel_scale(resp: Any, band_scales: Mapping[str, float]) -> Any:
    """Per-channel factor on the QPB expectation of ``resp`` from per-band scales (channels below 0.5 keV
    take band 1's, above 12 keV band 5's)."""
    import numpy as np

    e = 0.5 * (np.asarray(resp.eb_min, dtype=float) + np.asarray(resp.eb_max, dtype=float))
    f = np.full(e.shape, float(band_scales.get(QPB_BAND, 1.0)))
    for k, (lo, hi) in PPS_BANDS.items():
        f[(e >= lo) & (e < hi)] = float(band_scales.get(k, band_scales.get(QPB_BAND, 1.0)))
    f[e < PPS_BANDS["1000"][0]] = float(band_scales.get("1000", band_scales.get(QPB_BAND, 1.0)))
    return f


@cites("snowden2008", "deluca2004", "katayama2004", "leccardi2008")
def donor_background_record(measured: Mapping[str, Mapping[str, Mapping[str, float]]],
                            nominal: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """One row of the donor background table from :func:`measure_observation_background` (``measured``) and the
    nominal QPB rates (``nominal``: camera -> band -> rate, :func:`nominal_qpb_rates`).

    Per camera: the on-axis live time ``live_s`` (what the instrument stage reads under ``live_time =
    'donor-expmap'``); the measured flat and vignetted 4.5-12 keV rates ``b5``, ``a5``; and, as diagnostics, the flat
    level of each band over the nominal FWC level, ``qpb_<band>`` and ``qpb_scale`` (= the 4.5-12 keV one), NaN where
    the band is not measured.  The flat level of an image decomposition also holds the flat share of the residual
    soft protons and is degenerate with the vignetted part (De Luca & Molendi 2004, App. B.3), so it is not the
    particle level: the instrument stage takes that from the corners (:mod:`tngxmm.instrument.donor_corner_qpb`).
    No calibration, shrinkage or clipping is applied (removed at the integration of 2026-10-30)."""
    row: dict[str, float] = {}
    nan = float("nan")
    for cam, suf in CAMERAS.items():
        mc = measured.get(cam) or {}
        m = mc.get(QPB_BAND) or {}
        row[f"b5_{suf}"] = float(m.get("b", nan))
        row[f"a5_{suf}"] = float(m.get("a", nan))
        for k in PPS_BANDS:
            b = float((mc.get(k) or {}).get("b", nan))
            n = float((nominal.get(cam) or {}).get(k, nan))
            row[f"qpb_{k}_{suf}"] = b / n if (math.isfinite(b) and b > 0 and math.isfinite(n) and n > 0) else nan
        row[f"qpb_scale_{suf}"] = row[f"qpb_{QPB_BAND}_{suf}"]
        row[f"sp_rate_{suf}"] = 0.0
        row[f"live_s_{suf}"] = float((mc.get(LIVE_BAND) or {}).get("live", nan))
    return row

# ── spatial distribution of the particle background (FWC event lists) ───────

#: CCF filter-wheel-closed event lists (``E*_FWC``) per camera and the PATTERN selection of the PPS images
FWC_CCF = {"MOS1": "EMOS1_FWC", "MOS2": "EMOS2_FWC", "PN": "EPN_FWC"}
FWC_PATTERN_MAX = {"MOS1": 12, "MOS2": 12, "PN": 4}
#: detector coordinates -> the frame of the packaged detector masks (2.5" grid), per camera: ``(row x, row y, shift)``
#: with u = (row x . (DETX, DETY) + shift_x, row y . (DETX, DETY) + shift_y), DET and u in arcmin, u the offset from
#: the grid centre (u_x along columns, u_y along rows).  The masks are SAS ``emask`` images of observation 0827340901
#: (PA_PNT 114.2 deg); these rows are SAS's own DET -> sky solution for that exposure, read event by event from its PPS
#: event lists and the masks' sky -> pixel keywords (``tools/det_to_mask.py``; scatter 0.12" median, the attitude
#: drift).  A reflection (determinant -1): DETX runs opposite to the mask columns.
DET_TO_MASK = {
    "MOS1": ((+0.41021, -0.91199), (-0.91198, -0.41022), (-0.5237, -1.5192)),
    "MOS2": ((+0.91407, +0.40555), (+0.40556, -0.91407), (-0.3178, -1.6281)),
    "PN": ((-0.91044, -0.41364), (-0.41364, +0.91044), (-0.2603, -1.4703)),
}
#: the identity map (tests, synthetic detectors)
DET_IDENTITY = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
#: Gaussian smoothing of the FWC density [arcmin], within each CCD (normalised convolution)
FWC_SMOOTH_ARCMIN = 1.0
_FWC_MAPS: dict = {}


def _oldest_fwc(ccf_dir: str, inst: str) -> str:
    """The oldest FWC event list of ``inst`` in ``ccf_dir`` other than the latest (pn: the extended-full-frame set).

    The pn extended-full-frame set ``EPN_FWC_EFF_nnnn.CCF`` must not sit in the site CCF directory: the CCF scan of the
    upstream xmm_simulator reads the number after ``EPN_FWC_`` of every matching file and fails on ``EFF``.  It is
    looked up in ``ccf_dir`` and then in the site directory ``TNGXMM_CCF_EXTRA`` (environment or site file)."""
    import os

    from tngxmm.core import site

    stem = FWC_CCF[inst]
    if inst == "PN":
        for d in (ccf_dir, site.get("TNGXMM_CCF_EXTRA") or ""):
            if d and os.path.isdir(d) and any(f.startswith("EPN_FWC_EFF_") for f in os.listdir(d)):
                return _latest_ccf(d, "EPN_FWC_EFF")
        raise FileNotFoundError(f"no EPN_FWC_EFF CCF in {ccf_dir} or $TNGXMM_CCF_EXTRA")
    found = sorted(f for f in os.listdir(ccf_dir) if f.startswith(stem + "_") and f.endswith(".CCF"))
    if not found:
        raise FileNotFoundError(f"no {stem} CCF in {ccf_dir}")
    return os.path.join(ccf_dir, found[0])


def _latest_ccf(ccf_dir: str, stem: str) -> str:
    import os

    best, nb = None, -1
    for f in os.listdir(ccf_dir):
        if f.startswith(stem + "_") and f.endswith(".CCF") and f[len(stem) + 1:-4].isdigit():
            k = int(f.rsplit("_", 1)[1].split(".")[0])
            if k > nb:
                best, nb = f, k
    if best is None:
        raise FileNotFoundError(f"no {stem} CCF in {ccf_dir}")
    return os.path.join(ccf_dir, best)


def fwc_density_maps(detx_arcmin: Any, dety_arcmin: Any, pi_ev: Any, ccdnr: Any, det_to_mask: tuple,
                     shape: tuple[int, int] = (900, 900), pixsize_arcmin: float = 2.5 / 60.0,
                     smooth_arcmin: float = FWC_SMOOTH_ARCMIN, bands: Mapping[str, tuple] = PPS_BANDS,
                     ccd_exposure: Mapping[int, float] | None = None, exclude_ccds: Any = (),
                     gaps: Mapping[str, tuple] | None = None, support: tuple | None = None) -> dict:
    """Relative QPB surface density per PPS band on the stage's detector grid from filter-wheel-closed events.

    Every event is placed on the mask frame by ``det_to_mask`` (:data:`DET_TO_MASK`, SAS's DET -> sky solution of the
    observation behind the masks).  Per CCD the events of a band are histogrammed on the grid, smoothed with a Gaussian
    of ``smooth_arcmin`` by normalised convolution over that CCD's occupied pixels (so chip edges stay sharp and
    CCD-to-CCD level steps are kept) and divided by the CCD's summed FWC exposure (``ccd_exposure``, CCD number -> s; a
    CCD lost during the mission has less).  Each map is normalised to mean 1 over its occupied pixels and is NaN
    elsewhere.  The events of ``exclude_ccds`` are dropped: those CCDs are NaN too and do not enter the mean-1
    normalisation.  ``gaps`` (band -> (lo, hi) keV) leaves a PI range out of a band (the pn 4.5-12 keV band of the PPS
    images without the Cu line, ``instrument.pn_hard_band = 'pps'``).  ``support``: (DETX, DETY [arcmin], CCDNR) of the
    events whose occupied pixels make each CCD's area, isolated empty pixels filled (the closed exposures of an epoch
    in every CCD state, when the density comes from those in the normal state only; :func:`fwc_qpb_maps`); default:
    the density events."""
    import numpy as np
    from scipy.ndimage import binary_closing, gaussian_filter

    n = shape[0]
    edges = (np.arange(n + 1) - n / 2.0) * pixsize_arcmin
    sig = smooth_arcmin / pixsize_arcmin
    (a, b), (c, d), (sx, sy) = det_to_mask

    def to_mask(dx, dy):
        x0, y0 = np.asarray(dx, float), np.asarray(dy, float)
        return a * x0 + b * y0 + sx, c * x0 + d * y0 + sy

    x, y = to_mask(detx_arcmin, dety_arcmin)
    pi, cc = np.asarray(pi_ev), np.asarray(ccdnr)
    if support is not None:
        sxm, sym = to_mask(support[0], support[1])
        scc = np.asarray(support[2])
    occ_all = np.zeros(shape)
    dens = {k: np.zeros(shape) for k in bands}
    for ccd in np.unique(cc):
        if int(ccd) in {int(e) for e in exclude_ccds}:
            continue
        sc = cc == ccd
        if support is None:
            h_all, _, _ = np.histogram2d(y[sc], x[sc], bins=[edges, edges])
            occ = (h_all > 0).astype(float)
        else:
            ss = scc == ccd
            h_all, _, _ = np.histogram2d(sym[ss], sxm[ss], bins=[edges, edges])
            occ = binary_closing(h_all > 0, structure=np.ones((3, 3))).astype(float)
        if occ.sum() < 100:
            continue
        socc = gaussian_filter(occ, sig)
        occ_all += occ
        texp = float((ccd_exposure or {}).get(int(ccd), 1.0)) or 1.0
        for k, (lo, hi) in bands.items():
            sb = sc & (pi >= lo * 1000.0) & (pi < hi * 1000.0)
            if gaps and k in gaps:
                sb &= ~((pi >= gaps[k][0] * 1000.0) & (pi < gaps[k][1] * 1000.0))
            h, _, _ = np.histogram2d(y[sb], x[sb], bins=[edges, edges])
            with np.errstate(invalid="ignore", divide="ignore"):
                dens[k] += np.where(occ > 0, gaussian_filter(h, sig) / np.maximum(socc, 1e-12), 0.0) / texp
    sup = occ_all > 0
    out = {}
    for k, m in dens.items():
        m = np.where(sup & (m > 0), m, np.nan)
        out[k] = m / np.nanmean(m) if np.isfinite(m).any() else m      # no events in the band: NaN (flat)
    return out


@cites("esas_proton", "xmm_ccf_sas")
def spdetmap_stage(ccf_dir: str, inst: str, shape: tuple[int, int] = (900, 900),
                   pixsize_arcmin: float = 2.5 / 60.0) -> Any:
    """The ESAS soft-proton detector map of ``inst`` (CCF ``XMM_SPDETMAP``: 780 x 780 detector pixels of 2.5", rows
    DETY, columns DETX, centred on DET (0, 0)) on the stage's mask grid: embedded in the grid about its centre (the
    grid's pixel must be 2.5") and placed on the mask frame by :data:`DET_TO_MASK`, like the FWC density maps;
    normalised to mean 1 over its positive pixels, 0 elsewhere."""
    import os

    import numpy as np
    from astropy.io import fits
    from scipy.ndimage import rotate

    if abs(pixsize_arcmin - 2.5 / 60.0) > 1e-9:
        raise ValueError(f"spdetmap_stage needs a 2.5 arcsec grid, got {pixsize_arcmin * 60:.3f} arcsec")
    row = {"MOS1": "EMOS1", "MOS2": "EMOS2", "PN": "EPN"}[inst]
    path = _latest_ccf(str(ccf_dir), "XMM_SPDETMAP")
    with fits.open(path) as h:
        m = next(np.asarray(r["SPDETMAP"], float) for r in h["SPDETMAP"].data if str(r["INSTRUME"]).strip() == row)
    a = _detmap_on_grid(m, inst, shape)
    del os, rotate
    return a / a[a > 0].mean() if (a > 0).any() else a


def smooth_detmap(m: Any, smooth_arcmin: float = FWC_SMOOTH_ARCMIN, pix_arcmin: float = 2.5 / 60.0) -> Any:
    """A CCF flare map smoothed with a Gaussian of ``smooth_arcmin`` by normalised convolution over its pixels with
    data, clipped at 0.  Unsmoothed, the counting noise of the map would be printed on every mock and fit template.

    The flare maps are images of the flare counts minus the non-flare images scaled by the exposure times (Kuntz &
    Snowden 2008, Sect. 4.3), so a pixel holds a difference of counts: negative values are the noise of that
    difference and are data (55 per cent of the pn pixels in the field, Medium, 0.75-1.25 keV); they enter the local
    mean like the positive ones, which keeps it unbiased.  A pixel exactly 0 holds no data: outside the field, or a
    chip gap or bad column of the stacked exposures ("The gap between the inner chip and the outer chips can be seen
    at R ~ 5.25'").  Such a pixel is not a pixel without soft protons: inside the field it gets the smoothed value of
    its neighbourhood, and the dead pixels of the observation enter once, through its own footprint (the mock's
    efficiency map, which thins every event; the fit's unvignetted exposure map, which multiplies the template).  The
    pixels filled are those without data inside the half level of the smoothed occupancy (the edge of a filled region
    under a symmetric kernel): gaps and columns narrower than the kernel lie inside it, the space beyond the field
    does not.  The clip at 0 is the only constraint on the result: a soft-proton rate is not negative."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    m = np.asarray(m, float)
    occ = (np.isfinite(m) & (m != 0)).astype(float)
    a = np.where(occ > 0, m, 0.0)
    sig = smooth_arcmin / pix_arcmin
    socc = gaussian_filter(occ, sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where((occ > 0) | (socc >= 0.5), gaussian_filter(a, sig) / np.maximum(socc, 1e-12), 0.0)
    return np.clip(out, 0.0, None)


def _detmap_on_grid(m: Any, inst: str, shape: tuple[int, int]) -> Any:
    """A 780 x 780 ESAS detector map (2.5" pixels, rows DETY, columns DETX, centred on DET (0, 0)) resampled onto the
    stage's 2.5" mask grid through :data:`DET_TO_MASK` (bilinear); non-finite and non-positive pixels set to 0."""
    import numpy as np
    from scipy.ndimage import affine_transform

    (a, b), (c, d), (sx, sy) = DET_TO_MASK[inst]
    # output (row, col) -> u (arcmin from the grid centre) -> DET = R^-1 (u - s) -> input (row, col) of the 780 grid;
    # both grids have 2.5" pixels, so the pixel unit cancels in the rotation
    inv = np.linalg.inv(np.array([[a, b], [c, d]]))          # (u_x, u_y) -> (x, y)
    pix = 2.5 / 60.0
    co, ci = shape[0] / 2.0 - 0.5, m.shape[0] / 2.0 - 0.5
    # in (row, col) order: input_row = y/pix + ci, input_col = x/pix + ci, with (x, y) = inv @ (u_x - sx, u_y - sy)
    mat = np.array([[inv[1, 1], inv[1, 0]], [inv[0, 1], inv[0, 0]]])
    shift = np.array([sy, sx]) / pix
    off = np.array([ci, ci]) - mat @ (np.array([co, co]) + shift)
    src = np.where(np.isfinite(m) & (m > 0), m, 0.0)
    out = affine_transform(src, mat, offset=off, output_shape=tuple(shape), order=1, cval=0.0)
    return np.where(np.isfinite(out) & (out > 0), out, 0.0)


#: energy bands [keV] of the soft-proton flare maps, in the order of the CCF rows B1 ... B6: the six bands in which
#: Kuntz & Snowden (2008, Sect. 4.3) built the flare images behind the ESAS soft-proton maps
FLARE_BANDS_KEV = {"B1": (0.3, 0.75), "B2": (0.75, 1.25), "B3": (1.25, 2.0), "B4": (2.0, 4.0), "B5": (4.0, 8.0),
                   "B6": (8.0, 12.0)}
#: filters with a flare map in the CCF; another filter (Thin2) takes the Medium map (Kuntz & Snowden 2008, Sect. 4.3:
#: the maps of the other filters have the shape of the Medium-filter map, only the amplitude differs)
FLARE_FILTERS = ("Thin1", "Medium", "Thick")
#: radius [arcmin, from the pointing] of the region in which the IN/OUT excess, i.e. the donor's sp_rate, is measured
#: (donor_tables.mos_background.MAX_OFFAXIS_ARCMIN); the flare maps are normalised to mean 1 inside it
FLARE_NORM_RADIUS_ARCMIN = 12.0
_FLARE: dict = {}


@cites("kuntz2008", "esas_proton", "xmm_ccf_sas")
def flare_maps_stage(ccf_dir: str, inst: str, filt: str, shape: tuple[int, int] = (900, 900),
                     pixsize_arcmin: float = 2.5 / 60.0) -> dict[tuple[float, float], Any]:
    """The soft-proton flare maps of ``inst`` and filter ``filt`` (CCF ``XMM_FLARE``, the maps the SAS task ``proton``
    co-adds; Kuntz & Snowden 2008, Sect. 4.3) on the stage's mask grid, one per band of :data:`FLARE_BANDS_KEV`:
    ``{(lo, hi) [keV]: map}``, each placed like :func:`spdetmap_stage` and unnormalised (:func:`flare_band_weights`
    normalises them on the camera's field of view), each smoothed first (:func:`smooth_detmap`).  The file is looked up
    in ``ccf_dir`` and then in
    ``$TNGXMM_CCF_EXTRA``.  Cached per (file, camera, filter, grid)."""
    import numpy as np
    from astropy.io import fits

    from tngxmm.instrument.rga_obscuration import ccf_file

    if abs(pixsize_arcmin - 2.5 / 60.0) > 1e-9:
        raise ValueError(f"flare_maps_stage needs a 2.5 arcsec grid, got {pixsize_arcmin * 60:.3f} arcsec")
    row = {"MOS1": "EMOS1", "MOS2": "EMOS2", "PN": "EPN"}[inst]
    f = filt if filt in FLARE_FILTERS else "Medium"
    path = str(ccf_file(ccf_dir, "XMM_FLARE"))
    key = (path, row, f, tuple(shape))
    if key not in _FLARE:
        out = {}
        with fits.open(path) as h:
            for r in h["FLAREMAP"].data:
                band = str(r["BAND"]).strip()
                if str(r["INSTRUME"]).strip() == row and str(r["FILTER"]).strip() == f and band in FLARE_BANDS_KEV:
                    out[FLARE_BANDS_KEV[band]] = _detmap_on_grid(smooth_detmap(np.asarray(r["FLAREMAP"], float)),
                                                                  inst, shape)
        if len(out) != len(FLARE_BANDS_KEV):
            raise ValueError(f"{path}: {row} {f} has flare maps for {sorted(out)}, not the six bands")
        _FLARE[key] = out
    return _FLARE[key]


def flare_band_of(e_kev: Any, edges: list[tuple[float, float]]) -> Any:
    """Index into ``edges`` (sorted bands) of each energy; energies below the first band take it, above the last band
    the last."""
    import numpy as np

    lows = np.array([lo for lo, _ in edges])
    return np.clip(np.searchsorted(lows, np.asarray(e_kev, float), side="right") - 1, 0, len(edges) - 1)


class FlareMaps(dict):
    """The flare maps of one camera (band edges -> map, :func:`flare_maps_stage`) with the cluster region of the view
    that the donor's IN/OUT measurement left out: ``exclude`` = (x, y, r) of the cluster centre [mask px, detector
    frame] and the excluded radius [arcmin], or None."""

    def __init__(self, maps: Mapping, exclude: tuple | None = None) -> None:
        super().__init__(maps)
        self.exclude = exclude


def flare_band_weights(resp: Any, sp_maps: Mapping[tuple[float, float], Any]) -> tuple:
    """(field-of-view pixels (ay, ax), per-band weights at those pixels, sorted band edges): each band's map at the
    camera's field-of-view pixels divided by its mean over the region of the donor's IN/OUT measurement
    (:mod:`tngxmm.donor_tables.mos_background`): within :data:`FLARE_NORM_RADIUS_ARCMIN` of the pointing and, when
    ``sp_maps`` is a :class:`FlareMaps` with ``exclude``, beyond the excluded radius of the cluster, so that the band's
    mean rate there is the donor's measured rate."""
    import numpy as np

    from tngxmm.instrument.simulate_instrument import fov_map

    fov = fov_map(resp)
    ay, ax = np.nonzero(fov)
    r = np.hypot(ax - resp.cx, ay - resp.cy) * float(resp.pixsize)
    inner = r < FLARE_NORM_RADIUS_ARCMIN
    exc = getattr(sp_maps, "exclude", None)
    if exc is not None:
        inner &= np.hypot(ax - float(exc[0]), ay - float(exc[1])) * float(resp.pixsize) > float(exc[2])
    edges = sorted(sp_maps)
    w = []
    for b in edges:
        v = np.asarray(sp_maps[b], float)[ay, ax]
        v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
        mean_in = float(v[inner].mean()) if inner.any() else 0.0
        w.append(v / mean_in if mean_in > 0 else np.ones(len(ax)))
    return (ay, ax), w, edges


@cites("kuntz2008", "esas_proton", "leccardi2008", "deluca2004")
def flare_band_events(resp: Any, t_live: float, rate: float, rng: Any, sp_maps: Mapping[tuple[float, float], Any],
                      off_px: tuple = (0.0, 0.0), emin: float = 0.2, emax: float = 12.0) -> tuple:
    """Residual soft-proton events that follow the flare map of their energy band (``soft_proton_map = 'flare'``).

    ``rate`` [counts / s / arcmin^2, 0.5-12 keV]: the donor's IN/OUT level, a mean over the region within
    :data:`FLARE_NORM_RADIUS_ARCMIN` of the pointing, with the counts spectrum :func:`sp_counts_spectrum` there.  In
    band b the expected number is rate x (spectral share of b) x the sum over field-of-view pixels of the band map
    normalised to mean 1 in that region (:func:`flare_band_weights`) x pixel area x live time; positions follow the
    map, energies the spectrum within the band (no redistribution).  Below 0.3 keV the first band's map is used, above
    12 keV the last one's."""
    import numpy as np

    (ay, ax), w, edges = flare_band_weights(resp, sp_maps)
    z = np.zeros(0)
    if rate <= 0 or len(ax) == 0:
        return z, z, z
    grid = np.linspace(emin, emax, 2401)
    mid = 0.5 * (grid[1:] + grid[:-1])
    spec = sp_counts_spectrum(mid)
    norm = float(spec[(mid >= 0.5) & (mid < 12.0)].sum())
    kb = flare_band_of(mid, edges)
    pix_area = float(resp.pixsize) ** 2
    xs, ys, cs = [], [], []
    for b in range(len(edges)):
        sel = kb == b
        if not sel.any():
            continue
        wb = w[b]
        mu = rate * float(spec[sel].sum()) / norm * float(wb.sum()) * pix_area * float(t_live)
        n = int(rng.poisson(mu))
        if n == 0:
            continue
        cdf = np.cumsum(spec[sel]) / spec[sel].sum()
        k = np.clip(np.searchsorted(cdf, rng.random(n), side="right"), 0, int(sel.sum()) - 1)
        g_lo, g_hi = grid[:-1][sel], grid[1:][sel]
        cs.append(g_lo[k] + rng.random(n) * (g_hi[k] - g_lo[k]))
        c = np.cumsum(wb)
        p = np.minimum(np.searchsorted(c, rng.random(n) * c[-1], side="right"), len(ax) - 1)
        xs.append(ax[p] + rng.uniform(-0.5, 0.5, n) + off_px[0])
        ys.append(ay[p] + rng.uniform(-0.5, 0.5, n) + off_px[1])
    if not cs:
        return z, z, z
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def flare_band_shares(edges: list[tuple[float, float]], emin: float, emax: float) -> Any:
    """Share of the residual soft-proton counts in [emin, emax) keV that falls in each flare band of ``edges`` (sorted;
    energies below the first band count to it, above the last to the last, as in the 'flare' draw): the counts
    spectrum :func:`sp_counts_spectrum` on a 5 eV grid.  The weights with which XMM-ESAS ``proton`` co-adds the flare
    maps of the selected bands, here weighted by the spectrum instead of equally."""
    import numpy as np

    grid = np.linspace(0.2, 12.0, 2361)
    mid = 0.5 * (grid[1:] + grid[:-1])
    spec = np.where((mid >= emin) & (mid < emax), sp_counts_spectrum(mid), 0.0)
    kb = flare_band_of(mid, edges)
    return np.array([float(spec[kb == b].sum()) for b in range(len(edges))]) / max(float(spec.sum()), 1e-30)


def flare_broadband_shape(resp: Any, sp_maps: Mapping[tuple[float, float], Any], emin: float = 0.5,
                          emax: float = 12.0) -> Any:
    """The soft-proton surface density at the field-of-view pixels, summed over the bands of ``sp_maps`` with the
    shares of the counts spectrum in [emin, emax) (:func:`flare_band_shares`; the shape of the 'flare' draw in that
    energy range, for the model images)."""
    import numpy as np

    (ay, ax), w, edges = flare_band_weights(resp, sp_maps)
    share = flare_band_shares(edges, emin, emax)
    tot = np.zeros(len(ax))
    for b in range(len(edges)):
        tot += share[b] * w[b]
    return tot


def scale_maps_by_ccd(maps: dict, labels: Any, factors: Mapping[int, float]) -> dict:
    """QPB density maps with each CCD's pixels multiplied by its factor (``labels``: the CCD label map on the same
    grid; a CCD without a factor keeps 1), renormalised to mean 1 over the finite positive pixels: the per-CCD /
    per-quadrant corner levels of ``instrument.qpb_units = 'corners'`` redistribute the camera's QPB, the camera's
    band totals stay those of the band factors."""
    import numpy as np

    lab = np.asarray(labels)
    f = np.ones(lab.shape)
    for ccd, v in factors.items():
        if v is not None and math.isfinite(float(v)) and float(v) > 0:
            f[lab == int(ccd)] = float(v)
    out = {}
    for k, m in maps.items():
        a = np.asarray(m) * f
        ok = np.isfinite(a) & (a > 0)
        out[k] = a / a[ok].mean() if ok.any() else a
    return out


def working_ccds(inst: str) -> tuple[int, ...]:
    """CCD numbers of ``inst`` that still deliver data (the permanent losses of
    :data:`tngxmm.instrument.active_ccds.CCD_LOSSES` removed: MOS1 CCD3 and CCD6)."""
    from tngxmm.instrument.active_ccds import CCD_LOSSES, CCDS

    lost = {c for _, c in CCD_LOSSES.get(inst, ())}
    return tuple(c for c in CCDS[inst] if c not in lost)


#: the two disjoint halves of an FWC event list (even and odd event index): the mock stage draws the QPB with one, the
#: fit's QPB templates (mock and real alike) are built from the other, so that a template does not share the
#: calibration's counting noise with the data it models
FWC_HALVES = (0, 1)
FWC_GENERATOR_HALF = 0
FWC_TEMPLATE_HALF = 1


@cites("snowden2008", "leccardi2008", "xmm_uhb", "xmm_ccf_sas")
def fwc_qpb_maps(ccf_dir: str, inst: str, shape: tuple[int, int] = (900, 900),
                 pixsize_arcmin: float = 2.5 / 60.0, ccds: Any = None, which: str = "latest",
                 gaps: Mapping[str, tuple] | None = None, half: int | None = None) -> dict:
    """:func:`fwc_density_maps` of the camera's latest ``E*_FWC`` CCF event list (FLAG == 0, the PPS
    PATTERN selection), on the stage's detector grid; cached per (ccf_dir, camera, grid, CCDs).

    The template covers the CCDs ``ccds`` only (default :func:`working_ccds`): the FWC list of MOS1 holds
    events of CCD3 and CCD6, lost in 2012 and 2005, which an observation made after the loss does not image,
    so they are left out of the template (NaN, outside its normalisation).  The
    stage passes the CCDs the donor imaged, so a donor observed before the loss keeps the FWC density of the
    CCD.  The density comes from the closed exposures of the donor's epoch (the one those CCDs identify,
    :func:`tngxmm.instrument.fwc_state.select_epoch`) in the normal CCD state, each CCD's area from the epoch's
    exposures in every state, as the fit's templates of a real exposure are built
    (:func:`tngxmm.massfit.qpb_template.fwc_band_rates`).  ``half`` (:data:`FWC_HALVES`): only the events of even
    (0) or odd (1) index, None all of them.  The total QPB count does not depend on the template: it is the FWC
    spectrum times the imaged area (``resp.fwc_spec``, scaled to the active pixels); the template only distributes
    those counts."""
    import numpy as np
    from astropy.io import fits

    from tngxmm.instrument import fwc_state as S

    keep = tuple(sorted({int(c) for c in (working_ccds(inst) if ccds is None else ccds)}))
    gkey = tuple(sorted((k, tuple(v)) for k, v in (gaps or {}).items()))
    key = (str(ccf_dir), inst, tuple(shape), round(pixsize_arcmin, 9), keep, which, gkey, half, S.FWC_REFERENCE_RULE)
    if key not in _FWC_MAPS:
        if half is not None and half not in FWC_HALVES:
            raise ValueError(f"FWC half must be one of {FWC_HALVES} or None, got {half!r}")
        # which = 'oldest': the oldest FWC set of the CCF (the wrong-template test of the ESAS study; the pn has one
        # full-frame set, so its oldest other set is the extended-full-frame one)
        path = _latest_ccf(str(ccf_dir), FWC_CCF[inst]) if which == "latest" else _oldest_fwc(str(ccf_dir), inst)
        d = fits.getdata(path, "EVENTS", memmap=True)
        rows = S.exptime_rows(fits.getdata(path, "EXPTIME"))
        ep = S.select_epoch(S.epochs(rows), keep)
        normal = S.row_selection(rows, ep, normal=True)
        row_of = S.event_rows(np.asarray(d["TIME"], float), np.asarray(d["CCDNR"]), rows)
        texp = S.ccd_exposure(rows, normal)
        good = (np.asarray(d["FLAG"]) == 0) & (np.asarray(d["PATTERN"]) <= FWC_PATTERN_MAX[inst])
        area = good & S.selected_events(row_of, S.row_selection(rows, ep, normal=False))
        ok = good & S.selected_events(row_of, normal)
        if half is not None:
            ok &= (np.arange(len(ok)) % 2) == int(half)
        arcmin = 0.05 / 60.0                                      # DETX/DETY unit: 0.05"
        detx, dety, ccdnr = np.asarray(d["DETX"]), np.asarray(d["DETY"]), np.asarray(d["CCDNR"])
        _FWC_MAPS[key] = fwc_density_maps(detx[ok] * arcmin, dety[ok] * arcmin,
                                          np.asarray(d["PI"])[ok], ccdnr[ok],
                                          DET_TO_MASK[inst], shape, pixsize_arcmin, ccd_exposure=texp,
                                          exclude_ccds=sorted(set(texp) - set(keep)), gaps=gaps,
                                          support=(detx[area] * arcmin, dety[area] * arcmin, ccdnr[area]))
    return _FWC_MAPS[key]


def qpb_pixel_draw(resp: Any, chan_kev: Any, maps: Mapping[str, Any] | None, rng: Any) -> Any:
    """Indices into ``resp.active_x`` of QPB events with channel energies ``chan_kev``: uniform over the
    active pixels without ``maps`` (one ``rng.integers`` draw, the nominal stage), else, band by band,
    with probability proportional to the band's FWC density at the pixel (1 where the map has no value).
    Channels below 0.2 keV use band 1, above 12 keV band 5."""
    import numpy as np

    n = len(chan_kev)
    npx = len(resp.active_x)
    if maps is None:
        return rng.integers(0, npx, n)
    e = np.asarray(chan_kev, dtype=float)
    sel = np.empty(n, dtype=np.int64)
    codes = list(PPS_BANDS)
    which = np.full(n, codes.index(QPB_BAND))
    for i, (k, (lo, hi)) in enumerate(PPS_BANDS.items()):
        which[(e >= lo) & (e < hi)] = i
    which[e < PPS_BANDS["1000"][0]] = 0
    for i, k in enumerate(codes):
        m = which == i
        if not m.any():
            continue
        w = np.asarray(maps[k])[resp.active_y, resp.active_x]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        cdf = np.cumsum(w)
        sel[m] = np.minimum(np.searchsorted(cdf, rng.random(int(m.sum())) * cdf[-1], side="right"), npx - 1)
    return sel


__cites__ = collect_cites(__name__)
