"""EPIC-MOS CCDs in the "anomalous state": a soft (below ~1 keV) particle-like excess on whole CCDs.

Individual MOS CCDs (MOS1 CCD4 and CCD5, MOS2 CCD2 and CCD5 most often) are at times in an anomalous
state in which their background below ~1 keV is raised over the whole chip (Kuntz & Snowden 2008;
Snowden et al. 2008 exclude such CCDs from their cluster analysis).  The donors' 0.5-1 keV images show
it: the background of 40 donors, cut in 24 cells per MOS camera and fitted as vignetted plus flat
components, scatters 3.7-3.9 times the Poisson variance (90th percentile 33-43) in 0.5-1 keV, against
1.5 in the mocks and about 2 in every band above 1 keV; the cells that depart are whole CCDs.

The stage gives every MOS CCD the camera's flat (QPB) level.  This module measures, per donor, MOS
camera, CCD and PPS band, the CCD's flat excess in units of the camera's flat level,

    k_c = (N_c - mu_c) / (b sum_c F),     mu_c = sum_c (a E + b F),

with (a, b) the camera's vignetted and flat levels (:func:`tngxmm.instrument.donor_background.
fit_two_poisson`, F = E / V the unvignetted exposure) fitted on the CCDs not in the anomalous state,
over the background region (beyond 1.5 R500 of the cluster, compact sources clipped against the Poisson
expectation).  A CCD counts as anomalous when its 0.5-1 keV excess exceeds :data:`NSIG` Poisson sigma and its 1-2 keV
excess (same units) is below :data:`CONTROL_MAX_FRAC` of it (:func:`classify`); its k_c in 0.2-0.5 and
0.5-1 keV are then kept, and k_c = 0 otherwise.  The CCD of every donor sky pixel comes from the CCD label maps of
:mod:`tngxmm.instrument.active_ccds` moved through the donor pointing and roll (the transform of
:func:`tngxmm.donors.build_active_ccds.expmap_in_mask_frame`).

Instrument option ``mos_soft_ccd`` (:data:`MOS_SOFT_CCD`): ``'off'`` (default) keeps every product;
``'donor'`` adds to each anomalous CCD of the replica's camera the donor CCD's excess RATE (counts / s per
4" pixel, the excess over the donor's live time ``t_ref``, the on-axis level of its exposure map) times the
replica's live time over the CCD's active pixels, spread over the band's channels with the shape of the
camera's particle spectrum.  The rate, not the factor k_c, is carried: k_c divides by the fitted flat level,
which is poorly determined in the soft bands of short exposures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tngxmm.cites import cites, collect_cites

#: values of InstrumentOptions.mos_soft_ccd
MOS_SOFT_CCD = ("off", "donor")
#: PPS bands measured [keV]
BANDS = {"1000": (0.2, 0.5), "2000": (0.5, 1.0), "3000": (1.0, 2.0)}
#: bands of the anomalous state (below 1 keV); 1-2 keV is the control band
SOFT_BANDS = ("1000", "2000")
CONTROL_BAND = "3000"
#: a CCD's soft excess counts as the anomalous state only when its 1-2 keV excess (same units) is below this
#: fraction of its 0.5-1 keV excess: a soft-only excess (sky structure, cluster outskirts, extended sources
#: and soft protons raise the 1-2 keV band too)
CONTROL_MAX_FRAC = 0.25
#: significance of a CCD's excess for the anomalous state [Poisson sigma]
NSIG = 5.0
#: the CCDs considered (outer MOS CCDs; the central CCD 1 holds the cluster)
OUTER_CCDS = (2, 3, 4, 5, 6, 7)
#: the cameras
CAMERAS = ("MOS1", "MOS2")
#: the CCDs with anomalous states (Kuntz & Snowden 2008, Sect. 3.3: "these anomalous states occur only for chips
#: MOS1-4, MOS1-5, MOS2-2, and MOS2-5"); excesses on other CCDs are not given to the replica
KS_CCDS = {"MOS1": (4, 5), "MOS2": (2, 5)}
#: table columns
COLUMNS = ("obsid", "camera", "ccd", "band", "n", "mu", "k", "sig", "n_pix", "excess", "t_ref", "rate", "rate_raw")


def mask_to_sky_labels(labels: Any, sky_header: Any, sky_shape: tuple[int, int], ra_pnt: float, dec_pnt: float,
                       pa_pnt: float, cx: float = 450.0, cy: float = 450.0, pixsize_arcmin: float = 2.5 / 60.0,
                       pa_ref_deg: float | None = None) -> Any:
    """CCD label of every pixel of a donor sky image: each mask pixel's label is written at the sky pixel
    the mask-frame transform of :func:`tngxmm.donors.build_active_ccds.expmap_in_mask_frame` samples (the
    2.5" mask pixels oversample the 4" sky pixels, so every covered sky pixel is reached).  0 = no CCD."""
    import warnings

    import numpy as np
    from astropy.wcs import WCS, FITSFixedWarning

    if pa_ref_deg is None:
        from tngxmm.donors.build_core_offsets import PA_REF_DEG

        pa_ref_deg = PA_REF_DEG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        w = WCS(sky_header, naxis=2)
    px0, py0 = w.all_world2pix(float(ra_pnt), float(dec_pnt), 0)
    ppix = abs(float(sky_header["CDELT2"])) * 60.0
    lab = np.asarray(labels)
    yy, xx = np.indices(lab.shape)
    dx, dy = (xx - cx) * pixsize_arcmin, (yy - cy) * pixsize_arcmin
    a = math.radians(float(pa_pnt) - pa_ref_deg)
    ix = np.round(px0 + (dx * math.cos(a) - dy * math.sin(a)) / ppix).astype(np.int64)
    iy = np.round(py0 + (dx * math.sin(a) + dy * math.cos(a)) / ppix).astype(np.int64)
    ok = (lab > 0) & (ix >= 0) & (iy >= 0) & (ix < sky_shape[1]) & (iy < sky_shape[0])
    out = np.zeros(sky_shape, np.uint8)
    out[iy[ok], ix[ok]] = lab[ok]
    return out


@cites("kuntz2008", "snowden2008")
def measure_ccd_excess(counts: Any, expmap: Any, vig: Any, labels: Any, region: Any,
                       ccds: tuple[int, ...] = OUTER_CCDS, nsig: float = NSIG, n_iter: int = 6,
                       min_pix: int = 200) -> dict[int, dict[str, float]]:
    """Per CCD: counts ``n``, expectation ``mu`` of the camera model, excess ``k`` in units of the camera's
    flat level (0 unless the excess exceeds ``nsig`` sigma), significance ``sig`` and pixels ``n_pix``.
    The camera model (a, b) is fitted on the region's pixels of the CCDs not flagged, pixels above
    mu + 5 sqrt(mu) + 3 left out (compact sources), iterated until the flagged set is stable."""
    import numpy as np

    from tngxmm.instrument.donor_background import VIG_FLOOR, fit_two_poisson

    c = np.asarray(counts, dtype=float)
    e = np.asarray(expmap, dtype=float)
    v = np.asarray(vig, dtype=float)
    lab = np.asarray(labels)
    base = np.asarray(region, bool) & (e > 0) & np.isfinite(v)
    f = np.where(base, e / np.maximum(np.where(np.isfinite(v), v, 1.0), VIG_FLOOR), 0.0)
    present = [k for k in ccds if (base & (lab == k)).sum() >= min_pix]
    flagged: set[int] = set()
    keep = base.copy()
    out: dict[int, dict[str, float]] = {}
    for _ in range(n_iter):
        fit = keep & np.isin(lab, [k for k in present if k not in flagged] or present)
        if fit.sum() < 50:
            break
        a, b = fit_two_poisson(c[fit], e[fit], f[fit])
        mu = a * e + b * f
        keep = base & (c <= mu + 5.0 * np.sqrt(mu) + 3.0)
        out = {}
        new = set()
        for k in present:
            m = keep & (lab == k)
            n, mk, flat = float(c[m].sum()), float(mu[m].sum()), float(b * f[m].sum())
            sig = (n - mk) / math.sqrt(mk) if mk > 0 else 0.0
            kk = (n - mk) / flat if (sig > nsig and flat > 0) else 0.0
            if sig > nsig:
                new.add(k)
            out[k] = {"n": n, "mu": mk, "k": float(kk), "sig": float(sig), "n_pix": int(m.sum()),
                      "excess": float(n - mk) if sig > nsig else 0.0}
        if new == flagged:
            break
        flagged = new
    return out


def classify(rows: Any) -> Any:
    """The table rows with ``rate`` [counts / s per 4" sky pixel, = excess / (n_pix t_ref)] kept only for the
    anomalous state: soft bands (:data:`SOFT_BANDS`) of CCDs whose 1-2 keV rate is below
    :data:`CONTROL_MAX_FRAC` of their 0.5-1 keV rate; every other rate set to 0 (the control band always 0).
    ``rows``: frame with obsid, camera, ccd, band, excess, n_pix, t_ref."""
    import numpy as np
    import pandas as pd

    t = pd.DataFrame(rows).copy()
    t["band"] = t["band"].astype(str)
    t["rate_raw"] = np.where((t.n_pix > 0) & (t.t_ref > 0), t.excess / (t.n_pix * t.t_ref).where(t.t_ref > 0, 1.0), 0.0)
    key = ["obsid", "camera", "ccd"]
    r2 = t[t.band == "2000"].set_index(key)["rate_raw"]
    r3 = t[t.band == CONTROL_BAND].set_index(key)["rate_raw"]
    idx = pd.MultiIndex.from_frame(t[key])
    a = r2.reindex(idx).to_numpy(float)
    c = r3.reindex(idx).fillna(0.0).to_numpy(float)
    ok = (a > 0) & (c < CONTROL_MAX_FRAC * a) & t.band.isin(SOFT_BANDS).to_numpy()
    t["rate"] = np.where(ok, t["rate_raw"], 0.0)
    t.loc[~ok, "k"] = 0.0
    return t


class Unmeasured(dict):
    """The (empty) anomalous-CCD map of a donor that has no PPS MOS image to measure it on: no excess is
    applied, and the SOFTCCD header says ``unmeasured`` instead of ``none``."""

    unmeasured = True


def donor_soft_ccd(table: Any, obsid: str, camera_ccds: Mapping[str, Mapping[str, Any]] | None = None
                   ) -> dict[str, dict[int, dict[str, float]]]:
    """camera -> CCD -> {band code: rate [counts / s per 4" pixel]} of one donor, only the CCDs of
    :data:`KS_CCDS` and the bands with rate > 0 (empty when the donor has none).

    A donor absent from the table raises ``KeyError`` (it was not measured), except when
    ``camera_ccds`` (the donor's active CCDs, :func:`~tngxmm.instrument.active_ccds.package_camera_ccds`)
    shows that no MOS camera of the donor has a PPS exposure map (source other than ``pps-expmap``): there is
    nothing to measure the anomalous state on (3 of the 638 donors), and the donor is given no anomalous CCD
    (:class:`Unmeasured`)."""
    from tngxmm.instrument.donor_tables import donor_rows

    try:
        t = donor_rows(table, obsid, "MOS soft-CCD")
    except KeyError:
        if camera_ccds is not None and not any(
                str(v.get("source")) == "pps-expmap" for c, v in camera_ccds.items() if c.startswith("MOS")):
            return Unmeasured()
        raise
    out: dict[str, dict[int, dict[str, float]]] = {}
    for r in t.itertuples():
        k = float(r.rate)
        if int(r.ccd) not in KS_CCDS.get(str(r.camera), ()):
            continue
        if math.isfinite(k) and k > 0:
            out.setdefault(str(r.camera), {}).setdefault(int(r.ccd), {})[str(r.band)] = k
    return out


def anomalous_ccds_by_observation(table: Any) -> dict[tuple[str, str], list[int]]:
    """(obsid, camera) -> the MOS CCDs of an observation in the anomalous state: the rows of the soft-CCD table with a
    0.5-1 keV excess k > 0 (:func:`classify` keeps k only for those).  The ESAS fit leaves these CCDs out of its images
    (:func:`tngxmm.massfit.fit_hydrostatic.anomalous_ccds`) and the corner scales leave them out of the camera's level
    (:mod:`tngxmm.donor_tables.mos_background`, :mod:`tngxmm.donor_tables.qpb_corners`)."""
    t = table[(table.band.astype(str) == "2000") & (table.k > 0)]
    return {(str(o).zfill(10), str(c)): sorted(int(x) for x in g.ccd) for (o, c), g in t.groupby(["obsid", "camera"])}


@cites("kuntz2008")
def soft_ccd_events(resp: Any, lam_qpb: Any, ccd_labels: Any, rates: Mapping[int, Mapping[str, float]],
                    t_live: float, rng: Any, bands: Mapping[str, tuple[float, float]] = BANDS,
                    sky_pix_arcsec: float = 4.0) -> tuple[Any, Any, Any]:
    """Extra events of the anomalous CCDs of one camera: for every CCD c and band with a donor rate r [counts
    / s per ``sky_pix_arcsec`` pixel], Poisson counts of mean r t_live (active pixels of c) (mask pixel /
    sky pixel)^2, spread over the band's channels in proportion to the camera's QPB expectation ``lam_qpb``
    (the particle spectrum's shape), placed uniformly on the CCD's active pixels.  ``ccd_labels``: the CCD
    label of every pixel of the mask grid.  Returns detector pixel x, y (pixel centres plus a uniform
    sub-pixel offset) and channel energies; no draw when ``rates`` is empty."""
    import numpy as np

    z = np.zeros(0)
    if not rates:
        return z, z, z
    lab = np.asarray(ccd_labels)[resp.active_y, resp.active_x]
    emid = (np.asarray(resp.eb_min) + np.asarray(resp.eb_max)) / 2.0
    lam = np.asarray(lam_qpb, dtype=float)
    area = (float(resp.pixsize) * 60.0 / sky_pix_arcsec) ** 2
    xs, ys, cs = [], [], []
    for ccd in sorted(rates):
        idx = np.flatnonzero(lab == int(ccd))
        if not idx.size:
            continue
        for code in sorted(rates[ccd]):
            r = float(rates[ccd][code])
            lo, hi = bands[code]
            ch = (emid >= lo) & (emid < hi)
            if not (r > 0) or not ch.any():
                continue
            w = lam[ch] if lam[ch].sum() > 0 else np.ones(int(ch.sum()))
            spec = rng.poisson(r * float(t_live) * idx.size * area * w / w.sum()).astype(int)
            n = int(spec.sum())
            if not n:
                continue
            sel = idx[rng.integers(0, idx.size, n)]
            xs.append(resp.active_x[sel] + rng.uniform(-0.5, 0.5, n))
            ys.append(resp.active_y[sel] + rng.uniform(-0.5, 0.5, n))
            cs.append(np.repeat(emid[ch], spec))
    if not xs:
        return z, z, z
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


__cites__ = collect_cites(__name__)
