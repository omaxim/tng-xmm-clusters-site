"""Observing conditions of a view from one of the N nearest real observations (the conditions donor).

Under rule ``donor`` every view copies its donor exactly (rule ``nn``, the v1 presets' n=5, separates the roles below): the K replicas of a donor share its footprint, background,
live times, filters, modes and N_H, so the K views are correlated and a model can learn the donor's
identity from the data quality alone.  This module separates the two roles of the donor:

* the PHYSICS donor (``donor_package['obsid']``) is the observation whose mass and redshift the halo
  matches; it stays the donor of record (apertures, mass, z_obs, the count-matching target);
* the CONDITIONS donor is the real observation whose data quality the view is observed with: its
  per-camera live times (``live_s`` of the background table), background parameters, PPS exposure
  maps (detector footprint), active CCDs and readout windows, filters, PN mode and submodes, N_H,
  core offset and pointing, AGN normalisation row and SAS donor facts.  Every donor-keyed table of the
  instrument stage is looked up with its obsid.

Condition space (:func:`condition_matrix`): per camera the log10 live time and the log10 QPB scale of
the donor background table, log10 N_H, log10 theta500 of the donor's own cluster and its core offset
[arcmin] (how the cluster sits on the detector), each standardised by the pool's robust spread
(1.4826 MAD); a mismatch of PN filter, MOS filter or PN submode adds :data:`CATEGORY_PENALTY` (in
squared standard units).  The N nearest observations of the pool (the physics donor itself at
distance 0 first) are the candidates.

Draw (reproducible, no existing stream changes):

* the candidate: a permutation of the N candidates seeded per PHYSICS donor
  (:func:`tngxmm.core.rng.seed_for_donor`, purpose ``conditions``); replica k (1-based) takes
  element ``(k - 1) mod N`` of it, so the K replicas of a donor get K different conditions donors
  (without replacement while K <= N).  A view whose replica index is unknown draws the candidate
  uniformly from its own ``conditions`` stream;
* per projection (``per_projection``, release 1.2; :func:`draw_projection_conditions`): the projections of one
  halo-snapshot draw in view order, each uniformly from its own ``conditions`` stream among the N nearest pool
  observations of its ANCHOR (its physics donor; for a halo-snapshot without one, the nearest observation of the
  pairing, :mod:`tngxmm.donors.pair_projections`, whose views cannot be generated yet) that no earlier projection of
  the halo-snapshot has taken.  The conditions donors of a halo-snapshot's views are therefore all different, even
  where several projections share a physics donor (the candidate list then reaches past the N nearest).  A view
  recomputes the earlier projections' choices from the first integer of their own ``conditions`` streams, so it can be
  generated or replayed alone.  The order has an effect: where projections share an anchor, a later one draws from
  candidates the earlier ones left, further from the anchor, and the exposure jitter, whose width is the spread of
  those candidates, is wider.  On the production condition table, with six projections on one physics donor (every
  pool observation as the anchor, three halo-snapshots each), the median distance of the conditions donor from the
  anchor rises from 2.20 at projection 0 to 2.99 at projection 5 and the median width of the exposure jitter from 0.28
  to 0.33 in ln; with six different physics donors (1,325 halo-snapshots of a trial pairing) there is no trend.  The
  pairing ties the rank of the physics donor to the projection index as well (view 0 has the nearest), so a subset of
  views chosen by index is not a random subset;
* the jitter (:func:`draw_jitter`), from the view's own ``conditions`` stream, every number drawn
  whatever the switches so turning one on does not move the others:

  - ``exposure``: one common log-normal factor on all cameras' live times, sigma = the robust spread
    (floored, 0 below two candidates) of the log effective live time (ARF-weighted over cameras) among the N candidates, the result clipped to the candidates' range;
  - ``roll``: the position angle of the conditions donor's observation re-drawn as the nominal angle
    of its date (the position angle of the Sun seen from the pointing) plus an off-nominal angle
    drawn from the archive (the measured PA_PNT - PA_sun of the 638 bank observations, within the
    +-20 deg of the XMM-Newton Users Handbook section 4.2.2); the mock's POSANGLE follows the new
    angle and the cluster's detector-frame offset rotates by minus the change, so the cluster keeps
    its sky offset from the pointing and the detector (footprint, CCDs, gaps, every camera together)
    turns under it;
  - ``pointing``: an isotropic Gaussian offset of the aimpoint (sigma from repeated observations of
    the same clusters in the bank, radii above their 90th percentile are clipped onto that radius).

Count matching: the cluster's physical FLUX is held fixed, so the counts a view must have are the physics donor's
real net counts (0.15-1 R500, cameras summed) scaled to the conditions donor's observing time:
N_target = N_physics,real x t_cond / t_phys, the effective live times t = sum_cam w_cam t_cam / sum_cam w_cam
(:func:`effective_exposure_s`, ARF weights :data:`T_EFF_WEIGHTS_CM2`) of the products (the jittered live times) and of
the physics donor's REAL observation (``cameras_real_s``: N_physics,real was counted at those times, not at the jittered
times of the replica's target row).  The factor is f = N_target / N_mock, N_mock the net counts of the replica at f = 1
(:mod:`tngxmm.realism.matching`); thinning the photon list by f and replaying it under the conditions donor gives N_target.
The data quality (live times, background, footprint, filters, vignetting) therefore follows the conditions donor while the
flux stays the physics donor's; the filter, vignetting and footprint differences between the two donors are not corrected in
N_target (only the live times), so f absorbs them.  Under the rule 'donor' the ratio is 1.
The photon list was absorbed by the physics view's N_H; it is moved to the
conditions donor's N_H by keeping each photon with probability f T(E; N_H,c) / T(E; N_H,p), capped at
1 (:func:`retarget_keep_prob`; a lower N_H can only be reached where f T_c/T_p <= 1 and the deficit
is recorded).

References
----------
xmm_uhb : XMM-Newton Users Handbook, section 4.2.2 (solar aspect angle 70-110 deg; position angle
          within +-20 deg of the nominal one) and 4.4.2 (position angle definition).
rosen2016 : 3XMM (attitude frame shifts; the scale of the pointing registration).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: rules of the conditions donor: 'donor' (the physics donor's own conditions, the stored products) | 'nn'
CONDITION_RULES = ("donor", "nn")
#: jitter components (any subset)
JITTER_KINDS = ("exposure", "roll", "pointing")
CAMERAS = ("MOS1", "MOS2", "PN")
_SUF = {"MOS1": "mos1", "MOS2": "mos2", "PN": "pn"}
#: continuous dimensions of the condition space
CONTINUOUS = tuple(f"log_live_{s}" for s in ("mos1", "mos2", "pn")) + \
    tuple(f"log_qpb_{s}" for s in ("mos1", "mos2", "pn")) + ("log_nh", "log_theta500", "core_offset_arcmin")
#: categorical dimensions and the cost of a mismatch [squared standard units]: a filter change moves
#: the soft effective area (Thin1 / Medium / Thick) and costs as much as a 3-sigma difference of a
#: continuous dimension; the pn submode (Full Frame / extended Full Frame: out-of-time fraction 6.3 /
#: 2.3 %, frame time) costs a 2-sigma one
CATEGORICAL = ("filter_pn", "filter_mos", "pn_mode")
CATEGORY_PENALTY = {"filter_pn": 9.0, "filter_mos": 9.0, "pn_mode": 4.0}
#: XMM-Newton Users Handbook 4.2.2: the position angle stays within +-20 deg of the nominal one
ROLL_MAX_DEG = 20.0
#: aimpoint scatter of repeated on-axis (< 3') observations of the same cluster in the bank
#: (22 clusters, 45 observations; pointing minus catalogue position, demeaned per cluster):
#: median radius 0.865' -> sigma = 0.865 / sqrt(2 ln 2) = 0.735' per axis; 90th percentile radius 2.42'
POINTING_SIGMA_ARCMIN = 0.735
POINTING_MAX_ARCMIN = 2.42
#: ARF band-mean effective areas [cm^2] that weight the cameras' live times into one effective live time
#: (:func:`_t_eff`, :func:`effective_exposure_s`; the weights of donors.build_replica_targets)
T_EFF_WEIGHTS_CM2 = {"MOS1": 395.0, "MOS2": 400.0, "PN": 1259.0}
#: PN out-of-time fraction per submode (as donors.build_replica_targets)
PN_OOT = {"PrimeFullWindow": 0.063, "PrimeFullWindowExtended": 0.023,
          "PrimeLargeWindow": 0.0016, "PrimeSmallWindow": 0.011}
#: readout modes in which the central CCD gives no image (XMM-Newton Users Handbook, EPIC science modes: the MOS
#: timing mode and the pn timing and burst modes keep one spatial dimension).  The cluster region of such a camera has
#: no imaging events, so SAS cannot make its per-observation response (rmfgen refuses an empty detector map; obs-v1
#: set, 0780291001 MOS2), and the observation leaves the pool (:func:`build_condition_table`)
TIMING_MODES = frozenset({"FastUncompressed", "FastCompressed", "FastTiming", "FastBurst", "PrimeTiming", "Timing",
                          "Burst"})
#: version of the pool rule of :func:`build_condition_table`; the package copy must record it
#: (:func:`tngxmm.instrument.donor_tables.measurement_record`).  2: an observation with a camera in a timing mode
#: leaves the pool
MEASURE_VERSION = 2


# ── the condition table ──────────────────────────────────────────────────────

def sun_radec(jd: Any) -> tuple[Any, Any]:
    """(RA, Dec) [rad] of the Sun at Julian date ``jd``: the low-precision formula of the
    Astronomical Almanac (about 0.01 deg between 1950 and 2050)."""
    import numpy as np

    n = np.asarray(jd, float) - 2451545.0
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 4e-7 * n)
    return np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)), np.arcsin(np.sin(eps) * np.sin(lam))


def position_angle_deg(ra1: Any, dec1: Any, ra2: Any, dec2: Any) -> tuple[Any, Any]:
    """(position angle [deg, 0-360, N through E] of point 2 seen from point 1, separation [deg]);
    inputs in radians."""
    import numpy as np

    d = ra2 - ra1
    pa = np.arctan2(np.sin(d), np.cos(dec1) * np.tan(dec2) - np.sin(dec1) * np.cos(d))
    sep = np.arccos(np.clip(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(d), -1, 1))
    return np.rad2deg(pa) % 360.0, np.rad2deg(sep)


def nominal_pa_deg(ra_deg: Any, dec_deg: Any, date_obs: Any) -> tuple[Any, Any]:
    """(nominal position angle [deg], solar aspect angle [deg]) of a pointing at ``date_obs``
    (ISO strings): the position angle of the Sun seen from the pointing, which the bank's PA_PNT
    follows (median offset 0.0 deg, all 638 within +-19 deg)."""
    import numpy as np
    import pandas as pd

    jd = pd.to_datetime(pd.Series(np.atleast_1d(date_obs))).map(lambda x: x.to_julian_date()).to_numpy()
    rs, ds = sun_radec(jd)
    return position_angle_deg(np.deg2rad(np.atleast_1d(ra_deg).astype(float)),
                              np.deg2rad(np.atleast_1d(dec_deg).astype(float)), rs, ds)


def wrap180(x: Any) -> Any:
    """Angles [deg] wrapped to [-180, 180)."""
    import numpy as np

    return ((np.asarray(x, float) + 180.0) % 360.0) - 180.0


@cites("xmm_uhb", "xmm_mission")
def build_condition_table(targets: Any, background: Any, core_offsets: Any, active_ccds: Any,
                          seed_lx: Any) -> Any:
    """One row per real observation: its conditions (package fields), the condition-space columns,
    its pointing and roll, and ``pool`` (usable as a conditions donor: a background-table row with
    finite positive live times for every camera, finite positive ``qpb_scale_<cam>`` and ``nh_1e22``, finite ``theta500_arcmin`` and ``pa_pnt_deg``, a usable core offset, an active-CCD row and no camera in a
    timing mode, :data:`TIMING_MODES`).

    ``targets``: the replica target table (one or more rows per donor; ``donor_id``, ``ref_name``,
    ``ref_sample``, ``ref_z``, ``exposure_<cam>_donor_s``, ``nh_donor_1e22cm2``, ``filter_<cam>``,
    ``pn_mode``, ``submode_mos1/2``); ``background``: donor background table; ``core_offsets``:
    donor_core_offsets.csv; ``active_ccds``: donor_active_ccds.csv (``date_obs``); ``seed_lx``:
    obs_seed_lx (``obsid``, ``r500_kpc``)."""
    import numpy as np
    import pandas as pd

    from tngxmm.donors.build_replica_targets import _theta_arcmin

    z = lambda s: s.astype(str).str.zfill(10)  # noqa: E731
    t = targets.copy()
    t["obsid"] = z(t["donor_id"])
    t = t.drop_duplicates("obsid").set_index("obsid")
    out = pd.DataFrame(index=t.index)
    out["name"] = t.get("ref_name")
    out["sample"] = t.get("ref_sample")
    out["z_donor"] = t["ref_z"].astype(float)
    for c in CAMERAS:
        out[f"exposure_{_SUF[c]}_s"] = t[f"exposure_{_SUF[c]}_donor_s"].astype(float)
        out[f"filter_{_SUF[c]}"] = t[f"filter_{_SUF[c]}"]
    out["nh_1e22"] = t["nh_donor_1e22cm2"].astype(float)
    out["pn_mode"] = t["pn_mode"]
    out["submode_mos1"] = t["submode_mos1"]
    out["submode_mos2"] = t["submode_mos2"]
    b = background.copy()
    b.index = z(b["obsid"])
    for c in CAMERAS:
        out[f"live_s_{_SUF[c]}"] = b[f"live_s_{_SUF[c]}"].reindex(out.index).astype(float)
        out[f"qpb_scale_{_SUF[c]}"] = b[f"qpb_scale_{_SUF[c]}"].reindex(out.index).astype(float)
    co = core_offsets.copy()
    co.index = z(co["obsid"])
    for col in ("core_dx_mask_arcmin", "core_dy_mask_arcmin", "core_offset_arcmin", "ra_pnt_deg",
                "dec_pnt_deg", "pa_pnt_deg"):
        out[col] = co[col].reindex(out.index).astype(float)
    out["core_usable"] = co["usable"].astype(str).str.lower().eq("true").reindex(out.index).fillna(False)
    ac = active_ccds.copy()
    ac.index = z(ac["obsid"])
    out["date_obs"] = ac["date_obs"].reindex(out.index)
    s = seed_lx.copy()
    s.index = z(s["obsid"])
    r500 = s["r500_kpc"].reindex(out.index).astype(float)
    out["theta500_arcmin"] = _theta_arcmin(r500.to_numpy(), out["z_donor"].to_numpy())
    ok = out.date_obs.notna() & out.ra_pnt_deg.notna()
    out["pa_nominal_deg"] = np.nan
    out["saa_deg"] = np.nan
    if ok.any():
        pa, saa = nominal_pa_deg(out.ra_pnt_deg[ok], out.dec_pnt_deg[ok], out.date_obs[ok])
        out.loc[ok, "pa_nominal_deg"] = pa
        out.loc[ok, "saa_deg"] = saa
    out["dpa_deg"] = wrap180(out.pa_pnt_deg - out.pa_nominal_deg)
    pool = out.core_usable & ac.index.to_series().reindex(out.index).notna() & out.pa_pnt_deg.notna()
    for c in CAMERAS:
        pool &= np.isfinite(out[f"live_s_{_SUF[c]}"]) & (out[f"live_s_{_SUF[c]}"] > 0)
        pool &= np.isfinite(out[f"qpb_scale_{_SUF[c]}"]) & (out[f"qpb_scale_{_SUF[c]}"] > 0)
    pool &= np.isfinite(out.nh_1e22) & (out.nh_1e22 > 0) & np.isfinite(out.theta500_arcmin)
    for col in ("pn_mode", "submode_mos1", "submode_mos2"):
        pool &= ~out[col].astype(str).isin(TIMING_MODES)
    out["pool"] = pool.astype(bool)
    out.index.name = "obsid"
    return out.reset_index()


def condition_columns(table: Any) -> Any:
    """The condition-space columns of :data:`CONTINUOUS` and :data:`CATEGORICAL`."""
    import numpy as np
    import pandas as pd

    d = pd.DataFrame(index=table.index)
    for s in ("mos1", "mos2", "pn"):
        d[f"log_live_{s}"] = np.log10(table[f"live_s_{s}"].astype(float))
        d[f"log_qpb_{s}"] = np.log10(table[f"qpb_scale_{s}"].astype(float))
    d["log_nh"] = np.log10(table["nh_1e22"].astype(float))
    d["log_theta500"] = np.log10(table["theta500_arcmin"].astype(float))
    d["core_offset_arcmin"] = table["core_offset_arcmin"].astype(float)
    d["filter_pn"] = table["filter_pn"].astype(str)
    d["filter_mos"] = table["filter_mos1"].astype(str) + "/" + table["filter_mos2"].astype(str)
    d["pn_mode"] = table["pn_mode"].astype(str)
    return d


def robust_scale(x: Any) -> float:
    """1.4826 x the median absolute deviation (the standard deviation of a Gaussian), floored by
    a tenth of the standard deviation so a column with a degenerate MAD still has a scale."""
    import numpy as np

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 1.0
    s = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return max(s, 0.1 * float(np.std(x)), 1e-12)


def _missing_category(v: Any) -> bool:
    """A categorical condition value built from a missing entry (``astype(str)`` of NaN: 'nan', 'nan/nan', 'None')."""
    return any(p.strip().lower() in ("", "nan", "none", "<na>") for p in str(v).split("/"))


def missing_dimensions(table: Any, obsid: str) -> list[str]:
    """The condition-space dimensions (:data:`CONTINUOUS`, :data:`CATEGORICAL`) that ``obsid`` lacks (a camera without
    live time or background scale: a physics donor outside the pool) and that :func:`condition_distances` leaves out of ITS
    distances, in the order of the two tuples; empty for a complete row."""
    import numpy as np

    t = table.set_index(table["obsid"].astype(str).str.zfill(10), drop=False)
    key = str(obsid).zfill(10)
    if key not in t.index:
        raise KeyError(f"observation {key} is not in the condition table")
    c = condition_columns(t)
    return ([col for col in CONTINUOUS if not np.isfinite(c.at[key, col])]
            + [col for col in CATEGORICAL if _missing_category(c.at[key, col])])


def condition_distances(table: Any, obsid: str, weights: dict[str, float] | None = None,
                        folds: dict[str, int] | None = None) -> Any:
    """Squared distances of every POOL observation from ``obsid`` (a row of ``table``, in the pool
    or not) in the condition space; a Series indexed by obsid, ascending, the observation itself
    first when it is in the pool (distance 0).

    ``folds`` (obsid -> donor fold, :func:`read_donor_folds`) restricts the pool to the observations of the SAME donor
    fold as ``obsid`` (the standardising spreads stay those of the whole pool, so the distances are the unrestricted
    ones); ``KeyError`` when ``obsid`` has no fold.

    A dimension ``obsid`` lacks (:func:`missing_dimensions`) is left out of ITS distances: a continuous one so the
    candidate list is not emptied by a NaN, a categorical one because its mismatch with every pool row is a constant
    (the ranking is the same, the recorded distance is not inflated by it)."""
    import numpy as np
    import pandas as pd

    t = table.set_index(table["obsid"].astype(str).str.zfill(10), drop=False)
    key = str(obsid).zfill(10)
    if key not in t.index:
        raise KeyError(f"observation {key} is not in the condition table")
    c = condition_columns(t)
    pool = t["pool"].astype(bool).to_numpy()
    w = dict(weights or {})
    d2 = pd.Series(0.0, index=t.index)
    for col in CONTINUOUS:
        if not np.isfinite(c.at[key, col]):       # a physics donor outside the pool may lack a dimension (a camera
            continue                              # without live time): it does not enter ITS distances, so it still has candidates
        s = robust_scale(c[col][pool])
        d2 += float(w.get(col, 1.0)) * ((c[col] - c.at[key, col]) / s) ** 2
    for col in CATEGORICAL:
        if _missing_category(c.at[key, col]):     # a missing category mismatches EVERY pool row: a constant that
            continue                              # changes no ranking but inflates the recorded distance; left out
        d2 += float(w.get(col, 1.0)) * CATEGORY_PENALTY[col] * (c[col] != c.at[key, col]).astype(float)
    d2 = d2[pool & np.isfinite(d2.to_numpy())]
    if folds is not None:
        if key not in folds:
            raise KeyError(f"observation {key} has no donor fold: the conditions donor cannot be restricted to its fold")
        d2 = d2[[folds.get(k) == folds[key] for k in d2.index]]
    order = sorted(d2.index, key=lambda k: (d2[k], k != key, k))
    return d2.loc[order]


def neighbours(table: Any, obsid: str, n: int, weights: dict[str, float] | None = None,
               folds: dict[str, int] | None = None) -> list[str]:
    """The ``n`` nearest pool observations of ``obsid`` in the condition space (itself first when
    it is in the pool); with ``folds``, only those of its donor fold (:func:`condition_distances`).  A fold with fewer
    than ``n`` pool observations gives fewer candidates."""
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    return list(condition_distances(table, obsid, weights, folds).index[: int(n)])


#: the donor folds of the bank (obsid, donor_fold): THE assignment of the package, shared by the view tables
#: (``donor_fold`` of ``targets_v4.csv``), the bank manifest and the conditions rule's ``fold_restricted``
DONOR_FOLDS_FILE = "campaigns/v4/donor_folds.csv"


def donor_folds_path(path: str | Path | None = None) -> Path:
    """The donor-fold table: ``path`` when given, else the repository's :data:`DONOR_FOLDS_FILE` (also shipped in an
    installed package)."""
    from tngxmm.core.paths import repo_file

    return Path(path) if path else repo_file(DONOR_FOLDS_FILE)


_FOLDS: dict[str, dict[str, int]] = {}


def read_donor_folds(path: str | Path | None = None) -> dict[str, int]:
    """``{obsid (10 digits): donor fold}`` of the donor-fold table (:func:`donor_folds_path`)."""
    import pandas as pd

    p = str(donor_folds_path(path))
    if p not in _FOLDS:
        t = pd.read_csv(p, dtype={"obsid": str})
        _FOLDS[p] = {str(o).zfill(10): int(f) for o, f in zip(t["obsid"], t["donor_fold"])}
    return _FOLDS[p]


def folds_sha256(conditions: dict[str, Any]) -> str | None:
    """sha256 of the donor-fold table a ``fold_restricted`` conditions rule uses (``folds`` null = the repository's);
    None when the rule is not fold restricted, ``'absent'`` when the file is missing."""
    from tngxmm.core.hashing import sha256_file

    if not (conditions or {}).get("fold_restricted"):
        return None
    p = donor_folds_path(conditions.get("folds") or None)
    return sha256_file(p) if Path(p).is_file() else "absent"


# ── the draws ────────────────────────────────────────────────────────────────

@cites("numpy")
def choose_conditions(candidates: list[str], physics_obsid: str, replica_k: int | None, view_rng: Any,
                      campaign: str) -> tuple[str, int, str]:
    """(conditions obsid, its index among ``candidates``, method).

    With ``replica_k`` (1-based): element ``(k - 1) mod N`` of a permutation of the N candidates
    seeded per physics donor (:func:`tngxmm.core.rng.seed_for_donor`, purpose ``conditions``), the
    K replicas of a donor thereby taking different candidates (``'donor-permutation'``).  Without
    it, a uniform index from ``view_rng`` (``'view-uniform'``).  ``view_rng`` is drawn from in both
    cases (one integer), so the later jitter draws do not depend on the method."""
    import numpy as np

    from tngxmm.core.rng import seed_for_donor

    n = len(candidates)
    if n < 1:
        raise ValueError("no conditions candidate")
    u = int(view_rng.integers(n))
    if replica_k is not None and int(replica_k) >= 1:
        perm = np.random.default_rng(seed_for_donor(physics_obsid, "conditions", campaign=campaign)).permutation(n)
        i = int(perm[(int(replica_k) - 1) % n])
        return candidates[i], i, "donor-permutation"
    return candidates[u], u, "view-uniform"


def draw_jitter(view_rng: Any, cond_row: Any, cand_rows: Any, dpa_archive: Any,
                kinds: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """The jitter of one view (module docstring); every number is drawn whatever ``kinds`` holds,
    in a fixed order (exposure normal, archive index of the off-nominal roll, two pointing
    normals), and only the components in ``kinds`` are applied.

    ``cond_row``: the conditions donor's row of the condition table; ``cand_rows``: the rows of the
    N candidates; ``dpa_archive``: the archive's off-nominal roll angles [deg] (``dpa_deg``).
    Returns ``exposure_scale`` (1 when off), ``pa_new_deg`` and ``droll_deg`` (the change of the
    position angle against the conditions donor's, 0 when off), ``pointing_dx/dy_arcmin`` (0 when
    off) and the raw draws."""
    import numpy as np

    kinds = tuple(kinds or ())
    bad = [k for k in kinds if k not in JITTER_KINDS]
    if bad:
        raise ValueError(f"unknown jitter {bad}; known: {JITTER_KINDS}")
    z_exp = float(view_rng.standard_normal())
    dpa = np.asarray(dpa_archive, float)
    dpa = dpa[np.isfinite(dpa) & (np.abs(dpa) <= ROLL_MAX_DEG)]
    i_roll = int(view_rng.integers(len(dpa))) if len(dpa) else -1
    z_px, z_py = (float(v) for v in view_rng.standard_normal(2))

    rec: dict[str, Any] = {"jitter": list(kinds), "z_exposure": z_exp, "roll_archive_index": i_roll,
                           "z_pointing": [z_px, z_py]}
    # exposure: common log-normal factor, sigma = robust spread of log t_eff among the candidates
    teff = _t_eff(cand_rows)
    lt = np.log(teff[np.isfinite(teff) & (teff > 0)])
    t0 = float(_t_eff(cond_row))
    sig = robust_scale(lt) if len(lt) >= 2 else 0.0
    scale = 1.0
    if "exposure" in kinds and sig > 0 and len(lt):
        t_new = float(np.clip(t0 * math.exp(sig * z_exp), math.exp(lt.min()), math.exp(lt.max())))
        scale = t_new / t0
    rec.update(exposure_scale=scale, exposure_sigma_ln=sig, t_eff_s=t0, t_eff_range_s=[
        float(math.exp(lt.min())) if len(lt) else float("nan"), float(math.exp(lt.max())) if len(lt) else float("nan")])
    # roll: nominal angle of the donor's date + an archive off-nominal angle
    pa0 = float(cond_row["pa_pnt_deg"])
    pa_new, droll = pa0, 0.0
    if "roll" in kinds:
        nom = float(cond_row["pa_nominal_deg"])
        if not (math.isfinite(nom) and i_roll >= 0):
            raise ValueError("roll jitter needs the nominal position angle and an archive of off-nominal rolls")
        pa_new = (nom + float(dpa[i_roll])) % 360.0
        droll = float(wrap180(pa_new - pa0))
    rec.update(pa_donor_deg=pa0, pa_new_deg=pa_new, droll_deg=droll)
    # pointing: isotropic Gaussian aimpoint offset; a radius beyond POINTING_MAX_ARCMIN is CLIPPED onto that circle (not redrawn), so a
    # small mass of offsets sits exactly on the cap
    dx, dy = 0.0, 0.0
    if "pointing" in kinds:
        dx, dy = POINTING_SIGMA_ARCMIN * z_px, POINTING_SIGMA_ARCMIN * z_py
        r = math.hypot(dx, dy)
        if r > POINTING_MAX_ARCMIN:
            dx, dy = dx * POINTING_MAX_ARCMIN / r, dy * POINTING_MAX_ARCMIN / r
    rec.update(pointing_dx_arcmin=dx, pointing_dy_arcmin=dy)
    return rec


def _t_eff(rows: Any) -> Any:
    """Effective live time [s]: the band-mean-ARF weighted sum of the cameras' live times (the
    weights of donors.build_replica_targets: MOS 395/400, pn 1259 cm^2) over their sum."""
    import numpy as np

    a = {_SUF[c]: w for c, w in T_EFF_WEIGHTS_CM2.items()}
    num = sum(a[s] * np.asarray(rows[f"live_s_{s}"], float) for s in a)
    return num / sum(a.values())


def effective_exposure_s(cameras_s: dict[str, float]) -> float:
    """Effective live time [s] of per-camera live times ``cameras_s`` (keys PN, MOS1, MOS2): the ARF-weighted sum
    ``sum_c w_c t_c / sum_c w_c`` over ALL cameras (:data:`T_EFF_WEIGHTS_CM2`), a camera without data contributing
    0, so the ratio of two such times is the ratio of the counts a source of fixed flux gives."""
    return sum(w * float(cameras_s.get(c) or 0.0) for c, w in T_EFF_WEIGHTS_CM2.items()) / sum(T_EFF_WEIGHTS_CM2.values())


def placed_core_offset(core_xy: tuple[float, float], droll_deg: float, dx: float = 0.0,
                       dy: float = 0.0) -> tuple[float, float]:
    """Detector-frame (mask) core offset [arcmin] after a roll change ``droll_deg`` and an aimpoint
    offset (dx, dy): the cluster keeps its sky offset from the pointing, so in the detector frame it
    turns by minus the roll change (the products rotate the detector frame by +POSANGLE, and POSANGLE
    follows the position angle one for one), then moves by the aimpoint offset."""
    a = math.radians(-float(droll_deg))
    x, y = float(core_xy[0]), float(core_xy[1])
    return (x * math.cos(a) - y * math.sin(a) + float(dx), x * math.sin(a) + y * math.cos(a) + float(dy))


def posangle_for_pa(pa_deg: float) -> float:
    """The stage's POSANGLE [deg, 0-360) for an SAS position angle (``PA_PNT = POSANGLE + PA_REF``,
    :func:`tngxmm.instrument.sas_headers.pa_pnt_deg`)."""
    from tngxmm.donors.build_core_offsets import PA_REF_DEG

    return float((float(pa_deg) - PA_REF_DEG) % 360.0)


def conditions_package(package: dict[str, Any], cond_row: Any, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """The donor package of a view observed under the conditions donor ``cond_row``: the physics
    donor's identity moves to ``physics_obsid`` / ``physics_donor`` (z_donor and m500_donor_1e14 stay
    the physics donor's), every condition field is the conditions donor's (``obsid``, ``donor``,
    ``sample``, ``cameras_s``, ``nh_1e22``, ``filters``, ``pn_submode``, ``pn_oot_frac``, the MOS
    submodes, the core offset); ``conditions`` holds ``record``."""
    pkg = dict(package)
    pkg["physics_obsid"] = str(package.get("obsid"))
    pkg["physics_donor"] = package.get("donor")
    pkg["obsid"] = str(cond_row["obsid"]).zfill(10)
    pkg["donor"] = cond_row.get("name") if hasattr(cond_row, "get") else cond_row["name"]
    pkg["sample"] = cond_row["sample"]
    pkg["cameras_s"] = {c: float(cond_row[f"exposure_{_SUF[c]}_s"]) for c in CAMERAS
                        if _finite(cond_row[f"exposure_{_SUF[c]}_s"])}
    pkg["nh_1e22"] = float(cond_row["nh_1e22"])
    pkg["filters"] = {c: str(cond_row[f"filter_{_SUF[c]}"]) for c in CAMERAS}
    pkg["pn_submode"] = str(cond_row["pn_mode"])
    pkg["pn_oot_frac"] = pn_oot_fraction(pkg["pn_submode"], "PN" in pkg["cameras_s"], pkg["obsid"])
    pkg["submode_mos1"] = str(cond_row["submode_mos1"])
    pkg["submode_mos2"] = str(cond_row["submode_mos2"])
    pkg["core_dx_mask_arcmin"] = float(cond_row["core_dx_mask_arcmin"])
    pkg["core_dy_mask_arcmin"] = float(cond_row["core_dy_mask_arcmin"])
    pkg["jittered"] = False
    pkg["conditions"] = dict(record or {})
    return pkg


def pn_oot_fraction(mode: str, has_pn: bool, obsid: str = "") -> float:
    """The pn out-of-time fraction of the conditions donor's pn mode (:data:`PN_OOT`).  A donor with a pn exposure in
    a mode the table does not list is refused (``ValueError``): no fraction of another mode stands in for it.  Without
    a pn exposure no pn event is simulated and the fraction is never read; the Full Frame value is recorded."""
    if mode in PN_OOT:
        return float(PN_OOT[mode])
    if has_pn:
        raise ValueError(f"conditions donor {obsid}: pn mode {mode!r} has no out-of-time fraction (known: "
                         f"{sorted(PN_OOT)}); leave the donor out of the pool or add the mode's fraction")
    return float(PN_OOT["PrimeFullWindow"])


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def retarget_keep_prob(e_kev: Any, nh_from: float, nh_to: float, sigma: Any, scale: float = 1.0) -> tuple[Any, float]:
    """(keep probability per photon, deficit): ``scale`` x T(E; nh_to) / T(E; nh_from) capped at 1,
    T = exp(-sigma(E) N_H) with ``sigma`` a callable [cm^2 per H atom] and N_H in 1e22 cm^-2.
    The deficit is the fraction of the wanted photons the cap cannot provide (sum of the excess
    over the sum of the wanted probabilities; 0 when the cap is never reached)."""
    import numpy as np

    e = np.asarray(e_kev, float)
    w = float(scale) * np.exp(-np.asarray(sigma(e), float) * (float(nh_to) - float(nh_from)) * 1e22)
    p = np.minimum(w, 1.0)
    tot = float(w.sum())
    return p, (float((w - p).sum()) / tot if tot > 0 else 0.0)


# ── one view ─────────────────────────────────────────────────────────────────

_TABLES: dict[str, Any] = {}


def read_condition_table(path: str) -> Any:
    """The conditions table CSV at ``path`` (``obsid`` as string), read once per process and cached.  The data
    package's copy is read only when its provenance names the current core-offset, background and centre tables
    (:func:`tngxmm.instrument.donor_tables.check_table`): its core offsets place the cluster on the detector."""
    import pandas as pd

    if path not in _TABLES:
        from tngxmm.instrument.donor_tables import check_table, is_package_copy

        if is_package_copy("conditions", path):
            check_table("conditions", path)
        _TABLES[path] = pd.read_csv(path, dtype={"obsid": str})
    return _TABLES[path]


def replica_index(view_id: str, path: str | None) -> int | None:
    """The 1-based replica index of ``view_id`` from a table with ``view_id`` and ``replica_k``
    (None without a table or a row)."""
    if not path:
        return None
    t = read_condition_table(path)
    hit = t[t["view_id"].astype(str) == str(view_id)]
    return int(hit.iloc[0]["replica_k"]) if len(hit) else None


@cites("xmm_uhb")
def draw_view_conditions(package: dict[str, Any], view_id: str, view_rng: Any, *, campaign: str,
                         table: Any, n: int, replica_k: int | None = None,
                         jitter: tuple[str, ...] | list[str] = (),
                         weights: dict[str, float] | None = None,
                         folds: dict[str, int] | None = None) -> dict[str, Any]:
    """The conditions of one view: ``{'package': conditions package, 'record': ...}``; the record
    holds the candidates, the chosen donor, the method and the jitter (:func:`draw_jitter`), the
    placed core offset and the POSANGLE to use (None: keep the view's own) and ``dims_skipped`` (the condition dimensions the
    physics donor lacks, left out of its distances; empty for a complete row).  ``folds``: restrict the candidates to the
    physics donor's donor fold (:func:`condition_distances`); the record then holds ``fold`` (that fold)."""
    t = table.set_index(table["obsid"].astype(str).str.zfill(10), drop=False)
    phys = str(package["obsid"]).zfill(10)
    cands = neighbours(table, phys, n, weights, folds)
    cond, idx, method = choose_conditions(cands, phys, replica_k, view_rng, campaign)
    row = t.loc[cond]
    jit = draw_jitter(view_rng, row, t.loc[cands], table["dpa_deg"], jitter)
    core = placed_core_offset((float(row["core_dx_mask_arcmin"]), float(row["core_dy_mask_arcmin"])),
                              jit["droll_deg"], jit["pointing_dx_arcmin"], jit["pointing_dy_arcmin"])
    d2 = condition_distances(table, phys, weights, folds)
    rec = dict(rule="nn", n=int(n), physics_obsid=phys, view_id=str(view_id), replica_k=replica_k,
               fold_restricted=folds is not None, fold=(int(folds[phys]) if folds is not None else None),
               dims_skipped=missing_dimensions(table, phys),
               candidates=cands, chosen=cond, chosen_index=idx, method=method,
               distance=float(d2.get(cond, float("nan"))) ** 0.5,
               core_offset_arcmin=list(core),
               posangle_deg=(posangle_for_pa(jit["pa_new_deg"]) if "roll" in tuple(jitter) else None),
               nh_physics_1e22=float(package.get("nh_1e22", float("nan"))), nh_conditions_1e22=float(row["nh_1e22"]),
               **jit)
    pkg = conditions_package(package, row, rec)
    pkg["core_dx_mask_arcmin"], pkg["core_dy_mask_arcmin"] = core
    return {"package": pkg, "record": rec}


# ── per projection (release 1.2) ─────────────────────────────────────────────

#: the donor-package key of the per-projection draw: ``{projection (as a string, a JSON key): anchor obsid}`` of the
#: views of the package's halo-snapshot (:func:`tngxmm.core.pool.read_replica_packages`)
ANCHORS_KEY = "conditions_anchors"


def projection_anchors(package: dict[str, Any]) -> dict[int, str]:
    """``{projection: anchor obsid (10 digits)}`` of the halo-snapshot of a view, from its package key
    :data:`ANCHORS_KEY` (:func:`tngxmm.core.pool.halo_projection_anchors`: every projection of the scheme, read before
    any exclusion).  The anchor of a projection is the observation whose nearest pool observations are its conditions
    candidates: its physics donor, or for an unpaired halo-snapshot the nearest observation of the pairing
    (``conditions_anchor_id`` of :mod:`tngxmm.donors.pair_projections`).  ``ValueError`` without the key or when the
    projections are not 0, 1, ... without a gap."""
    a = package.get(ANCHORS_KEY)
    if not a:
        raise ValueError(f"per-projection conditions: the donor package has no {ANCHORS_KEY!r} (the targets of the "
                         "view's halo-snapshot; core.pool.read_replica_packages(projection_anchors=True))")
    out = {int(k): str(v).zfill(10) for k, v in dict(a).items()}
    if sorted(out) != list(range(len(out))):
        raise ValueError(f"per-projection conditions: the anchors name projections {sorted(out)}, not 0-{len(out) - 1}")
    return out


def sibling_conditions_rng(streams: Any, view: int) -> Any:
    """A new Generator of the ``conditions`` stream of projection ``view`` of the halo-snapshot of ``streams`` (a
    :class:`tngxmm.core.rng.ViewStreams`: same stream key, simulation, snapshot, halo, realisation and scheme).  Its
    first integer is the one that projection's own draw takes (:func:`projection_draws`)."""
    import numpy as np

    from tngxmm.core.rng import ViewId, seed_for

    v = streams.v
    return np.random.default_rng(seed_for(ViewId(v.sim, int(v.snap), int(v.halo_id), int(view)), "conditions",
                                          campaign=streams.campaign, realisation=streams.realisation,
                                          scheme=streams.scheme))


def projection_draws(table: Any, anchors: dict[int, str], n: int, first_integer: Any, upto: int,
                     weights: dict[str, float] | None = None,
                     folds: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The conditions donors of the projections of one halo-snapshot, drawn in view order up to and including
    projection ``upto``: projection q takes ``candidates[u]``, the candidates being the ``n`` nearest pool observations
    of ``anchors[q]`` (in its donor fold with ``folds``, :func:`condition_distances`) that no earlier projection has
    taken, and ``u = first_integer(q, len(candidates))`` (the first integer of projection q's ``conditions`` stream;
    with a single candidate numpy draws nothing for the range of one and u is 0).
    One dict per projection: ``projection``, ``anchor``, ``candidates``, ``chosen``, ``chosen_index``,
    ``taken_before`` (the earlier projections' conditions donors) and ``distance`` from the anchor.  ``ValueError`` when
    a projection is left without a candidate (its fold's pool is used up)."""
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    taken: list[str] = []
    out: list[dict[str, Any]] = []
    for q in sorted(anchors):
        if q > int(upto):
            break
        d2 = condition_distances(table, anchors[q], weights, folds)
        cands = [o for o in d2.index if o not in taken][: int(n)]
        if not cands:
            raise ValueError(f"projection {q}: no conditions candidate left for anchor {anchors[q]} "
                             f"after {len(taken)} earlier projections")
        u = int(first_integer(q, len(cands)))
        out.append(dict(projection=int(q), anchor=anchors[q], candidates=cands, chosen=cands[u], chosen_index=u,
                        taken_before=list(taken), distance=float(d2[cands[u]]) ** 0.5))
        taken.append(cands[u])
    return out


@cites("xmm_uhb")
def draw_projection_conditions(package: dict[str, Any], view_id: str, streams: Any, *, campaign: str, table: Any,
                               n: int, jitter: tuple[str, ...] | list[str] = (),
                               weights: dict[str, float] | None = None,
                               folds: dict[str, int] | None = None) -> dict[str, Any]:
    """The conditions of one view under the per-projection draw (release 1.2): ``{'package', 'record'}`` as
    :func:`draw_view_conditions`.

    The view is projection ``streams.v.view`` of its halo-snapshot; ``package`` carries the anchors of all projections
    of that halo-snapshot (:func:`projection_anchors`).  The earlier projections' choices are recomputed from their
    own ``conditions`` streams (:func:`sibling_conditions_rng`), this view's from ``streams.get('conditions')``, whose
    first integer picks the candidate (none with a single candidate) and whose following numbers are the jitter
    (:func:`draw_jitter`), as under :func:`draw_view_conditions`.  The physics donor of a paired view
    (``package['obsid']``) must be its anchor; a package whose ``paired`` flag reads False
    (:func:`tngxmm.donors.pair_projections.paired_value`) records no physics donor (such packages are not made yet:
    generating unpaired views is release 1.2 work, E1).  ``campaign`` is the stream key of the view (that of
    ``streams``), as in :func:`draw_view_conditions`."""
    from tngxmm.donors.pair_projections import paired_value

    t = table.set_index(table["obsid"].astype(str).str.zfill(10), drop=False)
    anchors = projection_anchors(package)
    view = int(streams.v.view)
    if view not in anchors:
        raise KeyError(f"projection {view} is not among the projections {sorted(anchors)} of its package")
    if str(campaign) != str(streams.campaign):
        raise ValueError(f"stream key {campaign!r} differs from the view's streams ({streams.campaign!r})")
    paired = paired_value(package.get("paired", True))
    phys = str(package.get("obsid")).zfill(10) if paired else None
    if paired and phys != anchors[view]:
        raise ValueError(f"projection {view}: physics donor {phys} of the package is not its anchor {anchors[view]}")
    own = streams.get("conditions")

    def first_integer(q: int, m: int) -> int:
        return int((own if q == view else sibling_conditions_rng(streams, q)).integers(m))

    me = projection_draws(table, anchors, n, first_integer, view, weights, folds)[-1]
    row = t.loc[me["chosen"]]
    jit = draw_jitter(own, row, t.loc[me["candidates"]], table["dpa_deg"], jitter)
    core = placed_core_offset((float(row["core_dx_mask_arcmin"]), float(row["core_dy_mask_arcmin"])),
                              jit["droll_deg"], jit["pointing_dx_arcmin"], jit["pointing_dy_arcmin"])
    anchor = anchors[view]
    rec = dict(rule="nn", n=int(n), per_projection=True, physics_obsid=phys, anchor=anchor, view_id=str(view_id),
               projection=view, projections=sorted(anchors), taken_before=me["taken_before"],
               fold_restricted=folds is not None, fold=(int(folds[anchor]) if folds is not None else None),
               dims_skipped=missing_dimensions(table, anchor),
               candidates=me["candidates"], chosen=me["chosen"], chosen_index=me["chosen_index"], method="projection",
               distance=me["distance"], core_offset_arcmin=list(core),
               posangle_deg=(posangle_for_pa(jit["pa_new_deg"]) if "roll" in tuple(jitter) else None),
               nh_physics_1e22=float(package.get("nh_1e22", float("nan"))), nh_conditions_1e22=float(row["nh_1e22"]),
               **jit)
    pkg = conditions_package(package, row, rec)
    pkg["physics_obsid"] = phys
    pkg["core_dx_mask_arcmin"], pkg["core_dy_mask_arcmin"] = core
    return {"package": pkg, "record": rec}


def draw_conditions(conditions: dict[str, Any], pkg: dict[str, Any], meta: dict[str, Any], streams: Any,
                    campaign: str) -> dict[str, Any]:
    """The draw half of :func:`stage_conditions`: the conditions donor of a view and its jitter, from the
    view's ``conditions`` stream, BEFORE any photon exists, so the photon list can be sized for the live times
    the instrument will use (the longest jittered camera time, :func:`longest_exposure_s`).

    Arguments as :func:`stage_conditions` (``meta`` needs only ``view_id`` here; under ``per_projection`` ``streams``
    is the view's :class:`~tngxmm.core.rng.ViewStreams` and ``pkg`` carries the anchors of its halo-snapshot,
    :func:`draw_projection_conditions`).  Returns ``package``,
    ``record`` (the draw; :func:`stage_events` completes it), ``conditions`` (the resolved setting), ``rng``
    (the view's ``conditions`` Generator, which :func:`stage_events` continues: the keep draws follow the
    jitter on that stream) and ``exposure_s`` (the per-camera live times of the instrument stage)."""
    c = dict(conditions)
    rule = c.get("rule", "nn")
    if rule not in CONDITION_RULES:
        raise ValueError(f"unknown conditions rule {rule!r}; known: {CONDITION_RULES}")
    vr = streams.get("conditions")
    if rule == "donor":                    # the physics donor's own conditions; only the count-matching thinning acts
        d = {"package": pkg, "record": {"rule": "donor", "exposure_scale": 1.0, "nh_conditions_1e22": None}}
    else:
        c = resolve_table(c)                  # null = the package's donor_conditions.csv
        table = read_condition_table(str(c["table"]))
        view_id = str(pkg.get("view_id") or meta.get("view_id") or "")
        folds = read_donor_folds(c.get("folds") or None) if c.get("fold_restricted") else None
        if c.get("per_projection"):           # release 1.2: the projections of a halo-snapshot draw in turn
            d = draw_projection_conditions(pkg, view_id, streams, campaign=campaign, table=table,
                                           n=int(c.get("n", 5)), jitter=tuple(c.get("jitter") or ()),
                                           weights=c.get("weights"), folds=folds)
        else:
            k = c.get("replica_k")
            if k is None:
                k = replica_index(view_id, c.get("replicas"))
            if k is None:                     # the replica index of the targets row (generation)
                k = pkg.get("replica_k")
            d = draw_view_conditions(pkg, view_id, vr, campaign=campaign, table=table, n=int(c.get("n", 5)),
                                     replica_k=k, jitter=tuple(c.get("jitter") or ()), weights=c.get("weights"),
                                     folds=folds)
    return {"package": d["package"], "record": d["record"], "conditions": c, "rng": vr,
            "exposure_s": scale_exposure(d["package"]["cameras_s"], d["record"])}


#: the recorded status of a replay whose conditions ask for a longer camera time than the stored photon list holds
REFUSED_STATUS = "refused: conditions longer than the stored photon list"
#: relative tolerance of the comparison of a camera time with the list's ``exp_time`` (the times are products of the same floats)
EXPOSURE_RTOL = 1e-9


class ConditionsRefused(ValueError):
    """A replay's conditions need a camera live time above the ``exp_time`` of the stored photon list: the list cannot
    supply the camera (the summed camera-assignment probability would exceed 1,
    :func:`tngxmm.instrument.simulate_instrument.check_camera_partition`).  A refusal of THIS view; ``status`` is the
    string recorded for it (:data:`REFUSED_STATUS`)."""

    status = REFUSED_STATUS


#: the recorded status of a replay whose redrawn conditions donor is not the one the stored tree was made under
STALE_STATUS = "refused: stored conditions donor differs"


class StoredConditionsDiffer(ConditionsRefused):
    """The conditions donor a replay redraws under the preset's own conditions (no re-seed) is not the stored view's
    ``obs_meta`` ``conditions_obsid``: the tree was made under another rule, table, fold restriction or release (a tree
    older than the fold-restricted rule replayed under it).  A refusal of THIS view (``status`` :data:`STALE_STATUS`)."""

    status = STALE_STATUS


def check_stored_conditions_donor(record: dict[str, Any], meta: dict[str, Any]) -> None:
    """Raise :class:`StoredConditionsDiffer` when the redrawn conditions donor ``record['chosen']`` differs from
    ``meta['conditions_obsid']`` (compared as 10-digit obsids).  A stored view without the key (a tree of an earlier
    release) or a record without a chosen donor (the rule ``donor``) is not checked."""
    stored, chosen = meta.get("conditions_obsid"), (record or {}).get("chosen")
    if not stored or not chosen:
        return
    if str(stored).zfill(10) != str(chosen).zfill(10):
        raise StoredConditionsDiffer(f"{STALE_STATUS} (stored {str(stored).zfill(10)}, redrawn {str(chosen).zfill(10)})")


def stored_list_exposure_s(evt_file: str | Path) -> float:
    """``parameters/exp_time`` [s] of a stored photon list: the exposure the list was sized for."""
    import h5py

    with h5py.File(str(evt_file), "r") as f:
        return float(f["parameters"]["exp_time"][()])


def check_list_covers(exposure_s: dict[str, float], exp_time_s: float) -> None:
    """Raise :class:`ConditionsRefused` when the longest camera time of ``exposure_s`` exceeds ``exp_time_s``, the
    exposure of the stored photon list (its keep probability is ARF / A_gen x t_cam / t_gen: above ``t_gen`` the list is
    too short for the camera).  A replay under conditions other than the generation's (a rule-donor replay of a tree made
    under the rule nn, a study's other conditions table) can ask for more than the list holds; the check is per view."""
    t = longest_exposure_s(exposure_s)
    if t > float(exp_time_s) * (1.0 + EXPOSURE_RTOL):
        raise ConditionsRefused(f"{REFUSED_STATUS} (camera time {t:g} s > exp_time {float(exp_time_s):g} s)")


def table_sha256(conditions: dict[str, Any]) -> str | None:
    """sha256 of the condition table the rule ``nn`` of ``conditions`` draws from (``table`` null = the data package's
    ``donor_conditions.csv``), a hash of the file's bytes; ``'absent'`` when the file is missing, None under the rule
    ``donor`` (no table)."""
    from tngxmm.core.hashing import sha256_file
    from tngxmm.instrument.donor_tables import table_path

    if (conditions or {}).get("rule") == "donor":
        return None
    p = table_path("conditions", conditions.get("table") or None)
    return sha256_file(p) if Path(p).is_file() else "absent"


def longest_exposure_s(exposure_s: dict[str, float]) -> float:
    """The photon-list exposure [s] for per-camera live times ``exposure_s``: the longest, so the keep
    probability of every camera, ARF_c / A_gen x t_c / t_gen, stays <= ARF_c / A_gen."""
    return float(max(exposure_s.values()))


def stage_events(draw: dict[str, Any], meta: dict[str, Any], evt_file: Path, out_dir: Path,
                 abund: str = "lodd") -> dict[str, Any]:
    """The staging half of :func:`stage_conditions`: the absorbed photon list ``evt_file`` of the physics view,
    thinned by the count-matching factor and retargeted to the conditions donor's N_H (``draw``: the output of
    :func:`draw_conditions`), written to ``out_dir/stage_events.h5``.  ``meta`` needs ``nh_1e22`` (the list's
    N_H when the file does not say).  Returns ``package``, ``record``, ``nh_1e22``, ``evt_file``."""
    import h5py
    import numpy as np

    c, rec, vr = draw["conditions"], draw["record"], draw["rng"]
    rule = c.get("rule", "nn")
    f = float(c.get("thin", 1.0))
    with h5py.File(str(evt_file), "r") as a:
        e = a["data/eobs"][()]
        params = a["parameters"]
        nh_list = float(params["nH_1e22"][()]) if "nH_1e22" in params else float(meta["nh_1e22"])
        model = params["absorb_model"][()] if "absorb_model" in params else b"tbabs"
        model = model.decode() if isinstance(model, bytes) else str(model)
        if model.lower() != "tbabs":
            raise ValueError(f"conditions: the photon list was absorbed by {model!r}; the N_H retarget is TBabs")
        from tngxmm.sim.apply_absorption import tbabs_sigma

        if rule == "donor":
            rec["nh_conditions_1e22"] = nh_list
        p, deficit = retarget_keep_prob(e, nh_list, rec["nh_conditions_1e22"], tbabs_sigma(abund), f)
        keep = vr.random(len(e)) < p
        stage = out_dir / "stage_events.h5"
        with h5py.File(str(stage), "w") as b:
            g = b.create_group("data")
            for col in ("eobs", "xsky", "ysky"):
                g.create_dataset(col, data=a["data"][col][()][keep])
            a.copy("parameters", b)
            if "nH_1e22" in b["parameters"]:
                b["parameters"]["nH_1e22"][()] = float(rec["nh_conditions_1e22"])
    rec.update(thin=f, nh_list_1e22=nh_list, absorb_model=model, abund_table=abund,
               n_parent=int(len(e)), n_kept=int(keep.sum()), nh_deficit=deficit,
               mean_keep=float(np.mean(p)) if len(p) else float("nan"))
    return {"package": draw["package"], "record": rec, "nh_1e22": float(rec["nh_conditions_1e22"]),
            "evt_file": stage}


def stage_conditions(conditions: dict[str, Any], pkg: dict[str, Any], meta: dict[str, Any], streams: Any,
                    campaign: str, evt_file: Path, out_dir: Path, abund: str = "lodd") -> dict[str, Any]:
    """Draw the conditions donor of a view and stage its photon list: the ONE implementation of the
    conditions rule.  The generation (:func:`tngxmm.sim.generate_view.generate_halo_views`) runs the two halves
    separately, :func:`draw_conditions` before the photons (they are sized for its live times) and
    :func:`stage_events` after; the replay (:func:`tngxmm.instrument.replay.replay_view`), whose photon list
    exists, calls both here.

    ``conditions``: ``{'rule': 'donor'|'nn', 'n', 'table', 'jitter', 'replica_k', 'replicas', 'thin',
    'weights', 'fold_restricted', 'folds', 'per_projection'}`` (:func:`conditions_of` gives the spec's, without ``weights``); ``pkg`` the physics donor's package; ``meta``
    needs ``nh_1e22`` and may hold ``view_id``; ``streams`` the view's :class:`~tngxmm.core.rng.ViewStreams`
    (stream ``conditions``); ``evt_file`` the absorbed photon list of the physics view; the thinned,
    N_H-retargeted list is written to ``out_dir/stage_events.h5``.  Returns ``package``, ``record``,
    ``nh_1e22``, ``evt_file``."""
    return stage_events(draw_conditions(conditions, pkg, meta, streams, campaign), meta, evt_file, out_dir,
                        abund=abund)


def conditions_of(spec: Any) -> dict[str, Any]:
    """The observing conditions of a preset (``spec.donors.observing_conditions``) as the ``conditions``
    argument of :func:`stage_conditions`, validated: ``{'rule': 'donor'}`` when unset, else
    ``{'rule': 'nn', 'n': N, 'jitter': [...], 'table': None|path, 'fold_restricted': bool}`` (the ``table`` null = the
    package's ``donor_conditions.csv``, resolved by :func:`resolve_table`; ``fold_restricted``: the candidates are the
    observations of the physics donor's donor fold, ``folds`` null = :data:`DONOR_FOLDS_FILE`), plus
    ``'per_projection': True`` when the spec sets it (release 1.2, :func:`draw_projection_conditions`; absent or false
    keeps the draw of release 1.1.1 and the returned mapping without the key).  The one source of the production
    generation, the replay and the frozen evaluation."""
    c = getattr(spec.donors, "observing_conditions", None)
    if c is None:
        return {"rule": "donor"}
    c = {str(k): v for k, v in dict(c).items()}
    rule = c.get("rule", "donor")
    if rule not in CONDITION_RULES:
        raise ValueError(f"donors.observing_conditions: rule {rule!r} not in {CONDITION_RULES}")
    if rule == "donor":
        extra = sorted(set(c) - {"rule"})
        if extra:
            raise ValueError(f"donors.observing_conditions: rule 'donor' takes no {extra}")
        return {"rule": "donor"}
    unknown = sorted(set(c) - {"rule", "n", "jitter", "table", "weights", "fold_restricted", "folds", "per_projection"})
    if unknown:
        raise ValueError(f"donors.observing_conditions: unknown keys {unknown}")
    per_projection = c.get("per_projection", False)
    if not isinstance(per_projection, bool):
        raise ValueError(f"donors.observing_conditions: per_projection must be true or false, not {per_projection!r}")
    n = int(c.get("n", 5))
    if n < 1:
        raise ValueError("donors.observing_conditions: n must be >= 1")
    jitter = [str(j) for j in (c.get("jitter") or ())]
    bad = sorted(set(jitter) - set(JITTER_KINDS))
    if bad:
        raise ValueError(f"donors.observing_conditions: jitter {bad} not in {JITTER_KINDS}")
    fold_restricted = bool(c.get("fold_restricted", False))
    if c.get("folds") and not fold_restricted:
        raise ValueError("donors.observing_conditions: 'folds' needs fold_restricted: true")
    return {"rule": "nn", "n": n, "jitter": jitter, "table": c.get("table") or None,
            "fold_restricted": fold_restricted, **({"folds": c["folds"]} if c.get("folds") else {}),
            **({"weights": dict(c["weights"])} if c.get("weights") else {}),
            **({"per_projection": True} if per_projection else {})}


def resolve_table(conditions: dict[str, Any]) -> dict[str, Any]:
    """``conditions`` with the condition table named: ``table`` null = the data package's
    ``donor_conditions.csv``, and, with ``fold_restricted``, ``folds`` null = the donor-fold table of the repository
    (rule 'nn' only)."""
    from tngxmm.instrument.donor_tables import table_path

    c = dict(conditions)
    if c.get("rule", "nn") != "donor":
        c["table"] = str(c["table"]) if c.get("table") else str(table_path("conditions"))
        if c.get("fold_restricted"):
            c["folds"] = str(donor_folds_path(c.get("folds") or None))
    return c


def scale_exposure(exposure_s: dict[str, float], record: dict[str, Any] | None) -> dict[str, float]:
    """The per-camera live times after the conditions record's ``exposure_scale`` (the common
    log-normal factor of the 'exposure' jitter).  The donor background parameters' live times follow
    through ``info['exposure_scale']`` of :func:`tngxmm.sim.generate_view.instrument_view_kwargs`."""
    sc = 1.0 if record is None else float(record.get("exposure_scale", 1.0))
    return exposure_s if sc == 1.0 else {c: float(t) * sc for c, t in exposure_s.items()}


__cites__ = collect_cites(__name__)
