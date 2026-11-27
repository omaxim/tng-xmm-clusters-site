"""Residual soft protons of a donor from the IN/OUT method of De Luca & Molendi (2004).

Soft protons are focused by the mirrors and reach only the field of view; the particle background reaches the whole
CCD.  At 10-12 keV, where the X-ray sky is faint, the in-FOV surface brightness of an observation over its out-of-FOV
(corner) brightness, relative to the same ratio of the filter-wheel-closed data, measures the in-FOV excess, i.e. the
residual soft protons (De Luca & Molendi 2004, Sect. 3 and App. B; Leccardi & Molendi 2008, App. B).  Here, within
the flare GTIs of the PPS images:

    excess = N_IN / (A T)_IN  -  s_corner q_FWC,IN  -  CXB_IN      [counts / s / arcmin^2, 10-12 keV]

with N_IN the 10-12 keV in-FOV events of the region beyond the cluster, (A T)_IN its unvignetted exposure (area times
live time), s_corner the corner scale of the band (instrument.donor_corner_qpb), q_FWC,IN the FWC in-FOV rate per
area of the band, and CXB_IN the cosmic X-ray background of De Luca & Molendi (2004: photon index 1.41, 11.6 photons
cm^-2 s^-1 sr^-1 keV^-1 at 1 keV) through the camera's effective area averaged over 3-12' off axis.  The excess
becomes the stage's SP rate (0.5-12 keV, :func:`tngxmm.instrument.donor_background.soft_proton_events`) through the
quiescent-SP counts spectrum of Leccardi & Molendi (2008, App. B: broken power law, slopes 0.4 and 0.8, break 5 keV),
spread flat over the field of view: the SP have "a much flatter radial distribution than the X-ray emission" (Kuntz
& Snowden 2008).  Instrument option ``soft_protons`` (:data:`SOFT_PROTONS`): ``'off'`` or ``'inout'``, which reads the
donor's rate from the MOS background table (columns ``sp_rate_<cam>``; the QPB is the corner level of
:mod:`tngxmm.instrument.donor_corner_qpb`, measured separately).

The table holds the pn as well (:mod:`tngxmm.donor_tables.mos_background`).  The pn rate is a lower bound: the pn
corners are not shielded from the sky, they receive focused soft protons and photons (Marelli et al. 2021), so the
corner-scaled QPB that is subtracted is too high and the excess too low.  No source calibrates the pn IN/OUT excess
against the soft-proton level, and the quiescent-SP spectrum is the MOS one of Leccardi & Molendi (2008); ledger
entry ``instrument.pn_soft_protons_inout``.  Every camera the stage simulates needs its ``sp_rate_<cam>`` column:
a table without the pn columns, such as the MOS-only table measured before the pn was added, is refused when the
pn is simulated (:func:`apply_to_cameras`) instead of leaving the pn without soft protons.

A donor can have no rate for a camera: the IN region is empty when 1.5 R500 of a nearby cluster covers the 12'
circle about the pointing (112 of the 588 donors of the conditions pool), or the camera has no usable corner level,
event list or flare series.  Such a donor had soft protons like any other observation (98.7 per cent of the measured
MOS1 rates are positive), so a rate of 0 would not be a measurement but the lowest possible value, and the donors
concerned are the nearby clusters (90 per cent of the pool donors at z < 0.1, 1 per cent at z >= 0.2).  Instrument
option ``sp_rate_unmeasured`` (:data:`SP_RATE_UNMEASURED`) says what happens instead: ``'nearest-in-fold'`` takes the
measured rate of the nearest donor of the conditions pool, in the same donor fold, that has one for the camera
(:func:`nearest_measured_in_fold`: the condition-space distance of :mod:`tngxmm.instrument.conditions_donor` from the
donor, ties by obsid), and the view records the donor that lent it (``SPRFLAG``/``SPSUBST`` in the camera headers,
``soft_protons`` in ``obs_meta.json`` and the conditions record); ``'refuse'`` stops the view
(:class:`UnmeasuredSoftProtons`).  No rate is ever set to 0 for want of a measurement.
"""

from __future__ import annotations

import math
from typing import Any

from collections.abc import Mapping

from tngxmm.cites import cites, collect_cites

#: values of InstrumentOptions.soft_protons
SOFT_PROTONS = ("off", "inout")
#: values of InstrumentOptions.sp_rate_unmeasured: what a camera without a measured rate gets (module docstring)
SP_RATE_UNMEASURED = ("nearest-in-fold", "refuse")
#: cameras and their table column suffixes
CAMERA_SUFFIX = {"MOS1": "mos1", "MOS2": "mos2", "PN": "pn"}
#: De Luca & Molendi (2004) CXB: photons cm^-2 s^-1 sr^-1 keV^-1 at 1 keV, photon index
CXB_NORM, CXB_INDEX = 11.6, 1.41
#: the IN/OUT band [keV] and the off-axis range of the CXB average [arcmin]
INOUT_BAND_KEV = (10.0, 12.0)
CXB_THETA_ARCMIN = (3.0, 12.0)
ARCMIN2_PER_SR = (180.0 * 60.0 / math.pi) ** 2


@cites("deluca2004")
def cxb_rate(arf_theta: Any, thetas: Any, e_mid: Any, e_wid: Any, band: tuple = INOUT_BAND_KEV,
             theta_range: tuple = CXB_THETA_ARCMIN) -> float:
    """CXB count rate [counts / s / arcmin^2] in ``band`` through ``arf_theta`` (theta x energy, cm^2), averaged over
    ``theta_range`` with weight theta (uniform sky area)."""
    import numpy as np

    th = np.asarray(thetas, dtype=float); a = np.asarray(arf_theta, dtype=float)
    e = np.asarray(e_mid, dtype=float); de = np.broadcast_to(np.asarray(e_wid, dtype=float), e.shape)
    sel_t = (th >= theta_range[0]) & (th <= theta_range[1])
    w = th[sel_t] / th[sel_t].sum()
    a_mean = (a[sel_t] * w[:, None]).sum(0)
    sel_e = (e >= band[0]) & (e < band[1])
    flux = CXB_NORM * e[sel_e] ** (-CXB_INDEX) * de[sel_e]          # photons / cm2 / s / sr per bin
    return float((flux * a_mean[sel_e]).sum() / ARCMIN2_PER_SR)


def cxb_band_rates(ccf_dir: str, filt: str = "Medium") -> dict[str, float]:
    """:func:`cxb_rate` of each camera with the stage's responses."""
    from tngxmm.instrument.load_responses import get_response

    out = {}
    for cam in ("MOS1", "MOS2", "PN"):
        r = get_response(cam, ccf_dir, filter=filt)
        out[cam] = cxb_rate(r.arf_theta, r.thetas, r.box_ene_mean, r.box_ene_wid)
    return out


@cites("deluca2004", "leccardi2008")
def inout_excess(n_in: float, area_time: float, qpb_in: float, cxb: float) -> dict[str, float]:
    """In-FOV 10-12 keV excess [counts / s / arcmin^2] over the corner-predicted QPB and the CXB, its Poisson error,
    and the IN/OUT ratio R (in-FOV over QPB + CXB; 1 for no excess)."""
    if not (area_time > 0) or not (qpb_in > 0):
        return {"excess": float("nan"), "excess_err": float("nan"), "R": float("nan")}
    rate = n_in / area_time
    return {"excess": rate - qpb_in - cxb, "excess_err": math.sqrt(max(n_in, 1.0)) / area_time,
            "R": rate / (qpb_in + cxb)}


@cites("leccardi2008")
def sp_rate_from_excess(excess: float, band: tuple = INOUT_BAND_KEV) -> float:
    """The stage's SP rate [counts / s / arcmin^2, 0.5-12 keV] whose quiescent-SP counts spectrum puts ``excess``
    into ``band``; 0 for a negative excess (no SP)."""
    import numpy as np

    from tngxmm.instrument.donor_background import sp_counts_spectrum

    if not math.isfinite(excess) or excess <= 0:
        return 0.0
    g = np.linspace(0.5, 12.0, 4601)
    m = 0.5 * (g[1:] + g[:-1])
    w = sp_counts_spectrum(m)
    frac = float(w[(m >= band[0]) & (m < band[1])].sum() / w.sum())
    return float(excess) / frac


class UnmeasuredSoftProtons(ValueError):
    """A camera of a donor has no measured IN/OUT soft-proton rate and the rule (``sp_rate_unmeasured``) gives it
    none: ``'refuse'``, or ``'nearest-in-fold'`` without a measured donor in the donor's fold."""


def measured_rate(row: Any, camera: str) -> float | None:
    """The measured ``sp_rate_<cam>`` [counts / s / arcmin^2] of one donor's table row (0 when its excess is not
    positive: a measured zero), or None when the row has no finite rate (no IN region, no usable corner level, no
    event list or flare series).  ValueError for a negative rate, which the table never holds."""
    rate = float(row[f"sp_rate_{CAMERA_SUFFIX[camera]}"])
    if not math.isfinite(rate):
        return None
    if rate < 0:
        raise ValueError(f"negative soft-proton rate {rate} for {camera} in the IN/OUT table")
    return rate


def nearest_measured_in_fold(conditions: Any, folds: Mapping[str, int], table: Any, obsid: str,
                             camera: str) -> dict[str, Any]:
    """The substitute rate of a donor without a measured one for ``camera``: ``{'obsid', 'sp_rate', 'distance'}`` of
    the nearest donor of the conditions pool, in the donor's fold, with a measured rate for the camera.  Nearest in
    the condition space of the conditions donor (``conditions``: the condition table;
    :func:`tngxmm.instrument.conditions_donor.condition_distances` with ``folds``, its standardised distance, ties by
    obsid); ``table``: the IN/OUT table.  :class:`UnmeasuredSoftProtons` when the donor has no row or fold to measure
    distances from, or its fold holds no pool donor with a measured rate."""
    from tngxmm.instrument.conditions_donor import condition_distances

    key = str(obsid).zfill(10)
    try:
        d2 = condition_distances(conditions, key, None, folds)
    except KeyError as e:
        why = e.args[0] if e.args else str(e)
        raise UnmeasuredSoftProtons(f"donor {key} has no measured {camera} soft-proton rate and no substitute: "
                                    f"{why}") from None
    rates = table.assign(obsid=table.obsid.astype(str).str.zfill(10)).set_index("obsid")[
        f"sp_rate_{CAMERA_SUFFIX[camera]}"].astype(float)
    for k in d2.index:
        if k == key or k not in rates.index:
            continue
        rate = float(rates[k])
        if math.isfinite(rate):
            return {"obsid": str(k), "sp_rate": rate, "distance": float(d2[k]) ** 0.5}
    raise UnmeasuredSoftProtons(f"donor {key} has no measured {camera} soft-proton rate and its fold holds no pool "
                                "donor with one")


def donor_soft_protons(table: Any, obsid: str, unmeasured: str = "refuse",
                       substitute: Any = None) -> dict[str, dict[str, Any]]:
    """camera -> ``{'sp_rate', 'source', 'substitute', 'distance'}`` of one donor, for the cameras the table has a
    column for (a camera without a column has no entry, and the stage refuses to simulate it,
    :func:`apply_to_cameras`); KeyError when the donor is absent.

    A measured rate (a zero included) has ``source`` ``'measured'``.  A camera without one follows ``unmeasured``
    (:data:`SP_RATE_UNMEASURED`): ``'refuse'`` raises :class:`UnmeasuredSoftProtons`; ``'nearest-in-fold'`` calls
    ``substitute(camera)`` (:func:`nearest_measured_in_fold` bound to the condition table, the folds and this table)
    and takes its rate, with ``source`` ``'substitute'``, the lending donor in ``substitute`` and its condition-space
    ``distance``."""
    from tngxmm.instrument.donor_tables import donor_rows

    if unmeasured not in SP_RATE_UNMEASURED:
        raise ValueError(f"unknown sp_rate_unmeasured {unmeasured!r}; known: {SP_RATE_UNMEASURED}")
    key = str(obsid).zfill(10)
    r = donor_rows(table, key, "MOS background").iloc[0]
    out: dict[str, dict[str, Any]] = {}
    for cam, suf in CAMERA_SUFFIX.items():
        if f"sp_rate_{suf}" not in r.index:
            continue
        rate = measured_rate(r, cam)
        if rate is not None:
            out[cam] = {"sp_rate": rate, "source": "measured", "substitute": None, "distance": None}
            continue
        if unmeasured == "refuse" or substitute is None:
            raise UnmeasuredSoftProtons(
                f"donor {key}: no measured {cam} soft-proton rate (sp_rate_{suf} is empty: no IN region, usable "
                f"corner level, event list or flare series) and sp_rate_unmeasured is {unmeasured!r}"
                + ("" if unmeasured == "refuse" else " without a substitute lookup"))
        sub = substitute(cam)
        out[cam] = {"sp_rate": float(sub["sp_rate"]), "source": "substitute", "substitute": str(sub["obsid"]),
                    "distance": float(sub["distance"])}
    return out


def apply_to_background(bkg: Mapping[str, Any] | None, sp: Mapping[str, Any]) -> dict[str, Any]:
    """The camera's donor-background parameters with the donor's flat SP rate set (the QPB level is untouched), and
    where it comes from (``sp_source``: ``'measured'`` or ``'substitute'``; ``sp_substitute``: the lending donor or
    None)."""
    out = dict(bkg or {"qpb_scale": 1.0, "sp_rate": 0.0})
    out["sp_rate"] = float(sp["sp_rate"])
    out["sp_source"] = str(sp.get("source") or "measured")
    out["sp_substitute"] = sp.get("substitute")
    return out


def apply_to_cameras(bkg_by_cam: Mapping[str, Any], donor_sp: Mapping[str, Mapping[str, float]],
                     cameras: Any) -> dict[str, dict[str, Any]]:
    """:func:`apply_to_background` for each of ``cameras``, the cameras the stage simulates.  Each needs an entry of
    :func:`donor_soft_protons`, that is an ``sp_rate_<cam>`` column in the table: a camera without one (the pn with
    the MOS-only table measured before the pn was added) raises ``ValueError``, because it would otherwise be
    simulated without soft protons and nothing would say so."""
    missing = [c for c in cameras if c not in donor_sp]
    if missing:
        cols = ", ".join(f"sp_rate_{str(c).lower()}" for c in missing)
        raise ValueError(f"soft_protons 'inout': the MOS background table has no column {cols} for the simulated "
                         f"camera(s) {', '.join(map(str, missing))}.  Use the table measured with every camera "
                         "(donor_tables.mos_background.cameras = [MOS1, MOS2, PN]) or set soft_protons to 'off'")
    return {c: apply_to_background(bkg_by_cam[c], donor_sp[c]) for c in cameras}


__cites__ = collect_cites(__name__)
