"""Quiescent particle background (QPB) of a donor from the unexposed (out-of-field-of-view) corners of its event lists.

The EPIC CCDs extend beyond the field of view of the mirrors; their corners see no X-rays and no soft protons, only the
particle background (De Luca & Molendi 2004, Sect. 3 and App. B; Kuntz & Snowden 2008).  The corner count rate,
divided by the corner rate of the filter-wheel-closed (FWC) event lists of the CCF on the same CCDs and in the same
band, is the observation's QPB level relative to the FWC data; the ratio of in-FOV to corner QPB is a property of the
detector, not of the observation (Leccardi & Molendi 2008, App. B, Eq. B.1), so the in-FOV QPB of the donor is that
ratio times the FWC in-FOV QPB.  This replaces the in-FOV decomposition of :mod:`tngxmm.instrument.donor_background`
(``background = 'donor'``), whose flat 4.5-12 keV level also holds residual soft protons and other flat counts and
is degenerate with the vignetted part (MOS1 / MOS2 flat ratio 1.25, against 1.01 in the corners).

Selection (same for donor and FWC): out-of-FOV flag (0x10000) set, no other bit of the camera's event-quality mask
(MOS 0x766aa000, pn 0xfa000c), PATTERN <= 12 (MOS) / <= 4 (pn), PI in :data:`CORNER_BAND_EV`, and for the MOS outside
the regions that XMM-ESAS leaves out of the unexposed pixels (:data:`MOS_CORNER_EXCLUDE`, Kuntz & Snowden 2008,
Table 3: the edge of the field of view, its extension, the light leaks of CCDs 2 and 7 and, on MOS1, three edges of
scattered light; De Luca & Molendi 2004, Sect. 3.1, mask the CCD 2 and 7 region "where both X-ray photons and low
energy particles can reach the detectors" without giving its outline); per CCD, counts over
the CCD's live time (event-list EXPOSUnn LIVETIME; FWC: EXPTIME).  pn: the out-of-time (OoT) events of the in-FOV
X-rays of a column spread over the whole column, corners included; the expected OoT counts of each corner column
(OoT fraction of the mode x the column's in-FOV counts above the particle expectation, distributed over the columns as in the FWC list, x the
column's corner share, measured on the FWC list) are subtracted.

Conversion to the stage: the stage draws the xmm_simulator FWC spectrum (``resp.fwc_spec``) scaled by the factors of
the donor background table; the conversion from the corner scale to those factors is done by the caller, not in this module.

Instrument use: ``InstrumentOptions.qpb_level`` (:data:`QPB_LEVELS`): ``'in-fov'`` (only with ``background =
'nominal'``) keeps the FWC level; ``'corners'`` sets each PPS band's QPB factor of every camera to that band's corner
scale (table ``donor_qpb_corners.csv``; a camera without a usable corner level, such as a pn Large or Small Window
exposure, whose corners are not read out, takes the FWC level, :func:`camera_qpb_level`) (the FWC spectra scaled to the observation band by band with the corner data, as Snowden et al. 2008,
Sect. 3.1.1, do spectral bin by spectral bin), so a donor whose particle spectrum is softer or harder than the FWC
keeps it; a single scale for all bands was tried and made the replicas LESS like their donors (DECISIONS, "QPB level
from the out-of-FOV corners"). The corner counts are taken within the flare GTIs of the PPS images
(:func:`flare_gti` from the PPS flare time series and its cut).
"""

from __future__ import annotations

import math
from typing import Any

from tngxmm.cites import cites, collect_cites

#: values of InstrumentOptions.qpb_level
QPB_LEVELS = ("in-fov", "corners")
#: PPS bands [eV] for the band-by-band corner scales ('corners'; Snowden et al. 2008 scale the FWC spectra to an
#: observation spectral bin by spectral bin with the corner data)
PPS_BANDS_EV = {"1000": (200, 500), "2000": (500, 1000), "3000": (1000, 2000), "4000": (2000, 4500), "5000": (4500, 12000)}
#: PI ranges [eV] the PPS images leave out of a band: the pn 4.5-12 keV image drops 7.8-8.2 keV, the Cu K-alpha line
#: of the pn particle background (selection expression ``PI in (4500:7800] || PI in (8200:12000]`` in the XPROC0
#: history of the PPS pn images, SAS 21 / PPS 21.51); the MOS images and the other pn bands are contiguous
PPS_BAND_GAPS_EV = {("PN", "5000"): (7800, 8200)}


def pps_band_parts(cam: str, band: str) -> tuple[tuple[int, int], ...]:
    """The PI ranges [eV] of PPS image band ``band`` of camera ``cam`` (one range, or two around a gap)."""
    lo, hi = PPS_BANDS_EV[band]
    gap = PPS_BAND_GAPS_EV.get((cam, band))
    return ((lo, gap[0]), (gap[1], hi)) if gap else ((lo, hi),)
#: out-of-FOV flag and the event-quality masks without it
OUT_OF_FOV = 0x10000
BAD_MASK = {"MOS1": 0x766AA000, "MOS2": 0x766AA000, "PN": 0xFA000C}
PATTERN_MAX = {"MOS1": 12, "MOS2": 12, "PN": 4}
#: regions left out of the MOS corners, in DETX, DETY (0.05" units): Kuntz & Snowden (2008), Table 3, the '!corner'
#: expression of each camera (SAS syntax: CIRCLE(x0, y0, r), BOX(x0, y0, half-width x, half-width y, angle [deg])),
#: the selection of the XMM-ESAS corner spectra.  A tuple ("circle", x0, y0, r) or ("box", x0, y0, a, b, angle), with
#: the table's comment.
MOS_CORNER_EXCLUDE = {
    "MOS1": (("circle", 100, -200, 17700, "FOV"), ("circle", 834, 135, 17100, "FOV"),
             ("circle", 770, -803, 17100, "FOV"), ("box", -20, -17000, 6500, 500, 0, "FOV extension"),
             ("box", 5880, -20500, 7500, 1500, 10, "light leak"), ("box", -5920, -20500, 7500, 1500, 350, "light leak"),
             ("box", -20, -20000, 5500, 500, 0, "light leak"), ("box", -12900, 16000, 250, 4000, 0, "scattered light"),
             ("box", 80, 18500, 150, 1300, 0, "scattered light"), ("box", -10, -18800, 125, 1500, 0, "scattered light")),
    "MOS2": (("circle", 435, 1006, 17100, "FOV"), ("circle", -34, 68, 17700, "FOV"),
             ("box", -20, -17000, 6500, 500, 0, "FOV extension"), ("box", 5880, -20500, 7500, 1500, 10, "light leak"),
             ("box", -5920, -20500, 7500, 1500, 350, "light leak"), ("box", -20, -20000, 5500, 500, 0, "light leak")),
}
#: corner band [eV]; pn starts higher (OoT and the low-energy noise of the pn corners)
CORNER_BAND_EV = {"MOS1": (2500, 12000), "MOS2": (2500, 12000), "PN": (5000, 12000)}
#: pn OoT fraction by submode (XMM Users Handbook, Table 3.3 / Sect. 3.3.2)
PN_OOT = {"PrimeFullWindow": 0.063, "PrimeFullWindowExtended": 0.023, "PrimeLargeWindow": 0.0016,
          "PrimeSmallWindow": 0.011}
#: table columns, per camera suffix
CAMS = {"MOS1": "mos1", "MOS2": "mos2", "PN": "pn"}


def pattern_max(cam: str, band: tuple | None = None) -> int:
    """PATTERN limit of the PPS images: MOS 12; pn 4, single pixels only below 0.5 keV."""
    if cam == "PN" and band is not None and band[1] <= 500:
        return 0
    return PATTERN_MAX[cam]


@cites("kuntz2008")
def corner_excluded(cam: str, detx: Any, dety: Any) -> Any:
    """Events inside the regions XMM-ESAS leaves out of the MOS corners (:data:`MOS_CORNER_EXCLUDE`); all False for
    the pn (Kuntz & Snowden 2008 give no pn regions)."""
    import numpy as np

    x = np.asarray(detx, dtype=float)
    y = np.asarray(dety, dtype=float)
    out = np.zeros(x.shape, bool)
    for r in MOS_CORNER_EXCLUDE.get(cam, ()):
        if r[0] == "circle":
            out |= (x - r[1]) ** 2 + (y - r[2]) ** 2 <= float(r[3]) ** 2
        else:
            _, x0, y0, a, b, ang, _ = r
            t = math.radians(ang)
            dx, dy = x - x0, y - y0
            u = dx * math.cos(t) + dy * math.sin(t)
            v = -dx * math.sin(t) + dy * math.cos(t)
            out |= (np.abs(u) <= a) & (np.abs(v) <= b)
    return out


def corner_mask(flag: Any, pattern: Any, pi: Any, cam: str, in_fov: bool = False, band: tuple | None = None,
                detx: Any = None, dety: Any = None) -> Any:
    """Events of the corner (or, with ``in_fov``, the in-FOV) selection; ``band`` [eV] defaults to
    :data:`CORNER_BAND_EV`.  The MOS corners leave out :data:`MOS_CORNER_EXCLUDE` and need the events' ``detx``,
    ``dety``."""
    import numpy as np

    f = np.asarray(flag).astype(np.int64)
    lo, hi = band or CORNER_BAND_EV[cam]
    base = ((np.asarray(pattern) <= pattern_max(cam, band)) & (np.asarray(pi) >= lo) & (np.asarray(pi) < hi)
            & ((f & BAD_MASK[cam]) == 0))
    oof = (f & OUT_OF_FOV) != 0
    if in_fov:
        return base & ~oof
    if cam in MOS_CORNER_EXCLUDE:
        if detx is None or dety is None:
            raise ValueError(f"{cam} corners: DETX and DETY are needed to leave out the regions of Kuntz & Snowden "
                             "(2008, Table 3)")
        oof = oof & ~corner_excluded(cam, detx, dety)
    return base & oof


def _corner_events(events: Any, cam: str, band: tuple | None = None) -> Any:
    """:func:`corner_mask` of an event table (FLAG, PATTERN, PI and, for the MOS, DETX, DETY)."""
    mos = cam in MOS_CORNER_EXCLUDE
    return corner_mask(events["FLAG"], events["PATTERN"], events["PI"], cam, band=band,
                       detx=events["DETX"] if mos else None, dety=events["DETY"] if mos else None)


def fwc_reference(events: Any, exptime: Any, cam: str, band: tuple | None = None) -> dict:
    """Per CCD of the FWC list: corner counts, in-FOV counts, exposure [s], and for the pn per (CCD, RAWX) the corner
    share of the column's particle events.  ``events``: FITS rec array (FLAG, PATTERN, PI, CCDNR, RAWX);
    ``exptime``: the EXPTIME extension (CCDNR, EXPOSURE)."""
    import numpy as np

    c = _corner_events(events, cam, band=band)
    i = corner_mask(events["FLAG"], events["PATTERN"], events["PI"], cam, in_fov=True, band=band)
    ccd = np.asarray(events["CCDNR"]).astype(int)
    tccd, texp = np.asarray(exptime["CCDNR"]).astype(int), np.asarray(exptime["EXPOSURE"], dtype=float)
    out: dict = {"ccd": {}}
    for k in np.unique(ccd):
        out["ccd"][int(k)] = {"corner": int((c & (ccd == k)).sum()), "infov": int((i & (ccd == k)).sum()),
                              "t": float(texp[tccd == k].sum())}
    if cam == "PN":
        rx = np.asarray(events["RAWX"]).astype(int)
        share, infov = {}, {}
        for k in np.unique(ccd):
            m = ccd == k
            tot = np.bincount(rx[m & (c | i)], minlength=70).astype(float)
            cor = np.bincount(rx[m & c], minlength=70).astype(float)
            inn = np.bincount(rx[m & i], minlength=70).astype(float)
            share[int(k)] = np.where(tot > 0, cor / np.maximum(tot, 1), 0.0)
            infov[int(k)] = inn / max(inn.sum(), 1.0)
        out["col_share"] = share
        out["col_infov"] = infov
    return out


@cites("deluca2004", "leccardi2008", "kuntz2008")
def corner_scale(events: Any, live: dict, fwc: dict, cam: str, submode: str | None = None,
                 min_fwc_counts: int = 50, band: tuple | None = None, keep: Any = None, exclude: Any = ()) -> dict:
    """The donor's corner QPB level relative to the FWC: sum over usable CCDs of corner counts / live time, over the
    same sum of the FWC; ``err`` from Poisson; pn: OoT counts of the in-FOV X-ray excess subtracted per column.
    ``live``: CCD -> live time [s] of the observation (within the GTIs of ``keep``); ``band`` [eV] (default
    :data:`CORNER_BAND_EV`); ``keep``: boolean event mask (e.g. the flare GTIs, :func:`gti_mask`); ``exclude``: CCDs
    left out (those the fit leaves out of its images: the MOS CCDs in the anomalous state)."""
    import numpy as np

    k_ = np.ones(len(np.asarray(events["FLAG"])), bool) if keep is None else np.asarray(keep, bool)
    c = _corner_events(events, cam, band=band) & k_
    ccd = np.asarray(events["CCDNR"]).astype(int)
    out_ccds = {int(x) for x in exclude}
    ccds = [k for k, v in fwc["ccd"].items() if v["corner"] >= min_fwc_counts and v["t"] > 0 and live.get(k, 0) > 0
            and int(k) not in out_ccds]
    n_corner = {k: float((c & (ccd == k)).sum()) for k in ccds}
    oot = {k: 0.0 for k in ccds}
    if cam == "PN" and fwc.get("col_share"):
        frac = PN_OOT.get(str(submode), PN_OOT["PrimeFullWindow"])
        i = corner_mask(events["FLAG"], events["PATTERN"], events["PI"], cam, in_fov=True, band=band) & k_
        rx = np.asarray(events["RAWX"]).astype(int)
        for k in ccds:
            ref = fwc["ccd"][k]
            q_in = ref["infov"] / ref["t"] * (n_corner[k] / live[k]) / (ref["corner"] / ref["t"])   # particle in-FOV rate
            cols = np.bincount(rx[i & (ccd == k)], minlength=70).astype(float)
            part_col = q_in * live[k] * fwc["col_infov"][k][:len(cols)]      # particles: the FWC column profile
            xray = np.clip(cols - part_col, 0.0, None)
            sh = fwc["col_share"].get(k, np.zeros(70))
            oot[k] = float((frac * xray * sh[:len(xray)]).sum())
    num = sum(max(n_corner[k] - oot[k], 0.0) / live[k] for k in ccds)
    den = sum(fwc["ccd"][k]["corner"] / fwc["ccd"][k]["t"] for k in ccds)
    n = sum(n_corner.values())
    scale = num / den if den > 0 else float("nan")
    return {"scale": scale, "err": scale / math.sqrt(n) if n > 0 else float("nan"), "n_corner": int(n),
            "oot": float(sum(oot.values())), "ccds": ",".join(map(str, ccds))}


def flare_gti(time: Any, rate: Any, timedel: float, threshold: float, timepixr: float = 0.5) -> Any:
    """Good time intervals (N, 2) of the PPS flare screening reconstructed from the flare background time series
    (FBKTSR: the >10 keV single-event rate per TIMEDEL bin) and its optimised cut ``FLCUTTHR``: bins with a finite
    rate at or below the cut, adjacent bins merged.  ``timepixr``: position of TIME within its bin."""
    import numpy as np

    t = np.asarray(time, dtype=float); r = np.asarray(rate, dtype=float)
    ok = np.isfinite(r) & (r <= float(threshold))
    lo = t - timepixr * float(timedel); hi = lo + float(timedel)
    out = []
    for a, b, g in zip(lo, hi, ok):
        if not g:
            continue
        if out and abs(out[-1][1] - a) < 1e-6 * float(timedel):
            out[-1][1] = b
        else:
            out.append([a, b])
    return np.asarray(out, dtype=float).reshape(-1, 2)


def gti_matching_ontime(time: Any, rate: Any, timedel: float, ontime: float, timepixr: float = 0.5) -> Any:
    """Good time intervals made of the lowest-rate bins of the flare time series whose total length reaches
    ``ontime`` (the PPS image's ONTIME): the optimised flare cut keeps the quietest bins, so this reproduces the
    image's screened time when the FBKTSR cut (:func:`flare_gti`) and the image GTIs differ (attitude GTIs,
    rebinning)."""
    import numpy as np

    t = np.asarray(time, dtype=float); r = np.asarray(rate, dtype=float)
    ok = np.isfinite(r)
    order = np.argsort(np.where(ok, r, np.inf), kind="stable")
    n = int(min(ok.sum(), math.ceil(float(ontime) / float(timedel) - 1e-9)))
    keep = np.zeros(t.size, bool); keep[order[:n]] = True
    lo = t - timepixr * float(timedel); hi = lo + float(timedel)
    out = []
    for a, b, g in zip(lo, hi, keep):
        if not g:
            continue
        if out and abs(out[-1][1] - a) < 1e-6 * float(timedel):
            out[-1][1] = b
        else:
            out.append([a, b])
    return np.asarray(out, dtype=float).reshape(-1, 2)


def gti_mask(time: Any, gti: Any) -> Any:
    """Events whose TIME lies in a good time interval."""
    import numpy as np

    t = np.asarray(time, dtype=float); g = np.asarray(gti, dtype=float).reshape(-1, 2)
    if not len(g):
        return np.zeros(t.size, bool)
    g = g[np.argsort(g[:, 0])]
    i = np.searchsorted(g[:, 0], t, side="right") - 1
    return (i >= 0) & (t < g[np.clip(i, 0, None), 1])


def gti_overlap(gti: Any, start: float, stop: float) -> float:
    """Length [s] of the good time intervals inside [start, stop)."""
    import numpy as np

    g = np.asarray(gti, dtype=float).reshape(-1, 2)
    return float(np.clip(np.minimum(g[:, 1], stop) - np.maximum(g[:, 0], start), 0.0, None).sum())


def fwc_infov_rate(fwc: dict, area_arcmin2: dict) -> float:
    """CCF FWC in-FOV QPB rate [counts / s / arcmin^2] of the corner band, area-weighted over the CCDs with an area."""
    num = sum(v["infov"] / v["t"] for k, v in fwc["ccd"].items() if v["t"] > 0 and area_arcmin2.get(k, 0) > 0)
    den = sum(a for k, a in area_arcmin2.items() if k in fwc["ccd"] and fwc["ccd"][k]["t"] > 0 and a > 0)
    return num / den if den > 0 else float("nan")


def donor_corner_band_scales(table: Any, obsid: str) -> dict[str, dict[str, float]]:
    """camera -> PPS band -> stage QPB factor (columns ``qpb_stage_<band>_<cam>``; pn band 1000 takes band 2000);
    cameras without a finite positive 0.5-12 keV set are absent.  KeyError when the donor is absent."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = donor_rows(table, obsid, "MOS background").iloc[0]
    out = {}
    for cam, suf in CAMS.items():
        vals = {}
        for b in PPS_BANDS_EV:
            v = r.get(f"qpb_stage_{b}_{suf}", float("nan"))
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                vals[b] = float(v)
        if cam == "PN" and "2000" in vals:
            vals["1000"] = vals["2000"]
        if all(b in vals for b in ("2000", "3000", "4000", "5000")):
            vals.setdefault("1000", vals["2000"])
            out[cam] = vals
    return out


#: pn quadrants (CCD numbers), the scaling units of the pn (XMM-ESAS pnback)
PN_QUADRANTS = {"q0": (1, 2, 3), "q1": (4, 5, 6), "q2": (7, 8, 9), "q3": (10, 11, 12)}


def donor_unit_factors(table: Any, obsid: str) -> dict[str, dict[int, float]]:
    """camera -> CCD -> the QPB level of the CCD (MOS) or of its quadrant (pn) relative to the mean of the camera's
    units with corner counts (columns ``unit_<u>_<cam>`` of the background table, :mod:`tngxmm.donor_tables.qpb_corners`);
    cameras without unit columns are absent.  KeyError when the donor is absent."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = donor_rows(table, obsid, "background").iloc[0]
    out: dict[str, dict[int, float]] = {}
    for cam, suf in CAMS.items():
        units = PN_QUADRANTS if cam == "PN" else {str(k): (k,) for k in range(1, 8)}
        vals = {u: float(r.get(f"unit_{u}_{suf}", float("nan"))) for u in units}
        good = [v for v in vals.values() if math.isfinite(v) and v > 0]
        if not good:
            continue
        mean = sum(good) / len(good)
        out[cam] = {int(c): (v / mean if (math.isfinite(v) and v > 0) else 1.0)
                    for u, v in vals.items() for c in units[u]}
    return out


def donor_corner_counts(table: Any, obsid: str) -> dict[str, dict[str, Any]]:
    """camera -> the counts behind the donor's corner measurement: ``band`` (PPS band -> corner counts,
    ``ncorner_<band>_<cam>``; the pn band 1000 takes band 2000, as its scale does), ``unit`` (unit -> corner-band
    counts, ``nunit_<u>_<cam>``) and ``unit_value`` (unit -> its corner scale, ``unit_<u>_<cam>``).  KeyError when the
    donor is absent."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = donor_rows(table, obsid, "corner QPB").iloc[0]
    out: dict[str, dict[str, Any]] = {}
    for cam, suf in CAMS.items():
        f = lambda k: float(r.get(k, float("nan")))  # noqa: E731
        band = {b: f(f"ncorner_{'2000' if (cam == 'PN' and b == '1000') else b}_{suf}") for b in PPS_BANDS_EV}
        units = PN_QUADRANTS if cam == "PN" else {str(k): (k,) for k in range(1, 8)}
        out[cam] = {"band": band, "unit": {u: f(f"nunit_{u}_{suf}") for u in units},
                    "unit_value": {u: f(f"unit_{u}_{suf}") for u in units}}
    return out


def simulated_corner_measurement(counts: dict | None, live_ratio: float, rng: Any) -> dict[str, Any]:
    """The corner measurement of a mock camera, drawn with the counting statistics of the donor's: for every band
    (and every scaling unit) whose donor measurement rests on n counts, the mock's corners hold N ~ Poisson(n x
    ``live_ratio``) (the mock camera's live time over the donor's), so the mock measures its QPB level as the true
    level times N / E with E = n x live_ratio, as the donor measured its own with an error of the same size.  Returns
    ``band`` (band -> N / E) and ``unit`` (unit -> measured relative level, the units' values times their N / E,
    normalised to mean 1 over the units with counts, as :func:`donor_unit_factors` normalises them); a band or unit
    without counts keeps 1.  No draw for ``counts`` None (both empty)."""
    if not counts:
        return {"band": {}, "unit": {}}
    def ratio(n: float) -> float:
        e = float(n) * float(live_ratio)
        if not (math.isfinite(e) and e > 0):
            return 1.0
        return float(rng.poisson(e)) / e
    band = {b: ratio(n) for b, n in sorted(counts["band"].items())}
    vals = {}
    for u, n in sorted(counts["unit"].items()):
        v = counts["unit_value"].get(u, float("nan"))
        if math.isfinite(v) and v > 0:
            vals[u] = v * ratio(n)
    mean = sum(vals.values()) / len(vals) if vals else float("nan")
    unit = {u: v / mean for u, v in vals.items()} if vals and mean > 0 else {}
    return {"band": band, "unit": unit}


def camera_qpb_level(bkg: dict | None, scales: dict | None) -> tuple[dict, str]:
    """The camera's background parameters with its QPB level under ``qpb_level = 'corners'``, and where the level
    comes from: the donor's corner scales (``'corners'``) when it has them, else the FWC level of the CCF
    (``'fwc'``: every band factor 1).  The in-FOV image-decomposition level of the donor background table is never
    used for the QPB (it carried a replay calibration; integration of 2026-10-30)."""
    if scales:
        return apply_corner_band_scales(bkg, scales), "corners"
    out = dict(bkg or {"sp_rate": 0.0})
    out["qpb_band"] = {b: 1.0 for b in PPS_BANDS_EV}
    out["qpb_scale"] = 1.0
    return out, "fwc"


def apply_corner_band_scales(bkg: dict | None, scales: dict | None) -> dict:
    """The camera's donor-background parameters with the QPB band factors set to ``scales`` (band -> factor) and
    the QPB scale to the 4.5-12 keV factor; unchanged when ``scales`` is None or empty."""
    out = dict(bkg or {"qpb_scale": 1.0, "sp_rate": 0.0})
    if not scales:
        return out
    out["qpb_band"] = {b: float(v) for b, v in scales.items()}
    out["qpb_scale"] = float(scales.get("5000", out.get("qpb_scale", 1.0)))
    return out


__cites__ = collect_cites(__name__)
