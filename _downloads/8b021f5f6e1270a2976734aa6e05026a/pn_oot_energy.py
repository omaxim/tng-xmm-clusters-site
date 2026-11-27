"""Energies of the EPIC-pn out-of-time events: the CTI correction applied at the wrong row.

An out-of-time (OoT) event is registered while its CCD is read out.  Its charge is
transferred over the rows between where the photon landed and the readout node, but
``epevents`` assigns it the RAWY of the readout phase and corrects its amplitude for the
charge transfer inefficiency (CTI) of that row: "These so called Out-of-Time (OoT)
events get a wrong RAWY value assigned and thus finally a wrong CTI correction" (Users'
Handbook 3.3.10).  ``epevents withoutoftime=yes`` simulates OoT events the same way: "a new
RAWY is simulated by randomly shifting the pattern along the RAWY axis and performing the
gain and CTI correction afterwards".

Model.  Let c(A, y) be the ratio of corrected to raw amplitude that the SAS applies to an
event of raw amplitude A at row y (RAWY, 1 next to the node).  An in-time event of energy E
at its true row y_t was recorded with A = E / c(A, y_t); the same charge registered at row
y_r becomes

    E_OoT = c(A, y_r) A = E c(A, y_r) / c(A, y_t).

Along a column ln c is a straight line in y to within 1.0 per cent (largest residual over
the rows each mode reads, all modes, CCDs, epochs and amplitudes of the table; 0.8 per cent in
Full Frame), ln c = b(A) + k(A) y, so

    E_OoT = E exp(k(A) (y_r - y_t)),   A = E exp(-b(A) - k(A) y_t).

k > 0: an event assigned closer to the node than where it landed (y_r < y_t) is
under-corrected and loses energy, one assigned farther away gains it, as the Handbook
describes.

The table.  k and b come from the CCF through the SAS calibration access layer itself:
``tools/pn_cti/build_pn_cti_probe.sh`` runs isolated events (no precursor in their column)
through ``CAL_pnCtiCorrect`` and ``CAL_pnAdditionalGainCtiCorrect``, the two RAWY-dependent
steps of ``epevents`` with its default settings (``gainctiaccuracy = 2``,
``withctilongterm``, ``ctilongtermsoft``, ``ctilongtermy``), and fits the line to the
column mean of ln c over the rows the mode reads (Full Frame and Extended Full Frame RAWY
13-200, Large Window 101-200, Small Window 136-200).  The CAL state carries the CCD
temperature, on which the CTI model depends ('the dependence of the CTI on energy and
temperature', calpnalgo): 183.2 K, the pn focal-plane temperature of the housekeeping
parameter F1128 (-89.95 C in the PPS event lists of 2000-2025), as ``epevents`` sets it.
Left unset, the CAL returned a k 0.74-0.88 times as large (the table before 2026-11-03).
``tools/pn_cti/pps_cti_slope_check.py`` compares the table with the RAWY slope of
ln(PI / 5 PHA) of single-pixel events in PPS event lists, i.e. with the correction the
pipeline applied (0.990, 16-84 per cent 0.974-1.008, over 21 observations of 2000-2025).
The table holds k and b per pn imaging mode, epoch (2000-04-01, then 1 January 2001 ...
2027), CCD (Small Window: CCD 4, the only one it reads) and 15 amplitudes from 40 to 2400
ADU (0.2-12 keV at the 5 eV per ADU of the gain normalisation, calpnalgo).
Interpolation: linear in time, linear in ln A; outside the grid the end values.  The
temperature correction of ``epevents`` is a factor common to all rows of a CCD, and the gain
of a readout channel does not depend on RAWY, so neither enters c(A, y_r) / c(A, y_t).
``epspatialcti`` is not part of the pipeline processing (its documentation) and is left out.

Not modelled: the column-to-column CTI within a CCD (the table holds the mean over the 64
columns; the 16-84 per cent range of k over the columns of CCD 4 is about +-8 per cent in Full
Frame, 2010); the precursor dependence of the CTI (the probe events have no precursor; with a
1-5 keV event ahead in its frame and column the CAL lowers the correction by 2-4 per cent, and
the source columns of the real Full Frame streaks carry 0.005 events per column and frame, at
most 0.03, so the term moves the streak hardness by less than 0.003); and the split events,
whose pixels the SAS corrects one by one at their own, smaller amplitudes (the factor is taken at
the event's energy).

References
----------
xmm_uhb (Sect. 3.3.10), xmm_sas_epevents (out-of-time events and the CTI correction),
xmm_sas_cal (the CAL routines), xmm_cal_tn0018 (the long-term CTI correction), xmm_ccf_sas
(EPN_CTI), dennerl2002 (the operating temperature of the pn camera).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: values of ``instrument.pn_oot_energy`` / ``massfit.pn_oot_energy``
OOT_ENERGY_RULES = ("none", "ccf-cti")
#: CAL mode identifiers of the pn imaging modes (caloaldefs.f90 of the SAS)
CAL_MODE_IDS = {"PrimeFullWindow": 0, "PrimeFullWindowExtended": 20, "PrimeLargeWindow": 21,
                "PrimeSmallWindow": 22}
#: first RAWY each mode reads (and the probe fits from): rows 1-12 of the full-frame modes hold no
#: event after the PPS selection; the windows of EPN_MODEPARAM
FIRST_ROW = {"PrimeFullWindow": 13, "PrimeFullWindowExtended": 13, "PrimeLargeWindow": 101,
             "PrimeSmallWindow": 136}
#: amplitude unit of the CAL energies: the gains are "normalized to yield the constant conversion factor
#: 1 adu = 5.00 eV" (calpnalgo, Gain correction)
EV_PER_ADU = 5.0
TABLE_NAME = "pn_cti_slopes.csv"
#: where the CCD temperature of the CAL state comes from (the provenance of the table)
TEMPERATURE_SOURCE = ("pn focal-plane temperature, HK parameter F1128 (H_CE_TTMPFPF): -89.95 C (183.2 K) in the HKAUX "
                      "headers of the PPS pn event lists, the same to 0.01 C in 23 observations of 2000-2025 and on "
                      "every CCD; Dennerl et al. 2002 (astro-ph/0204242, p. 3): -90 C, the operating temperature of "
                      "the pn camera in orbit")
#: lowest amplitude of the table [ADU]: 40 ADU = 0.2 keV, the lower edge of the PPS bands.  The probe's first amplitude
#: (30 ADU) is dropped: there the CAL returns no correction for part of the Large and Small Window rows
MIN_AMPLITUDE_ADU = 40.0

_CACHE: dict = {}


def table_path() -> Path:
    """The packaged table (``tngxmm/instrument/calibration/pn_cti_slopes.csv``)."""
    return Path(__file__).resolve().parent / "calibration" / TABLE_NAME


def date_to_mjd(date: Any) -> float:
    """MJD of an ISO date or date-time (``DATE-OBS``)."""
    s = str(date).strip().replace("Z", "")
    t = _dt.datetime.fromisoformat(s[:19]) if "T" in s else _dt.datetime.fromisoformat(s[:10])
    return (t - _dt.datetime(1858, 11, 17)).total_seconds() / 86400.0


@cites("xmm_sas_cal", "xmm_ccf_sas")
def read_table(path: str | Path | None = None) -> dict[str, Any]:
    """The table as arrays: ``modes`` (submode -> index), ``mjd`` (epochs), ``e_kev`` (amplitude grid
    times 5 eV), ``k`` and ``b`` of shape (mode, epoch, CCD 1-12, amplitude).  Cached per path."""
    import numpy as np

    p = str(path or table_path())
    if p in _CACHE:
        return _CACHE[p]
    with open(p, newline="") as fh:
        rows = [r for r in csv.DictReader(line for line in fh if not line.startswith("#"))]
    kcols = [c for c in rows[0] if c.startswith("k_")]
    e_ev = np.array([float(c[2:]) for c in kcols])
    modes = sorted({r["submode"] for r in rows}, key=lambda m: CAL_MODE_IDS[m])
    mjd = np.array(sorted({float(r["mjd"]) for r in rows}))
    shape = (len(modes), len(mjd), 12, len(e_ev))
    k = np.full(shape, np.nan)
    b = np.full(shape, np.nan)
    for r in rows:
        i, j, c = modes.index(r["submode"]), int(np.searchsorted(mjd, float(r["mjd"]))), int(r["ccd"]) - 1
        k[i, j, c] = [float(r[f"k_{int(e)}"]) for e in e_ev]
        b[i, j, c] = [float(r[f"b_{int(e)}"]) for e in e_ev]
    # a CCD a mode does not read (Small Window: only CCD 4) takes the mean of the CCDs it reads; no event of that
    # mode lands there, the value only keeps the arrays rectangular
    for i in range(len(modes)):
        have = np.isfinite(k[i]).all(axis=(0, 2))
        if not have.any():
            raise ValueError(f"{p}: no CCD of mode {modes[i]}")
        k[i][:, ~have] = k[i][:, have].mean(axis=1, keepdims=True)
        b[i][:, ~have] = b[i][:, have].mean(axis=1, keepdims=True)
    if not (np.isfinite(k).all() and np.isfinite(b).all()):
        raise ValueError(f"{p}: incomplete table")
    out = {"modes": {m: n for n, m in enumerate(modes)}, "mjd": mjd, "e_kev": e_ev / 1000.0, "k": k, "b": b}
    _CACHE[p] = out
    return out


class PnCtiLine:
    """k(A) and b(A) of every CCD for one pn mode and epoch (:func:`cti_line`)."""

    def __init__(self, submode: str, mjd: float, e_kev, k, b):
        self.submode, self.mjd = submode, float(mjd)
        self.e_kev, self.k, self.b = e_kev, k, b          # k, b: (12, n_e)

    def _at(self, arr, e_kev, ccd):
        import numpy as np

        e = np.atleast_1d(np.asarray(e_kev, float))
        c = np.broadcast_to(np.atleast_1d(np.asarray(ccd, np.int64)), e.shape)
        x = np.interp(np.log(np.clip(e, self.e_kev[0], self.e_kev[-1])), np.log(self.e_kev),
                      np.arange(len(self.e_kev)))
        i0 = np.minimum(np.floor(x).astype(np.int64), len(self.e_kev) - 2)
        w = x - i0
        tab = np.vstack([arr, arr.mean(axis=0)])          # row 12: the mean of the twelve CCDs
        ci = np.where((c >= 1) & (c <= 12), c - 1, 12)
        return tab[ci, i0] * (1 - w) + tab[ci, i0 + 1] * w

    def oot_factor(self, e_kev, ccd, rawy_true, rawy_assigned):
        """E_OoT / E of events of energy ``e_kev`` [keV] (corrected, in-time scale) on CCD ``ccd`` (1-12; any
        other value takes the mean of the twelve) that landed at row ``rawy_true`` and are registered at
        ``rawy_assigned`` (RAWY, 1 next to the readout node; arrays of one length)."""
        import numpy as np

        e = np.asarray(e_kev, float)
        yt = np.asarray(rawy_true, float)
        yr = np.asarray(rawy_assigned, float)
        ccd = np.broadcast_to(np.asarray(ccd), e.shape)
        a = e.copy()
        for _ in range(2):          # A = E exp(-b(A) - k(A) y_t): two fixed-point steps (the shift is < 30 %)
            a = e * np.exp(-self._at(self.b, a, ccd) - self._at(self.k, a, ccd) * yt)
        return np.exp(self._at(self.k, a, ccd) * (yr - yt))

    def mean_k(self, e_kev):
        """k [per row] at energies ``e_kev`` averaged over the twelve CCDs (the fit template)."""
        import numpy as np

        e = np.atleast_1d(np.asarray(e_kev, float))
        return self._at(self.k, e, np.zeros(e.shape, np.int64))


@cites("xmm_sas_epevents", "xmm_sas_cal", "xmm_ccf_sas", "xmm_cal_tn0018", "dennerl2002")
def cti_line(submode: str | None, date_obs: Any, path: str | Path | None = None) -> PnCtiLine:
    """The CTI line of the pn mode ``submode`` (None: Full Frame) at ``date_obs`` (ISO date, or an MJD as a
    number), linear in time between the epochs of the table (the end epochs outside them)."""
    import numpy as np

    t = read_table(path)
    sub = submode or "PrimeFullWindow"
    if sub not in t["modes"]:
        raise ValueError(f"pn mode {sub!r} has no CTI line; known: {sorted(t['modes'])}")
    mjd = float(date_obs) if isinstance(date_obs, (int, float)) else date_to_mjd(date_obs)
    m = t["mjd"]
    x = float(np.interp(mjd, m, np.arange(len(m))))
    j = min(int(math.floor(x)), len(m) - 2)
    w = x - j
    i = t["modes"][sub]
    k = t["k"][i, j] * (1 - w) + t["k"][i, j + 1] * w
    b = t["b"][i, j] * (1 - w) + t["b"][i, j + 1] * w
    return PnCtiLine(sub, mjd, t["e_kev"], k, b)


# -- the table from the probe output (tools/pn_cti) -------------------------------------------------

@cites("xmm_sas_cal", "xmm_ccf_sas")
def write_table(probe_txt: str | Path, out_csv: str | Path, provenance: dict | None = None) -> dict[str, Any]:
    """Write the package table from the output of ``tools/pn_cti/pn_cti_probe.f90`` (one line per mode,
    epoch, CCD and amplitude: mode id, MJD, CCD, amplitude [ADU], k, b, largest residual of the line,
    columns used) and its provenance JSON next to it; returns the provenance record.  A
    ``ccd_temperature_k`` in ``provenance`` (the CCD temperature of the CAL state the probe ran with) is
    also written into the table's header line."""
    import numpy as np

    names = {v: k for k, v in CAL_MODE_IDS.items()}
    d = np.loadtxt(probe_txt)
    if d.ndim != 2 or d.shape[1] != 8:
        raise ValueError(f"{probe_txt}: expected 8 columns")
    d = d[d[:, 3] >= MIN_AMPLITUDE_ADU - 1e-6]
    if not np.isfinite(d).all():
        raise ValueError(f"{probe_txt}: non-finite probe values above {MIN_AMPLITUDE_ADU} ADU")
    amps = np.unique(np.round(d[:, 3], 4))
    e_ev = [int(round(a * EV_PER_ADU)) for a in amps]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    groups: dict = {}
    for r in d:
        groups.setdefault((int(r[0]), float(r[1]), int(r[2])), {})[round(float(r[3]), 4)] = r
    temp = (provenance or {}).get("ccd_temperature_k")
    with open(out_csv, "w", newline="") as fh:
        fh.write("# pn out-of-time energy model: ln(corrected / raw amplitude) = b + k RAWY per mode, epoch and CCD "
                 "(column mean), from the SAS CAL on EPN_CTI"
                 + ("" if temp is None else f" at a CCD temperature of {float(temp):g} K")
                 + "; tngxmm.instrument.pn_oot_energy\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["submode", "mjd", "date", "ccd", "first_row"] + [f"k_{e}" for e in e_ev] + [f"b_{e}" for e in e_ev])
        for (mode, mjd, ccd) in sorted(groups):
            g = groups[(mode, mjd, ccd)]
            if len(g) != len(amps):
                raise ValueError(f"mode {mode} MJD {mjd} CCD {ccd}: {len(g)} amplitudes, expected {len(amps)}")
            date = (_dt.datetime(1858, 11, 17) + _dt.timedelta(days=mjd)).date().isoformat()
            sub = names[mode]
            w.writerow([sub, f"{mjd:.2f}", date, ccd, FIRST_ROW[sub]]
                       + [f"{g[round(float(a), 4)][4]:.5e}" for a in amps]
                       + [f"{g[round(float(a), 4)][5]:.5e}" for a in amps])
    rec = {"table": out_csv.name, "rows": len(groups), "amplitudes_adu": [float(a) for a in amps],
           "largest_line_residual": {names[m]: float(d[d[:, 0] == m, 6].max()) for m in np.unique(d[:, 0]).astype(int)},
           "columns_used_min": int(d[:, 7].min())}
    rec.update(provenance or {})
    with open(out_csv.with_suffix(".provenance.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    return rec


# -- the check against the pipeline (tools/pn_cti/pps_cti_slope_check.py) -----------------------------------------

#: raw-amplitude bins [ADU] of the PPS check, narrow so that k changes little within a bin (0.45-5.5 keV at 5 eV/ADU)
PPS_CHECK_PHA_BINS = ((90, 110), (140, 160), (190, 210), (280, 320), (380, 420), (560, 640), (900, 1100))
#: smallest number of single-pixel events of a CCD and amplitude bin that enters the check
PPS_CHECK_MIN_EVENTS = 2000
#: largest |median(k_PPS / k_table) - 1| the check accepts, a design parameter (OPEN_ITEMS section P): the ratio
#: scatters by 1.8 per cent from cell to cell (711 cells of 21 observations, 2000-2025) and by 0.5 per cent between
#: the observations' means about 0.99, so 0.05 is about three times the cell scatter; the table built without the
#: CCD temperature gives 1.24 and fails
PPS_CHECK_TOLERANCE = 0.05


def pps_column_sums(rawx, rawy, pha, pi, ccdnr, pha_bins=PPS_CHECK_PHA_BINS) -> list[dict[str, Any]]:
    """Per CCD, raw-amplitude bin and column (RAWX) of selected single-pixel PPS pn events (PATTERN == 0, the PPS
    flag selection, RAWY > 12): the sums n, sum y, sum y^2, sum v, sum v y with y = RAWY and v = ln(PI / (5 PHA)), the
    logarithm of the energy correction epevents applied (corrected over raw amplitude at 5 eV per ADU).  These sums
    are all :func:`pps_slopes` needs, so a sample can be stored as them."""
    import numpy as np

    rawx, rawy = np.asarray(rawx, np.int64), np.asarray(rawy, float)
    pha, pi, ccdnr = np.asarray(pha, float), np.asarray(pi, float), np.asarray(ccdnr, np.int64)
    ok = (pha > 0) & (pi > 0)
    v = np.full(len(pha), np.nan)
    v[ok] = np.log(pi[ok] / (EV_PER_ADU * pha[ok]))
    rows = []
    for lo, hi in pha_bins:
        inb = ok & (pha >= lo) & (pha < hi)
        for ccd in np.unique(ccdnr[inb]):
            m = inb & (ccdnr == ccd)
            x, y, vv = rawx[m], rawy[m], v[m]
            for col in np.unique(x):
                c = x == col
                rows.append(dict(ccd=int(ccd), pha_lo=float(lo), pha_hi=float(hi), rawx=int(col), n=int(c.sum()),
                                 sum_y=float(y[c].sum()), sum_yy=float((y[c] ** 2).sum()), sum_v=float(vv[c].sum()),
                                 sum_vy=float((vv[c] * y[c]).sum())))
    return rows


def pps_slopes(sums, min_events: int = PPS_CHECK_MIN_EVENTS) -> list[dict[str, Any]]:
    """The RAWY slope of ln(PI / 5 PHA) per CCD and amplitude bin from :func:`pps_column_sums`: least squares in
    RAWY after the mean of every column is removed (the gain of a readout channel does not depend on RAWY), i.e.
    sum_c (S_vy - S_v S_y / n) / sum_c (S_yy - S_y^2 / n).  Cells with fewer than ``min_events`` events are left out."""
    cells: dict = {}
    for r in sums:
        key = (int(r["ccd"]), float(r["pha_lo"]), float(r["pha_hi"]))
        a = cells.setdefault(key, [0, 0.0, 0.0])
        n, sy, syy, sv, svy = (float(r["n"]), float(r["sum_y"]), float(r["sum_yy"]), float(r["sum_v"]),
                               float(r["sum_vy"]))
        if n < 2:
            continue
        a[0] += int(n)
        a[1] += svy - sv * sy / n
        a[2] += syy - sy * sy / n
    out = []
    for (ccd, lo, hi), (n, num, den) in sorted(cells.items()):
        if n >= min_events and den > 0:
            out.append(dict(ccd=ccd, pha_lo=lo, pha_hi=hi, n=n, k_pps=num / den))
    return out


@cites("xmm_sas_epevents", "xmm_sas_cal")
def pps_check(sums, submode: str | None, date_obs: Any, path: str | Path | None = None,
              min_events: int = PPS_CHECK_MIN_EVENTS) -> dict[str, Any]:
    """Compare the table with the correction the pipeline applied: per CCD and amplitude bin the ratio of the PPS
    slope (:func:`pps_slopes`) to k of the table at ``date_obs``, mode ``submode``, the CCD and the bin's centre
    amplitude.  Returns the cells (with ``k_table`` and ``ratio``), their median ratio and whether
    |median - 1| <= :data:`PPS_CHECK_TOLERANCE`."""
    import numpy as np

    line = cti_line(submode, date_obs, path)
    cells = pps_slopes(sums, min_events)
    for c in cells:
        e_kev = 0.5 * (c["pha_lo"] + c["pha_hi"]) * EV_PER_ADU / 1000.0
        c["k_table"] = float(line._at(line.k, np.array([e_kev]), np.array([c["ccd"]]))[0])
        c["ratio"] = c["k_pps"] / c["k_table"]
    med = float(np.median([c["ratio"] for c in cells])) if cells else float("nan")
    return dict(cells=cells, median_ratio=med, n_cells=len(cells), tolerance=PPS_CHECK_TOLERANCE,
                passed=bool(cells) and abs(med - 1.0) <= PPS_CHECK_TOLERANCE)


def main(argv: list[str] | None = None) -> int:
    """``python -m tngxmm.instrument.pn_oot_energy --from-probe <probe.txt> [--out <csv>] [--note ...]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-probe", required=True, help="output of tools/pn_cti/build_pn_cti_probe.sh")
    ap.add_argument("--out", default=str(table_path()))
    ap.add_argument("--ccf-file", default="", help="the EPN_CTI file the CIF selected")
    ap.add_argument("--sas", default="", help="SAS version string")
    ap.add_argument("--ccd-temperature-k", type=float, required=True,
                    help="the CCD temperature of the CAL state the probe ran with [K] (PN_CCD_TEMPERATURE_K)")
    ap.add_argument("--temperature-source", default=TEMPERATURE_SOURCE, help="where the temperature comes from")
    a = ap.parse_args(argv)
    rec = write_table(a.from_probe, a.out, {"ccf_cti_file": a.ccf_file, "sas": a.sas,
                                            "ccd_temperature_k": a.ccd_temperature_k,
                                            "ccd_temperature_source": a.temperature_source,
                                            "builder": "tools/pn_cti/build_pn_cti_probe.sh"})
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__cites__ = collect_cites(__name__)
