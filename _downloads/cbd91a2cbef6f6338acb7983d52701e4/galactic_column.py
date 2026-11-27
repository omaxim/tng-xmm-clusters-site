"""Total Galactic hydrogen column (atomic plus molecular) for the X-ray absorption of a view.

The views are absorbed (``tngxmm.sim.apply_absorption``, TBabs) by the donor field's HI4PI column, which is
the 21-cm column of neutral ATOMIC hydrogen.  TBabs normalises its cross-section to the TOTAL hydrogen
column (atomic, molecular and ionised; Wilms, Allen & McCray 2000, Eq. 2), and X-rays are absorbed by the
molecular gas too.  Willingale et al. (2013, Eq. 6-7) calibrate the molecular column from N_HI and the
dust reddening E(B-V) of Schlegel, Finkbeiner & Davis (1998):

    N_H2 = N_H2max [1 - exp(-N_HI E(B-V) / N_c)]^alpha,   N_H,tot = N_HI + 2 N_H2,

with N_H2max = 7.2e20 cm^-2, N_c = 3.0e20 cm^-2 and alpha = 1.1.  The ratio N_H,tot / N_HI is 1 at high
latitude and rises to 1.3-1.6 near N_HI ~ 1e21 cm^-2.

Instrument option ``galactic_nh`` (:data:`GALACTIC_NH`): ``'hi4pi'`` (default) keeps the stored absorption
and every product; ``'willingale13'`` absorbs the view's cluster photons by the missing column
dN = N_H,view (ratio - 1) (exact: exp(-sigma N_tot) = exp(-sigma N_view) exp(-sigma dN), so an already
absorbed list is thinned with probability exp(-sigma(E) dN)), and the sky background and the AGN are
absorbed by N_H,view x ratio.  The ratio is the DONOR's (table :func:`build_table`: obsid, nh_hi_1e22,
ebv_sfd, nh_ratio), so a view whose N_H was jittered keeps its jitter.
"""

from __future__ import annotations

import math
from typing import Any

from tngxmm.cites import cites, collect_cites

#: values of InstrumentOptions.galactic_nh
GALACTIC_NH = ("hi4pi", "willingale13")
#: Willingale et al. (2013) Eq. 7 constants: N_H2max [molecules cm^-2], N_c [atoms cm^-2], alpha
W13_NH2_MAX = 7.2e20
W13_NC = 3.0e20
W13_ALPHA = 1.1
#: table columns
COLUMNS = ("obsid", "nh_hi_1e22", "ebv_sfd", "nh_h2_1e22", "nh_tot_1e22", "nh_ratio")


@cites("willingale2013")
def molecular_column(nh_hi_cm2: float, ebv_mag: float) -> float:
    """N_H2 [molecules cm^-2] of Willingale et al. (2013, Eq. 7) from N_HI [cm^-2] and E(B-V) [mag]."""
    if not (math.isfinite(nh_hi_cm2) and math.isfinite(ebv_mag)) or nh_hi_cm2 <= 0 or ebv_mag <= 0:
        return 0.0
    return W13_NH2_MAX * (1.0 - math.exp(-nh_hi_cm2 * ebv_mag / W13_NC)) ** W13_ALPHA


@cites("willingale2013")
def total_column(nh_hi_cm2: float, ebv_mag: float) -> float:
    """N_H,tot = N_HI + 2 N_H2 [cm^-2] (Willingale et al. 2013, Eq. 6)."""
    return float(nh_hi_cm2) + 2.0 * molecular_column(nh_hi_cm2, ebv_mag)


def column_ratio(nh_hi_1e22: float, ebv_mag: float) -> float:
    """N_H,tot / N_HI (1 when N_HI or E(B-V) is missing or not positive)."""
    nh = float(nh_hi_1e22) * 1e22
    if not math.isfinite(nh) or nh <= 0:
        return 1.0
    return total_column(nh, ebv_mag) / nh


def build_table(donors: Any) -> Any:
    """The donor table from a frame with columns obsid, nh_hi_1e22 (the donor's HI4PI column) and ebv_sfd."""
    import pandas as pd

    t = pd.DataFrame({"obsid": donors["obsid"].astype(str).str.zfill(10),
                      "nh_hi_1e22": donors["nh_hi_1e22"].astype(float), "ebv_sfd": donors["ebv_sfd"].astype(float)})
    t["nh_h2_1e22"] = [molecular_column(n * 1e22, e) / 1e22 for n, e in zip(t.nh_hi_1e22, t.ebv_sfd)]
    t["nh_tot_1e22"] = t.nh_hi_1e22 + 2.0 * t.nh_h2_1e22
    t["nh_ratio"] = [column_ratio(n, e) for n, e in zip(t.nh_hi_1e22, t.ebv_sfd)]
    return t[list(COLUMNS)]


def donor_ratio(table: Any, obsid: str) -> float:
    """The donor's N_H,tot / N_HI; KeyError when the donor is absent."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = float(donor_rows(table, obsid, "total-column").iloc[0]["nh_ratio"])
    if not (math.isfinite(r) and r >= 1.0):
        raise ValueError(f"donor {str(obsid).zfill(10)}: unusable column ratio {r}")
    return r


@cites("willingale2013", "wilms2000_tbabs")
def extra_absorption_keep(eobs: Any, nh_view_1e22: float, ratio: float, rng: Any,
                          abund_table: str = "lodd") -> Any:
    """Keep mask of an absorbed photon list for the missing column dN = N_H,view (ratio - 1): each photon is
    kept with probability exp(-sigma(E) dN), sigma the TBabs cross-section of ``abund_table``.  ratio = 1
    keeps every photon and draws nothing."""
    import numpy as np

    from tngxmm.sim.apply_absorption import tbabs_sigma

    e = np.asarray(eobs, dtype=float)
    dnh = float(nh_view_1e22) * (float(ratio) - 1.0)
    if not (dnh > 0) or e.size == 0:
        return np.ones(e.size, dtype=bool)
    return rng.random(e.size) < np.exp(-dnh * 1e22 * tbabs_sigma(abund_table)(e))


__cites__ = collect_cites(__name__)
