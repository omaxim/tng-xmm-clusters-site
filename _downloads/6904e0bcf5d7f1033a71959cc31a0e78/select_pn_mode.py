"""EPIC-pn observing modes: out-of-time and live-time fractions per submode.

The out-of-time (OoT) fraction of a pn mode is the readout transfer time over
the frame cycle; the live-time fraction is the fraction of the frame time the
CCD integrates.  Values from the XMM-Newton Users' Handbook (EPIC-pn modes
table; the table is also copied in ``conditions_donor``, ``donor_corner_qpb``, ``donors.build_replica_targets`` and ``simulate_instrument``).  The PPS exposure-map header keyword
``OOTFRAC`` = live fraction x (1 - OoT fraction) reproduces the real headers
(FF 0.9364, eFF 0.9766, LW 0.9471, SW 0.7019) to 1e-3.

The CCF carries no OoT table; the handbook is the reference.

References
----------
XMM-Newton Users' Handbook, section 3.3.2 (EPIC science modes, Table 3);
Strueder et al. 2001 (EPIC-pn); Jansen et al. 2001 (XMM-Newton).
"""

from __future__ import annotations

from dataclasses import dataclass

from tngxmm.cites import cites, collect_cites


@dataclass(frozen=True)
class PnMode:
    """One EPIC-pn submode: its name, the out-of-time event fraction, the live-time fraction of the frame and whether
    the imaging geometry is modelled (``readout_windows``)."""
    submode: str
    oot_frac: float      # out-of-time event fraction
    live_frac: float     # live-time fraction of the frame
    supported: bool      # imaging geometry modelled (readout_windows)

    @property
    def ootfrac_pps(self) -> float:
        """PPS ``OOTFRAC`` = live fraction x (1 - OoT fraction)."""
        return self.live_frac * (1.0 - self.oot_frac)


#: pn submodes (Users' Handbook Table 3); the Large/Small Window geometry is
#: modelled by :mod:`tngxmm.instrument.readout_windows`
PN_MODES: dict[str, PnMode] = {
    "PrimeFullWindow": PnMode("PrimeFullWindow", 0.063, 0.999, True),
    "PrimeFullWindowExtended": PnMode("PrimeFullWindowExtended", 0.023, 1.000, True),
    "PrimeLargeWindow": PnMode("PrimeLargeWindow", 0.0016, 0.949, True),
    "PrimeSmallWindow": PnMode("PrimeSmallWindow", 0.011, 0.710, True),
}
DEFAULT_PN_MODE = "PrimeFullWindow"


@cites("xmm_mission", "epic_pn")
def select_pn_mode(submode: str | None) -> PnMode:
    """The mode record of a donor's pn submode (``None``/unknown -> Full Frame,
    the Full Frame OoT fraction 0.063)."""
    if submode in PN_MODES:
        return PN_MODES[submode]
    return PN_MODES[DEFAULT_PN_MODE]


__cites__ = collect_cites(__name__)
