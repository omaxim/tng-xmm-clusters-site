"""PATTERN selection efficiency of the EPIC images, from the CCF pattern tables.

The PPS images select PATTERN <= 4 (pn: singles and doubles) and
PATTERN <= 12 (MOS: singles to quadruples).  The upstream ARF
(xmm_simulator ``calc_arf``) multiplies by ``QE_TOTAL`` (all patterns), so a
mock without a selection loss has slightly too many pn counts.  The fraction
of X-ray events in each pattern class is tabulated per mode and CCD region
in the ``FRACTION_ENERGY`` extension of ``E<cam>_QUANTUMEF_<n>.CCF`` (columns
``FRAC_S, FRAC_D, FRAC_T, FRAC_Q, FRAC_SD, FRAC_SDTQ`` on ``EBINS_FRACTION``).

Efficiency used here (energy dependent, averaged over the region rows of
the mode): pn ``FRAC_SD / FRAC_SDTQ``; MOS 1 (PATTERN <= 12 keeps every valid
pattern).  On the current CCF (EPN_QUANTUMEF_0018) the FRACTION_ENERGY table lists no
triple or quadruple events, so FRAC_SDTQ = FRAC_SD and the applied pn ratio is 1.0000 in
0.5-2 keV (ledger instrument.pattern_loss); FRAC_SD itself is 0.985-0.988 at
1 keV in Full Frame.  With ``rule='sas'`` (the campaign setting) the pn efficiency is FRAC_SD,
the fraction SAS ``arfgen`` applies to a PATTERN <= 4 selection.  FLAG losses (bad pixels, CCD edges) are already in the
detection masks and are not applied again.

References
----------
XMM-Newton SOC, CCF ``EPN_QUANTUMEF``/``EMOS*_QUANTUMEF`` (FRACTION_ENERGY);
Gabriel et al. 2004 (SAS); Strueder et al. 2001 (pn); Turner et al. 2001 (MOS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tngxmm.cites import cites, collect_cites

#: SAS mode identifiers of the pn submodes in the QUANTUMEF fraction tables
PN_MODE_ID = {"PrimeFullWindow": 0, "PrimeFullWindowExtended": 1,
              "PrimeLargeWindow": 2, "PrimeSmallWindow": 3}
#: PPS pattern selection per camera (DSVAL of the PPS images)
PATTERN_SELECTION = {"PN": "PATTERN<=4", "MOS1": "PATTERN<=12", "MOS2": "PATTERN<=12"}


@dataclass(frozen=True)
class PatternEfficiency:
    """Energy-dependent PATTERN efficiency per camera (tables read from the CCF)."""
    pn_mode: str = "PrimeFullWindow"
    energy_kev: dict = field(default_factory=dict, compare=False)   # camera -> (n,)
    eff: dict = field(default_factory=dict, compare=False)          # camera -> (n,)
    qe_file: dict = field(default_factory=dict, compare=False)      # camera -> CCF name

    rule: str | None = None

    @property
    def key(self) -> tuple:
        """Cache key of this efficiency: ``('pattern-ccf', pn_mode[, rule])``."""
        return ("pattern-ccf", self.pn_mode) if self.rule is None else ("pattern-ccf", self.pn_mode, self.rule)

    def on_grid(self, inst: str, e_kev):
        """Efficiency of camera ``inst`` interpolated on ``e_kev`` [keV]."""
        import numpy as np

        if inst not in self.eff:
            return np.ones_like(np.asarray(e_kev, dtype=np.float64))
        return np.interp(e_kev, self.energy_kev[inst], self.eff[inst])

    def band_mean(self, inst: str, elo: float = 0.5, ehi: float = 2.0, n: int = 151) -> float:
        """Mean efficiency of camera ``inst`` over ``elo``-``ehi`` keV (an unweighted mean over ``n`` equally spaced
        energies; 1 for a camera without a table)."""
        import numpy as np

        e = np.linspace(elo, ehi, n)
        return float(np.mean(self.on_grid(inst, e)))


@cites("xmm_ccf_sas", "epic_pn", "epic_mos")
def pattern_efficiency_table(ccf_dir: str, inst: str, pn_mode: str = "PrimeFullWindow", rule: str | None = None):
    """(energy [keV], efficiency, CCF file) for one camera from the highest-issue
    ``E<inst>_QUANTUMEF`` CCF (the file the ARF uses). pn: ``rule`` None = FRAC_SD / FRAC_SDTQ,
    'sas' = FRAC_SD (the fraction SAS arfgen applies to a PATTERN<=4 selection); MOS: 1 (its ARF carries the
    PATTERN<=12 loss already, through mos_areacorr)."""
    import os

    import numpy as np
    from astropy.io import fits

    ccf = str(ccf_dir).rstrip('/') + '/'
    best, nbest = None, 0
    for tf in os.listdir(ccf):
        if f"E{inst}_QUANTUMEF" in tf:
            tn = int(tf.split('_')[2].split('.')[0])
            if tn > nbest:
                best, nbest = tf, tn
    if best is None:
        raise FileNotFoundError(f"no E{inst}_QUANTUMEF CCF in {ccf}")
    with fits.open(ccf + best) as f:
        e = np.asarray(f['EBINS_FRACTION'].data['ENERGY']).ravel() / 1e3
        d = f['FRACTION_ENERGY'].data
        if inst == "PN":
            mid = PN_MODE_ID.get(pn_mode, 0)
            rows = d[d['MODE_ID'] == mid]
            num = np.mean(np.stack([r['FRAC_SD'] for r in rows]), axis=0)
            den = (np.ones_like(num) if rule == "sas"
                   else np.mean(np.stack([r['FRAC_SDTQ'] for r in rows]), axis=0))
            with np.errstate(invalid='ignore', divide='ignore'):
                eff = np.where(den > 0, num / den, 1.0)
        else:
            eff = np.ones_like(e)
    eff = np.clip(np.nan_to_num(eff, nan=1.0), 0.0, 1.0)
    return e, eff, best


@cites("xmm_ccf_sas", "epic_pn", "epic_mos")
def apply_pattern_loss(ccf_dir: str, pn_mode: str = "PrimeFullWindow",
                       cameras: tuple[str, ...] = ("MOS1", "MOS2", "PN"), rule: str | None = None) -> PatternEfficiency:
    """The :class:`PatternEfficiency` to hand to ``load_responses.get_response(pattern=...)``:
    it multiplies the X-ray ARF (cluster, sky background, AGN), not the
    particle background."""
    energy, eff, files = {}, {}, {}
    for c in cameras:
        e, ef, f = pattern_efficiency_table(ccf_dir, c, pn_mode, rule)
        energy[c], eff[c], files[c] = e, ef, f
    return PatternEfficiency(pn_mode=pn_mode, energy_kev=energy, eff=eff, qe_file=files, rule=rule)


_SINGLE_TABLES: dict = {}


@cites("xmm_ccf_sas", "epic_pn")
def pn_single_pixel_table(ccf_dir: str, pn_mode: str = "PrimeFullWindow"):
    """(energy [keV], FRAC_S / FRAC_SDTQ) of the pn: the fraction of the valid X-ray events
    (patterns 0-12 of the QUANTUMEF table: singles, doubles, triples, quadruples) that are
    single-pixel (PATTERN 0), averaged over the region rows of the mode, from the same
    ``EPN_QUANTUMEF`` CCF as the SD efficiency.  Cached per (CCF directory, mode)."""
    import os

    import numpy as np
    from astropy.io import fits

    key = (str(ccf_dir), pn_mode)
    if key in _SINGLE_TABLES:
        return _SINGLE_TABLES[key]
    ccf = str(ccf_dir).rstrip('/') + '/'
    best, nbest = None, 0
    for tf in os.listdir(ccf):
        if "EPN_QUANTUMEF" in tf:
            tn = int(tf.split('_')[2].split('.')[0])
            if tn > nbest:
                best, nbest = tf, tn
    if best is None:
        raise FileNotFoundError(f"no EPN_QUANTUMEF CCF in {ccf}")
    with fits.open(ccf + best) as f:
        e = np.asarray(f['EBINS_FRACTION'].data['ENERGY']).ravel() / 1e3
        d = f['FRACTION_ENERGY'].data
        rows = d[d['MODE_ID'] == PN_MODE_ID.get(pn_mode, 0)]
        num = np.mean(np.stack([r['FRAC_S'] for r in rows]), axis=0)
        den = np.mean(np.stack([r['FRAC_SDTQ'] for r in rows]), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(den > 0, num / den, 1.0)
    out = (e, np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0))
    _SINGLE_TABLES[key] = out
    return out


@cites("xmm_ccf_sas", "epic_pn")
def pn_single_pixel_keep(ccf_dir: str, e_kev, pn_mode: str = "PrimeFullWindow",
                         sd_applied: bool = True):
    """Probability that a pn event of energy ``e_kev`` [keV] that is already a valid
    PATTERN <= 4 event is a single-pixel one: FRAC_S / FRAC_SD, with FRAC_SD / FRAC_SDTQ
    divided out when the ARF carries it (``sd_applied``; ``instrument.pattern_loss = 'ccf'``),
    FRAC_S / FRAC_SDTQ otherwise.  Clipped to [0, 1]."""
    import numpy as np

    e, single = pn_single_pixel_table(ccf_dir, pn_mode)
    p = np.interp(np.asarray(e_kev, dtype=np.float64), e, single)
    if sd_applied:
        _, sd, _ = pattern_efficiency_table(ccf_dir, "PN", pn_mode)
        p = p / np.maximum(np.interp(np.asarray(e_kev, dtype=np.float64), e, sd), 1e-6)
    return np.clip(p, 0.0, 1.0)


_QPB_SINGLE_TABLES: dict = {}
#: energy bins [keV] of the FWC single-pixel fraction (the band 0.2-0.5 keV in 20 eV steps)
_QPB_BANDEDGES = (0.2, 0.5)
_QPB_BIN_KEV = 0.02


@cites("xmm_ccf_sas", "epic_pn")
def pn_qpb_single_table(ccf_dir: str):
    """(energy centres [keV], fraction) of the pn particle-background events of the
    0.2-0.5 keV band that the PPS band-1 image keeps: ``FLAG == 0 && PATTERN == 0 &&
    RAWY > 12`` over ``FLAG == 0``, per 20 eV bin, from the events of the filter-wheel-closed
    CCF (``EPN_FWC``, the file the simulator's QPB spectrum is built from; that spectrum
    counts every pattern).  Below 0.3 keV the FWC events are mostly doubles (electronic
    noise: PATTERN <= 4 is 10 times the singles at 0.2-0.3 keV), which is why the
    PATTERN <= 4 band-1 image of the SAS thread is so much noisier than the PPS image.
    Cached per CCF directory."""
    import os

    import numpy as np
    from astropy.io import fits

    key = str(ccf_dir)
    if key in _QPB_SINGLE_TABLES:
        return _QPB_SINGLE_TABLES[key]
    ccf = key.rstrip('/') + '/'
    best, nbest = None, 0
    for tf in os.listdir(ccf):
        if tf.startswith("EPN_FWC"):
            tn = int(tf.split('_')[2].split('.')[0])
            if tn > nbest:
                best, nbest = tf, tn
    if best is None:
        raise FileNotFoundError(f"no EPN_FWC CCF in {ccf}")
    edges = np.arange(_QPB_BANDEDGES[0], _QPB_BANDEDGES[1] + 1e-9, _QPB_BIN_KEV) * 1e3
    with fits.open(ccf + best, memmap=True) as f:
        d = f['EVENTS'].data
        pi = np.asarray(d['PI'])
        ok = np.asarray(d['FLAG']) == 0
        single = ok & (np.asarray(d['PATTERN']) == 0) & (np.asarray(d['RAWY']) > 12)
        n_all = np.histogram(pi[ok], bins=edges)[0].astype(np.float64)
        n_one = np.histogram(pi[single], bins=edges)[0].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(n_all > 0, n_one / n_all, 0.0)
    out = (0.5 * (edges[1:] + edges[:-1]) / 1e3, np.clip(frac, 0.0, 1.0))
    _QPB_SINGLE_TABLES[key] = out
    return out


@cites("xmm_ccf_sas", "epic_pn")
def pn_band1_qpb_scale(ccf_dir: str, channel_lo_kev, channel_hi_kev,
                       pn_mode: str = "PrimeFullWindow", sd_applied: bool = True):
    """Per-channel factor of the pn QPB expectation (``CameraResponse.fwc_spec``, all patterns)
    that makes the band-1 events of the QPB stand for the PPS image's singles.

    The band-1 selection of :func:`~tngxmm.instrument.simulate_instrument.pn_band1_single_pixel`
    thins every event of 0.2-0.5 keV with :func:`pn_single_pixel_keep`; the QPB is meant to
    keep its own FWC single-pixel fraction (:func:`pn_qpb_single_table`), so its expectation
    is scaled by that fraction divided by the keep probability: the net retention is the FWC
    fraction.  1 outside 0.2-0.5 keV.  Channels given by their edges ``channel_lo/hi_kev``."""
    import numpy as np

    lo = np.asarray(channel_lo_kev, dtype=np.float64)
    mid = 0.5 * (lo + np.asarray(channel_hi_kev, dtype=np.float64))
    e, f0 = pn_qpb_single_table(ccf_dir)
    inb = (mid >= _QPB_BANDEDGES[0]) & (mid < _QPB_BANDEDGES[1])
    keep = np.maximum(pn_single_pixel_keep(ccf_dir, mid, pn_mode, sd_applied=sd_applied), 1e-6)
    want = np.interp(mid, e, f0)
    return np.where(inb, np.clip(want / keep, 0.0, 1.0), 1.0)


@cites("xmm_ccf_sas")
def describe(pat: Any) -> str:
    """One-line provenance string (CCF files and band means)."""
    if pat is None:
        return "none"
    parts = [f"{c}:{pat.qe_file.get(c, '-')}:{pat.band_mean(c):.4f}" for c in sorted(pat.eff)]
    return f"ccf-fraction-energy pn_mode={pat.pn_mode} " + " ".join(parts)


__cites__ = collect_cites(__name__)
