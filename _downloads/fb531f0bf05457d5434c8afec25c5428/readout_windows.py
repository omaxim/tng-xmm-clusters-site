"""EPIC readout windows: the detector area each camera images in the donor's mode.

Under ``instrument.readout_windows = 'donor'`` every camera images only the
silicon its donor mode reads out.  The window is a 0/1 mask on the 900 x 900
detector grid of :class:`~tngxmm.instrument.load_responses.CameraResponse`
(2.5" pixels) multiplied into ``resp.mask``; the windowed response
(:func:`windowed_response`) is a copy with that mask, its active pixels and a
QPB normalisation scaled to the reduced area.  Every mask lookup of the
instrument stage (cluster, sky background, QPB, AGN, PN out-of-time events,
the out-of-FOV rule) and both exposure-map constructions read ``resp.mask``,
so events and EXPMAPs share one footprint by construction.  Full-frame modes
return the response unchanged (``'full-frame'`` images every mode on the full mask).

Geometry (detector frame of the packaged masks, which are the sky-frame
footprints of obsid 0827340901 at PA_PNT = 114.2 deg; the CCD edges make the
angle ``-pn_readout_tilt_deg`` = 24.5 deg with the mask axes, measured on the
PN row boundary and column gaps to 24.44 +- 0.02 deg; the MOS gaps agree within
about 1 deg; q across the PN columns, s along them, as in
:func:`tngxmm.instrument.simulate_instrument.pn_out_of_time`):

* PN pixel = 4.1" = one CCD row (Users' Handbook Table 3).  The window rows
  are counted from the INNER silicon edge of each CCD row (the edge at the
  boundary between the two CCD rows, i.e. high RAWY; RAWY increases away from
  the readout node, EPN_LINCOORD): ``PrimeLargeWindow`` keeps 99 rows of all
  12 CCDs (EPN_MODEPARAM mode 2: RAWY 101-199; Handbook 198 x 384 pixels for
  the whole array); ``PrimeSmallWindow`` keeps 64 rows (EPN_MODEPARAM mode 3:
  RAWY 136-199; the Handbook table gives 63 x 64 pixels; the half-maximum
  edges of the PPS small-window exposure maps span 64.0 rows, as the large
  window spans 99.0) of CCD 4 only (Handbook): the CCD of the upper row
  (s > boundary) left of the central column gap, q in [-3.52', +0.96']
  between the centres of the dead gap lanes either side (mask pixels -84.5
  and +23.0; the PPS window spans 64.3 columns there).
  ``FastTiming``/``FastBurst`` have no imaging and are refused.
* MOS pixel = 1.1".  ``PrimePartialW2`` reads 100 x 100 pixels (1.83' square)
  and ``PrimePartialW3`` 300 x 300 (5.5') of the central CCD (Handbook
  Table 3; EMOS*_MODEPARAM modes 2/3 list 110 x 100 and 310 x 300 RAWX x RAWY
  pixels, the PPS exposure maps show squares, 42-45 and 126-133 mask pixels
  on both axes); the outer CCDs image normally.  The windows are centred on
  the central CCD, whose centre in the mask frame is
  :data:`MOS_CENTRAL_CCD_CENTRE_ARCMIN` (centroid of the PPS windows of
  0506210101, 0780291001 and 0810231001; consistent with the
  central-CCD edges of the mask within 3 px).  ``FastUncompressed`` /
  ``FastCompressed`` (timing): the central CCD (600 x 600 pixels, 11.0') gives
  no image.  ``PrimePartialRFS`` reads the full central CCD.  W1/W4/W5/W6
  (no donor uses them; their window geometry is not in the Handbook) are
  refused.

References
----------
XMM-Newton Users Handbook section 3.3.2 and Table 3 (window sizes, CCD 4);
XMM-Newton CCF EPN_MODEPARAM_0003, EMOS1_MODEPARAM_0006 (WINDOWX0/Y0/DX/DY),
EPN_LINCOORD_0009 (CCD positions, RAWY orientation); Strueder et al. 2001
(EPIC-pn); Turner et al. 2001 (EPIC-MOS).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tngxmm.cites import cites, collect_cites

#: rules of ``instrument.readout_windows``
READOUT_WINDOW_RULES = ("full-frame", "donor")

#: pixel sizes on the sky [arcsec] (Users' Handbook Table 3)
PN_PIXEL_ARCSEC = 4.1
MOS_PIXEL_ARCSEC = 1.1
#: PN window rows counted from the inner silicon edge of each CCD row
PN_LW_ROWS = 99
PN_SW_ROWS = 64
#: PN CCD 4 across the columns [arcmin, q from the detector centre]: centres of
#: the gap lanes either side (measured on the mask at the earlier tilt -24.5 deg; the 0.066 deg to -24.434 moves them by
#: under 0.01 arcmin at the 7 arcmin lever arm)
PN_SW_Q_RANGE_ARCMIN = (-3.521, 0.958)
#: MOS window side [pixels] per submode
MOS_WINDOW_PIXELS = {"PrimePartialW2": 100, "PrimePartialW3": 300}
#: MOS central CCD side [pixels]
MOS_CCD_PIXELS = 600
#: centre of the MOS central CCD = of its windows [arcmin, (q, s) frame]
MOS_CENTRAL_CCD_CENTRE_ARCMIN = {"MOS1": (-1.158, -1.188), "MOS2": (-0.975, -1.308)}

PN_FULL = ("PrimeFullWindow", "PrimeFullWindowExtended")
PN_NO_IMAGING = ("FastTiming", "FastBurst")
MOS_FULL = ("PrimeFullWindow", "PrimePartialRFS")
MOS_TIMING = ("FastUncompressed", "FastCompressed")
MOS_UNSUPPORTED = ("PrimePartialW1", "PrimePartialW4", "PrimePartialW5", "PrimePartialW6")

_WINDOW_MASKS: dict = {}
_WINDOWED: dict = {}


def _norm(submode: Any) -> str | None:
    if submode is None:
        return None
    if isinstance(submode, float) and submode != submode:
        return None
    s = str(submode).strip()
    return s or None


@cites("xmm_uhb", "xmm_ccf_sas")
def window_kind(inst: str, submode: str | None) -> str:
    """'full' | 'pn-lw' | 'pn-sw' | 'mos-w2' | 'mos-w3' | 'mos-timing' for the
    camera's submode (None = mode unknown: full frame); raises for modes
    without imaging (PN timing/burst) or without a verified geometry."""
    sm = _norm(submode)
    if sm is None:
        return "full"
    if inst == "PN":
        if sm in PN_FULL:
            return "full"
        if sm == "PrimeLargeWindow":
            return "pn-lw"
        if sm == "PrimeSmallWindow":
            return "pn-sw"
        if sm in PN_NO_IMAGING:
            raise ValueError(f"PN submode {sm!r} has no imaging: not usable as an imaging donor")
        raise ValueError(f"unknown PN submode {sm!r}")
    if inst in ("MOS1", "MOS2"):
        if sm in MOS_FULL:
            return "full"
        if sm == "PrimePartialW2":
            return "mos-w2"
        if sm == "PrimePartialW3":
            return "mos-w3"
        if sm in MOS_TIMING:
            return "mos-timing"
        if sm in MOS_UNSUPPORTED:
            raise ValueError(f"{inst} submode {sm!r}: window geometry not modelled")
        raise ValueError(f"unknown {inst} submode {sm!r}")
    raise ValueError(f"unknown camera {inst!r}")


def ccd_frame(resp: Any, tilt_deg: float):
    """(q, s) [mask px from the detector centre] of every mask pixel in the
    CCD-aligned frame (s along the PN readout columns; the same rotation for
    the MOS CCD edges)."""
    import numpy as np

    a = np.deg2rad(tilt_deg)
    yy, xx = np.indices(resp.mask.shape)
    dx, dy = xx - resp.cx, yy - resp.cy
    q = dx * np.cos(a) - dy * np.sin(a)
    s = dx * np.sin(a) + dy * np.cos(a)
    return q, s


def _pn_inner_edges(resp: Any) -> tuple[float, float]:
    from tngxmm.instrument.simulate_instrument import _pn_row_inner_edges

    return _pn_row_inner_edges(resp)


@cites("xmm_uhb", "xmm_ccf_sas", "epic_pn", "epic_mos")
def window_mask(resp: Any, submode: str | None, tilt_deg: float):
    """0/1 uint8 window of ``resp.inst`` in mode ``submode`` on the mask grid
    (1 everywhere for full-frame modes); cached per (camera, mask, mode, tilt)."""
    import numpy as np

    kind = window_kind(resp.inst, submode)
    key = (resp.inst, id(resp.mask), kind, float(tilt_deg))
    if key in _WINDOW_MASKS:
        return _WINDOW_MASKS[key]
    if kind == "full":
        win = np.ones(resp.mask.shape, dtype=np.uint8)
    else:
        q, s = ccd_frame(resp, tilt_deg)
        if kind in ("pn-lw", "pn-sw"):
            row = PN_PIXEL_ARCSEC / 60.0 / resp.pixsize
            top_in, bot_in = _pn_inner_edges(resp)
            nrow = PN_LW_ROWS if kind == "pn-lw" else PN_SW_ROWS
            top = (s >= top_in) & (s <= top_in + nrow * row)
            if kind == "pn-lw":
                w = top | ((s <= bot_in) & (s >= bot_in - nrow * row))
            else:
                qlo, qhi = (v / resp.pixsize for v in PN_SW_Q_RANGE_ARCMIN)
                w = top & (q >= qlo) & (q <= qhi)
        else:
            pix = MOS_PIXEL_ARCSEC / 60.0 / resp.pixsize
            qc, sc = (v / resp.pixsize for v in MOS_CENTRAL_CCD_CENTRE_ARCMIN[resp.inst])
            if kind == "mos-timing":
                half = 0.5 * MOS_CCD_PIXELS * pix
                w = ~((np.abs(q - qc) <= half) & (np.abs(s - sc) <= half))
            else:
                side = MOS_WINDOW_PIXELS["PrimePartialW2" if kind == "mos-w2" else "PrimePartialW3"]
                half_win = 0.5 * side * pix
                half_ccd = 0.5 * MOS_CCD_PIXELS * pix
                central = (np.abs(q - qc) <= half_ccd) & (np.abs(s - sc) <= half_ccd)
                inwin = (np.abs(q - qc) <= half_win) & (np.abs(s - sc) <= half_win)
                w = ~central | inwin
        win = w.astype(np.uint8)
    win.setflags(write=False)
    _WINDOW_MASKS[key] = win
    return win


@cites("xmm_uhb", "xmm_ccf_sas", "xmm_simulator")
def windowed_response(resp: Any, submode: str | None, tilt_deg: float) -> Any:
    """``resp`` imaged through the readout window of ``submode``: mask x window,
    the active pixels of that product, the QPB spectrum scaled to the read-out
    area (the full-frame normalisation is ``area_mask`` = active pixels x pixel
    area) and ``window`` = the submode.  Full-frame modes return ``resp``
    itself.  Cached (the object identity keys the downstream caches)."""
    import numpy as np

    kind = window_kind(resp.inst, submode)
    if kind == "full":
        return resp
    key = (id(resp), kind, float(tilt_deg))
    if key in _WINDOWED:
        return _WINDOWED[key][1]
    win = window_mask(resp, submode, tilt_deg)
    mask = (resp.mask * win).astype(np.uint8)
    ay, ax = np.nonzero(mask)
    frac = float(mask.sum()) / float(resp.mask.sum())
    out = dataclasses.replace(resp, mask=mask, active_x=ax, active_y=ay,
                              fwc_spec=np.asarray(resp.fwc_spec) * frac,
                              window=_norm(submode))
    _WINDOWED[key] = (resp, out)          # keep resp alive: its id is in the key
    return out


@cites("xmm_uhb")
def read_out_area_arcmin2(resp: Any) -> float:
    """Active (read-out) detector area of a response [arcmin^2]."""
    return float((resp.mask == 1).sum()) * resp.pixsize ** 2


@cites("xmm_uhb")
def apply_readout_windows(resps: dict[str, Any], submodes: dict[str, str | None],
                          tilt_deg: float, rule: str = "donor") -> dict[str, Any]:
    """Per-camera responses through the donor's windows (``rule='donor'``) or
    unchanged (``'full-frame'``)."""
    if rule not in READOUT_WINDOW_RULES:
        raise ValueError(f"unknown readout_windows rule {rule!r}; known: {READOUT_WINDOW_RULES}")
    if rule == "full-frame":
        return dict(resps)
    return {c: windowed_response(r, submodes.get(c), tilt_deg) for c, r in resps.items()}


__cites__ = collect_cites(__name__)
