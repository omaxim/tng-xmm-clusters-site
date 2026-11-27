"""SAS-readable headers of the per-camera mock products (``instrument.sas_headers = 'on'``).

ESA's standard source detection (``edetect_chain``: ``eboxdetect``, ``esplinemap``,
``emldetect``; SAS 22.1) reads, besides the WCS, a handful of keywords that the mock
headers of the writer (:mod:`tngxmm.instrument.write_camera_products`) do not carry.  This module adds them, to the IMAGE and the
EXPMAP of every camera and band, without touching the WCS the fitter uses:

``TELESCOP``, ``OBS_MODE``, ``DATAMODE``
    ``XMM``, ``POINTING``, ``IMAGING`` (every mock is an imaging exposure).
``OBS_ID``, ``EXP_ID``, ``EXPIDSTR``, ``DATE-OBS``
    The donor exposure the mock imitates: its observation id, the exposure id of that
    camera (``<obsid><3-digit exposure number>``, a numeric string as in the PPS
    products; the number is read from the name of the donor's PPS exposure map in the
    donor table ``donor_active_ccds.csv``) and the start of the donor observation.  The
    donor's date selects the CCF issue (``cifbuild``) and so the PSF calibration of
    ``emldetect``.  ``MOCK = T`` and ``DONORID`` mark the product as simulated.
``RA_PNT``, ``DEC_PNT``, ``PA_PNT``
    Sky position of the camera's optical axis and the position angle of the detector.
    The position is the WCS of the pixel ``OPTAXX/OPTAXY`` (zero camera offsets:
    the image centre).  The angle is ``PA_PNT = (POSANGLE + PA_REF) mod 360`` with
    ``PA_REF = 114.2157 deg``, the ``PA_PNT`` of the observation (0827340901) whose
    detector masks the instrument stage rolls by ``POSANGLE`` (counter-clockwise on the
    north-up, east-left image): see :func:`pa_pnt_deg`.
``ONTIMEnn``, ``TIMEDEL``, ``FRMTIME``
    Live time of every imaged CCD (``nn`` = CCD number, as in the PPS headers), and for
    the pn the frame time ``FRMTIME`` [ms] and ``TIMEDEL`` [s] = frame time x
    ``OOTFRAC`` of the mode (reproduces the PPS values 0.0687 s for Full Frame and
    0.1945 s for Extended Full Frame to 1e-3).
physical sky coordinates
    ``REFX/REFY*``, ``CTYPEnL``, ``CRPIXnL``, ``CRVALnL``, ``CDELTnL``, ``LTVn``,
    ``LTMn_m``, ``WCSNAMEL``: the PPS convention of 0.05" physical sky pixels, the
    pixel 25921 of both axes at the optical axis, mapped to the 4" image (80 physical
    pixels per image pixel).

Integer keywords beyond 32 bits (``VIEWSEED``) are written as strings: SAS reads FITS
headers through the DAL, which cannot parse 64-bit integer keywords
(:func:`stringify_wide_ints`).

References
----------
XMM-Newton SOC / SAS: PPS image and exposure-map headers (SAS 22.1, PPS 21.51), the
keywords listed in the ``emldetect``, ``eboxdetect`` and ``eexpmap`` documentation;
Gabriel et al. 2004 (SAS).  XMM-Newton Users' Handbook, section 3.3 (EPIC modes,
frame times).
"""

from __future__ import annotations

from typing import Any

from tngxmm.cites import cites, collect_cites

#: signed 32-bit limit of a FITS integer keyword as read by the SAS DAL
INT32_MAX = 2**31 - 1
#: exposure number of a camera whose donor record names none (PPS: M1 = 001, M2 = 002, pn = 003)
DEFAULT_EXPOSURE_NUMBER = {"MOS1": "001", "MOS2": "002", "PN": "003"}
#: CCD numbers of an unwindowed camera
DEFAULT_CCDS = {"MOS1": tuple(range(1, 8)), "MOS2": tuple(range(1, 8)), "PN": tuple(range(1, 13))}
#: PPS physical sky pixel [arcsec]; image pixels are 80 of them (4")
PHYS_PIX_ARCSEC = 0.05
PHYS_REF_PIXEL = 25921
PHYS_MAX_PIXEL = 51840
#: pn frame time [ms] per submode (Users' Handbook Table 3)
PN_FRAME_MS = {"PrimeFullWindow": 73.4, "PrimeFullWindowExtended": 199.1,
               "PrimeLargeWindow": 47.7, "PrimeSmallWindow": 5.7}

_CAMERAS = ("MOS1", "MOS2", "PN")


@cites("xmm_ccf_sas")
def pa_pnt_deg(posangle_deg: float) -> float:
    """SAS ``PA_PNT`` [deg, 0-360) of a mock rolled by ``posangle_deg``.

    The instrument stage rolls events and masks counter-clockwise by ``POSANGLE`` about
    the optical axis, on an image with north up and east left; the packaged masks have
    the orientation of obsid 0827340901, ``PA_PNT = 114.2157 deg`` (position angle of
    the detector, north through east).  Counter-clockwise on that image is increasing
    position angle, so ``PA_PNT = POSANGLE + 114.2157`` (mod 360).  Checked against
    five real donors: the exposure-map
    footprint of the stage rolled by ``PA_PNT - 114.2157`` correlates 0.92-0.95 with the
    PPS exposure map of the observation, against 0.39-0.89 for the mirrored
    ``114.2157 - PA_PNT`` and 0.87-0.92 for the 180 degree flip."""
    from tngxmm.donors.build_core_offsets import PA_REF_DEG

    return float((float(posangle_deg) + PA_REF_DEG) % 360.0)


@cites("xmm_ccf_sas")
def exposure_id(obsid: str, camera: str, pps_name: str | None = None) -> tuple[str, str]:
    """``(EXP_ID, EXPIDSTR)`` of one camera: the donor's exposure number read from the name
    of its PPS exposure map (``P<obsid>M1S001EXPMAP2000.FTZ`` -> ``0745120101001``,
    ``S001``), the default number of the camera without one."""
    import re

    obsid = str(obsid).zfill(10)
    m = re.search(r"(?:M1|M2|PN)([SU])(\d{3})", str(pps_name or ""))
    if m:
        return obsid + m.group(2), m.group(1) + m.group(2)
    num = DEFAULT_EXPOSURE_NUMBER[camera]
    return obsid + num, "S" + num


@cites("xmm_ccf_sas")
def package_sas_donor(package: dict[str, Any], table: Any = None) -> dict[str, Any]:
    """The donor facts of the SAS headers from a donor package (``obsid`` /
    ``donor_id``) and the donor table ``donor_active_ccds.csv``: ``{'obsid', 'date_obs',
    'exposure': {camera: (EXP_ID, EXPIDSTR)}}``.  A donor absent from the table keeps the
    package's own ``date_obs`` (else None) and default exposure numbers."""
    from tngxmm.instrument.active_ccds import read_active_ccds

    obsid = str(package.get("obsid") or package.get("donor_id") or "").zfill(10)
    date = package.get("date_obs")
    names: dict[str, Any] = {}
    t = read_active_ccds() if table is None else table
    hit = t[t.obsid.astype(str).str.zfill(10) == obsid]
    if len(hit):
        r = hit.iloc[0]
        d = r.get("date_obs")
        date = str(d) if d is not None and str(d) not in ("", "nan") else date
        names = {c: r.get(f"pps_{c.lower()}") for c in _CAMERAS}
    return {"obsid": obsid, "date_obs": None if date is None else str(date),
            "exposure": {c: exposure_id(obsid, c, names.get(c)) for c in _CAMERAS}}


def pn_timing(submode: str | None) -> tuple[int, float] | None:
    """``(FRMTIME [ms], TIMEDEL [s])`` of a pn submode (None for an unknown one)."""
    from tngxmm.instrument.select_pn_mode import PN_MODES

    if submode not in PN_FRAME_MS or submode not in PN_MODES:
        return None
    frame_s = PN_FRAME_MS[submode] / 1000.0
    return int(round(PN_FRAME_MS[submode])), frame_s * PN_MODES[submode].ootfrac_pps


def stringify_wide_ints(h: Any) -> list[str]:
    """Rewrite every integer keyword of header ``h`` that does not fit a signed 32-bit
    integer as a string (comment kept); returns the keywords changed."""
    changed = []
    for card in list(h.cards):
        v = card.value
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) > INT32_MAX:
            h[card.keyword] = (str(v), card.comment)
            changed.append(card.keyword)
    return changed


@cites("xmm_ccf_sas")
def add_sas_keys(h: Any, *, inst: str, kind: str, context: dict[str, Any]) -> Any:
    """Add the SAS keywords to the IMAGE/EXPMAP header ``h`` of camera ``inst`` (see the
    module docstring).  ``context`` carries ``POSANGLE``, ``OPTAXX``, ``OPTAXY`` (0-indexed
    optical-axis pixel), the camera exposure ``EXPOSURE`` (``h``), ``_sas_ccds`` (CCDs
    imaged), ``_PNTX``/``_PNTY`` (the pointing pixel, else the optical axis), the PN mode ``_pn_submode`` and the donor facts ``_sas_donor``
    (:func:`package_sas_donor`).  Keys whose source is absent are skipped."""
    from astropy.wcs import WCS

    donor = context.get("_sas_donor") or {}
    h["TELESCOP"] = ("XMM", "Telescope (mission) name")
    h["OBS_MODE"] = ("POINTING", "Observation mode (pointing or slew)")
    h["DATAMODE"] = ("IMAGING", "Instrument mode (IMAGING, TIMING, BURST, etc.)")
    if donor.get("obsid"):
        h["OBS_ID"] = (donor["obsid"], "Observation identifier of the donor")
        exp_id, exp_str = donor["exposure"][inst]
        h["EXP_ID"] = (exp_id, "Exposure identifier of the donor")
        h["EXPIDSTR"] = (exp_str, "Exposure identifier (Xnnn)")
    if donor.get("date_obs"):
        h["DATE-OBS"] = (donor["date_obs"], "Start Time (UTC) of the donor exposure")
    if context.get("OPTAXX") is not None and context.get("POSANGLE") is not None:
        # the pointing: the optical-axis pixel, or ``_PNTX/_PNTY`` when the stage centres the
        # vignetting off the pointing (simulate_instrument vig_axis); FITS pixel
        px = float(context.get("_PNTX", context["OPTAXX"])) + 1.0
        py = float(context.get("_PNTY", context["OPTAXY"])) + 1.0
        ra, dec = (float(v) for v in WCS(h, naxis=2).all_pix2world([[px, py]], 1)[0])
        pa = pa_pnt_deg(float(context["POSANGLE"]))
        h["AVRG_PNT"] = ("MEDIAN", "Meaning of PNT values (mean or median)")
        h["RA_PNT"] = (ra, "[deg] Pointing RA of the optical axis")
        h["DEC_PNT"] = (dec, "[deg] Pointing Dec of the optical axis")
        h["PA_PNT"] = (pa, "[deg] Position angle of the detector")
        h["RA_NOM"] = (ra, "[deg] Ra of boresight")
        h["DEC_NOM"] = (dec, "[deg] Dec of boresight")
        _physical_keys(h, px, py, ra, dec)
    t_live = float(h.get("EXPOSURE", 0.0))
    ccds = context.get("_sas_ccds") or DEFAULT_CCDS[inst]
    for c in sorted({int(c) for c in ccds if 1 <= int(c) <= 12}):
        h[f"ONTIME{c:02d}"] = (t_live, "Sum of GTIs for a particular CCD")
    if inst == "PN":
        timing = pn_timing(context.get("_pn_submode"))
        if timing is not None:
            h["FRMTIME"] = (timing[0], "[ms] Frame Time, nearest integer")
            h["TIMEDEL"] = (timing[1], "[s] Length of exposure entry interval")
    return h


def _physical_keys(h: Any, px: float, py: float, ra: float, dec: float) -> None:
    """The PPS physical sky-pixel keywords: physical pixel 25921 at the optical axis (FITS
    pixel ``px``, ``py``), 80 physical pixels per image pixel."""
    scale = 80.0
    deg = PHYS_PIX_ARCSEC / 3600.0
    for ax, ctype, crval, cdelt in (("X", "RA---TAN", ra, -deg), ("Y", "DEC--TAN", dec, deg)):
        h[f"REF{ax}CTYP"] = (ctype, "WCS Coord. type")
        h[f"REF{ax}CRPX"] = (PHYS_REF_PIXEL, "WCS axis reference pixel of projected image")
        h[f"REF{ax}CRVL"] = (crval, "[deg] WCS coord. at the reference pixel")
        h[f"REF{ax}CDLT"] = (cdelt, "[deg/pix] WCS increment at the reference pixel")
        h[f"REF{ax}LMIN"] = (1, "WCS minimum legal projected image value")
        h[f"REF{ax}LMAX"] = (PHYS_MAX_PIXEL, "WCS maximum legal projected image value")
        h[f"REF{ax}CUNI"] = ("deg", "WCS Physical units")
    for n, (axis, pix) in enumerate((("X", px), ("Y", py)), start=1):
        crval_l = PHYS_REF_PIXEL - (pix - 1.0) * scale
        h[f"CTYPE{n}L"] = (axis, "WCS coordinate name")
        h[f"CRPIX{n}L"] = (1, "WCS reference pixel")
        h[f"CRVAL{n}L"] = (crval_l, "WCS reference pixel value")
        h[f"CDELT{n}L"] = (scale, "WCS pixel size")
        h[f"LTV{n}"] = 1.0 - crval_l / scale
        h[f"LTM{n}_{n}"] = 1.0 / scale
    h["LTM1_2"] = 0
    h["LTM2_1"] = 0
    h["WCSNAMEL"] = ("PHYSICAL", "WCS L name")
    h["WCSAXESL"] = (2, "No. of axes for WCS L")


__cites__ = collect_cites(__name__)
