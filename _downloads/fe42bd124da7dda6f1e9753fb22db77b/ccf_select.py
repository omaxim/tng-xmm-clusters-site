"""Which file of the calibration directory a response takes: the highest issue number of the file name (the rule of
upstream xmm_simulator, ``utils.get_ccf_file_names``, and of ``massfit.build_responses``) or the constituent SAS
selects for the observation date (``cifbuild``).

The two rules differ where a directory keeps a constituent SAS no longer uses.  The site CCF directory holds
``EMOS1_QUANTUMEF_0026`` and ``EMOS2_QUANTUMEF_0024``: files of 2002 whose header ``ISSUE`` is 14, valid from
2000-01-01; the highest-number rule takes them for every epoch, while cifbuild takes ``EMOSn_QUANTUMEF_0020`` (header
issue 20, valid from 2000-01-01 until ``EVALDATE`` 2002-11-08) or ``EMOSn_QUANTUMEF_0021`` (valid from 2002-11-08).
For every other file type the stage reads the two rules agree.

Scope.  The rule is applied where the response is made: the stage's response (``load_responses.get_response``:
quantum efficiency, filter, mirror area and its vignetting, PSF, the FWC file of upstream's particle spectrum) and
the fit's shared ARFs (``massfit.build_responses``).  The other readers of the CCF keep the highest issue:
``instrument.calibration_axis`` (XMM_MISCDATA, LINCOORD, XAREAEF), ``donor_background`` and ``donor_tables.fwc``
(the FWC sets, XMM_SPDETMAP), ``rga_obscuration``, ``area_perturbation`` (ABSCORRAREA), ``apply_pattern_loss``
(EPN_QUANTUMEF), ``compute_ecf`` (the ECF table), and in the fit ``fit_psf`` (XPSF), ``prepare_images``
(XAREAEF) and ``qpb_template`` (XMM_SPDETMAP).  In the site CCF directory each of those types has the same file
under both rules at every date of 2000-2025 (of its 52 scope-types, 15 differ: the MOS quantum efficiency and
ADU conversion, bad pixels, CTI, dark frames, housekeeping limits and the MOS redistribution, which no reader above
takes), so the two rules give the same products there; another directory would need the same check
(``validity_breaks`` and :func:`select` under both rules).

``rule``:

``'highest-issue'``
    the highest issue number in the file name ``<SCOPE>_<TYPEID>_<nnnn>.CCF`` (today's selection of the presets);
``'valid-at-date'``
    the selection of SAS cifbuild (its documented algorithm): among the constituents of the scope and type
    (header ``SCOPE``, ``TYPEID``, ``CATEGORY`` XMMCCF) with ``VALDATE`` <= observation date, ``EVALDATE`` (when
    present) >= observation date and ``SUBDATE`` <= analysis date, the highest header ``ISSUE``.  The analysis
    date is the day the valid CCF set was downloaded (:data:`tngxmm.instrument.sas_detect.CCF_ANALYSIS_DATE`), the
    date the per-observation responses are built with.

References
----------
sas_cifbuild (the SAS 22.0 task document of cifbuild: section 3.6, membership of the current calibration file, and
section 8, the algorithm), xmm_ccf_sas (the CCF constituents), xmm_simulator (``get_ccf_file_names``).
"""

from __future__ import annotations

import functools
import glob
import os
import re
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: the rules of :func:`select` (and of ``instrument.ccf_selection`` / ``massfit.ccf_selection``)
CCF_RULES = ("highest-issue", "valid-at-date")
_NAME = re.compile(r"^([A-Z0-9]+)_([A-Z0-9_]+?)_(\d{4})\.CCF$", re.I)


def rule_of(value: Any) -> str:
    """The rule of a spec value: None or '' -> ``'highest-issue'``; else the value, checked."""
    rule = str(value or "highest-issue")
    if rule not in CCF_RULES:
        raise ValueError(f"unknown CCF selection {value!r}; known: {CCF_RULES}")
    return rule


def iso_date(text: Any) -> str:
    """A header date (``VALDATE``, ``DATE-OBS``: ``YYYY-MM-DD`` or ``YYYY-MM-DDThh:mm:ss[.sss]``) as
    ``YYYY-MM-DDThh:mm:ss``, which orders as a string; '' for none.  The CIF form ``yyyy:dd:mmZhh:mm:ss`` is refused."""
    s = str(text or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "T")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d*)?Z?)?", s):
        raise ValueError(f"unexpected date form {text!r}: YYYY-MM-DD[Thh:mm:ss[.sss]]")
    return (s + "T00:00:00")[:19] if len(s) == 10 else s[:19]


@functools.lru_cache(maxsize=8)
def constituents(ccf_dir: str) -> tuple:
    """Every ``*.CCF`` file of ``ccf_dir`` with its selection attributes: tuples
    ``(file, scope, typeid, file_issue, header_issue, valdate, evaldate, subdate, category)`` read from the
    primary header (cached per directory)."""
    from astropy.io import fits

    out = []
    for p in sorted(glob.glob(os.path.join(str(ccf_dir), "*.CCF"))):
        m = _NAME.match(os.path.basename(p))
        if not m:
            continue
        try:
            h = fits.getheader(p, 0)
        except OSError:
            continue
        out.append((os.path.basename(p), str(h.get("SCOPE", m.group(1))).strip(),
                    str(h.get("TYPEID", m.group(2))).strip(), int(m.group(3)), int(h.get("ISSUE", m.group(3))),
                    iso_date(h.get("VALDATE")), iso_date(h.get("EVALDATE")), iso_date(h.get("SUBDATE")),
                    str(h.get("CATEGORY", "XMMCCF")).strip()))
    return tuple(out)


def default_analysis_date(analysis_date: str | None = None) -> str:
    """``analysis_date``, else the analysis date of the per-observation responses
    (:data:`tngxmm.instrument.sas_detect.CCF_ANALYSIS_DATE`)."""
    from tngxmm.instrument.sas_detect import CCF_ANALYSIS_DATE

    return iso_date(analysis_date or CCF_ANALYSIS_DATE)


@cites("sas_cifbuild", "xmm_ccf_sas", "xmm_simulator")
def select(ccf_dir: str | Path, scope: str, typeid: str, date: str | None = None, rule: str = "highest-issue",
           analysis_date: str | None = None) -> str:
    """The file name of the ``scope``/``typeid`` constituent ``ccf_dir`` gives under ``rule`` (see the module
    docstring); ``date`` (the observation date, ISO) is required by ``'valid-at-date'``.  FileNotFoundError when no
    constituent qualifies."""
    rule = rule_of(rule)
    rows = [r for r in constituents(str(ccf_dir)) if r[0].upper().startswith(f"{scope}_{typeid}_".upper())
            and _NAME.match(r[0]).group(2).upper() == typeid.upper()]
    if rule == "highest-issue":
        if not rows:
            raise FileNotFoundError(f"no {scope}_{typeid} CCF in {ccf_dir}")
        return max(rows, key=lambda r: r[3])[0]
    if not date:
        raise ValueError(f"CCF selection 'valid-at-date' of {scope}_{typeid} needs the observation date")
    d, a = iso_date(date), default_analysis_date(analysis_date)
    ok = [r for r in rows if r[1] == scope and r[2] == typeid and r[8] == "XMMCCF"
          and r[5] and r[5] <= d and (not r[6] or r[6] >= d) and (not r[7] or r[7] <= a)]
    if not ok:
        raise FileNotFoundError(f"no {scope}_{typeid} CCF in {ccf_dir} valid at {d} (analysis date {a})")
    return max(ok, key=lambda r: (r[4], r[5], r[0]))[0]


def camera_files(ccf_dir: str | Path, inst: str, date: str | None = None, rule: str = "highest-issue",
                 mirror: str | None = None) -> dict[str, str]:
    """The files of one camera the stage and the shared ARFs read: ``qe``, ``filter``, ``area``, ``psf``, ``fwc``
    (file names in ``ccf_dir``).  ``mirror``: the XRT of the area file (default the camera's own: XRT1 MOS1, XRT2
    MOS2, XRT3 PN); the PSF is always the camera's own XRT."""
    xrt = {"MOS1": "XRT1", "MOS2": "XRT2", "PN": "XRT3"}[inst]
    out = {"qe": select(ccf_dir, f"E{inst}", "QUANTUMEF", date, rule),
           "filter": select(ccf_dir, f"E{inst}", "FILTERTRANSX", date, rule),
           "area": select(ccf_dir, mirror or xrt, "XAREAEF", date, rule),
           "psf": select(ccf_dir, xrt, "XPSF", date, rule)}
    try:
        out["fwc"] = select(ccf_dir, f"E{inst}", "FWC", date, rule)
    except FileNotFoundError:
        pass
    return out


def validity_breaks(ccf_dir: str | Path, scopes_types: list[tuple[str, str]], start: str, end: str) -> list[str]:
    """The dates in [start, end] at which the ``valid-at-date`` selection of one of ``scopes_types`` can change
    (every VALDATE, and every EVALDATE plus one second, inside the interval), sorted, with ``start`` first."""
    import datetime as dt

    out = {iso_date(start)}
    s, e = iso_date(start), iso_date(end)
    for r in constituents(str(ccf_dir)):
        if (r[1], r[2]) not in scopes_types:
            continue
        valdate, evaldate = r[5], r[6]
        starts = [valdate] if valdate else []
        if evaldate:                     # EVALDATE is the last valid instant: the next selection starts after it
            starts.append((dt.datetime.fromisoformat(evaldate) + dt.timedelta(seconds=1)).isoformat()[:19])
        out.update(d for d in starts if s < d <= e)
    return sorted(out)


__cites__ = collect_cites(__name__)
