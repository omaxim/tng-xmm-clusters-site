"""Per-observation EPIC responses: the RMF and on-axis ARF of each exposure of the real observations, made by SAS for
the observation's date, camera, mode, filter and pattern selection.

The shared set (``massfit.build_responses``, MBPROJ/responses_v1) has one RMF per camera, made in 2020 with rmfgen of
SAS 16.1 for one observation, and on-axis ARFs computed from the CCF tables.  Generation and fit both use it, so a mock
fit never meets the response of another epoch, while a real fit does.  This module builds, for every exposure that a
real observation's fit images come from, the response SAS would give that exposure, and serves it to

* the fit (``massfit.responses = 'per-observation'``): every image takes the RMF and ARF of its own exposure, the
  conditions donor's (header ``DONORID``) for a mock and its own (``OBS_ID``) for a real image;
* the instrument stage (``instrument.responses``): the photons of a view are redistributed with the conditions
  donor's RMF (``'donor-rmf'``), and with ``'donor'`` the on-axis energy dependence of the donor's ARF replaces that
  of the stage's CCF ARF, whose off-axis (vignetting) dependence is kept.

Layout of a response set (root ``$TNGXMM_OBS_RESP``, site key ``TNGXMM_OBS_RESP``)::

    <root>/<obsid>/<CAMERA>_<EXPID>.rmf      rmfgen, the cluster region, detector-map weighted
    <root>/<obsid>/<CAMERA>_<EXPID>.arf      arfgen, on axis, on the energy grid of that RMF
    <root>/<obsid>/provenance.json           SAS version, CCF set and the constituents used, inputs, settings, sha256
    <root>/<obsid>/logs/                     the SAS logs
    <root>/manifest.csv                      one row per exposure (written by :func:`write_manifest`)

``CAMERA`` is PN, MOS1 or MOS2, ``EXPID`` the PPS exposure identifier (``S003``, ``U002``, ...; header ``EXPIDSTR``).

Per exposure (:func:`entry_command`; SAS 22.1 through the ``sasrun`` container wrapper; the SAS threads' settings):

1. ``cifbuild`` for the observation date (``withobservationdate``) with the fixed analysis date of the CCF download
   (:data:`tngxmm.instrument.sas_detect.CCF_ANALYSIS_DATE`), once per observation;
2. ``evselect``: the spectrum of the cluster region of the PPS event list, PATTERN <= 4 and FLAG == 0 for the pn,
   PATTERN <= 12 and #XMMEA_EM for the MOS (the pattern selections of the PPS images and of the SAS threads), PI
   channels of 5 eV (pn, 0-20479) and 15 eV (MOS, 0-11999); and the detector image (DETX, DETY, 4 arcsec) of the same
   events at 0.5-7 keV;
3. ``rmfgen`` with that image as detector map (``detmaptype=dataset``): the redistribution averaged over the
   cluster's counts in the region, which carries the epoch (the MOS redistribution changes with time in the CCF,
   EMOSn_REDIST issues by validity date) and the position (MOS boresight patch, pn RAWY);
4. ``arfgen`` on axis (``sourcecoords=tel``, theta = 0; flat detector map, no encircled-energy, bad-pixel or
   out-of-time correction: the PPS exposure maps carry the vignetting, the gaps and the dead time of an image), with
   the corrections of SAS 22 at their defaults written out (MOS contamination, MOS-to-pn CORRAREA, EPIC-to-NuSTAR
   ABSCORRAREA; :class:`ResponseSettings`).

The region is the circle of the cluster centre the fit uses (``obs_centres_guarded_v4.csv``: the guarded X-ray peak)
with radius theta500 of the table, at most :data:`REGION_MAX_ARCSEC`.

Resumable: an observation is finished when its ``provenance.json`` says ``ok`` and every response it lists exists;
:func:`run_queue` works off the rest, one single-core observation per core.

References
----------
xmm_ccf_sas (SAS rmfgen, arfgen, cifbuild, evselect; CCF), xmm_cal_tn0018 (the time-dependent MOS redistribution and
contamination, the SAS 22 default effective-area corrections).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: version label of the response set (provenance and manifest)
RESPONSE_SET_VERSION = "obs-v1"
#: site key / environment variable of the response-set root
ROOT_KEY = "TNGXMM_OBS_RESP"
#: cameras -> PPS file code
CAMERA_CODES = {"PN": "PN", "MOS1": "M1", "MOS2": "M2"}
#: header INSTRUME -> camera
INSTRUME = {"EPN": "PN", "EMOS1": "MOS1", "EMOS2": "MOS2"}
#: event selection of the spectra and detector maps (SAS threads; the PATTERN cuts of the PPS images)
SELECTION = {"PN": "(FLAG==0)&&(PATTERN<=4)", "MOS1": "#XMMEA_EM&&(PATTERN<=12)", "MOS2": "#XMMEA_EM&&(PATTERN<=12)"}
#: spectral binning of the SAS threads: (spectralbinsize [eV], specchannelmax)
BINNING = {"PN": (5, 20479), "MOS1": (15, 11999), "MOS2": (15, 11999)}
#: the largest radius of the region [arcsec]: the cluster's counts within theta500, at most 5 arcmin (a design
#: parameter, docs/assumptions/OPEN_ITEMS.md section P)
REGION_MAX_ARCSEC = 300.0
#: sky pixel of the event lists [arcsec]
SKY_PIXEL_ARCSEC = 0.05
#: the CCF constituents a response depends on (recorded from the CIF in the provenance)
RESPONSE_CCF_TYPES = ("REDIST", "QUANTUMEF", "FILTERTRANSX", "XAREAEF", "CONTAMINATION", "XPSF", "ADUCONV", "CTI")
#: PPS event list of an exposure: P<obsid><code><S|U><3 digits><PI|MI>EVLI0000.FTZ
EVLI_PATTERN = re.compile(r"^P(\d{10})(M1|M2|PN)([SU]\d{3})(MI|PI)EVLI0000\.FTZ$")


@dataclasses.dataclass(frozen=True)
class ResponseSettings:
    """The settings of every response of a set (recorded in each provenance)."""
    detmap: str = "cluster"                  # rmfgen weighting: 'cluster' (detector image, dataset) | 'psf' (centre)
    region_max_arcsec: float = REGION_MAX_ARCSEC
    detmap_pi: tuple = (500, 7000)           # PI range [eV] of the detector image
    detmap_bin: int = 80                     # detector image pixel [0.05 arcsec] (4 arcsec)
    contamination: bool = True               # arfgen modelcontamination (SAS 22 default)
    xcal: bool = True                        # arfgen applyxcaladjustment: MOS-to-pn CORRAREA (SAS 22 default)
    absflux: bool = True                     # arfgen applyabsfluxcorr: EPIC-to-NuSTAR ABSCORRAREA (SAS 22 default)
    analysisdate: str | None = None          # cifbuild analysis date; None = sas_detect.CCF_ANALYSIS_DATE
    cpus: str | None = "0-3"                 # host cores (taskset); None = no pinning
    sasrun: str | None = None                # sasrun wrapper; None = $TNGXMM_SASRUN, then the site

    def cif_analysis_date(self) -> str:
        """The cifbuild analysis date: the setting, else the date of the CCF download (``sas_detect``)."""
        from tngxmm.instrument.sas_detect import CCF_ANALYSIS_DATE

        return self.analysisdate or CCF_ANALYSIS_DATE

    def record(self) -> dict[str, Any]:
        """The settings that make the responses (no host or path setting)."""
        d = dataclasses.asdict(self)
        for k in ("cpus", "sasrun"):
            d.pop(k)
        d["analysisdate"] = self.cif_analysis_date()
        d["detmap_pi"] = list(self.detmap_pi)
        return d


# ------------------------------------------------------------------ lookup ----

def response_root(root: str | Path | None = None) -> Path:
    """The response-set root: ``root`` > ``$TNGXMM_OBS_RESP`` > the site key ``TNGXMM_OBS_RESP``."""
    from tngxmm.core import site

    if root:
        return Path(root)
    val = os.environ.get(ROOT_KEY) or site.get(ROOT_KEY)
    if not val:
        raise site.SiteNotConfigured(f"no per-observation response set: pass the root or set ${ROOT_KEY}")
    return Path(val)


def response_name(camera: str, expid: str) -> str:
    """``<CAMERA>_<EXPID>`` (the stem of an exposure's RMF and ARF)."""
    return f"{camera}_{str(expid).strip()}"


def response_pair(root: str | Path | None, obsid: str, camera: str, expid: str | None = None) -> tuple[Path, Path]:
    """``(rmf, arf)`` of one exposure of ``obsid`` in the set.  With ``expid`` (the header's EXPIDSTR) exactly that
    exposure, else FileNotFoundError; without one the camera's only exposure, and two or more raise.  There is no
    fallback to another exposure, another observation or the shared set."""
    if camera not in CAMERA_CODES:
        raise ValueError(f"unknown camera {camera!r}; known: {tuple(CAMERA_CODES)}")
    d = response_root(root) / str(obsid).strip().zfill(10)
    found = sorted(p.stem for p in d.glob(f"{camera}_*.rmf") if (d / f"{p.stem}.arf").exists())
    if expid is not None and str(expid).strip():
        stem = response_name(camera, str(expid).strip())
        if (d / f"{stem}.rmf").exists() and (d / f"{stem}.arf").exists():
            return d / f"{stem}.rmf", d / f"{stem}.arf"
        raise FileNotFoundError(f"{camera} of observation {obsid}: no response of exposure {str(expid).strip()!r} "
                                f"(EXPIDSTR) in {d}; the set has {found or 'none'}")
    if len(found) == 1:
        return d / f"{found[0]}.rmf", d / f"{found[0]}.arf"
    if not found:
        raise FileNotFoundError(f"no {camera} response of observation {obsid} in {d} (per-observation set: build it "
                                f"with tngxmm instrument obs-responses run)")
    raise FileNotFoundError(f"{camera} of observation {obsid}: exposures {found} in {d} and the header names none "
                            "(no EXPIDSTR)")


def unplanned_observations(entries: list[dict[str, Any]], observations: str | Path | None = None) -> list[str]:
    """The observations of the real-fit list (``observations``, default the package's ``obs/catalogue/observations.csv``)
    without an entry in ``entries``: a per-observation fit of each needs its own responses, donor or not."""
    import pandas as pd

    if observations is None:
        from tngxmm.massfit.build_realfit_targets import observations_csv

        observations = observations_csv()
    want = set(pd.read_csv(observations, dtype={"obsid": str}).obsid.astype(str).str.zfill(10))
    have = {str(e["obsid"]).zfill(10) for e in entries}
    return sorted(want - have)


def count_donors(entries: list[dict[str, Any]], conditions: Any = None) -> int:
    """How many of the planned observations are conditions donors (rows of the package's condition table)."""
    if conditions is None:                  # a count for the log: the table's rows, not its currency
        import pandas as pd

        from tngxmm.instrument.donor_tables import table_path

        conditions = pd.read_csv(table_path("conditions"), dtype={"obsid": str})
    donors = set(conditions.obsid.astype(str).str.zfill(10))
    return sum(1 for e in entries if str(e["obsid"]).zfill(10) in donors)


def missing_responses(donors: Any, root: str | Path | None = None, active_ccds: Any = None) -> dict[str, str]:
    """``{obsid: reason}`` for the donors whose per-observation responses a view generated under
    ``instrument.responses`` 'donor' (or a fit of it under 'per-observation') would not find: no entry in the set, an
    entry whose ``provenance.json`` status is not ok, or a camera of the entry without a response of the exposure the
    mock headers name (EXPIDSTR of :func:`tngxmm.instrument.sas_headers.package_sas_donor`, the same lookup as
    :func:`response_pair`).  ``active_ccds``: the table ``package_sas_donor`` reads (default the package's)."""
    from tngxmm.instrument.active_ccds import read_active_ccds
    from tngxmm.instrument.sas_headers import package_sas_donor

    base = response_root(root)
    table = read_active_ccds() if active_ccds is None else active_ccds
    out: dict[str, str] = {}
    donors = sorted({str(x).strip().zfill(10) for x in donors})
    for d in donors:
        p = base / d / "provenance.json"
        if not p.exists():
            out[d] = "no entry in the set"
            continue
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            out[d] = "provenance.json unreadable"
            continue
        if rec.get("status") != "ok":
            out[d] = f"entry status {rec.get('status')!r}"
            continue
        want = package_sas_donor({"obsid": d}, table)["exposure"]
        gone = []
        for cam in sorted({e["camera"] for e in rec.get("exposures", []) if e.get("camera") in want}):
            try:
                response_pair(base, d, cam, want[cam][1])
            except FileNotFoundError:
                gone.append(f"{cam}_{want[cam][1]}")
        if gone:
            out[d] = f"no response of {', '.join(gone)}"
    return out


def header_obsid(header: Any) -> str:
    """The observation whose response an image header takes: ``DONORID`` (a mock: its conditions donor), else
    ``OBS_ID``."""
    for k in ("DONORID", "OBS_ID"):
        v = str(header.get(k, "") or "").strip()
        if v:
            return v.zfill(10)
    raise KeyError("image header has neither DONORID nor OBS_ID: no per-observation response")


# ---------------------------------------------------------- the generator side ----

class RmfMatrix:
    """An OGIP redistribution matrix as the instrument stage uses threeML's ``OGIPResponse``: ``monte_carlo_energies``
    (the ENERG_LO edges and the last ENERG_HI), ``ebounds`` (the E_MIN edges and the last E_MAX), ``matrix``
    (channels x energies) and ``convolve`` (counts per channel of counts per energy bin).  The matrix is read as
    threeML reads it (F_CHAN counted from its TLMIN, variable-length groups)."""

    def __init__(self, path: str | Path) -> None:
        """Read the matrix and the energy and channel edges of the RMF file ``path``."""
        import numpy as np
        from astropy.io import fits

        with fits.open(path) as f:
            names = [h.name for h in f]
            m = f["MATRIX" if "MATRIX" in names else "SPECRESP MATRIX"]
            d, h = m.data, m.header
            nchan = int(h["DETCHANS"])
            first = int(h.get(f"TLMIN{d.columns.names.index('F_CHAN') + 1}", 1))
            rsp = np.zeros((len(d), nchan), dtype=np.float64)
            for i in range(len(d)):
                ng = int(np.squeeze(d["N_GRP"][i]))
                fch = np.atleast_1d(d["F_CHAN"][i]).astype(int) - first
                nch = np.atleast_1d(d["N_CHAN"][i]).astype(int)
                row = np.atleast_1d(np.asarray(d["MATRIX"][i], dtype=np.float64))
                k = 0
                for j in range(ng):
                    rsp[i, fch[j]:fch[j] + nch[j]] = row[k:k + nch[j]]
                    k += nch[j]
            eb = f["EBOUNDS"].data
            self.monte_carlo_energies = np.append(np.asarray(d["ENERG_LO"], float), float(d["ENERG_HI"][-1]))
            self.ebounds = np.append(np.asarray(eb["E_MIN"], float), float(eb["E_MAX"][-1]))
        self.matrix = rsp.T
        self._matrix_transpose = rsp
        self.rsp_filename = Path(path)

    def convolve(self, fluxes: Any) -> Any:
        """Counts per channel of ``fluxes`` (counts per Monte-Carlo energy bin); non-finite entries count zero."""
        import numpy as np

        x = np.array(fluxes, dtype=np.float64)
        x[~np.isfinite(x)] = 0.0
        return x @ self._matrix_transpose


@functools.lru_cache(maxsize=6)
def _rmf_tables(rmf_path: str) -> tuple:
    """(RmfMatrix, mc_edges, mc_ene, mc_wid, emb, rmf_cdf) of an RMF, as
    :func:`tngxmm.instrument.load_responses.get_response` builds them for the packaged RMF (a small cache, two
    donors' cameras: a pn matrix and its row CDFs are about 140 MB)."""
    import numpy as np

    rmf = RmfMatrix(rmf_path)
    nch = len(rmf.monte_carlo_energies) - 1
    mc_edges = np.asarray(rmf.monte_carlo_energies)
    mc_ene = (mc_edges[:nch] + mc_edges[1:]) / 2.0
    mc_wid = mc_edges[1:] - mc_edges[:nch]
    emb = (np.asarray(rmf.ebounds[1:]) + np.asarray(rmf.ebounds[:-1])) / 2.0
    rmf_cdf = np.cumsum(np.asarray(rmf.matrix).T.copy(), axis=1)
    return rmf, mc_edges, mc_ene, mc_wid, emb, rmf_cdf


def read_arf(arf_path: str | Path) -> tuple[Any, Any]:
    """(bin-centre energies [keV], SPECRESP [cm^2]) of an OGIP ARF."""
    import numpy as np
    from astropy.io import fits

    with fits.open(arf_path) as f:
        d = f["SPECRESP"].data
        return 0.5 * (np.asarray(d["ENERG_LO"], float) + np.asarray(d["ENERG_HI"], float)), \
            np.asarray(d["SPECRESP"], float)


def arf_bin_average(arf_path: str | Path, edges: Any) -> tuple[Any, Any]:
    """(mean SPECRESP [cm^2] over each bin of ``edges`` [keV], whether the ARF's bins cover the bin entirely): the
    ARF taken as constant over each of its own energy bins, as XSPEC folds it, and averaged over the overlap.  Where
    the two grids share their edges this is the ARF's own value bin by bin; elsewhere no edge (O-K at 0.532 keV, the
    Si-K and Au-M edges) is sampled at a point."""
    import numpy as np
    from astropy.io import fits

    with fits.open(arf_path) as f:
        d = f["SPECRESP"].data
        lo = np.asarray(d["ENERG_LO"], float)
        hi = np.asarray(d["ENERG_HI"], float)
        a = np.asarray(d["SPECRESP"], float)
    x = np.asarray(edges, dtype=float)
    node = np.concatenate([lo[:1], hi])                   # the ARF's bins are contiguous in OGIP files
    cum = np.concatenate([[0.0], np.cumsum(a * (hi - lo))])
    c = np.interp(x, node, cum)                           # the integral of the step function up to each edge
    width = np.diff(x)
    mean = np.where(width > 0, np.diff(c) / np.where(width > 0, width, 1.0), 0.0)
    tol = 1e-5                                           # [keV]: single-precision ARF energies
    covered = (x[:-1] >= node[0] - tol) & (x[1:] <= node[-1] + tol)
    return mean, covered


def onaxis_ratio(resp: Any, arf_path: str | Path) -> Any:
    """g(E) on the stage's energy bins ``box_ene``: the ARF of ``arf_path`` averaged over each bin
    (:func:`arf_bin_average`; the stage's bins under ``instrument.arf_model = 'fit-linear'`` are the RMF's, which are
    the bins of the donor's ARF, so the average is its own value) over the stage's on-axis area ``arf_theta[0]``, so
    that ``arf_theta[0] * g`` is that ARF within its energy range; beyond it g keeps its value at the last bin inside
    (the stage's own shape there, scaled), and where the stage has no area g is zero."""
    import numpy as np

    on = np.asarray(resp.arf_theta, dtype=float)[0]
    mean, covered = arf_bin_average(arf_path, np.asarray(resp.box_ene, dtype=float))
    g = np.where(on > 0, mean / np.where(on > 0, on, 1.0), 0.0)
    inside = np.flatnonzero(covered & (on > 0))
    if not len(inside):
        raise ValueError(f"{arf_path}: no bin of the stage's grid inside the ARF's range with area")
    g[:inside[0]] = g[inside[0]]
    g[inside[-1] + 1:] = g[inside[-1]]
    return np.where(on > 0, g, 0.0)


_OBS: dict = {}


def with_observation_response(resp: Any, rmf_path: str | Path, arf_path: str | Path | None = None) -> Any:
    """``resp`` redistributed with the RMF of ``rmf_path`` (Monte-Carlo grid, channel energies, row CDFs and the
    response used to fold the sky background and the AGN) and, with ``arf_path``, its effective area times
    :func:`onaxis_ratio` (the on-axis energy dependence of that ARF, the stage's vignetting kept).  The FWC particle
    spectrum keeps its own channel grid.  Cached by response identity and file (a few entries)."""
    import dataclasses as _dc

    import numpy as np

    key = (id(resp), str(rmf_path), str(arf_path))
    if key in _OBS:
        return _OBS[key][1]
    rmf, mc_edges, mc_ene, mc_wid, emb, rmf_cdf = _rmf_tables(str(rmf_path))
    kw: dict[str, Any] = dict(rmf=rmf, mc_edges=mc_edges, mc_ene=mc_ene, mc_wid=mc_wid, emb=emb, rmf_cdf=rmf_cdf)
    if arf_path is not None:
        kw["arf_theta"] = np.asarray(resp.arf_theta) * onaxis_ratio(resp, arf_path)[None, :]
    if len(_OBS) >= 12:
        _OBS.pop(next(iter(_OBS)))
    out = _dc.replace(resp, **kw)
    _OBS[key] = (resp, out)
    return out


#: values of ``instrument.responses`` (InstrumentOptions.responses)
GENERATOR_RESPONSES = ("packaged", "donor-rmf", "donor")


def apply_observation_responses(resps: dict, mode: str, obsid: str | None, expids: dict | None = None,
                                root: str | Path | None = None) -> tuple[dict, dict]:
    """``(resps, used)``: camera -> response with the conditions donor's responses (``mode`` 'donor-rmf': its RMF;
    'donor': its RMF and the on-axis energy dependence of its ARF; 'packaged': unchanged), and camera -> (rmf, arf)
    file names used."""
    if mode not in GENERATOR_RESPONSES:
        raise ValueError(f"unknown instrument responses {mode!r}; known: {GENERATOR_RESPONSES}")
    if mode == "packaged":
        return resps, {}
    if not obsid:
        raise ValueError(f"instrument responses {mode!r} need the conditions donor's obsid")
    out, used = {}, {}
    for inst, r in resps.items():
        rmf, arf = response_pair(root, str(obsid), inst, (expids or {}).get(inst))
        out[inst] = with_observation_response(r, rmf, arf if mode == "donor" else None)
        used[inst] = (rmf.name, arf.name)
    return out, used


# ------------------------------------------------------------------ the plan ----

def sky_xy(header: Any, ra: float, dec: float) -> tuple[float, float]:
    """Sky pixel (X, Y) of (ra, dec) [deg] in an event list's tangent-plane frame (``REFXCRVL``, ``REFXCRPX``,
    ``REFXCDLT`` and the Y keywords of the EVENTS header)."""
    import math

    ra0, dec0 = math.radians(float(header["REFXCRVL"])), math.radians(float(header["REFYCRVL"]))
    a, d = math.radians(float(ra)), math.radians(float(dec))
    cosc = math.sin(dec0) * math.sin(d) + math.cos(dec0) * math.cos(d) * math.cos(a - ra0)
    xi = math.cos(d) * math.sin(a - ra0) / cosc
    eta = (math.cos(dec0) * math.sin(d) - math.sin(dec0) * math.cos(d) * math.cos(a - ra0)) / cosc
    return (float(header["REFXCRPX"]) + math.degrees(xi) / float(header["REFXCDLT"]),
            float(header["REFYCRPX"]) + math.degrees(eta) / float(header["REFYCDLT"]))


def image_exposures(img_root: str | Path, obsid: str) -> list[dict[str, Any]]:
    """The per-camera exposures of the fit images of ``obsid``: one per ``P<obsid><code><expid>IMAGE_2000.FTZ`` of the
    image tree (camera, expid, filter, submode, date)."""
    from astropy.io import fits

    out = []
    for p in sorted(Path(img_root).glob(f"*/*/P{obsid}*IMAGE_2000.FTZ")):
        m = re.match(rf"^P{obsid}(M1|M2|PN)([SU]\d{{3}})IMAGE_2000\.FTZ$", p.name)
        if not m:
            continue
        h = fits.getheader(p)
        out.append(dict(camera={v: k for k, v in CAMERA_CODES.items()}[m.group(1)], expid=m.group(2),
                        filter=str(h.get("FILTER", "")).strip(), submode=str(h.get("SUBMODE", "")).strip(),
                        date_obs=str(h.get("DATE-OBS", "")).strip(), image=p.name))
    return out


@cites("xmm_ccf_sas")
def plan_entries(centres: Any, img_root: str | Path, pps_root: str | Path, out_root: str | Path,
                 settings: ResponseSettings | None = None, obsids: list[str] | None = None) -> list[dict[str, Any]]:
    """One entry per observation of ``centres`` (the table of the fit centres: ``obsid``, ``ra_peak``, ``dec_peak``,
    ``theta500_arcsec``): its image exposures, each with the PPS event list of the same exposure (``evli`` None when
    the archive has none), the region and the output directory ``<out_root>/<obsid>``."""
    st = settings or ResponseSettings()
    rows = centres if hasattr(centres, "iterrows") else None
    if rows is None:
        import pandas as pd

        rows = pd.DataFrame(centres)
    out = []
    for _, r in rows.iterrows():
        obsid = str(r["obsid"]).zfill(10)
        if obsids and obsid not in obsids:
            continue
        exps = image_exposures(img_root, obsid)
        evl = {}
        pdir = Path(pps_root) / obsid
        if pdir.is_dir():
            for n in sorted(os.listdir(pdir)):
                m = EVLI_PATTERN.match(n)
                if m:
                    evl[(m.group(2), m.group(3))] = str(pdir / n)
        for e in exps:
            e["evli"] = evl.get((CAMERA_CODES[e["camera"]], e["expid"]))
        dates = sorted(e["date_obs"] for e in exps if e["date_obs"])
        th = float(r["theta500_arcsec"])
        out.append(dict(obsid=obsid, date_obs=dates[0] if dates else "", ra=float(r["ra_peak"]),
                        dec=float(r["dec_peak"]),
                        radius_arcsec=min(th, st.region_max_arcsec) if th == th else st.region_max_arcsec,
                        exposures=exps, out=str(Path(out_root) / obsid)))
    return out


# --------------------------------------------------------------- the commands ----

def exposure_commands(entry: dict[str, Any], exp: dict[str, Any], settings: ResponseSettings) -> list[str]:
    """The SAS command lines of one exposure, run in the entry's work directory with ``SAS_CCF`` set: spectrum and
    detector image of the region, rmfgen, arfgen (outputs ``<CAMERA>_<EXPID>.{pi,img,rmf,arf}``, logs in ``logs/``)."""
    from astropy.io import fits

    cam = exp["camera"]
    stem = response_name(cam, exp["expid"])
    if not (entry["ra"] == entry["ra"] and entry["dec"] == entry["dec"]):
        raise ValueError(f"{entry['obsid']}: no fit centre (the region of the response is undefined)")
    h = fits.getheader(exp["evli"], "EVENTS")
    x, y = sky_xy(h, entry["ra"], entry["dec"])
    rad = entry["radius_arcsec"] / SKY_PIXEL_ARCSEC
    sel = f"{SELECTION[cam]}&&((X,Y) IN circle({x:.1f},{y:.1f},{rad:.1f}))"
    binsize, chmax = BINNING[cam]
    lo, hi = settings.detmap_pi
    yes = {True: "yes", False: "no"}
    cmds = [
        f"evselect table={exp['evli']}:EVENTS withspectrumset=yes spectrumset={stem}.pi energycolumn=PI "
        f"spectralbinsize={binsize} withspecranges=yes specchannelmin=0 specchannelmax={chmax} "
        f"expression='{sel}' > logs/{stem}_evselect_spectrum.log 2>&1",
        f"evselect table={exp['evli']}:EVENTS withimageset=yes imageset={stem}.img xcolumn=DETX ycolumn=DETY "
        f"imagebinning=binSize ximagebinsize={settings.detmap_bin} yimagebinsize={settings.detmap_bin} "
        f"expression='{sel}&&(PI in [{lo}:{hi}])' > logs/{stem}_evselect_detmap.log 2>&1",
    ]
    if settings.detmap == "cluster":
        cmds.append(f"rmfgen spectrumset={stem}.pi rmfset={stem}.rmf detmaptype=dataset detmaparray={stem}.img "
                    f"> logs/{stem}_rmfgen.log 2>&1")
    elif settings.detmap == "psf":
        cmds.append(f"rmfgen spectrumset={stem}.pi rmfset={stem}.rmf detmaptype=psf > logs/{stem}_rmfgen.log 2>&1")
    else:
        raise ValueError(f"unknown detmap {settings.detmap!r}; known: ('cluster', 'psf')")
    cmds.append(
        f"arfgen spectrumset={stem}.pi arfset={stem}.arf withrmfset=yes rmfset={stem}.rmf withsourcepos=yes "
        f"sourcecoords=tel sourcex=0 sourcey=0 extendedsource=no detmaptype=flat withbadpixcorr=no "
        f"badpixlocation={exp['evli']} modelee=no modelootcorr=no "
        f"modelcontamination={yes[settings.contamination]} applyxcaladjustment={yes[settings.xcal]} "
        f"applyabsfluxcorr={yes[settings.absflux]} > logs/{stem}_arfgen.log 2>&1")
    return cmds


def entry_command(entry: dict[str, Any], settings: ResponseSettings, workdir: str | Path) -> str:
    """The shell command of one observation (run inside the SAS container in ``workdir``): the CIF of the
    observation date, then every exposure with an event list; a failing exposure does not stop the others."""
    exps = [e for e in entry["exposures"] if e.get("evli")]
    if not entry.get("date_obs"):
        raise ValueError(f"{entry['obsid']}: no DATE-OBS (the CIF cannot be chosen)")
    head = (f"export OMP_NUM_THREADS=1 && cd {workdir} && mkdir -p logs && "
            f"cifbuild withobservationdate=yes observationdate={entry['date_obs']} "
            f"analysisdate={settings.cif_analysis_date()} calindexset=ccf.cif > logs/cifbuild.log 2>&1 && "
            f"export SAS_CCF={workdir}/ccf.cif")
    body = ["sasversion > logs/sasversion.log 2>&1"]
    body += ["( " + " && ".join(exposure_commands(entry, e, settings)) + " )" for e in exps]
    return f"{head} && {{ {'; '.join(body)}; }}"


# ------------------------------------------------------------------ running ----

def _sha256(path: str | Path) -> str:
    from tngxmm.core.hashing import sha256_file

    return sha256_file(path)


def cif_constituents(cif: str | Path) -> dict[str, int]:
    """``{'<SCOPE>_<TYPEID>': issue}`` of the response-relevant constituents of a CIF (:data:`RESPONSE_CCF_TYPES`)."""
    from astropy.io import fits

    out = {}
    with fits.open(cif) as f:
        for r in f["CALINDEX"].data:
            t = str(r["TYPEID"]).strip()
            if t in RESPONSE_CCF_TYPES:
                out[f"{str(r['SCOPE']).strip()}_{t}"] = int(r["ISSUE"])
    return dict(sorted(out.items()))


def sas_version(log_text: str) -> str | None:
    """The SAS release of a SAS log (``[22.1.0-a8f2c2afa-20250304]``), or None."""
    m = re.search(r"\[(\d+\.\d+\.\d+-[0-9a-f]+-\d+)\]", log_text)
    return m.group(1) if m else None


def _run_sas(cmd: str, workdir: Path, settings: ResponseSettings) -> None:
    """Run ``cmd`` inside the SAS container (``sasrun``), pinned to ``settings.cpus``."""
    from tngxmm.instrument.sas_detect import DetectSettings, sasrun_path

    env = dict(os.environ, SAS_WORK=str(workdir), OMP_NUM_THREADS="1")
    argv = [sasrun_path(DetectSettings(sasrun=settings.sasrun)), cmd]
    if settings.cpus:
        argv = ["taskset", "-c", settings.cpus, *argv]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    if proc.returncode != 0 and not (Path(workdir) / "ccf.cif").exists():
        raise RuntimeError(f"SAS failed (exit {proc.returncode}): {proc.stderr[-400:]}")


def finished(entry: dict[str, Any]) -> bool:
    """True when ``<out>/provenance.json`` says ok and every response it lists exists."""
    p = Path(entry["out"]) / "provenance.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text())
    except ValueError:
        return False
    if rec.get("status") != "ok":
        return False
    return all((Path(entry["out"]) / e[k]).exists() for e in rec.get("exposures", []) if e.get("status") == "ok"
               for k in ("rmf", "arf"))


@cites("xmm_ccf_sas")
def run_entry(entry: dict[str, Any], settings: ResponseSettings | None = None, *, work_dir: str | Path | None = None,
              run: Any = None, keep_work: bool = False) -> dict[str, Any]:
    """Build the responses of one observation into ``entry['out']`` and write its ``provenance.json``; returns it.
    ``run(cmd, workdir, settings)`` replaces the SAS call (tests).  Status ``ok`` when every exposure with an event
    list has an RMF and an ARF; an exposure without an event list is recorded as ``no-events`` and leaves the status
    ok (its fit images have no per-observation response)."""
    if Path(entry["out"]).is_symlink():
        # a set that links another set's entries: building here would rewrite the files of that set
        raise ValueError(f"{entry['out']} is a link into another response set; replace it by a directory "
                         "(obs-responses adopt takes over its finished entries) before building")
    _drop_links(Path(entry["out"]))                   # an adopted entry: its files are links into the earlier set
    st = settings or ResponseSettings()
    out = Path(entry["out"])
    out.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    t0 = time.time()
    cmd = entry_command(entry, st, work)
    (run or _run_sas)(cmd, work, st)
    (out / "logs").mkdir(exist_ok=True)
    for lf in (work / "logs").glob("*.log") if (work / "logs").is_dir() else []:
        shutil.copyfile(lf, out / "logs" / lf.name)
    logs = " ".join(p.read_text(errors="replace") for p in (out / "logs").glob("*.log"))
    exps = []
    n_ok = 0
    for e in entry["exposures"]:
        stem = response_name(e["camera"], e["expid"])
        rec = dict(e, rmf=f"{stem}.rmf", arf=f"{stem}.arf")
        if e.get("evli"):
            rec["evli"] = Path(e["evli"]).name
        if not e.get("evli"):
            rec.update(status="no-events")
        elif (work / f"{stem}.rmf").exists() and (work / f"{stem}.arf").exists():
            for k in ("rmf", "arf"):
                shutil.copyfile(work / rec[k], out / rec[k])
                rec[f"{k}_sha256"] = _sha256(out / rec[k])
            rec.update(status="ok", n_counts=_spectrum_counts(work / f"{stem}.pi"))
            n_ok += 1
        else:
            rec.update(status="FAIL")
        exps.append(rec)
    cif = work / "ccf.cif"
    prov = dict(set=RESPONSE_SET_VERSION, obsid=entry["obsid"], date_obs=entry["date_obs"], ra=entry["ra"],
                dec=entry["dec"], radius_arcsec=entry["radius_arcsec"],
                region="circle about the fit centre (obs_centres_guarded_v4 ra_peak, dec_peak), radius theta500 "
                       f"at most {st.region_max_arcsec:g} arcsec",
                centres_sha256=entry.get("centres_sha256"),
                sas_version=sas_version(logs), settings=st.record(),
                ccf=dict(analysisdate=st.cif_analysis_date(), cif_sha256=_sha256(cif) if cif.exists() else None,
                         constituents=cif_constituents(cif) if cif.exists() else {},
                         **_ccf_manifest()),
                exposures=exps, command=cmd, seconds=round(time.time() - t0, 1),
                status="ok" if all(x["status"] in ("ok", "no-events") for x in exps) and n_ok else "FAIL")
    if cif.exists():
        shutil.copyfile(cif, out / "ccf.cif")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return prov


def _ccf_manifest() -> dict[str, str]:
    from tngxmm.instrument.sas_detect import CCF_MANIFEST_SHA256

    return {"ccf_manifest_sha256": CCF_MANIFEST_SHA256}


def _spectrum_counts(pi: Path) -> int | None:
    """Total counts of a spectrum (None when unreadable)."""
    try:
        from astropy.io import fits

        with fits.open(pi) as f:
            return int(f["SPECTRUM"].data["COUNTS"].sum())
    except Exception:                               # noqa: BLE001 - a record field only
        return None


@cites("xmm_ccf_sas")
def run_queue(entries: list[dict[str, Any]], tmp: str | Path, *, cores: str | list[int] | None = None,
              workers: int | None = None, only: str | None = None, limit: int | None = None,
              settings: ResponseSettings | None = None, run: Any = None, attempts: int = 2,
              progress: str | Path | None = None, stagger: float = 2.0) -> dict[str, Any]:
    """Build the unfinished entries, one worker per core (each pinned with ``taskset``), two attempts per entry; one
    JSON line per entry in ``progress`` (default ``<tmp>/progress.jsonl``).  SIGTERM/SIGINT stop taking new entries.
    Returns ``{"entries", "todo", "done", "failed", "wall_s"}``."""
    import signal

    from tngxmm.instrument.sas_queue import select_cores

    base = settings or ResponseSettings()
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    sel = [e for e in entries if not only or e["obsid"] in only.split(",")]
    if limit:
        sel = sel[:limit]
    todo = [e for e in sel if not finished(e)]
    pool = select_cores(cores, workers)
    print(f"{len(sel)} observations, {len(sel) - len(todo)} finished, {len(todo)} to build on {len(pool)} cores",
          flush=True)
    q: queue.Queue = queue.Queue()
    for e in todo:
        q.put(e)
    lock, stop, t0 = threading.Lock(), threading.Event(), time.time()
    count = {"done": 0, "fail": 0}
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
    log = Path(progress) if progress else tmp / "progress.jsonl"

    def worker(core: int, delay: float) -> None:
        time.sleep(delay)
        while not stop.is_set():
            try:
                e = q.get_nowait()
            except queue.Empty:
                return
            rec: dict[str, Any] = dict(obsid=e["obsid"], core=core, host=os.uname().nodename)
            t1 = time.time()
            for attempt in range(1, attempts + 1):
                try:
                    prov = run_entry(e, dataclasses.replace(base, cpus=str(core)),
                                     work_dir=tmp / f"work_{e['obsid']}", run=run)
                    rec.update(status=prov["status"], attempt=attempt,
                               exposures=[f"{x['camera']}_{x['expid']}:{x['status']}" for x in prov["exposures"]])
                    if prov["status"] == "ok":
                        break
                except Exception as exc:                 # noqa: BLE001 - keep the queue going
                    rec.update(status="FAIL", error=str(exc)[-400:] or repr(exc), attempt=attempt)
                finally:
                    shutil.rmtree(tmp / f"work_{e['obsid']}", ignore_errors=True)
            rec["seconds"] = round(time.time() - t1, 1)
            with lock:
                count["done" if rec["status"] == "ok" else "fail"] += 1
                with open(log, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                n = count["done"] + count["fail"]
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] {e['obsid']} {rec['status']} {rec['seconds']:.0f}s elapsed {el / 60:.0f} "
                      f"min, ETA {el / n * (len(todo) - n) / 60:.0f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(c, i * stagger)) for i, c in enumerate(pool)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    print(f"done {count['done']}, failed {count['fail']}, wall {wall / 60:.1f} min", flush=True)
    return dict(entries=len(sel), todo=len(todo), done=count["done"], failed=count["fail"], wall_s=round(wall, 1))


def _drop_links(d: Path) -> None:
    """Remove the links below ``d`` (never what they point to), so that nothing is written through one."""
    if not d.is_dir():
        return
    for f in sorted(d.rglob("*"), reverse=True):
        if f.is_symlink():
            f.unlink()


def _link_tree(src: Path, dst: Path) -> None:
    """``dst`` as a directory tree of links to the files of ``src`` (directories are made, not linked)."""
    dst.mkdir()
    for f in sorted(src.iterdir()):
        if f.is_dir():
            _link_tree(f, dst / f.name)
        elif f.name != "provenance.json":
            (dst / f.name).symlink_to(f)


def adopt_entries(entries: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Take over the finished responses that an earlier centre table planned and the current one plans identically.

    A centre table that gains rows (observations added to the bank) has another sha256, although the rows of the
    observations it already held are the same; their responses need no rebuild.  For every plan entry whose
    ``<out>/provenance.json`` is finished under another ``centres_sha256``: when the set, the region (``ra``, ``dec``,
    ``radius_arcsec``), the exposures (camera, exposure id, event list) and, if given, the ``settings`` of the record
    equal the plan's, the record is rewritten with the plan's ``centres_sha256`` and the earlier one appended to
    ``centres_sha256_built``; an entry that is a link into another set becomes a directory of links to that set's
    files with its own record (the other set is not touched; the directory is prepared beside the link and takes
    its place in one rename, and :func:`run_entry` removes such links before it builds).  An entry that differs in
    any of these, or whose record lacks one, is left as it is and listed, for :func:`run_queue` to rebuild after its
    directory is removed.  Without ``settings`` the settings are not compared.

    Returns ``{"adopted": [...], "current": [...], "differs": [...], "unfinished": [...]}`` (obsids)."""
    import math

    out: dict[str, list[str]] = dict(adopted=[], current=[], differs=[], unfinished=[])
    for e in entries:
        d = Path(e["out"])
        left = d.with_name(d.name + ".adopt")
        if left.exists() and not d.is_symlink():      # an interrupted run got as far as removing the link
            shutil.rmtree(left)
        if not finished(e):
            out["unfinished"].append(e["obsid"])
            continue
        rec = json.loads((d / "provenance.json").read_text())
        if rec.get("centres_sha256") == e.get("centres_sha256"):
            out["current"].append(e["obsid"])
            continue
        same = (rec.get("set") == RESPONSE_SET_VERSION
                and all(rec.get(k) is not None and math.isclose(float(rec[k]), float(e[k]), rel_tol=1e-9,
                                                                abs_tol=1e-9)
                        for k in ("ra", "dec", "radius_arcsec"))
                and [(x["camera"], x["expid"], x.get("evli") or "") for x in rec.get("exposures", [])]
                == [(x["camera"], x["expid"], Path(x["evli"]).name if x.get("evli") else "")
                    for x in e["exposures"]]
                and (settings is None or rec.get("settings") == settings))
        if not same:
            out["differs"].append(e["obsid"])
            continue
        rec["centres_sha256_built"] = list(rec.get("centres_sha256_built", [])) + [rec.get("centres_sha256")]
        rec["centres_sha256"] = e.get("centres_sha256")
        text = json.dumps(rec, indent=1, default=str) + "\n"
        if d.is_symlink():
            new = d.with_name(d.name + ".adopt")
            if new.exists():
                shutil.rmtree(new)                    # links only: left by an interrupted run
            _link_tree(d.resolve(), new)
            (new / "provenance.json").write_text(text)
            d.unlink()
            new.rename(d)
        else:
            tmp = d / "provenance.json.new"
            tmp.write_text(text)
            tmp.replace(d / "provenance.json")
        out["adopted"].append(e["obsid"])
    return out


def write_manifest(root: str | Path) -> Path:
    """``<root>/manifest.csv``: one row per exposure of every ``<root>/<obsid>/provenance.json`` (obsid, camera,
    expid, date, filter, submode, status, files and sha256, SAS version, CIF sha256, the REDIST, QUANTUMEF and
    CONTAMINATION issues of the camera, and the sha256 of the centre table the region was planned from)."""
    import pandas as pd

    root = Path(root)
    rows = []
    for p in sorted(root.glob("*/provenance.json")):
        if p.parent.name.endswith(".adopt"):           # left by an interrupted adopt_entries, not an entry
            continue
        rec = json.loads(p.read_text())
        con = {k: str(v) for k, v in rec.get("ccf", {}).get("constituents", {}).items()}
        for e in rec.get("exposures", []):
            scope = {"PN": "EPN", "MOS1": "EMOS1", "MOS2": "EMOS2"}[e["camera"]]
            rows.append(dict(obsid=rec["obsid"], camera=e["camera"], expid=e["expid"], date_obs=e.get("date_obs"),
                             filter=e.get("filter"), submode=e.get("submode"), status=e["status"],
                             rmf=e["rmf"] if e["status"] == "ok" else "", arf=e["arf"] if e["status"] == "ok" else "",
                             rmf_sha256=e.get("rmf_sha256", ""), arf_sha256=e.get("arf_sha256", ""),
                             n_counts=e.get("n_counts"), sas_version=rec.get("sas_version"),
                             cif_sha256=rec.get("ccf", {}).get("cif_sha256"),
                             redist=con.get(f"{scope}_REDIST"), quantumef=con.get(f"{scope}_QUANTUMEF"),
                             contamination=con.get(f"{scope}_CONTAMINATION"), set=rec.get("set"),
                             centres_sha256=rec.get("centres_sha256")))
    out = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def manifest_digest(root: str | Path) -> str:
    """sha256 of the sorted (obsid, camera, expid, rmf sha256, arf sha256) rows of the manifest (16 hex digits): the
    identity of a response set, recorded by the fits and the instrument stage that use it."""
    import pandas as pd

    m = pd.read_csv(Path(root) / "manifest.csv", dtype=str).fillna("")
    m = m[m.status == "ok"].sort_values(["obsid", "camera", "expid"])
    text = "\n".join(",".join(r) for r in m[["obsid", "camera", "expid", "rmf_sha256", "arf_sha256"]].values)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def set_digest(root: str | Path | None = None) -> str:
    """:func:`manifest_digest` of the set ``root`` (default ``$TNGXMM_OBS_RESP``), cached by the manifest's path and
    modification time: the ``RESPSET`` a mock records and a per-observation fit checks.  FileNotFoundError without a
    manifest (``tngxmm instrument obs-responses manifest`` writes it at the end of the node job)."""
    p = response_root(root) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: the per-observation set has no manifest (tngxmm instrument "
                                "obs-responses manifest)")
    return _set_digest(str(p.parent), p.stat().st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _set_digest(root: str, mtime_ns: int) -> str:
    return manifest_digest(root)


def check_set(root: str | Path | None = None, centres: str | Path | None = None) -> str:
    """The digest of the set (:func:`set_digest`; FileNotFoundError without a manifest) after checking that every
    entry was planned from the current fit-centre table (``centres``, default the data package's
    ``obs_centres_guarded_v4.csv``): the manifest's ``centres_sha256`` must be that table's sha256, else ValueError.
    The rule: a set whose cluster regions came from another centre table is rebuilt before it is used, since a mock
    and a real fit would otherwise take responses of regions the fit does not use.  Called where the set is used:
    the planners (``donor_tables.missing_donors``, the closure grid's coverage) and the per-observation fit
    (``fit_hydrostatic.response_record``)."""
    import pandas as pd

    from tngxmm.core.hashing import sha256_file

    digest = set_digest(root)
    if centres is None:
        from tngxmm.photometry.build_obs_centres_guarded import default_table_path

        centres = default_table_path()
    want = sha256_file(Path(centres))
    m = pd.read_csv(response_root(root) / "manifest.csv", dtype=str).fillna("")
    got = sorted(set(m.get("centres_sha256", pd.Series([""] * len(m)))))
    if got != [want]:
        raise ValueError(f"the per-observation set {response_root(root)} was planned from the centre table(s) "
                         f"{got} (sha256), not the current {Path(centres).name} ({want}): rebuild the set "
                         "(tngxmm instrument obs-responses plan | run | manifest)")
    return digest


__cites__ = collect_cites(__name__)
