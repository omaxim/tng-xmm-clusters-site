"""Per-camera EPIC responses for the fast instrument stage.

``get_response`` builds, once per process, everything per-camera that is
identical across views: the ARF on the off-axis angle grid, the detector mask,
the RMF redistribution tables, the FWC particle-background spectrum, the King
PSF parameters and (pn) the readout-column geometry.  Three options shape the
response:

* ``filter``: the EPIC optical filter whose CCF transmission enters the ARF.
  Upstream xmm_simulator 0.2.0 hard-codes ``FILTER-MEDIUM``
  (``simulator.py:127``); ``None`` keeps that.  ``'Thin1'``, ``'Thin2'``,
  ``'Medium'``, ``'Thick'`` replace the transmission curve with the named
  extension of the camera's ``E*_FILTERTRANSX`` CCF before the ARF is computed
  (no fork needed: the simulator object only carries the curve).
* ``mos2_mirror``: upstream gives MOS2 the MOS1 mirror file ``XRT1_XAREAEF``
  (``utils.py:205-210``); ``'XRT2'`` loads MOS2's own ``XRT2_XAREAEF`` CCF.
* ``pattern``: ``None`` (no PATTERN selection loss) or a
  :class:`tngxmm.instrument.apply_pattern_loss.PatternEfficiency` table that
  multiplies the X-ray ARF (cluster, sky background, AGN; the FWC particle
  background is already a screened event spectrum and is not thinned).
* ``ccf_selection``: which calibration file of each type the response reads.
  ``'highest-issue'`` (the default) keeps upstream's scan (the highest number in
  the file name, ``utils.get_ccf_file_names``); ``'valid-at-date'`` takes the
  constituent SAS cifbuild selects for ``obs_date``
  (:mod:`tngxmm.instrument.ccf_select`; with the site CCF directory this changes
  the MOS quantum efficiency only).

References
----------
Eckert, D., xmm_simulator 0.2.0 (github.com/domeckert/xmm_simulator, commit
9fdedb4e): CCF file selection, ``calc_arf``, the detector masks, RMFs and FWC
spectra used here.
XMM-Newton SOC, Current Calibration Files (XRT*_XAREAEF, E*_QUANTUMEF,
E*_FILTERTRANSX, XRT*_XPSF); Gabriel et al. 2004 (SAS).
Vianello et al. 2015 (threeML ``OGIPResponse``).
Jansen et al. 2001; Strueder et al. 2001; Turner et al. 2001 (XMM-Newton, EPIC).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

#: camera loop order of the stage (the order of the random draws depends on it)
INSTRUMENTS: tuple[str, ...] = ("MOS1", "MOS2", "PN")
_INSTRUMENTS = list(INSTRUMENTS)

#: PN readout/CCD-long-edge tilt from detector Y [deg] (CampaignSpec ``instrument.pn_readout_tilt_deg``): the angle
#: of the pn DETY axis on the packaged masks in SAS's DET -> sky solution of the observation behind them,
#: atan2(-0.41364, 0.91044) (``donor_background.DET_TO_MASK``, tools/det_to_mask.py); line fits to the CCD edges
#: of the mask gave 24.44 +- 0.02 deg
PN_READOUT_TILT_DEG = -24.434

#: rules for the effective area beyond 15' off axis (``get_response(vignetting_beyond=...)``)
VIGNETTING_BEYOND = ("zero", "ccf-edge")
#: models of the off-axis ARF grid (``get_response(arf_model=...)``, ``instrument.arf_model``)
ARF_MODELS = ("xmm_simulator", "fit-linear")

#: filter names accepted by ``get_response(filter=...)`` -> CCF extension
FILTER_EXTENSION: dict[str, str] = {
    "Thin1": "FILTER-THIN1", "Thin2": "FILTER-THIN2", "Medium": "FILTER-MEDIUM",
    "Thick": "FILTER-THICK",
}
#: the filter upstream xmm_simulator 0.2.0 always simulates
UPSTREAM_FILTER = "Medium"

# module-level caches: {key: value}; keys carry every response-shaping option
_RESPONSES: dict = {}


@dataclass
class CameraResponse:
    """Everything per-camera that is identical across all halos/views."""
    inst        : str
    sim         : object          # XMMSimulator (CCF curves only)
    pixsize     : float           # mask pixel [arcmin]
    npix        : int             # 900
    cx          : float
    cy          : float
    mask        : Any             # (900, 900) uint8
    active_x    : Any             # active-pixel coordinates (mask > 0)
    active_y    : Any
    thetas      : Any             # (200,) off-axis grid [arcmin]
    arf_theta   : Any             # (200, nene) float64 [cm^2]
    box_ene     : Any             # (nene+1,) coarse energy edges [keV]
    box_ene_mean: Any             # (nene,)
    box_ene_wid : float
    theta_idx_pix: Any            # (900, 900) int16: nearest theta bin per pixel
    mc_edges    : Any             # RMF MC energy edges
    mc_ene      : Any             # RMF MC bin centres
    mc_wid      : Any             # RMF MC bin widths
    emb         : Any             # RMF output channel centre energies
    rmf_cdf     : Any             # (n_mc, n_chan) float64 row-wise CDF
    rmf         : object          # threeML OGIPResponse (for convolve)
    fwc_spec    : Any             # FWC QPB spectrum x active-pixel (mask) area [arcmin^2]
    eb_min      : Any             # EBOUNDS E_MIN
    eb_max      : Any             # EBOUNDS E_MAX
    king_r0     : float           # [arcmin]
    king_alpha  : float
    # PN only: per-column active-pixel extents of each detector half
    # (for out-of-time event redistribution along readout columns)
    pn_top_lo   : Any = None
    pn_top_hi   : Any = None
    pn_bot_lo   : Any = None
    pn_bot_hi   : Any = None
    # PN only: measured CCD-row boundary along the readout axis, relative to
    # the detector centre [mask px] (it is at -2.17', NOT 0)
    pn_half_s   : float = 0.0
    # provenance: the filter and mirror file simulated (defaults: Medium and upstream's mirror choice)
    filter      : str = UPSTREAM_FILTER
    area_file   : str = ""
    pattern_eff : Any = None      # (nene,) applied PATTERN efficiency or None
    # readout window imaged (readout_windows.windowed_response): None = full frame
    window      : str | None = None
    # CCDs imaged (active_ccds.ccd_masked_response): None = the packaged mask
    ccds        : tuple | None = None
    # detector footprint (donor_footprint.footprint_response): None = the 0/1 mask; else the
    # (900, 900) float32 good-silicon fraction that multiplies the EXPMAPs and thins the events
    efficiency  : Any = None
    footprint   : str | None = None
    # vignetting centre relative to the mask centre [mask px] (simulate_instrument.with_vignetting_axis):
    # theta_idx_pix and every off-axis angle of the stage are measured from (cx + x, cy + y)
    vig_axis    : tuple = (0.0, 0.0)
    # CCF files of the response under ccf_selection 'valid-at-date' (((type, file), ...)); None = upstream's scan
    ccf_files   : tuple | None = None
    # MOS only, instrument.mos_rga_vignetting 'ccf' (simulate_instrument.with_rga_modulation): None, or
    # (a(E) [1/arcmin] on box_ene_mean, phi0 [deg, mask frame]): throughput 1 + a r cos(phi - phi0) about the
    # vignetting centre (instrument.rga_obscuration)
    rga         : Any = None


@cites("atomdb", "xmm_simulator")
def ensure_atomdb() -> str:
    """Fail fast unless ``$ATOMDB`` names an existing directory.

    pyatomdb (imported by xmm_simulator and threeML's APEC) asks interactively
    "Set up AtomDB files now?" when ``ATOMDB`` is unset and blocks forever in a
    batch job.  Call before the first xmm_simulator import."""
    import os

    path = os.environ.get("ATOMDB", "")
    if not path or not os.path.isdir(path):
        raise RuntimeError("ATOMDB is unset or not a directory: source "
                           "environment/runtime_env.sh sim <id> before importing xmm_simulator "
                           "(pyatomdb would otherwise block on an interactive prompt)")
    return path


def _response_key(inst: str, ccf_dir: str | Path, filter: str | None,
                  mos2_mirror: str, pattern: Any, readout_tilt_deg: float,
                  energy_grid_max_kev: float | None = None,
                  vignetting_beyond: str = "zero", arf_model: str = "xmm_simulator") -> tuple:
    key: tuple = (inst, str(ccf_dir))
    extra = (filter, mos2_mirror if inst == "MOS2" else "upstream",
             None if pattern is None else pattern.key, readout_tilt_deg, energy_grid_max_kev)
    if vignetting_beyond != "zero":
        extra = extra + (vignetting_beyond,)
    if arf_model != "xmm_simulator":
        extra = extra + (arf_model,)
    return key + extra


@cites("xmm_ccf_sas", "xmm_simulator")
def set_filter_transmission(sim: Any, ccf_str: str, filter: str) -> None:
    """Replace the simulator's filter curve with ``filter`` from its own
    ``E*_FILTERTRANSX`` CCF (the file upstream already selected)."""
    from astropy.io import fits

    ext = FILTER_EXTENSION[filter]
    with fits.open(ccf_str + sim.filter_file) as f:
        sim.ene_filter = f["EBINS"].data["ENERGY"] / 1e3
        sim.filter = f[ext].data["TRANSMISSION"]


def _load_area_file(sim: Any, ccf_str: str, name: str) -> None:
    """Read the mirror effective-area CCF ``name`` into the simulator as upstream ``XMMSimulator.__init__`` reads its
    own choice (on-axis area, vignetting, CORRAREA, D_THETA)."""
    import numpy as np
    from astropy.io import fits

    sim.area_file = name
    with fits.open(ccf_str + name) as f:
        sim.onaxis = f[1].data
        sim.vignetting = f[2].data
        sim.corrarea = f[3].data
        sim.dtheta = f[2].header["D_THETA"] * 60.0
    sim.nrad = len(sim.vignetting[0][1])
    sim.rads_vignetting = np.arange(0.0, 16.0, sim.dtheta)


@cites("xmm_ccf_sas", "xmm_simulator")
def set_mirror_area_file(sim: Any, ccf_str: str, mirror: str) -> None:
    """Load the mirror effective-area CCF ``<mirror>_XAREAEF_<n>`` (highest
    issue) into the simulator, exactly as upstream ``XMMSimulator.__init__``
    reads its own choice (on-axis area, vignetting, CORRAREA, D_THETA)."""
    import os

    best, nbest = None, 0
    for tf in os.listdir(ccf_str):
        if f"{mirror}_XAREAEF" in tf:
            tn = int(tf.split("_")[2].split(".")[0])
            if tn > nbest:
                best, nbest = tf, tn
    if best is None:
        raise FileNotFoundError(f"no {mirror}_XAREAEF CCF in {ccf_str}")
    _load_area_file(sim, ccf_str, best)


@cites("xmm_ccf_sas", "xmm_simulator")
def use_ccf_files(sim: Any, ccf_str: str, files: dict[str, str]) -> None:
    """Point the simulator at the CCF files ``files`` (:func:`tngxmm.instrument.ccf_select.camera_files`: ``qe``,
    ``filter``, ``area``, ``psf``, ``fwc``) and re-read each curve whose file differs from the one upstream's scan
    loaded, the way ``XMMSimulator.__init__`` reads it (QE_TOTAL; the FILTER-MEDIUM transmission, which
    :func:`set_filter_transmission` then replaces by the donor's filter from the same file; the mirror tables).  The
    PSF file is read later by name; the FWC file only feeds upstream's particle spectrum of the packaged response
    (``read_qpb_spectrum``): the donor background, the QPB maps and the fit's QPB template pick their FWC set by
    their own rule (``donor_background``, fit-fix ``qpb_fwc``), the highest issue, which equals cifbuild's choice
    in the site CCF directory."""
    from astropy.io import fits

    if files["qe"] != sim.qe_file:
        sim.qe_file = files["qe"]
        with fits.open(ccf_str + sim.qe_file) as f:
            sim.ene_qeff = f["EBINS_FRACTION"].data["ENERGY"] / 1e3
            sim.qeff = f["QE_TOTAL"].data["QE_TOTAL"]
    if files["filter"] != sim.filter_file:
        sim.filter_file = files["filter"]
        with fits.open(ccf_str + sim.filter_file) as f:
            sim.ene_filter = f["EBINS"].data["ENERGY"] / 1e3
            sim.filter = f["FILTER-MEDIUM"].data["TRANSMISSION"]
    if files["area"] != sim.area_file:
        _load_area_file(sim, ccf_str, files["area"])
    sim.psf_file = files["psf"]
    if "fwc" in files:
        sim.fwc_file = files["fwc"]


def ccf_choice(inst: str, ccf_dir: str | Path, mos2_mirror: str = "upstream",
               ccf_selection: str | None = None, obs_date: str | None = None) -> dict[str, str] | None:
    """The CCF files of a camera's response under ``ccf_selection`` (:func:`tngxmm.instrument.ccf_select.camera_files`
    for ``obs_date``; the area of MOS2 from the mirror ``mos2_mirror`` picks, XRT1 upstream), or None for the default
    rule (upstream's scan, unchanged)."""
    from tngxmm.instrument import ccf_select as CS

    if CS.rule_of(ccf_selection) == "highest-issue":
        return None
    if not obs_date:
        raise ValueError(f"ccf_selection {ccf_selection!r} needs the observation date of the conditions donor "
                         "(context['_sas_donor']['date_obs'])")
    mirror = None
    if inst == "MOS2":
        mirror = "XRT1" if mos2_mirror in ("upstream", "XRT1") else mos2_mirror
    return CS.camera_files(ccf_dir, inst, obs_date, ccf_selection, mirror=mirror)


@cites("xmm_ccf_sas", "sas_cifbuild", "xmm_simulator", "threeml")
def get_response(inst: str, ccf_dir: str | Path, *, filter: str | None = None,
                 mos2_mirror: str = "upstream", pattern: Any = None,
                 readout_tilt_deg: float = PN_READOUT_TILT_DEG,
                 energy_grid_max_kev: float | None = None,
                 vignetting_beyond: str = "zero", ccf_selection: str | None = None,
                 obs_date: str | None = None, arf_model: str = "xmm_simulator") -> CameraResponse:
    """Build (once per process) the cached response for one camera.

    ``energy_grid_max_kev``: upper edge of the coarse ARF energy grid; None =
    upstream's ``linspace(0.1, 10.01, 496)`` (photons above 10.01 keV then use
    the last bin's ARF); e.g. 15.0 gives 0.02 keV bins to 15 keV (pairs with
    the 20 keV photon cut of the sim stage).

    ``vignetting_beyond``: the effective area beyond 15' off axis.  ``'zero'`` (the default,
    upstream's ARF_Box): none, so the pn and MOS corners image only the particle background
    and have no exposure.  ``'ccf-edge'``: the ARF at the last off-axis angle of the CCF
    vignetting table (15.6' for D_THETA = 0.01 deg), held constant to the end of the grid:
    the PPS exposure maps of the donors are flat at that value out to 18' (pn) and their
    count rates there follow them (:data:`VIGNETTING_BEYOND`).

    ``ccf_selection`` / ``obs_date``: see the module docstring (:func:`ccf_choice`); the cache key carries the files
    chosen, not the date, so donors of one validity interval share a response.

    ``arf_model`` (:data:`ARF_MODELS`): ``'xmm_simulator'`` (upstream ``calc_arf`` sampled at the centres of the
    20 eV grid: the 0.52-0.54 keV bin takes the area below the O-K edge for all of it, the MOS 0.5-1 keV area 1.1-1.2
    per cent high) or ``'fit-linear'`` (the fit's own effective area on the RMF's energy bins,
    :func:`tngxmm.instrument.effective_area.effective_area_theta`: the on-axis ARF of ``massfit.build_responses``
    times the CCF vignetting, linear in angle and energy)."""
    import numpy as np

    if vignetting_beyond not in VIGNETTING_BEYOND:
        raise ValueError(f"unknown vignetting_beyond {vignetting_beyond!r}; known: {VIGNETTING_BEYOND}")
    if arf_model not in ARF_MODELS:
        raise ValueError(f"unknown arf_model {arf_model!r}; known: {ARF_MODELS}")
    if arf_model == "fit-linear" and inst == "MOS2" and mos2_mirror in ("upstream", "XRT1"):
        raise ValueError("arf_model 'fit-linear' is the fit's effective area, which reads MOS2's own mirror file "
                         "(mos2_mirror 'XRT2')")
    chosen = ccf_choice(inst, ccf_dir, mos2_mirror, ccf_selection, obs_date)
    key = _response_key(inst, ccf_dir, filter, mos2_mirror, pattern, readout_tilt_deg,
                        energy_grid_max_kev, vignetting_beyond, arf_model)
    if chosen is not None:
        key = key + (("ccf",) + tuple(sorted(chosen.items())),)
    if key in _RESPONSES:
        return _RESPONSES[key]

    ensure_atomdb()
    from astropy.io import fits
    from threeML.utils.OGIP.response import OGIPResponse
    from xmm_simulator import XMMSimulator
    from xmm_simulator.background import read_qpb_spectrum
    from xmm_simulator.utils import calc_arf, get_data_file_path

    ccf_str = str(ccf_dir).rstrip('/') + '/'
    if arf_model == "fit-linear":
        # the fit's energy grid: the camera RMF's own bins (where XSPEC reads the ARF), and one bin of zero area
        # above them (XSPEC folds nothing beyond the RMF)
        from tngxmm.instrument.effective_area import rmf_energy_edges

        sim = XMMSimulator(ccf_str, inst, 1.0, evtfile_input='x',
                           box_ene=rmf_energy_edges(inst, emin_kev=0.1,
                                                    emax_kev=max(float(energy_grid_max_kev or 0.0), 100.0)))
    elif energy_grid_max_kev is None:
        sim = XMMSimulator(ccf_str, inst, 1.0, evtfile_input='x')
    else:
        nbin = int(round((float(energy_grid_max_kev) - 0.1) / 0.02))
        sim = XMMSimulator(ccf_str, inst, 1.0, evtfile_input='x',
                           box_ene=np.linspace(0.1, float(energy_grid_max_kev), nbin + 1))
    if chosen is not None:
        use_ccf_files(sim, ccf_str, chosen)           # the area of chosen is already the mirror mos2_mirror picks
    if filter is not None:
        set_filter_transmission(sim, ccf_str, filter)
    if inst == "MOS2" and mos2_mirror not in ("upstream", "XRT1") and chosen is None:
        set_mirror_area_file(sim, ccf_str, mos2_mirror)

    # detector mask
    with fits.open(get_data_file_path(f'imgs/{inst}_mask.fits.gz')) as f:
        mask    = f[1].data.astype(np.uint8)
        pixsize = f[1].header['CDELT2'] * 60.0     # arcmin
    npix = mask.shape[0]
    cx = cy = npix / 2.0
    ay, ax = np.nonzero(mask)

    # theta table: exactly ARF_Box's grid (200 points to 22', zero beyond 15')
    nthetas = 200
    thetas  = np.linspace(0.0, 22.0, nthetas)
    ene_lo, ene_hi = sim.box_ene[:-1], sim.box_ene[1:]
    arf_theta = np.zeros((nthetas, len(ene_lo)), dtype=np.float64)
    th_edge = float(np.max(sim.rads_vignetting))            # last angle of the CCF vignetting table
    if arf_model == "fit-linear":
        from tngxmm.instrument.effective_area import effective_area_theta

        arf_theta[:] = effective_area_theta(inst, UPSTREAM_FILTER if filter is None else filter, ene_lo, ene_hi,
                                            thetas, ccf_str, beyond=vignetting_beyond, date=obs_date,
                                            rule=ccf_selection or "highest-issue",
                                            area_file=None if chosen is None else chosen["area"])
        arf_theta[:, -1] = 0.0                                  # the bin above the RMF
    for i, th in enumerate(thetas if arf_model == "xmm_simulator" else ()):
        if th <= 15.0:
            arf_theta[i] = calc_arf(theta=th, ebound_lo=ene_lo,
                                    ebound_hi=ene_hi, xmmsim=sim)
        elif vignetting_beyond == "ccf-edge":
            arf_theta[i] = calc_arf(theta=min(float(th), th_edge), ebound_lo=ene_lo,
                                    ebound_hi=ene_hi, xmmsim=sim)
    pattern_eff = None
    if pattern is not None:
        pattern_eff = pattern.on_grid(inst, np.asarray(sim.box_ene_mean))
        arf_theta = arf_theta * pattern_eff[None, :]

    # nearest theta bin per mask pixel (0.11' step; vignetting slope ~3%/',
    # so nearest-bin vs linear interp differs by < 0.2%)
    yy, xx = np.indices(mask.shape)
    th_pix = np.hypot(xx - cx, yy - cy) * pixsize
    theta_idx_pix = np.clip(np.round(th_pix / (thetas[1] - thetas[0])), 0,
                            nthetas - 1).astype(np.int16)

    # RMF: per-MC-bin channel CDFs for vectorized redistribution sampling.
    # (The upstream sample_rmf normalised columns in place per photon; here the
    # matrix is copied and left untouched.)
    rmf = OGIPResponse(rsp_file=get_data_file_path(f'rmfs/{inst}.rmf'))
    nch = len(rmf.monte_carlo_energies) - 1
    mc_edges = np.asarray(rmf.monte_carlo_energies)
    mc_ene   = (mc_edges[:nch] + mc_edges[1:]) / 2.0
    mc_wid   = mc_edges[1:] - mc_edges[:nch]
    emb      = (np.asarray(rmf.ebounds[1:]) + np.asarray(rmf.ebounds[:-1])) / 2.0
    rmf_cdf  = np.cumsum(np.asarray(rmf.matrix).T.copy(), axis=1)

    # FWC (QPB) spectrum, already scaled to the ACTIVE-pixel area:
    # upstream = ExtractFWC (x tot_area) then gen_qpb_evt (x area_mask/area_tot)
    # = read_qpb_spectrum x area_mask.
    area_mask = float(mask.sum()) * pixsize ** 2
    fwc_spec  = read_qpb_spectrum(sim) * area_mask

    # King PSF parameters (on-axis row), the same file the PSF patch reads
    with fits.open(ccf_str + sim.psf_file) as f:
        pars = f['KING_PARAMS'].data['PARAMS'][0]
    king_r0, king_alpha = pars[0] / 60.0, pars[1]

    # PN: per-readout-column active extents of each detector half, used to
    # place out-of-time events, computed in the READOUT-ALIGNED frame (the
    # mask rotated by -tilt makes the CCD columns vertical).  The two halves
    # clock charge toward OPPOSITE outer edges.
    pn_ext = {}
    pn_half_s = 0.0
    if inst == 'PN':
        from scipy.ndimage import rotate as _rot
        act = _rot(mask == 1, -readout_tilt_deg, reshape=False, order=0,
                   cval=0) > 0
        # MEASURE the boundary between the two CCD rows: the single internal
        # dead lane crossing the readout direction.  It sits 2.17' BELOW the
        # detector centre (silicon runs -15.04' to +10.71').
        _rowfrac = act[:, int(cx - 150):int(cx + 150)].mean(axis=1)
        _inside  = np.flatnonzero(_rowfrac > 0.5)
        _dead    = [y for y in range(_inside.min(), _inside.max() + 1)
                    if _rowfrac[y] < 0.30]
        yb = float(np.mean(_dead)) if _dead else float(cy)
        pn_half_s = yb - cy                       # [mask px] relative to centre
        yy2 = np.arange(npix)
        for half, m in (('top', act & (yy2[:, None] >= yb)),
                        ('bot', act & (yy2[:, None] < yb))):
            has = m.any(axis=0)
            lo = np.where(has, np.argmax(m, axis=0), -1)
            hi = np.where(has, npix - 1 - np.argmax(m[::-1], axis=0), -1)
            pn_ext[f'{half}_lo'] = lo.astype(np.int32)
            pn_ext[f'{half}_hi'] = hi.astype(np.int32)

    resp = CameraResponse(
        inst=inst, sim=sim, pixsize=pixsize, npix=npix, cx=cx, cy=cy,
        mask=mask, active_x=ax, active_y=ay,
        thetas=thetas, arf_theta=arf_theta,
        box_ene=np.asarray(sim.box_ene), box_ene_mean=np.asarray(sim.box_ene_mean),
        box_ene_wid=(np.asarray(sim.box_ene_width, dtype=np.float64) if arf_model == "fit-linear"
                     else float(np.atleast_1d(sim.box_ene_width)[0])),
        theta_idx_pix=theta_idx_pix,
        mc_edges=mc_edges, mc_ene=mc_ene, mc_wid=mc_wid, emb=emb,
        rmf_cdf=rmf_cdf, rmf=rmf,
        fwc_spec=np.asarray(fwc_spec),
        eb_min=np.asarray(sim.ebounds['E_MIN']), eb_max=np.asarray(sim.ebounds['E_MAX']),
        king_r0=king_r0, king_alpha=king_alpha,
        pn_top_lo=pn_ext.get('top_lo'), pn_top_hi=pn_ext.get('top_hi'),
        pn_bot_lo=pn_ext.get('bot_lo'), pn_bot_hi=pn_ext.get('bot_hi'),
        pn_half_s=pn_half_s,
        filter=UPSTREAM_FILTER if filter is None else filter,
        area_file=str(sim.area_file), pattern_eff=pattern_eff,
        ccf_files=None if chosen is None else tuple(sorted(chosen.items())),
    )
    _RESPONSES[key] = resp
    return resp


@cites("xmm_ccf_sas", "xmm_simulator")
def warm_responses(ccf_dir: str | Path, cameras: tuple[str, ...] = INSTRUMENTS,
                   **response_kw: Any) -> None:
    """Pre-build all camera responses (call BEFORE forking workers so the
    tables are shared copy-on-write)."""
    for inst in cameras:
        get_response(inst, ccf_dir, **response_kw)


@cites("xmm_ccf_sas", "xmm_simulator")
def load_responses(ccf_dir: str | Path, cameras: tuple[str, ...] = INSTRUMENTS,
                   filters: dict[str, str | None] | None = None,
                   **response_kw: Any) -> dict[str, CameraResponse]:
    """Responses of ``cameras`` keyed by camera; ``filters`` maps camera to
    filter (``None``/absent = upstream Medium)."""
    filters = filters or {}
    return {c: get_response(c, ccf_dir, filter=filters.get(c), **response_kw)
            for c in cameras}


__cites__ = collect_cites(__name__)
