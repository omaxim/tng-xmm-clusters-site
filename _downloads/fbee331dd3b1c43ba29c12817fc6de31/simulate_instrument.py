"""Fast EPIC instrument stage: camera partition, PSF, masks, RMF, sky and
particle backgrounds, AGN, pn out-of-time events, event-level roll and
rebinning, per-camera products, merged image and coverage map.

Why a fast stage: the upstream xmm_simulator path spends 10-20 min per camera
on machinery whose outputs are statistically identical to closed-form draws
(494 FFT PSF convolutions of a flat sky background; a per-pixel loop over
~2e5 active pixels although the ARF cube is radially symmetric; a per-photon
double loop for the cluster).  The replacements are exact applications of
Poisson thinning and splitting.  Camera partitioning: each parent photon goes
to at most ONE camera by a single categorical draw with
p_i = ARF_i(theta, E)/A_gen x t_i/t_gen, so the per-camera streams are
independent Poisson processes with the single-camera rates (needs
A_gen >= sum_i max ARF_i, hence AREA_CM2 = 4000).  Event-level merge: the roll
and the 2.5"->4" rebin act on event coordinates, so the merged mock is integer
Poisson counts like the real mosaics.

The stage follows the donor observation: the filter, the SUBMODE, the active
CCDs and the readout window of every camera, and the cluster placement (the
donor's core offset) come from the donor package.  The ARF and sky-background
energy grid extends beyond upstream's 10.01 keV (``arf_energy_max_kev``); the
pn effective area carries the PATTERN efficiency; exposure maps are built by
inverse mapping with a count-weighted vignetting energy; the headers are
written from scratch (optical axis, filter, PSF model, vignetting energy,
provenance); the AGN list is a FITS table; the gzip container is
deterministic.

References
----------
Eckert, D., xmm_simulator 0.2.0 (point_sources, event_file, imaging,
background: the models and calibration products this stage re-implements).
Moretti et al. 2003 (logN-logS and unresolved CXB of the sky model).
XMM-Newton SOC CCF; Jansen et al. 2001; Strueder et al. 2001; Turner et al. 2001.
Vianello et al. 2015 (threeML spectral models).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites
from tngxmm.instrument.load_responses import (
    _INSTRUMENTS,
    INSTRUMENTS,
    PN_READOUT_TILT_DEG,
    CameraResponse,
    get_response,
)

TARGET_PIX_ARCSEC = 4.0
MERGED_NPIX = 648
#: measured inter-camera boresight offsets [arcmin, detector frame]
#: (:mod:`tngxmm.instrument.calibrate_boresight`).  The campaign does not apply them:
#: ``CampaignSpec.instrument.camera_offset_arcmin`` is (0, 0) for every camera.
CAMERA_OFFSET_ARCMIN = {
    'MOS1': (+0.66, +0.23),
    'MOS2': (+1.34, +0.85),
    'PN':   (-1.99, -1.08),
}
#: pn out-of-time fraction, Full Frame (4.6 ms / 73.4 ms, Users' Handbook)
PN_OOT_FRAC = 0.063
#: the camera-assignment probability sum may exceed 1 by this much (rounding) before the stage refuses the view
P_CUM_TOL = 1e-9
#: pn CCD column length on the sky: 200 rows x 150 um = 200 x 4.1" [arcmin]
#: (Strueder et al. 2001; Users' Handbook section 3.3)
PN_COLUMN_ARCMIN = 200 * 4.1 / 60.0
#: pn out-of-time model (header OOTMODEL): every X-ray event is out-of-time with
#: probability oot_frac and lands uniformly anywhere on its own CCD column
PN_OOT_MODEL = "full-column"
#: PPS pn band 1 (0.2-0.5 keV) keeps single-pixel events (PATTERN == 0) with RAWY above
#: PN_BAND1_RAWY_MIN, i.e. drops the rows next to the readout node
PN_BAND1_RAWY_MIN = 12
#: pn pixel row pitch on the sky [arcsec] (150 um; Strueder et al. 2001; Handbook section 3.3)
PN_ROW_ARCSEC = 4.1
#: rules of ``pn_edge_rows`` (the pn rows next to the readout node): 'drop' removes the events of the 12 rows inside
#: the footprint's edge (pn_rawy counts from that edge); 'footprint' leaves the cut to the footprint, which already
#: ends at the readout rows (the PPS exposure maps are zero at RAWY <= 12 and the cleaned event lists empty there)
PN_EDGE_ROWS = ("drop", "footprint")

_SKYBKG_MODELS: dict = {}
_FOV_MAPS: dict = {}


# ── options ──────────────────────────────────────────────────────────────────

#: PSF model of the fast stage (header PSFMODEL): the on-axis King profile of
#: the camera's XRT*_XPSF CCF, independent of energy and off-axis angle
PSF_MODEL = "KING-ONAXIS"
#: filters a donor camera may carry (CCF E*_FILTERTRANSX extensions)
DONOR_FILTERS = ("Thin1", "Thin2", "Medium", "Thick")


#: rules of ``vig_axis``: where each camera's vignetting is centred in the mask frame: 'pointing', or 'ccf' (the
#: telescope axis of the XMM calibration files, :func:`tngxmm.instrument.calibration_axis.telescope_axis_arcmin`)
VIG_AXIS_RULES = ("pointing", "ccf")

#: the value of each switch the stage implements; any other value is refused
_SUPPORTED = {
    "filter_mode": "per-donor",
    "submode_mode": "per-donor",
    "pattern_mode": "ccf",
    "header_style": "v4",
    "obsid_style": "v4",
    "expmap_method": "inverse-map",
    "placement": "donor-offset",
    "pn_oot_model": PN_OOT_MODEL,
    "out_of_fov_events": "drop",
    "readout_windows": "donor",
    "active_ccds": "donor",
    "product_bands": "pps-1-5",
    "pn_band1_events": "single-pixel",
    "sas_headers": "on",
    "agn_norm": "moretti03",
}


#: the observing-condition switches of the donor-conditioned stage: the spec fields (``InstrumentSpec``) that
#: ``InstrumentOptions.from_spec`` copies one for one, with ``vig_energy_rule``
REALISM_SWITCHES = (
    "vig_energy_rule", "detector_footprint", "vignetting_beyond", "vig_axis", "live_time", "background",
    "qpb_spatial", "qpb_level", "soft_protons", "sp_rate_unmeasured", "agn_lognlogs", "agn_psf", "agn_galactic_nh",
    "agn_norm", "galactic_nh", "mos_soft_ccd", "sky_model", "agn_population",
)


@dataclass(frozen=True)
class InstrumentOptions:
    """Every number and switch of the instrument stage (from CampaignSpec).

    The defaults reproduce the unconditioned stage; the v1 presets set the donor-conditioned values.  Switches that have a
    single supported value (``_SUPPORTED``) stay fields because the products
    and the callers read them; any other value is refused by
    :func:`simulate_instrument`."""
    ccf_dir: str = ""
    emin_image_kev: float = 0.5
    emax_image_kev: float = 2.0
    target_pix_arcsec: float = TARGET_PIX_ARCSEC
    merged_npix: int = MERGED_NPIX
    camera_offset_arcmin: tuple = (("MOS1", (0.0, 0.0)), ("MOS2", (0.0, 0.0)), ("PN", (0.0, 0.0)))
    pn_readout_tilt_deg: float = PN_READOUT_TILT_DEG
    ra0_deg: float = 150.0
    dec0_deg: float = 2.0
    filter_mode: str = "per-donor"         # each camera simulates the donor's filter
    submode_mode: str = "per-donor"        # SUBMODE headers from the donor record
    pattern_mode: str = "ccf"              # PATTERN efficiency folded into the pn/MOS ARF
    pattern_rule: str | None = "sas"       # 'sas' (FRAC_SD, as arfgen) | None (FRAC_SD / FRAC_SDTQ)
    expmap_method: str = "inverse-map"
    expmap_ss: int = 4
    header_style: str = "v4"
    mos2_mirror: str = "XRT2"              # mirror file of MOS2: 'XRT2' | 'upstream' (XRT1)
    obsid_style: str = "v4"
    arf_energy_max_kev: float | None = 15.0  # upper edge of the ARF grid (None: upstream grid, 10.01 keV)
    vig_energy_rule: str = "count-weighted"  # EXPMAP vignetting energy: count-weighted | band-centre (SAS eexpmap) | geometric-mean
    placement: str = "donor-offset"
    pn_oot_model: str = PN_OOT_MODEL
    pn_oot_energy: str = "none"            # pn_oot_energy.OOT_ENERGY_RULES: 'ccf-cti' = CTI correction of the assigned row
    out_of_fov_events: str = "drop"
    readout_windows: str = "donor"
    active_ccds: str = "donor"
    product_bands: str = "pps-1-5"         # write_camera_products.PRODUCT_BANDS
    pn_band1_events: str = "single-pixel"
    sas_headers: str = "on"                # instrument.sas_headers
    pn_edge_rows: str = "drop"             # PN_EDGE_ROWS: 'drop' (12 rows from the footprint edge) | 'footprint'
    arf_model: str = "xmm_simulator"       # load_responses.ARF_MODELS: 'fit-linear' = the fit's effective area
    agn: str = "on"                        # 'on' | 'off': every AGN draw is made, the AGN events are dropped (diagnostic)
    # -- the donor-conditioned stage (REALISM_SWITCHES: the spec fields, same names and values) --
    # defaults are the unconditioned stage; the production presets set the conditioned values
    detector_footprint: str = "mask"       # 'mask' (packaged 0/1 mask) | 'donor-pps' (donor_footprint)
    vignetting_beyond: str = "zero"        # load_responses.VIGNETTING_BEYOND: effective area beyond 15' ('ccf-edge')
    vig_axis: str = "pointing"             # VIG_AXIS_RULES: vignetting centre at the pointing | 'ccf'
    live_time: str = "package"             # 'package' (cameras_s) | 'donor-expmap' (donor's on-axis PPS EXPMAP level)
    background: str = "nominal"            # 'nominal' (FWC level) | 'donor' (instrument.donor_background)
    qpb_spatial: str = "flat"              # QPB over the detector: 'flat' | 'fwc' (FWC event density per band)
    qpb_level: str = "in-fov"              # donor_corner_qpb.QPB_LEVELS: 'corners' = QPB level from the out-of-FOV corners
    soft_protons: str = "off"              # donor_inout_sp.SOFT_PROTONS: 'inout' adds the donor's residual SP
    sp_rate_unmeasured: str = "refuse"     # donor_inout_sp.SP_RATE_UNMEASURED: a camera without a measured rate
    agn_lognlogs: str = "moretti03-hard"   # point_sources.AGN_LOGNLOGS: number counts and flux band of the AGN
    agn_psf: str = "king-onaxis"           # point_sources.AGN_PSF: PSF of the AGN photons
    psf_model: str = "king-onaxis"         # CLUSTER_PSF_MODELS: PSF of the cluster photons
    agn_galactic_nh: str = "off"           # point_sources.AGN_GALACTIC_NH: 'view' absorbs the AGN by the view's N_H
    agn_norm: str = "moretti03"            # number counts of Moretti et al. (2003) at their own normalisation
    galactic_nh: str = "hi4pi"             # galactic_column.GALACTIC_NH: 'willingale13' absorbs by N_HI + 2 N_H2
    mos_soft_ccd: str = "off"              # donor_soft_ccd.MOS_SOFT_CCD: 'donor' adds the donor's anomalous-CCD soft excess
    agn_population: str = "xmm_simulator" # point_sources.AGN_POPULATIONS: 'ueda14' = z, N_H, Gamma of Ueda et al. (2014)
    sky_model: str = "xmm_simulator"       # SKY_MODELS: 'donor-rass' = the donor field's RASS foreground (donor_tables.rass_sky)
    # -- replacements of the data package's donor tables (donor_tables); '' = the package table --
    background_table: str = ""
    mos_background_table: str = ""         # 'qpb_level' corners, 'qpb_units' corners and 'soft_protons' inout
    nh_total_table: str = ""
    mos_soft_ccd_table: str = ""
    sky_table: str = ""
    # -- XMM-ESAS / X-COP background study (release 1.1.3; spec fields hashed only when set) --
    qpb_units: str = "none"                # 'corners': per MOS CCD / pn quadrant QPB level from the background table
    soft_proton_map: str = "flat"          # SOFT_PROTON_MAPS: 'spdetmap' (CCF XMM_SPDETMAP) | 'flare' (CCF XMM_FLARE per band)
    soft_proton_template: str = ""         # the map of the fit's SPMAP when it differs from the draw ('' = the same)
    # -- MOS vignetting (spec field hashed only when set) --
    mos_rga_vignetting: str = "off"        # rga_obscuration.MOS_RGA_RULES: 'ccf' = RGA azimuthal modulation of the CAL
    qpb_model_images: str = "off"          # 'on' | 'on+alt': write QPBEXP / SPMAP (and QPBALT) images per band
    pn_hard_band: str = "contiguous"       # 'pps': pn 4.5-12 keV images without 7.8-8.2 keV (the PPS selection)
    # -- per-observation responses and the calibration-error arms (spec fields hashed only when set) --
    responses: str = "packaged"            # obs_responses.GENERATOR_RESPONSES: 'donor-rmf' | 'donor' (conditions donor's)
    response_set: str = ""                 # root of the per-observation set ('' = $TNGXMM_OBS_RESP, the site key)
    area_perturbation: str = ""            # area_perturbation '<model>:+1' | '<model>:-1' ('' = none)
    ccf_selection: str = "highest-issue"   # ccf_select.CCF_RULES: 'valid-at-date' = the CCF files cifbuild selects

    @property
    def camera_offsets(self) -> dict[str, tuple[float, float]]:
        """``camera_offset_arcmin`` as ``{camera: (dx, dy)}`` [arcmin, detector frame]."""
        return {k: tuple(v) for k, v in self.camera_offset_arcmin}

    @classmethod
    def legacy(cls, ccf_dir: str | Path) -> InstrumentOptions:
        """The options at their defaults (``InstrumentOptions(ccf_dir=...)``)."""
        return cls(ccf_dir=str(ccf_dir))

    @classmethod
    def from_spec(cls, spec: Any, ccf_dir: str | Path) -> InstrumentOptions:
        """Options of a CampaignSpec preset.

        Three spec fields select behaviour the stage implements in one way only and are
        checked here: ``pattern_loss`` (``'ccf'``: the CCF pattern efficiency
        ``pattern_efficiency``), ``write_source_lists`` (the AGN table is
        always written) and ``deterministic_gzip`` (products are always byte-reproducible).
        ``dither_clear_arcmin`` has no effect: the cluster sits at the donor's core offset."""
        ins = spec.instrument
        if str(ins.pattern_loss) != "ccf":
            raise ValueError(f"instrument.pattern_loss {ins.pattern_loss!r}: the stage applies the CCF pattern "
                             "efficiency ('ccf')")
        if not bool(ins.write_source_lists) or not bool(ins.deterministic_gzip):
            raise ValueError("instrument.write_source_lists and deterministic_gzip must be True")
        return cls(
            ccf_dir=str(ccf_dir),
            emin_image_kev=float(ins.emin_image_kev), emax_image_kev=float(ins.emax_image_kev),
            target_pix_arcsec=float(ins.target_pix_arcsec), merged_npix=int(ins.merged_npix),
            camera_offset_arcmin=tuple((k, tuple(v)) for k, v in ins.camera_offset_arcmin),
            pn_readout_tilt_deg=float(ins.pn_readout_tilt_deg),
            ra0_deg=float(ins.ra0_deg), dec0_deg=float(ins.dec0_deg),
            filter_mode=str(ins.filter),
            submode_mode=str(ins.pn_submode_header),
            pattern_rule=ins.pattern_efficiency,
            header_style=ins.header_style,
            expmap_method=ins.expmap_method,
            mos2_mirror=ins.mos2_mirror,
            obsid_style=ins.obsid_style,
            expmap_ss=int(ins.expmap_subsamples),
            arf_energy_max_kev=(None if ins.arf_energy_max_kev is None
                                else float(ins.arf_energy_max_kev)),
            placement=str(ins.placement),
            pn_oot_model=str(ins.pn_oot_model),
            pn_oot_energy=str(getattr(ins, "pn_oot_energy", None) or "none"),
            out_of_fov_events=str(ins.out_of_fov_events),
            readout_windows=str(ins.readout_windows),
            active_ccds=str(ins.active_ccds),
            product_bands=str(ins.product_bands),
            pn_band1_events=str(ins.pn_band1_events),
            sas_headers=str(ins.sas_headers),
            pn_edge_rows=str(ins.pn_edge_rows),
            arf_model=str(getattr(ins, "arf_model", None) or "xmm_simulator"),
            psf_model=str(getattr(ins, "psf_model", None) or "king-onaxis"),
            mos_background_table=str(getattr(ins, "mos_background_table", None) or ""),
            qpb_units=str(getattr(ins, "qpb_units", None) or "none"),
            soft_proton_map=str(getattr(ins, "soft_proton_map", None) or "flat"),
            soft_proton_template=str(getattr(ins, "soft_proton_template", None) or ""),
            mos_rga_vignetting=str(getattr(ins, "mos_rga_vignetting", None) or "off"),
            qpb_model_images=str(getattr(ins, "qpb_model_images", None) or "off"),
            pn_hard_band=str(getattr(ins, "pn_hard_band", None) or "contiguous"),
            responses=str(getattr(ins, "responses", None) or "packaged"),
            area_perturbation=str(getattr(ins, "area_perturbation", None) or ""),
            ccf_selection=str(getattr(ins, "ccf_selection", None) or "highest-issue"),
            **{name: str(getattr(ins, name)) for name in REALISM_SWITCHES},
        )

    def replace(self, **kw: Any) -> InstrumentOptions:
        """A copy with the given fields changed (``dataclasses.replace``)."""
        return dataclasses.replace(self, **kw)


_FILTER_COLUMNS = {"MOS1": "filter_mos1", "MOS2": "filter_mos2", "PN": "filter_pn"}


def donor_filters(*records: Any) -> dict[str, str] | None:
    """Per-camera donor filters from the first record that carries them.

    A record is a mapping (donor package, view-table row, targets row): a
    ``filters`` or ``filter_by_camera`` dict (camera -> filter), or the
    targets columns ``filter_mos1``, ``filter_mos2``, ``filter_pn``
    (``campaigns/v4/targets_v4.csv``; empty/NaN = camera absent).  Returns
    None when no record names a filter."""
    import math

    for rec in records:
        if rec is None:
            continue
        get = rec.get if hasattr(rec, "get") else (lambda k, d=None, _r=rec: getattr(_r, k, d))
        for key in ("filters", "filter_by_camera"):
            val = get(key, None)
            if isinstance(val, dict) and val:
                return {str(k): str(v) for k, v in val.items()}
        out = {}
        for cam, col in _FILTER_COLUMNS.items():
            v = get(col, None)
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
                continue
            out[cam] = str(v).strip()
        if out:
            return out
    return None


_SUBMODE_COLUMNS = {"MOS1": ("submode_mos1",), "MOS2": ("submode_mos2",),
                    "PN": ("pn_submode", "pn_mode", "submode_pn")}


def donor_submodes(*records: Any) -> dict[str, str] | None:
    """Per-camera donor submodes (camera -> SAS SUBMODE) from the first record
    that carries any: a ``submodes`` dict (camera -> mode), or the columns
    ``submode_mos1``, ``submode_mos2`` and ``pn_submode``/``pn_mode`` (donor
    package, targets row; empty/NaN = unknown).  None when no record names one."""
    import math

    for rec in records:
        if rec is None:
            continue
        get = rec.get if hasattr(rec, "get") else (lambda k, d=None, _r=rec: getattr(_r, k, d))
        val = get("submodes", None)
        if isinstance(val, dict) and val:
            return {str(k): str(v) for k, v in val.items() if v is not None}
        out = {}
        for cam, cols in _SUBMODE_COLUMNS.items():
            for col in cols:
                v = get(col, None)
                if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
                    continue
                out[cam] = str(v).strip()
                break
        if out:
            return out
    return None


def view_context(spec: Any = None, *, snap: int | None = None, halo_id: int | None = None,
                 view: int | None = None, donor_package: dict | None = None,
                 view_id: str | None = None, seed: int | None = None,
                 z_obs: float | None = None, nh_1e22: float | None = None,
                 mockobs: str | None = None) -> dict[str, Any]:
    """View-level provenance keys of the product headers (``add_v4_keys``).

    Sim-side keys (``APECVER``, ``EPHOTMAX``, ``TFLOORK``, ``GASLOAD``; ``SNAPMINI`` and
    ``PERIODIC`` from the snapshot number and ``sim.periodic_wrap``) are
    read from ``spec.sim`` when the spec carries them (fields
    ``apec_version``, ``emax_phot_kev``, ``t_min_k``, ``loader``/``gas_selection``);
    absent fields are left out, never guessed."""
    ctx: dict[str, Any] = {}
    pkg = donor_package or {}
    if spec is not None:
        ctx["SPECHASH"] = spec.spec_hash()
        sim = getattr(spec, "sim", None)
        ctx["SIMNAME"] = getattr(sim, "sim_name", None)
        for key, names in (("APECVER", ("apec_version", "apec_vers")),
                           ("EPHOTMAX", ("emax_phot_kev",)),
                           ("TFLOORK", ("t_min_k", "t_floor_k")),
                           ("GASLOAD", ("gas_selection", "loader"))):
            for n in names:
                v = getattr(sim, n, None)
                if v is not None:
                    ctx[key] = v if isinstance(v, (int, float, bool)) else str(v)
                    break
    from tngxmm import __version__

    ctx["TNGXVER"] = str(__version__)
    if spec is not None and snap is not None and getattr(spec, "sim", None) is not None \
            and hasattr(spec.sim, "periodic_wrap"):
        from tngxmm.sim.snapshots import snap_is_mini

        ctx["SNAPMINI"] = bool(snap_is_mini(int(snap)))
        ctx["PERIODIC"] = bool(spec.sim.periodic_wrap)
    ctx.update(VIEW_ID=view_id, SNAPNUM=None if snap is None else int(snap),
               HALOID=None if halo_id is None else int(halo_id),
               VIEWSEED=None if seed is None else int(seed),
               Z_OBS=None if z_obs is None else float(z_obs),
               NH_1E22=None if nh_1e22 is None else float(nh_1e22),
               MOCKOBS=mockobs,
               DONORID=(None if pkg.get("obsid", pkg.get("donor_id")) is None
                        else str(pkg.get("obsid", pkg.get("donor_id")))))
    return {k: v for k, v in ctx.items() if v is not None}


def _donor_expids(context: dict | None) -> dict[str, str]:
    """Camera -> ``EXPIDSTR`` of the conditions donor's exposure (``context['_sas_donor']``,
    :func:`tngxmm.instrument.sas_headers.package_sas_donor`): the exposure whose response ``instrument.responses``
    takes, the one the mock's headers name."""
    donor = (context or {}).get("_sas_donor") or {}
    return {str(c): str(v[1]) for c, v in (donor.get("exposure") or {}).items() if v is not None and len(v) > 1}


# ── cluster placement and field of view ─────────────────────────────────────

def _window_key(resp: CameraResponse) -> tuple:
    """Cache-key suffix of a windowed response (empty for the full frame)."""
    win = getattr(resp, 'window', None)
    ccds = getattr(resp, 'ccds', None)
    foot = getattr(resp, 'footprint', None)
    if foot is not None:
        return (win, ccds, foot, id(resp.mask))
    return () if win is None and ccds is None else (win, ccds, id(resp.mask))


def fov_map(resp: CameraResponse):
    """Detector pixels inside the field of view: on the mask (``mask == 1``)
    AND with non-zero effective area at some energy (the vignetting CCF ends at
    15'; the pn and MOS CCD corners beyond it are active silicon that sees only
    the particle background).  These are exactly the pixels where the
    inverse-map EXPMAP is non-zero."""
    # a response with a shifted vignetting centre shares arf_theta and mask with its parent: its theta map and axis key it
    key = (resp.inst, str(getattr(resp.sim, 'ccfpath', '')), id(resp.arf_theta), id(resp.theta_idx_pix),
           _vig_axis(resp)) + _window_key(resp)
    if key not in _FOV_MAPS:
        has_area = (resp.arf_theta > 0).any(axis=1)          # per off-axis bin
        _FOV_MAPS[key] = (resp.mask == 1) & has_area[resp.theta_idx_pix]
    return _FOV_MAPS[key]


_AXIS: dict = {}


@cites("xmm_ccf_sas", "xmm_cal_tn0018")
def with_vignetting_axis(resp: CameraResponse, axis_arcmin: tuple) -> CameraResponse:
    """``resp`` with its vignetting centred at ``axis_arcmin`` [arcmin, mask frame] from the
    mask centre: ``theta_idx_pix`` about the new centre and ``vig_axis`` [mask px], which every
    off-axis angle of the stage adds (camera partition, AGN).  The mask is not moved.  The
    response itself for a zero offset; cached (object identities key downstream caches)."""
    import dataclasses as _dc

    import numpy as np

    ax, ay = (float(axis_arcmin[0]) / resp.pixsize, float(axis_arcmin[1]) / resp.pixsize)
    if ax == 0.0 and ay == 0.0:
        return resp
    key = (id(resp), ax, ay)
    if key not in _AXIS:
        yy, xx = np.indices(resp.mask.shape)
        th = np.hypot(xx - resp.cx - ax, yy - resp.cy - ay) * resp.pixsize
        tidx = np.clip(np.round(th / (resp.thetas[1] - resp.thetas[0])), 0,
                       len(resp.thetas) - 1).astype(np.int16)
        _AXIS[key] = (resp, _dc.replace(resp, theta_idx_pix=tidx, vig_axis=(ax, ay)))
    return _AXIS[key][1]


def _vig_axis(resp: Any) -> tuple:
    return tuple(getattr(resp, "vig_axis", (0.0, 0.0)) or (0.0, 0.0))


@cites("xmm_cal_tn0018", "xmm_ccf_sas")
def apply_vignetting_calibration(resps: dict, opt: InstrumentOptions) -> dict:
    """``resps`` (camera -> response) with the vignetting centre that the options ask for: ``vig_axis = 'ccf'``
    centres each camera on the telescope axis of the calibration files in ``opt.ccf_dir``
    (:mod:`tngxmm.instrument.calibration_axis`); ``mos_rga_vignetting = 'ccf'`` adds to the MOS the azimuthal
    modulation by the RGA that the SAS calibration layer computes (:mod:`tngxmm.instrument.rga_obscuration`), about
    the same centre.  The defaults return ``resps`` unchanged (radially symmetric vignetting about the pointing)."""
    from tngxmm.instrument.rga_obscuration import MOS_RGA_RULES

    if opt.vig_axis not in VIG_AXIS_RULES:
        raise ValueError(f"unknown vig_axis {opt.vig_axis!r}; known: {VIG_AXIS_RULES}")
    if opt.mos_rga_vignetting not in MOS_RGA_RULES:
        raise ValueError(f"unknown mos_rga_vignetting {opt.mos_rga_vignetting!r}; known: {MOS_RGA_RULES}")
    if opt.vig_axis == "ccf":
        from tngxmm.instrument.calibration_axis import telescope_axis_arcmin

        resps = {i: with_vignetting_axis(r, telescope_axis_arcmin(
            i, opt.ccf_dir, opt.pn_readout_tilt_deg, r.pn_half_s * r.pixsize if i == "PN" else None))
            for i, r in resps.items()}
    if opt.mos_rga_vignetting == "ccf":
        resps = {i: with_rga_modulation(r, opt.ccf_dir) if i in ("MOS1", "MOS2") else r for i, r in resps.items()}
    return resps


_RGA: dict = {}


@cites("xmm_calhb", "xmm_ccf_sas")
def with_rga_modulation(resp: CameraResponse, ccf_dir: str) -> CameraResponse:
    """``resp`` (MOS1 or MOS2) carrying its RGA modulation ``rga`` = (a(E) on ``box_ene_mean`` [1/arcmin], phi0 [deg])
    of :func:`tngxmm.instrument.rga_obscuration.camera_dipole`; cached (object identities key downstream caches)."""
    import dataclasses as _dc

    from tngxmm.instrument.rga_obscuration import camera_dipole

    key = (id(resp), str(ccf_dir))
    if key not in _RGA:
        a, phi0 = camera_dipole(ccf_dir, resp.inst, resp.box_ene_mean)
        _RGA[key] = (resp, _dc.replace(resp, rga=(a, float(phi0))))
    return _RGA[key][1]


def rga_factor(resp: Any, dx_px, dy_px, energy_index=None):
    """RGA throughput factor of ``resp`` at offsets (dx, dy) [mask px, detector frame] from its vignetting centre:
    1 without a modulation; ``energy_index`` None evaluates every energy of the grid (trailing axis), an integer array
    (broadcast with the offsets) one energy per point."""
    import numpy as np

    from tngxmm.instrument.rga_obscuration import factor

    rga = getattr(resp, "rga", None)
    dx = np.asarray(dx_px, float) * resp.pixsize
    dy = np.asarray(dy_px, float) * resp.pixsize
    if rga is None:
        return np.ones(np.broadcast(dx, dy).shape)
    a, phi0 = rga
    if energy_index is None:
        return factor(np.asarray(a)[None, :], phi0, dx[..., None], dy[..., None])
    return factor(np.asarray(a)[np.asarray(energy_index)], phi0, dx, dy)


def rga_offset_map(resp: Any):
    """(dx, dy) [mask px] of every mask-grid pixel from the vignetting centre of ``resp`` (detector frame)."""
    import numpy as np

    vx, vy = _vig_axis(resp)
    yy, xx = np.indices(resp.mask.shape)
    return xx - resp.cx - vx, yy - resp.cy - vy


def drop_out_of_fov(resp: CameraResponse, X, Y, chan, off_px: tuple = (0.0, 0.0)):
    """Remove the events (sky X/Y [mask px], channels) whose detector pixel
    ``round(sky - off)`` lies outside :func:`fov_map`: the OUT_OF_FOV selection
    of the PPS images.  The pixel rule is the one of the inverse-map EXPMAP
    (``make_exposure_maps.inverse_map_samples``).  No random draw."""
    import numpy as np

    fov = fov_map(resp)
    xd = np.round(np.asarray(X, dtype=np.float64) - off_px[0]).astype(np.int64)
    yd = np.round(np.asarray(Y, dtype=np.float64) - off_px[1]).astype(np.int64)
    ok = (xd > 0) & (xd < resp.npix - 1) & (yd > 0) & (yd < resp.npix - 1)
    ok[ok] = fov[yd[ok], xd[ok]]
    return X[ok], Y[ok], chan[ok]


#: PSF models of the cluster photons (``InstrumentSpec.psf_model``): the on-axis King profile (the default, what the
#: fit convolves with), the King parameters interpolated in energy and off-axis angle, or the ELLBETA envelope the AGN
#: are drawn with (a second calibration of the same PSF; the closure arm ``v1_psfell``)
CLUSTER_PSF_MODELS = ("king-onaxis", "king-energy-angle", "ellbeta")


def check_psf_model(opt: InstrumentOptions) -> None:
    """``ValueError`` unless ``psf_model`` names an implemented cluster PSF (:data:`CLUSTER_PSF_MODELS`)."""
    if opt.psf_model not in CLUSTER_PSF_MODELS:
        raise ValueError(f"unknown psf_model {opt.psf_model!r}; known: {CLUSTER_PSF_MODELS}")


_KING_TABLES: dict[str, dict] = {}


@cites("xmm_ccf_sas")
def king_table(psf_file: str) -> dict:
    """The KING_PARAMS extension of an XRT PSF calibration file as a grid: ``e_kev`` (energies), ``theta_arcmin``
    (off-axis angles), ``r0_arcmin`` and ``alpha`` (energy x angle), without the energy rows whose slope alpha is
    not above 1 at every angle.  Cached per file."""
    import numpy as np
    from astropy.io import fits

    if psf_file not in _KING_TABLES:
        with fits.open(psf_file) as f:
            d = f["KING_PARAMS"].data
            e = np.asarray(d["ENERGY"], float) / 1000.0
            t = np.degrees(np.asarray(d["THETA"], float)) * 60.0
            par = np.asarray(d["PARAMS"], float)
        es, ts = np.unique(e), np.unique(t)
        r0 = np.full((len(es), len(ts)), np.nan)
        al = np.full_like(r0, np.nan)
        for ei, ti, (rr, aa) in zip(np.searchsorted(es, e), np.searchsorted(ts, t), par, strict=True):
            r0[ei, ti], al[ei, ti] = rr / 60.0, aa
        if not (np.isfinite(r0).all() and np.isfinite(al).all()):
            raise ValueError(f"{psf_file}: KING_PARAMS is not a full energy x angle grid")
        # a King profile with alpha <= 1 has no finite flux (the EEF inverse does not exist): the energy rows with
        # such a slope (13.25 keV in XRT3_XPSF_0018) are dropped, so photons above the last valid row take its values
        ok = (al > 1.0).all(axis=1)
        _KING_TABLES[psf_file] = dict(e_kev=es[ok], theta_arcmin=ts, r0_arcmin=r0[ok], alpha=al[ok])
    return _KING_TABLES[psf_file]


def _bilinear(grid: Any, xs: Any, ys: Any, x: Any, y: Any) -> Any:
    """``grid`` (len(xs) x len(ys)) linearly interpolated at the points (x, y), clipped to the grid's range."""
    import numpy as np

    x = np.clip(np.asarray(x, float), xs[0], xs[-1])
    y = np.clip(np.asarray(y, float), ys[0], ys[-1])
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    fx = (x - xs[i]) / (xs[i + 1] - xs[i])
    fy = (y - ys[j]) / (ys[j + 1] - ys[j])
    return ((1 - fx) * (1 - fy) * grid[i, j] + fx * (1 - fy) * grid[i + 1, j]
            + (1 - fx) * fy * grid[i, j + 1] + fx * fy * grid[i + 1, j + 1])


@cites("xmm_ccf_sas", "xmm_simulator")
def king_offsets_energy_angle(resp: CameraResponse, table: dict, e_kev: Any, theta_arcmin: Any, rng) -> tuple:
    """PSF offsets [mask pixels] of photons of energy ``e_kev`` at off-axis angle ``theta_arcmin``: the King EEF
    inverse of :func:`king_offsets` with r0 and alpha interpolated (bilinearly, clipped to the tabulated range) in
    energy and angle in the camera's KING_PARAMS (:func:`king_table`).  The random draws are those of
    :func:`king_offsets` (n uniforms, then n angles), so the two models consume the stream identically."""
    import numpy as np

    n = len(e_kev)
    u = rng.uniform(0.0, 1.0, n)
    r0 = _bilinear(table["r0_arcmin"], table["e_kev"], table["theta_arcmin"], e_kev, theta_arcmin)
    al = _bilinear(table["alpha"], table["e_kev"], table["theta_arcmin"], e_kev, theta_arcmin)
    r = r0 * np.sqrt((1.0 / u) ** (1.0 / (al - 1.0)) - 1.0)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    return (r * np.cos(ang) / resp.pixsize, r * np.sin(ang) / resp.pixsize)


@cites("xmm_ccf_sas", "read2011")
def ellbeta_offsets_energy_angle(resp: CameraResponse, table: dict, e_kev: Any, x_det: Any, y_det: Any,
                                 rng) -> tuple:
    """PSF offsets [mask pixels] of cluster photons of energy ``e_kev`` at detector mask pixel (``x_det``, ``y_det``)
    from the ELLBETA envelope of the CCF (:func:`tngxmm.instrument.point_sources.ellbeta_table`): per photon the
    parameters at the nearest tabulated energy and linear in the off-axis angle from the vignetting centre, the major
    axis tangential, the profile of :func:`tngxmm.instrument.point_sources.ellbeta_offsets` (King inside its
    normalisation radius plus, where tabulated, the Gaussian core).  One uniform per photon picks the component and
    the radius (inverse CDF of the mixture), then one angle per photon: the stream is consumed as by
    :func:`king_offsets`, so a cell drawn with this PSF is the King cell with only the PSF changed."""
    import numpy as np

    from tngxmm.instrument import point_sources as P

    e = np.asarray(e_kev, dtype=float)
    n = len(e)
    u = rng.uniform(0.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    vx, vy = _vig_axis(resp)
    ddx = np.asarray(x_det, float) - resp.cx - vx
    ddy = np.asarray(y_det, float) - resp.cy - vy
    es, ts = table["energy_kev"], table["theta_arcmin"]
    theta = np.clip(np.hypot(ddx, ddy) * resp.pixsize, ts[0], ts[-1])
    tang = np.arctan2(ddy, ddx) + 0.5 * np.pi
    ie = np.argmin(np.abs(e[:, None] - es[None, :]), axis=1)
    par = np.empty((n, 5))
    for k in np.unique(ie):
        m = ie == k
        for j in range(5):
            par[m, j] = np.interp(theta[m], ts, table["params"][k][:, j])
    r0, alpha, fwhm, nrat = par[:, 0], par[:, 1], par[:, 3], par[:, 4]
    eps = np.clip(par[:, 2], 0.0, 0.95)
    umax = 1.0 - (1.0 + (P.ELLBETA_RMAX_ARCSEC / r0) ** 2) ** (1.0 - alpha)       # King EEF at the radius
    w_k = r0 ** 2 / (alpha - 1.0) * umax
    w_g = np.where((fwhm > 0) & (nrat > 0), nrat * fwhm ** 2 / (4.0 * np.log(2.0)), 0.0)
    pg = w_g / (w_k + w_g)
    gauss = u < pg
    r = np.empty(n)
    k = ~gauss
    r[k] = P._king_radius((u[k] - pg[k]) / (1.0 - pg[k]) * umax[k], r0[k], alpha[k])
    sig = fwhm[gauss] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    r[gauss] = sig * np.sqrt(-2.0 * np.log(1.0 - u[gauss] / pg[gauss]))
    xp, yp = r * np.cos(phi), (1.0 - eps) * r * np.sin(phi)
    c, s = np.cos(tang), np.sin(tang)
    scale = 1.0 / (resp.pixsize * 60.0)                              # arcsec -> mask px
    return (xp * c - yp * s) * scale, (xp * s + yp * c) * scale


def soft_protons_on(with_qpb: bool, donor_bkg: dict | None) -> bool:
    """Whether :func:`camera_events` draws residual soft protons: only with the particle background on (``with_qpb``)
    and a donor background record with a positive ``sp_rate``.  Production always runs with the particle background,
    so this only matters for a view made without it (the ``clean`` and ``agnonly`` closure cells)."""
    return bool(with_qpb) and donor_bkg is not None and float(donor_bkg.get('sp_rate', 0.0)) > 0


def check_agn_switches(opt: InstrumentOptions) -> None:
    """``ValueError`` unless the AGN switches name implemented choices (``agn_lognlogs``, ``agn_psf``,
    ``agn_galactic_nh`` and ``agn_norm``, whose only choice is the published Moretti et al. counts)."""
    from tngxmm.instrument import point_sources as PS

    if opt.agn_lognlogs not in PS.AGN_LOGNLOGS:
        raise ValueError(f"unknown agn_lognlogs {opt.agn_lognlogs!r}; known: {tuple(PS.AGN_LOGNLOGS)}")
    if opt.agn_psf not in PS.AGN_PSF:
        raise ValueError(f"unknown agn_psf {opt.agn_psf!r}; known: {PS.AGN_PSF}")
    if opt.agn_norm not in PS.AGN_NORM:
        raise ValueError(f"unknown agn_norm {opt.agn_norm!r}; known: {PS.AGN_NORM}")
    if opt.agn_galactic_nh not in PS.AGN_GALACTIC_NH:
        raise ValueError(f"unknown agn_galactic_nh {opt.agn_galactic_nh!r}; known: {PS.AGN_GALACTIC_NH}")
    if opt.agn_population not in PS.AGN_POPULATIONS:
        raise ValueError(f"unknown agn_population {opt.agn_population!r}; known: {PS.AGN_POPULATIONS}")
    if opt.agn_population == "ueda14" and opt.agn_lognlogs != "moretti03-soft":
        raise ValueError("agn_population 'ueda14' draws z and N_H at the observed soft flux: agn_lognlogs 'moretti03-soft'")


def donor_core_offset(*records: Any) -> tuple[float, float] | None:
    """The donor's core offset (``core_dx_mask_arcmin``, ``core_dy_mask_arcmin``,
    mask frame [arcmin]) from the first record (mapping or object) that carries
    both as finite numbers; None otherwise."""
    import math

    for rec in records:
        if rec is None:
            continue
        get = rec.get if hasattr(rec, "get") else (lambda k, d=None, _r=rec: getattr(_r, k, d))
        try:
            dx = float(get("core_dx_mask_arcmin", None))
            dy = float(get("core_dy_mask_arcmin", None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(dx) and math.isfinite(dy):
            return dx, dy
    return None


def donor_offset_displacement(core_offset_arcmin: tuple[float, float],
                              max_offset_arcmin: float | None = None) -> tuple[float, float, bool]:
    """Cluster displacement [arcmin, mask frame] of the ``donor-offset`` placement:
    the donor's core offset, shrunk along its direction to ``max_offset_arcmin``
    (the dither cap that keeps the largest aperture in the field; None = no
    cap; a cap <= 0 puts the cluster at the pointing).  Returns
    (dx, dy, capped)."""
    import math

    dx, dy = float(core_offset_arcmin[0]), float(core_offset_arcmin[1])
    r = math.hypot(dx, dy)
    if max_offset_arcmin is None or r <= float(max_offset_arcmin):
        return dx, dy, False
    cap = max(0.0, float(max_offset_arcmin))
    f = cap / r
    return dx * f, dy * f, True


# ── vectorized samplers ──────────────────────────────────────────────────────

@cites("xmm_ccf_sas", "xmm_simulator")
def king_offsets(resp: CameraResponse, n: int, rng) -> tuple:
    """PSF offsets [mask pixels] from the analytic King EEF inverse
    r = r0 sqrt(u^(-1/(alpha-1)) - 1) (the corrected upstream sampler)."""
    import numpy as np

    u = rng.uniform(0.0, 1.0, n)
    r = resp.king_r0 * np.sqrt((1.0 / u) ** (1.0 / (resp.king_alpha - 1.0)) - 1.0)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    return (r * np.cos(ang) / resp.pixsize, r * np.sin(ang) / resp.pixsize)


@cites("xmm_simulator")
def sample_channels(resp: CameraResponse, e_true, rng):
    """RMF channel energies for photons of true energy e_true [keV]
    (grouped inverse-CDF draws; replaces the per-photon np.random.choice)."""
    import numpy as np

    k = np.clip(np.searchsorted(resp.mc_edges, e_true, side='right') - 1,
                0, len(resp.mc_ene) - 1)
    out   = np.empty(len(e_true), dtype=np.float64)
    order = np.argsort(k, kind='stable')
    ks    = k[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    for grp in np.split(np.arange(len(ks)), bounds):
        row = resp.rmf_cdf[ks[grp[0]]]
        tot = row[-1]
        if tot <= 0:                      # MC bin with no response
            out[order[grp]] = resp.emb[0]
            continue
        idx = np.searchsorted(row, rng.random(len(grp)) * tot)
        out[order[grp]] = resp.emb[np.clip(idx, 0, len(resp.emb) - 1)]
    return out


@cites("xmm_simulator", "threeml")
def channel_cdf_from_coarse(resp: CameraResponse, coarse_spec):
    """(cdf, total) of the RMF-convolved channel distribution for a spectrum
    given as a DENSITY [per keV] on box_ene_mean (interp to the MC grid, x bin
    width, rmf.convolve), done once."""
    import numpy as np

    w = np.interp(resp.mc_ene, resp.box_ene_mean, coarse_spec, left=0., right=0.)
    w[w < 0] = 0.0
    conv = np.asarray(resp.rmf.convolve(w * resp.mc_wid))
    conv[conv < 0] = 0.0
    tot = conv.sum()
    return (np.cumsum(conv), tot)


def _draw_from_cdf(cdf, values, n: int, rng):
    import numpy as np

    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1])
    return values[np.clip(idx, 0, len(values) - 1)]


# ── sky background & AGN spectral models ─────────────────────────────────────

#: values of InstrumentOptions.sky_model
SKY_MODELS = ("xmm_simulator", "donor-rass")


@cites("xmm_simulator", "moretti2003", "threeml", "atomdb", "snowden2008", "yeung2024", "henley2013", "deluca2004")
def skybkg_spectrum(nh: float, box_ene_mean, sky: dict | None = None):
    """Sky background photon density [ph/s/cm^2/keV/arcmin^2] on the coarse grid: LHB APEC + PhAbs(NH) x (halo APEC
    + unresolved CXB power law; AGN are injected explicitly), all at solar abundance of Anders & Grevesse (1989).

    ``sky`` None: the xmm_simulator.event_file defaults (one fit to the MACS J0949 field).  ``sky`` = the donor's
    ``{'lhb_norm', 'halo_norm'}`` (``instrument.sky_model = 'donor-rass'``, :mod:`tngxmm.donor_tables.rass_sky`): the
    LHB at kT = 0.111 keV and the halo at 0.191 keV with those normalisations, and the CXB the drawn AGN leave out
    (:func:`tngxmm.instrument.point_sources.unresolved_cxb_norm`, photon index 1.41).  Cache keyed on round(nh, 6), the
    sky and the energy grid, so an extended ARF grid (``arf_energy_max_kev``) extends the spectrum with it."""
    import numpy as np
    from threeML import APEC, PhAbs, Powerlaw

    grid = np.asarray(box_ene_mean, dtype=np.float64)
    skey = None if sky is None else (round(float(sky["lhb_norm"]), 15), round(float(sky["halo_norm"]), 15))
    key = (round(float(nh), 6), len(grid), float(grid[0]), float(grid[-1]), skey)
    if key in _SKYBKG_MODELS:
        return _SKYBKG_MODELS[key]

    if sky is None:
        from xmm_simulator.event_file import cxb_m13_unres, ghn_ref, ght_ref, lhb_ref

        lhb = (0.11, lhb_ref)
        halo = (ght_ref, ghn_ref)
        cxb = (1.46, cxb_m13_unres)
    else:
        from tngxmm.donor_tables.rass_sky import HALO_KT, LHB_KT
        from tngxmm.instrument.point_sources import CXB_INDEX, unresolved_cxb_norm

        lhb = (LHB_KT, float(sky["lhb_norm"]))
        halo = (HALO_KT, float(sky["halo_norm"]))
        cxb = (CXB_INDEX, unresolved_cxb_norm())
    modlhb = APEC(); modgh = APEC(); modcxb = Powerlaw(); modphabs = PhAbs()  # noqa: E702
    modlhb.kT = lhb[0];  modlhb.K = lhb[1];  modlhb.redshift = 0.0  # noqa: E702
    modgh.kT  = halo[0]; modgh.K = halo[1]; modgh.redshift  = 0.0  # noqa: E702
    modcxb.index = -cxb[0]; modcxb.K = cxb[1]  # noqa: E702
    modphabs.NH = nh
    model = modlhb + modphabs * (modgh + modcxb)
    spec  = np.asarray(model(box_ene_mean), dtype=np.float64)
    _SKYBKG_MODELS[key] = spec
    return spec


@cites("xmm_simulator", "moretti2003")
def generate_agn_sources(resp: CameraResponse, rng, lognlogs: str = "moretti03-hard",
                         population: str = "xmm_simulator"):
    """Draw the field AGN population (Moretti+03 logN-logS), as
    xmm_simulator.point_sources.gen_sources but Generator-driven.  Returns the
    source table (srcnum, X, Y, Fx, Gamma, NH, z, 0, 0) in the 512-px
    simulation-box frame.

    ``lognlogs`` (:data:`~tngxmm.instrument.point_sources.AGN_LOGNLOGS`):
    ``'moretti03-hard'`` is the xmm_simulator draw (Fx: 2-10 keV, hard-band
    counts); ``'moretti03-soft'`` draws Fx, the observed 0.5-2 keV flux, from
    the soft-band counts and log NH up to ``point_sources.SOFT_NH_MAX``
    (:mod:`tngxmm.instrument.point_sources`).  ``population = 'ueda14'`` (soft counts only) keeps the soft fluxes
    and draws each source's redshift, log NH and photon index from the Ueda et al. (2014) population at that flux
    (:func:`tngxmm.instrument.point_sources.ueda14_draw`) instead of the upstream uniform z in [0, 1.5], log NH in
    [20.5, SOFT_NH_MAX] and Gaussian photon index; the uniform draws are still made, so the random stream of the
    positions is the same."""
    import numpy as np
    from xmm_simulator import point_sources as ps

    sim = resp.sim
    field_area = (sim.box_size / 60.0) ** 2
    if lognlogs == "moretti03-soft":
        from tngxmm.instrument import point_sources as P

        n_src = rng.poisson(P.moretti03_expected_number("soft", field_area))
        flux = P.moretti03_draw(n_src, "soft", rng)
        npix_box = sim.boxshape0
        X_src = rng.random(n_src) * npix_box
        Y_src = rng.random(n_src) * npix_box
        gamma = ps.gamma_mean + ps.gamma_sig * rng.standard_normal(n_src)
        nh_s = ps.nh_min + (P.SOFT_NH_MAX - ps.nh_min) * rng.random(n_src)
        zs = rng.random(n_src) * 1.5
        if population == "ueda14":
            zs, nh_s, gamma = P.ueda14_draw(flux, rng)
        return np.column_stack([np.arange(1, n_src + 1), X_src, Y_src, flux, gamma, nh_s, zs,
                                np.zeros(n_src), np.zeros(n_src)])
    if lognlogs != "moretti03-hard":
        raise ValueError(f"unknown agn_lognlogs {lognlogs!r}")
    n_src = rng.poisson(ps.N_gt15 * field_area)
    npix_box = sim.boxshape0

    S_grid = np.logspace(-15, -12, 1000)
    dS     = (np.log10(S_grid[1]) - np.log10(S_grid[0])) * S_grid * np.log(10.0)
    unnorm = ps.dNdS(S_grid) * dS
    prob   = unnorm / unnorm.sum()
    flux   = S_grid[_draw_from_cdf(np.cumsum(prob), np.arange(1000), n_src, rng)]

    X_src = rng.random(n_src) * npix_box
    Y_src = rng.random(n_src) * npix_box
    gamma = ps.gamma_mean + ps.gamma_sig * rng.standard_normal(n_src)
    nh_s  = ps.nh_min + (ps.nh_max - ps.nh_min) * rng.random(n_src)
    zs    = rng.random(n_src) * 1.5
    srcnum = np.arange(1, n_src + 1)

    table = np.column_stack([srcnum, X_src, Y_src, flux, gamma, nh_s, zs,
                             np.zeros(n_src), np.zeros(n_src)])
    return table


@cites("xmm_simulator", "threeml")
def agn_channel_cdfs(resp: CameraResponse, sources, t_live: float,
                     off_px: tuple = (0.0, 0.0), flux_band_kev: tuple | None = None,
                     nh_gal: float | None = None):
    """Per-source (x, y, expected counts, channel CDF) for one camera: the
    point_sources.pts_box model PhAbs(NH, z) x Powerlaw, flux normalised
    through templates_pts_hard.dat; off-axis angles in the detector frame.

    ``flux_band_kev`` (e1, e2): Fx is instead the observed energy flux of the
    source's own model (its NH and z) in [e1, e2] keV, normalised per source
    (``agn_lognlogs='moretti03-soft'``).  ``nh_gal`` [1e22 cm^-2]: the photon spectrum
    is also absorbed by this Galactic column (PhAbs at z = 0) AFTER the normalisation, so Fx
    stays the flux before Galactic absorption (``agn_galactic_nh='view'``)."""
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    from threeML import PhAbs, Powerlaw
    from xmm_simulator.utils import get_data_file_path

    template = np.loadtxt(get_data_file_path('pts/templates_pts_hard.dat'))
    fint = LinearNDInterpolator(template[:, :2], template[:, 2],
                                fill_value=np.mean(template[:, 2]))

    X_b, Y_b = sources[:, 1], sources[:, 2]          # 512-box pixels
    Fx, Gam, NH_s, z_s = sources[:, 3], sources[:, 4], sources[:, 5], sources[:, 6]
    norm = Fx / fint(Gam, NH_s)
    if flux_band_kev is not None:
        from tngxmm.instrument.point_sources import band_energy_flux

        unit = PhAbs() * Powerlaw()
        norm = np.empty(len(Fx))
        for j in range(len(Fx)):
            unit.NH_1 = 10.0 ** (NH_s[j] - 22.0)
            unit.index_2 = -Gam[j]
            unit.K_2 = 1.0
            unit.redshift_1 = z_s[j]
            norm[j] = Fx[j] / band_energy_flux(lambda e: np.asarray(unit(e)), *flux_band_kev)

    # box pixel -> mask pixel (sky); off-axis angle in the DETECTOR frame
    sim = resp.sim
    ox, oy = off_px
    x_mask = (X_b - sim.boxshape1 / 2.0) * sim.pixsize_ori / resp.pixsize + resp.cx
    y_mask = (Y_b - sim.boxshape0 / 2.0) * sim.pixsize_ori / resp.pixsize + resp.cy
    vx, vy = _vig_axis(resp)
    th_src = np.hypot(x_mask - resp.cx - ox - vx, y_mask - resp.cy - oy - vy) * resp.pixsize
    rga = (rga_factor(resp, x_mask - resp.cx - ox - vx, y_mask - resp.cy - oy - vy)
           if getattr(resp, "rga", None) is not None else None)      # (n_src, n_energy) MOS RGA modulation
    ith    = np.clip(np.round(th_src / (resp.thetas[1] - resp.thetas[0])), 0,
                     len(resp.thetas) - 1).astype(int)

    mod = PhAbs() * Powerlaw()
    gal = 1.0
    if nh_gal is not None:
        g = PhAbs()
        g.NH = float(nh_gal)
        g.redshift = 0.0
        gal = np.asarray(g(resp.box_ene_mean), dtype=np.float64)
    out = []
    for j in range(len(Fx)):
        mod.NH_1 = 10.0 ** (NH_s[j] - 22.0)
        mod.index_2 = -Gam[j]
        mod.K_2 = norm[j]
        mod.redshift_1 = z_s[j]
        dens = np.asarray(mod(resp.box_ene_mean)) * gal * resp.arf_theta[ith[j]] * t_live
        if rga is not None:
            dens = dens * rga[j]
        mu   = (dens * resp.box_ene_wid).sum()      # expected counts
        cdf, tot = channel_cdf_from_coarse(resp, dens)
        out.append((x_mask[j], y_mask[j], mu if tot > 0 else 0.0, cdf))
    return out


# ── pn out-of-time events ───────────────────────────────────────────────────

_PN_INNER_EDGE: dict = {}


def _pn_row_inner_edges(resp: CameraResponse) -> tuple[float, float]:
    """Inner (row-boundary) edges of the two pn CCD rows along the readout
    direction [mask px from the centre, readout-aligned frame]: the median
    over columns of the first active row on each side of the boundary."""
    import numpy as np

    key = id(resp)
    if key not in _PN_INNER_EDGE:
        top = resp.pn_top_lo[resp.pn_top_lo >= 0]
        bot = resp.pn_bot_hi[resp.pn_bot_hi >= 0]
        _PN_INNER_EDGE[key] = (float(np.median(top)) - 0.5 - resp.cy,
                               float(np.median(bot)) + 0.5 - resp.cy)
    return _PN_INNER_EDGE[key]


@cites("xmm_mission", "epic_pn", "xmm_ccf_sas", "xmm_sas_epevents")
def pn_out_of_time(resp: CameraResponse, X, Y, oot_frac: float, rng,
                   off_px: tuple = (0.0, 0.0),
                   readout_tilt_deg: float = PN_READOUT_TILT_DEG,
                   column_arcmin: float = PN_COLUMN_ARCMIN,
                   chan=None, cti=None, ccd_map=None) -> tuple:
    """pn out-of-time events, full-column model.

    A photon that arrives while the CCD is being read out (charge shifted
    along the column towards the node) is registered with a RAWY that has
    nothing to do with where it landed: the readout phase is uniform over
    the frame, so the assigned row is uniform over the whole column, on both
    sides of the source, and the event stays in its own column (RAWX) and on
    its own CCD (each pn CCD is read out through its own node at the outer
    edge of the camera; the two CCD rows meet at the boundary near the
    detector centre).  This is how SAS ``epchain withoutoftime=Y`` simulates
    them: "a new RAWY is simulated by randomly shifting the pattern along the
    RAWY axis".  Each event is out-of-time independently with probability
    ``oot_frac`` (Poisson splitting: in-time and OoT streams are independent
    Poisson processes).

    Energy (``cti``, a :class:`~tngxmm.instrument.pn_oot_energy.PnCtiLine`):
    the charge of an OoT event is transferred over the rows between where it
    landed and the node, but ``epevents`` corrects it for the CTI of its
    assigned RAWY, so its energy becomes E x c(A, y_rec) / c(A, y_true)
    (:mod:`tngxmm.instrument.pn_oot_energy`, from the EPN_CTI calibration):
    too high for events assigned farther from the node than where they
    landed, too low for those assigned closer (Users' Handbook 3.3.10).  The
    row of an event is its distance to the node in rows plus one half (the
    centre of its RAWY); its CCD is read from ``ccd_map`` (the CCD label map
    of the camera on the mask grid) at its true position.  ``cti=None``:
    energies unchanged.

    X, Y: event positions, sky mask-pixel frame (detector = sky - off_px);
    chan: event energies [keV] (optional, returned shifted when ``cti`` is given).
    Returns (X, Y, chan, kept): positions (OoT ones moved), energies, and the
    boolean mask of events still on active silicon (OoT events landing on a
    dead pixel, the row-boundary lane or outside the field are lost, as in
    the real camera); X, Y and chan are already filtered by ``kept``.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64).copy()
    Y = np.asarray(Y, dtype=np.float64).copy()
    C = None if chan is None else np.asarray(chan, dtype=np.float64).copy()
    n = len(X)
    kept = np.ones(n, dtype=bool)
    if n == 0 or oot_frac <= 0:
        return X, Y, C, kept
    pick = np.flatnonzero(rng.random(n) < oot_frac)
    if len(pick) == 0:
        return X, Y, C, kept
    ox, oy = off_px
    a = np.deg2rad(readout_tilt_deg)
    gx, gy = np.sin(a), np.cos(a)              # readout (RAWY) unit vector
    px_, py_ = np.cos(a), -np.sin(a)           # across-column (RAWX) unit vector
    dx = X[pick] - ox - resp.cx                # detector frame, from the centre
    dy = Y[pick] - oy - resp.cy
    s = dx * gx + dy * gy
    q = dx * px_ + dy * py_
    top_in, bot_in = _pn_row_inner_edges(resp)
    lcol = column_arcmin / resp.pixsize        # column length [mask px]
    top = s >= resp.pn_half_s
    u = rng.random(len(pick))
    s_new = np.where(top, top_in + u * lcol, bot_in - u * lcol)
    if C is not None and cti is not None:
        # rows to the node (outer end of the full 200-row column, also in the window modes, whose RAWY count
        # from the node too): true and assigned
        row_px = (PN_ROW_ARCSEC / 60.0) / resp.pixsize
        lfull = PN_COLUMN_ARCMIN / resp.pixsize
        node = np.where(top, top_in + lfull, bot_in - lfull)
        y_true = np.abs(node - s) / row_px + 0.5
        y_rec = np.abs(node - s_new) / row_px + 0.5
        if ccd_map is None:
            ccd = np.zeros(len(pick), np.int64)
        else:
            xi = np.clip(np.round(X[pick] - ox).astype(np.int64), 0, resp.npix - 1)
            yi = np.clip(np.round(Y[pick] - oy).astype(np.int64), 0, resp.npix - 1)
            ccd = np.asarray(ccd_map)[yi, xi].astype(np.int64)
        C[pick] = C[pick] * cti.oot_factor(C[pick], ccd, y_true, y_rec)
    X[pick] = s_new * gx + q * px_ + resp.cx + ox
    Y[pick] = s_new * gy + q * py_ + resp.cy + oy
    Xi = np.round(X[pick] - ox).astype(np.int64)
    Yi = np.round(Y[pick] - oy).astype(np.int64)
    on = (Xi >= 0) & (Xi < resp.npix) & (Yi >= 0) & (Yi < resp.npix)
    live = np.zeros(len(pick), dtype=bool)
    live[on] = resp.mask[Yi[on], Xi[on]] == 1
    kept[pick[~live]] = False
    return X[kept], Y[kept], (None if C is None else C[kept]), kept


@cites("xmm_ccf_sas", "read2011")
def ellbeta_scatter(resp: CameraResponse, table: dict, xs: float, ys: float, chan,
                    off_px: tuple, rng) -> tuple:
    """ELLBETA PSF offsets [mask px] of the photons of one point source at sky mask pixel
    (xs, ys): per tabulated energy nearest to the photons' energies ``chan`` [keV], the envelope
    at the source's off-axis angle (detector frame), major axis tangential.  Photons are
    processed in the order of ``chan`` within each energy group, so the result depends only on
    the inputs and ``rng``."""
    import numpy as np

    from tngxmm.instrument import point_sources as P

    vx, vy = _vig_axis(resp)
    ddx, ddy = xs - resp.cx - off_px[0] - vx, ys - resp.cy - off_px[1] - vy
    theta = float(np.hypot(ddx, ddy)) * resp.pixsize                 # arcmin
    tang = float(np.arctan2(ddy, ddx)) + 0.5 * np.pi
    es = table["energy_kev"]
    ie = np.argmin(np.abs(np.asarray(chan, float)[:, None] - es[None, :]), axis=1)
    dx = np.empty(len(chan)); dy = np.empty(len(chan))  # noqa: E702
    for k in np.unique(ie):
        m = np.flatnonzero(ie == k)
        ox, oy = P.ellbeta_offsets(P.ellbeta_params(table, es[k], theta), len(m), rng, tang)
        dx[m], dy[m] = ox, oy
    scale = 1.0 / (resp.pixsize * 60.0)                              # arcsec -> mask px
    return dx * scale, dy * scale


# ── per-camera event generation ──────────────────────────────────────────────

@cites("xmm_simulator", "xmm_ccf_sas", "epic_pn", "epic_mos")
def _ccd_labels(inst: str, shape: tuple[int, int]) -> Any:
    """The CCD label map of a camera on the mask grid (:func:`tngxmm.instrument.active_ccds.read_ccd_maps`)."""
    from tngxmm.instrument.active_ccds import read_ccd_maps

    lab = read_ccd_maps()[inst]
    if tuple(lab.shape) != tuple(shape):
        raise ValueError(f"{inst}: CCD label map {lab.shape} does not match the mask grid {shape}")
    return lab


def camera_events(
    resp      : CameraResponse,
    eobs      ,                      # cluster photon energies (this camera's share)
    x_pix     ,                      # cluster photon SKY position [mask-px frame]
    y_pix     ,
    t_live    : float,
    nh        : float,
    sources   ,
    rng       ,
    with_skybkg: bool = True,
    with_qpb  : bool = True,
    oot_frac  : float = PN_OOT_FRAC,
    off_px    : tuple = (0.0, 0.0),
    readout_tilt_deg: float = PN_READOUT_TILT_DEG,
    agn_stats : dict | None = None,
    oot_model : str = PN_OOT_MODEL,
    oot_cti   : Any = None,
    oot_column_arcmin: float = PN_COLUMN_ARCMIN,
    qpb_scale  = None,
    agn_events : bool = True,
    donor_bkg  : dict | None = None,
    agn_flux_band_kev: tuple | None = None,
    agn_psf_table: dict | None = None,
    agn_nh_gal: float | None = None,
    qpb_maps   : dict | None = None,
    soft_ccd   : tuple | None = None,
    cluster_psf_table: dict | None = None,
    sp_map=None,
    sky: dict | None = None,
) -> tuple:
    """All events for one camera: (X, Y, chan), X/Y in SKY coordinates
    (mask-pixel frame of the pointing axis).

    oot_frac: pn out-of-time fraction of the donor's mode (MOS: ignored).
    oot_model: 'full-column' (:func:`pn_out_of_time`), the only model.
    oot_cti: the CTI line of the donor's pn mode and epoch
    (:func:`tngxmm.instrument.pn_oot_energy.cti_line`) that shifts the OoT
    energies; None leaves them unchanged.
    oot_column_arcmin: the rows an OoT event can be assigned to (the full
    column; in the pn Large/Small Window modes the rows read out).
    qpb_scale: optional per-channel factor on the QPB expectation (pn band 1,
    :func:`~tngxmm.instrument.apply_pattern_loss.pn_band1_qpb_scale`); None = unscaled.
    off_px: the camera's boresight offset [mask px]; mask lookups and off-axis
    angles use detector coordinates (sky - off), detector-born events (sky
    background, QPB) map back to sky as (pixel + off).  Cluster photons are
    already partitioned and ARF-thinned; here they get PSF, mask and RMF.
    agn_stats: if a dict, filled with per-source 'mu' and 'n' (detected on
    silicon); bookkeeping only, no draw.
    agn_events: False drops the AGN events AFTER making every AGN draw (the random stream of
    the rest of the camera is unchanged): the no-AGN twin of a view.
    donor_bkg: the camera's donor-conditioned background (``background = 'donor'``,
    :func:`~tngxmm.instrument.donor_background.donor_background_parameters`): ``qpb_band`` (band ->
    factor) multiplies the QPB expectation band by band, else ``qpb_scale`` all channels;
    ``sp_rate`` > 0 adds the residual soft protons
    (:func:`~tngxmm.instrument.donor_background.soft_proton_events`); None = the nominal
    background, with no extra random draw.
    agn_flux_band_kev: band of the AGN fluxes (None: the xmm_simulator 2-10 keV template;
    see :func:`agn_channel_cdfs`).  agn_psf_table: None scatters the AGN photons with the
    on-axis King profile; an ELLBETA table (:func:`tngxmm.instrument.point_sources.ellbeta_table`)
    draws each photon's channel first and scatters it with the envelope at that energy and the
    source's off-axis angle, major axis tangential.  agn_nh_gal: Galactic column [1e22 cm^-2]
    applied to the AGN spectra (None: none, the xmm_simulator model).
    qpb_maps: PPS band -> relative QPB density on the detector grid
    (:func:`~tngxmm.instrument.donor_background.fwc_qpb_maps`, ``qpb_spatial = 'fwc'``); None = flat.
    soft_ccd: (CCD label map of the mask grid, CCD -> band -> donor rate) of the camera's anomalous CCDs
    (``mos_soft_ccd = 'donor'``, :func:`~tngxmm.instrument.donor_soft_ccd.soft_ccd_events`), drawn after
    the QPB; None or no factor = no draw.
    cluster_psf_table: None scatters the cluster photons with the camera's on-axis King profile
    (:func:`king_offsets`, ``psf_model = 'king-onaxis'``); a KING_PARAMS grid (:func:`king_table`) scatters each
    photon with the King parameters at its energy and at the off-axis angle of its detector position before the
    PSF (``psf_model = 'king-energy-angle'``, :func:`king_offsets_energy_angle`); an ELLBETA grid
    (:func:`tngxmm.instrument.point_sources.ellbeta_table`, key ``params``) with the envelope the AGN are drawn with,
    at the photon's energy and detector position (``psf_model = 'ellbeta'``, :func:`ellbeta_offsets_energy_angle`)."""
    import numpy as np

    ox, oy = off_px
    ev_x, ev_y, ev_c = [], [], []
    if agn_stats is not None:
        nsrc = 0 if sources is None else len(sources)
        agn_stats['mu'] = np.zeros(nsrc)
        agn_stats['n'] = np.zeros(nsrc, dtype=np.int64)

    # event positions stay FLOATS (sub-pixel); integers only for mask lookup
    # (rounding would imprint a rotated-lattice moire on the 4" image)

    # 1) cluster: PSF scatter -> bounds -> mask (detector frame) -> RMF
    if len(eobs):
        if cluster_psf_table is None:
            dx, dy = king_offsets(resp, len(eobs), rng)
        elif "params" in cluster_psf_table:
            dx, dy = ellbeta_offsets_energy_angle(resp, cluster_psf_table, np.asarray(eobs, float),
                                                  np.asarray(x_pix) - ox, np.asarray(y_pix) - oy, rng)
        else:
            xi0 = np.clip(np.round(np.asarray(x_pix) - ox).astype(np.int64), 0, resp.npix - 1)
            yi0 = np.clip(np.round(np.asarray(y_pix) - oy).astype(np.int64), 0, resp.npix - 1)
            theta = np.asarray(resp.thetas)[resp.theta_idx_pix[yi0, xi0]]
            dx, dy = king_offsets_energy_angle(resp, cluster_psf_table, np.asarray(eobs, float), theta, rng)
        Xf = x_pix + dx                      # sky
        Yf = y_pix + dy
        Xi = np.round(Xf - ox).astype(np.int64)   # detector
        Yi = np.round(Yf - oy).astype(np.int64)
        ok = (Xi > 0) & (Xi < resp.npix - 1) & (Yi > 0) & (Yi < resp.npix - 1)
        Xf, Yf, Xi, Yi, E = Xf[ok], Yf[ok], Xi[ok], Yi[ok], eobs[ok]
        ok = resp.mask[Yi, Xi] == 1
        Xf, Yf, E = Xf[ok], Yf[ok], E[ok]
        if len(E):
            ev_x.append(Xf); ev_y.append(Yf)  # noqa: E702
            ev_c.append(sample_channels(resp, E, rng))

    # 2) sky background: radial expectation, Poisson per active pixel,
    #    channels from per-theta-bin RMF-convolved CDFs
    if with_skybkg:
        S = skybkg_spectrum(nh, resp.box_ene_mean, sky)            # per arcmin^2
        lam_theta = (resp.pixsize ** 2) * t_live * \
            (S[None, :] * resp.arf_theta * resp.box_ene_wid).sum(axis=1)  # (200,)
        th_act = resp.theta_idx_pix[resp.active_y, resp.active_x]
        lam_pix = lam_theta[th_act]
        if getattr(resp, "rga", None) is not None:
            # MOS RGA modulation 1 + a(E) d (d = r cos(phi - phi0)): linear in a, so the expectation adds
            # d x sum_E a(E) S A dE; the channels keep the spectrum of the off-axis bin (a varies by < 8 % in E)
            lam_a = (resp.pixsize ** 2) * t_live * \
                (S[None, :] * resp.arf_theta * np.asarray(resp.rga[0])[None, :] * resp.box_ene_wid).sum(axis=1)
            vx, vy = _vig_axis(resp)
            phi0 = np.radians(resp.rga[1])
            d = ((resp.active_x - resp.cx - vx) * np.cos(phi0)
                 + (resp.active_y - resp.cy - vy) * np.sin(phi0)) * resp.pixsize
            lam_pix = np.maximum(lam_pix + d * lam_a[th_act], 0.0)
        n_pix  = rng.poisson(lam_pix)
        tot    = int(n_pix.sum())
        if tot:
            # detector pixel + sub-pixel jitter, mapped to sky via the offset
            X = np.repeat(resp.active_x, n_pix) + rng.uniform(-0.5, 0.5, tot) + ox
            Y = np.repeat(resp.active_y, n_pix) + rng.uniform(-0.5, 0.5, tot) + oy
            th_ev = np.repeat(th_act, n_pix)
            chan  = np.empty(tot, dtype=np.float64)
            for tb in np.unique(th_ev):
                m = th_ev == tb
                cdf, ctot = channel_cdf_from_coarse(
                    resp, S * resp.arf_theta[tb])
                chan[m] = (_draw_from_cdf(cdf, resp.emb, int(m.sum()), rng)
                           if ctot > 0 else resp.emb[0])
            ev_x.append(X); ev_y.append(Y); ev_c.append(chan)  # noqa: E702

    # 3) AGN point sources: per-source Poisson counts, King scatter, RMF
    if sources is not None and len(sources):
        per_src = agn_channel_cdfs(resp, sources, t_live, off_px=off_px, flux_band_kev=agn_flux_band_kev,
                                   nh_gal=agn_nh_gal)
        for j, (xs, ys, mu, cdf) in enumerate(per_src):
            if agn_stats is not None:
                agn_stats['mu'][j] = mu
            n = rng.poisson(mu)
            if n == 0:
                continue
            if agn_psf_table is not None:
                chan_all = _draw_from_cdf(cdf, resp.emb, n, rng)
                dx, dy = ellbeta_scatter(resp, agn_psf_table, xs, ys, chan_all, off_px, rng)
            else:
                dx, dy = king_offsets(resp, n, rng)
            Xf, Yf = xs + dx, ys + dy                 # sky
            Xi = np.round(Xf - ox).astype(np.int64)   # detector
            Yi = np.round(Yf - oy).astype(np.int64)
            ok = (Xi > 0) & (Xi < resp.npix - 1) & (Yi > 0) & (Yi < resp.npix - 1)
            Xf, Yf, Xi, Yi = Xf[ok], Yf[ok], Xi[ok], Yi[ok]
            ok2 = resp.mask[Yi, Xi] == 1
            Xf, Yf = Xf[ok2], Yf[ok2]
            if agn_stats is not None:
                agn_stats['n'][j] = len(Xf)
            if len(Xf):
                if agn_psf_table is not None:
                    chan_agn = chan_all[ok][ok2]
                else:
                    chan_agn = _draw_from_cdf(cdf, resp.emb, len(Xf), rng)
                if agn_events:
                    ev_x.append(Xf); ev_y.append(Yf)  # noqa: E702
                    ev_c.append(chan_agn)

    # 3b) pn out-of-time events: every X-ray event so far (cluster, sky
    #     background, AGN) is out-of-time with probability oot_frac; the QPB is
    #     added below and is not touched (its FWC spectrum is empirical).
    #     Applied before the QPB and before the roll.
    if oot_model != PN_OOT_MODEL:
        raise ValueError(f"unknown pn out-of-time model {oot_model!r}; known: {PN_OOT_MODEL!r}")
    if resp.inst == 'PN' and oot_frac > 0:
        for k in range(len(ev_x)):
            ev_x[k], ev_y[k], ev_c[k], _ = pn_out_of_time(
                resp, ev_x[k], ev_y[k], oot_frac, rng, off_px=off_px,
                readout_tilt_deg=readout_tilt_deg, column_arcmin=oot_column_arcmin,
                chan=ev_c[k], cti=oot_cti,
                ccd_map=(None if oot_cti is None else _ccd_labels(resp.inst, resp.mask.shape)))

    # 4) QPB: flat over active pixels, spectrum from the FWC file
    if with_qpb:
        lam_qpb = qpb_lambda(resp, t_live, qpb_scale, donor_bkg)
        spec = rng.poisson(lam_qpb).astype(int)
        n    = int(spec.sum())
        if n:
            chan = np.repeat((resp.eb_min + resp.eb_max) / 2.0, spec)
            if qpb_maps is None:
                sel = rng.integers(0, len(resp.active_x), n)
            else:
                from tngxmm.instrument.donor_background import qpb_pixel_draw

                sel = qpb_pixel_draw(resp, chan, qpb_maps, rng)
            ev_x.append(resp.active_x[sel] + rng.uniform(-0.5, 0.5, n) + ox)
            ev_y.append(resp.active_y[sel] + rng.uniform(-0.5, 0.5, n) + oy)
            ev_c.append(chan)
        if soft_ccd is not None and soft_ccd[1]:
            # 4b) MOS CCDs in the anomalous state: the donor CCD's soft excess, in units of the QPB
            from tngxmm.instrument.donor_soft_ccd import soft_ccd_events

            sx, sy, sc = soft_ccd_events(resp, lam_qpb, soft_ccd[0], soft_ccd[1], t_live, rng)
            if len(sx):
                ev_x.append(sx + ox); ev_y.append(sy + oy); ev_c.append(sc)  # noqa: E702

    # 5) residual soft protons (donor background only): focused, not X-rays; part of the particle background, so they
    #    go with ``with_qpb`` (a background-free view, e.g. a clean closure cell, has none)
    if soft_protons_on(with_qpb, donor_bkg):
        from tngxmm.instrument.donor_background import soft_proton_events

        X, Y, chan = soft_proton_events(resp, t_live, donor_bkg, rng, off_px=off_px,
                                        **({} if sp_map is None else {"sp_map": sp_map}))
        if len(X):
            ev_x.append(X); ev_y.append(Y); ev_c.append(chan)  # noqa: E702

    if not ev_x:
        z = np.zeros(0)
        return z.astype(np.int64), z.astype(np.int64), z
    X, Y, C = np.concatenate(ev_x), np.concatenate(ev_y), np.concatenate(ev_c)
    if getattr(resp, 'efficiency', None) is not None:
        # 5) detector footprint (donor_footprint): every event survives with the good-silicon
        #    fraction of its detector pixel (the exposure the attitude track gave it on good
        #    pixels); one draw per event, made only under this footprint
        keep = efficiency_thinning(resp, X, Y, rng, off_px)
        X, Y, C = X[keep], Y[keep], C[keep]
    return X, Y, C


def qpb_lambda(resp: CameraResponse, t_live: float, qpb_scale=None, donor_bkg: dict | None = None):
    """Expected QPB counts per channel of one camera: the FWC spectrum of the response times the live time and the
    channel width, times the per-channel factor ``qpb_scale`` and the donor's band factors (``qpb_band``) or single
    scale (``qpb_scale``).  No random draw."""
    bw = resp.eb_max - resp.eb_min
    lam_qpb = resp.fwc_spec * t_live * bw
    if qpb_scale is not None:
        lam_qpb = lam_qpb * qpb_scale
    if donor_bkg is not None:
        if donor_bkg.get('qpb_band'):           # the donor's flat spectrum, band by band
            from tngxmm.instrument.donor_background import qpb_channel_scale

            lam_qpb = lam_qpb * qpb_channel_scale(resp, donor_bkg['qpb_band'])
        else:
            lam_qpb = lam_qpb * float(donor_bkg.get('qpb_scale', 1.0))
    return lam_qpb


@cites("esas_mosback", "esas_pnback")
def qpb_expectation(resp: CameraResponse, t_live: float, bands, qpb_scale=None, donor_bkg: dict | None = None,
                    qpb_maps: dict | None = None, off_px: tuple = (0.0, 0.0), gaps: dict | None = None) -> tuple:
    """(X, Y, {band code: expected QPB counts}) at the active pixels of one camera (sky X, Y [mask px] of the pixel
    centres): the expectation of the QPB draw of :func:`camera_events` (:func:`qpb_lambda` summed over the channels of
    each band, spread with the band's FWC density ``qpb_maps`` as ``qpb_pixel_draw`` spreads the events, uniform
    without maps), times the footprint efficiency of the pixel; ``gaps`` (band code -> PI range [keV], :func:`band_gaps`)
    leaves those channels out of the band.  No random draw."""
    import numpy as np

    from tngxmm.instrument.donor_background import PPS_BANDS

    lam = qpb_lambda(resp, t_live, qpb_scale, donor_bkg)
    mid = (resp.eb_min + resp.eb_max) / 2.0
    ax, ay = resp.active_x, resp.active_y
    eff = np.ones(len(ax))
    if getattr(resp, 'efficiency', None) is not None:
        eff = np.asarray(resp.efficiency)[ay, ax]
    out = {}
    for elo, ehi, code in bands:
        n_b = float(lam[in_band(mid, elo, ehi, (gaps or {}).get(code))].sum())
        if qpb_maps is not None and code in PPS_BANDS:
            w = np.asarray(qpb_maps[code])[ay, ax]
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        else:
            w = np.ones(len(ax))
        out[code] = n_b * w / w.sum() * eff
    return ax + off_px[0], ay + off_px[1], out


def soft_proton_shape(resp: CameraResponse, sp_map=None, off_px: tuple = (0.0, 0.0),
                      with_efficiency: bool = True, band_kev: tuple = (0.5, 12.0)) -> tuple:
    """(X, Y, weights) of the residual soft protons of one camera at the pixels of the field of view: flat, or the
    detector map ``sp_map`` (:func:`tngxmm.instrument.donor_background.spdetmap_stage`), times the footprint
    efficiency (unless ``with_efficiency`` is False); the relative shape the soft-proton draw follows in the energy
    range ``band_kev`` (only the 'flare' maps depend on it).  No random draw."""
    import numpy as np

    fov = fov_map(resp)
    ay, ax = np.nonzero(fov)
    if isinstance(sp_map, dict):                   # 'flare': the band maps weighted by the counts spectrum
        from tngxmm.instrument.donor_background import flare_broadband_shape

        w = flare_broadband_shape(resp, sp_map, *band_kev)
    else:
        w = np.ones(len(ax)) if sp_map is None else np.nan_to_num(np.asarray(sp_map)[ay, ax], nan=0.0)
    if with_efficiency and getattr(resp, 'efficiency', None) is not None:
        w = w * np.asarray(resp.efficiency)[ay, ax]
    return ax + off_px[0], ay + off_px[1], w


#: values of InstrumentOptions.pn_hard_band
PN_HARD_BANDS = ("contiguous", "pps")


def band_gaps(opt: InstrumentOptions, inst: str) -> dict[str, tuple[float, float]]:
    """Band code -> PI range [keV] left out of that band's image: the pn 4.5-12 keV band without 7.8-8.2 keV with
    ``pn_hard_band = 'pps'`` (:data:`tngxmm.instrument.donor_corner_qpb.PPS_BAND_GAPS_EV`), else none."""
    from tngxmm.instrument.donor_corner_qpb import PPS_BAND_GAPS_EV

    if opt.pn_hard_band != "pps":
        return {}
    return {b: (lo / 1000.0, hi / 1000.0) for (cam, b), (lo, hi) in PPS_BAND_GAPS_EV.items() if cam == inst}


def in_band(chan, elo: float, ehi: float, gap: tuple | None = None):
    """Channel mask of a band [elo, ehi) keV without the PI range ``gap`` [keV] (None: no gap)."""
    sel = (chan >= elo) & (chan < ehi)
    if gap is not None:
        sel &= ~((chan >= gap[0]) & (chan < gap[1]))
    return sel


def check_pn_gap_level(opt: InstrumentOptions, background: str, corner_qpb: dict | None) -> None:
    """``pn_hard_band = 'pps'`` with a donor background needs ``qpb_level = 'corners'``: the pn FWC spectrum is drawn
    at the donor's corner scale (or at the FWC level when the donor has no pn corner level) and binned in the PPS PI
    ranges, with no further factor.  The level of the data package's image decomposition is referred to the contiguous
    band and is refused (the author's rule: no calibration factor in this path)."""
    if opt.pn_hard_band != "pps" or background != "donor":
        return
    if opt.qpb_level == "in-fov":
        raise ValueError("pn_hard_band 'pps' with a donor background needs qpb_level 'corners'; the "
                         "image-decomposition level is referred to the contiguous band and is not rescaled")


#: values of the ESAS particle-background options of InstrumentOptions
QPB_UNITS = ("none", "corners")
SOFT_PROTON_MAPS = ("flat", "spdetmap", "flare")
QPB_MODEL_IMAGES = ("off", "on", "on+alt")


def check_qpb_model_options(opt: InstrumentOptions, qpb_unit_factors: dict | None, insts) -> None:
    """Validate ``qpb_units``, ``soft_proton_map`` and ``qpb_model_images``; ``qpb_units = 'corners'`` needs the
    donor's unit factors for every camera of the view."""
    if opt.pn_hard_band not in PN_HARD_BANDS:
        raise ValueError(f"unknown pn_hard_band {opt.pn_hard_band!r}; known: {PN_HARD_BANDS}")
    if opt.qpb_units not in QPB_UNITS:
        raise ValueError(f"unknown qpb_units {opt.qpb_units!r}; known: {QPB_UNITS}")
    if opt.soft_proton_map not in SOFT_PROTON_MAPS:
        raise ValueError(f"unknown soft_proton_map {opt.soft_proton_map!r}; known: {SOFT_PROTON_MAPS}")
    if opt.soft_proton_template and opt.soft_proton_template not in SOFT_PROTON_MAPS:
        raise ValueError(f"unknown soft_proton_template {opt.soft_proton_template!r}; known: {SOFT_PROTON_MAPS}")
    if opt.qpb_model_images not in QPB_MODEL_IMAGES:
        raise ValueError(f"unknown qpb_model_images {opt.qpb_model_images!r}; known: {QPB_MODEL_IMAGES}")
    if opt.qpb_units == "corners":
        missing = [i for i in insts if not (qpb_unit_factors or {}).get(i)]
        if missing:
            raise ValueError(f"qpb_units 'corners' needs the donor's unit factors (qpb_unit_factors) for {missing}")


def camera_qpb_maps(opt: InstrumentOptions, inst: str, rp: CameraResponse, ccf_dir, qpb_unit_factors: dict | None,
                    which: str = "latest", half: int | None = None) -> tuple:
    """(QPB density maps or None, soft-proton detector map or None) of one camera on the mask grid.

    The maps are the FWC density maps (``qpb_spatial = 'fwc'``; ``which`` = 'latest' | 'oldest' FWC set) or flat; with
    ``qpb_units = 'corners'`` each CCD is multiplied by the donor's corner level of its CCD (MOS) or quadrant (pn)
    relative to the camera (:func:`tngxmm.instrument.donor_background.scale_maps_by_ccd`, the camera's band totals
    unchanged).  None when both are off (the nominal uniform draw).  The soft-proton map is the ESAS detector map with
    ``soft_proton_map = 'spdetmap'``, or the per-band flare maps of the camera's filter with ``'flare'``
    (:func:`tngxmm.instrument.donor_background.flare_maps_stage`).  ``half``: the FWC half of the density maps
    (:data:`tngxmm.instrument.donor_background.FWC_HALVES`; the draw uses ``FWC_GENERATOR_HALF``, the fit templates
    ``FWC_TEMPLATE_HALF``)."""
    import numpy as np

    from tngxmm.instrument import donor_background as DB

    qmaps = None
    if opt.qpb_spatial == "fwc" or opt.qpb_units == "corners":
        if opt.qpb_spatial == "fwc":
            qmaps = DB.fwc_qpb_maps(ccf_dir, inst, rp.mask.shape, rp.pixsize, ccds=rp.ccds, which=which,
                                    gaps=band_gaps(opt, inst) or None, half=half)
        else:
            qmaps = {k: np.ones(rp.mask.shape) for k in DB.PPS_BANDS}
        if opt.qpb_units == "corners":
            qmaps = DB.scale_maps_by_ccd(qmaps, _ccd_labels(inst, rp.mask.shape), qpb_unit_factors[inst])
    sp_map = None
    if opt.soft_proton_map == "spdetmap":
        sp_map = DB.spdetmap_stage(ccf_dir, inst, rp.mask.shape, rp.pixsize)
    elif opt.soft_proton_map == "flare":
        sp_map = DB.flare_maps_stage(ccf_dir, inst, rp.filter, rp.mask.shape, rp.pixsize)
    return qmaps, sp_map


def camera_model_images(opt: InstrumentOptions, inst: str, rp: CameraResponse, t_live: float, bands, qpb_scale,
                        donor_bkg: dict | None, qmaps, sp_map, off_px: tuple, edges, do_rot: bool, ca: float,
                        sa: float, ccf_dir, qpb_unit_factors: dict | None, corner_counts: dict | None = None,
                        rng=None) -> dict:
    """The model images of one camera that the fit's ESAS background reads (``qpb_model_images``), per band code:

    * ``QPBEXP``: the QPB template of the fit, built as for a real exposure: the expected QPB counts per image pixel
      (:func:`qpb_expectation` carried to the image like the events, the pn rows at RAWY <= 12 dropped) with the
      levels of the mock's own corner measurement (:func:`tngxmm.instrument.donor_corner_qpb.simulated_corner_measurement`:
      Poisson counts with the donor's statistics, ``corner_counts``, drawn from ``rng``) and the spatial form of the
      FWC half that the draw did not use (``FWC_TEMPLATE_HALF``).  It is not the QPB the stage drew from: like a real
      template it carries the corners' counting error and its own calibration noise;
    * ``SPMAP``: the soft-proton shape of the band (:func:`soft_proton_shape` without the footprint efficiency, which
      the fit's exposure map carries), normalised to mean 1 over its positive pixels; band-dependent only with the
      'flare' maps;
    * ``QPBALT`` (``'on+alt'``): ``QPBEXP`` (the same simulated corner levels) with the oldest FWC set as the spatial
      template, for the wrong-template test;
    * ``CCDMAP``: the CCD number of each image pixel (the CCD covering most of it; 0 outside the field of view), from
      the CCD label map of the mask grid (:func:`_ccd_labels`) carried to the image like the events; the fit leaves
      out the MOS CCDs in the anomalous state with it, as for a real image.

    Empty when ``qpb_model_images = 'off'``."""
    import numpy as np

    if opt.qpb_model_images == "off":
        return {}
    from tngxmm.instrument import donor_corner_qpb as CQ
    from tngxmm.instrument.donor_background import FWC_TEMPLATE_HALF

    cut = inst == "PN" and opt.pn_edge_rows == "drop"
    out: dict = {"QPBEXP": {}, "SPMAP": {}}
    counts = (corner_counts or {}).get(inst)
    meas = CQ.simulated_corner_measurement(counts, float((counts or {}).get("live_ratio", 1.0)), rng) \
        if counts else {"band": {}, "unit": {}}
    t_bkg = dict(donor_bkg or {"qpb_scale": 1.0, "sp_rate": 0.0})
    if t_bkg.get("qpb_band") and meas["band"]:
        t_bkg["qpb_band"] = {b: v * meas["band"].get(b, 1.0) for b, v in t_bkg["qpb_band"].items()}
        t_bkg["qpb_scale"] = float(t_bkg["qpb_band"].get("5000", t_bkg.get("qpb_scale", 1.0)))
    t_units = qpb_unit_factors
    if meas["unit"] and qpb_unit_factors and qpb_unit_factors.get(inst):
        units = CQ.PN_QUADRANTS if inst == "PN" else {str(k): (k,) for k in range(1, 8)}
        t_units = {**qpb_unit_factors, inst: {int(c): meas["unit"].get(u, 1.0) for u, cs in units.items() for c in cs}}
    t_maps, _ = camera_qpb_maps(opt, inst, rp, ccf_dir, t_units, half=FWC_TEMPLATE_HALF)
    X, Y, per_band = qpb_expectation(rp, t_live, bands, qpb_scale, t_bkg, t_maps, off_px, band_gaps(opt, inst))
    for code, w in per_band.items():
        out["QPBEXP"][code] = expectation_image(rp, X, Y, w, off_px, edges, do_rot, ca, sa, pn_rawy_cut=cut)
    s_cache: dict = {}
    for lo, hi, code in bands:
        key = (lo, hi) if isinstance(sp_map, dict) else None
        if key not in s_cache:
            sx, sy, sw = soft_proton_shape(rp, sp_map, off_px, with_efficiency=False, band_kev=(lo, hi))
            s_img = expectation_image(rp, sx, sy, sw, off_px, edges, do_rot, ca, sa, pn_rawy_cut=cut)
            pos = s_img > 0
            s_cache[key] = s_img / s_img[pos].mean() if pos.any() else s_img
        out["SPMAP"][code] = s_cache[key]
    lab = np.asarray(_ccd_labels(inst, rp.mask.shape))[rp.active_y, rp.active_x]
    cover = []
    ccds = [int(c) for c in np.unique(lab) if c > 0]
    for c in ccds:
        cover.append(expectation_image(rp, rp.active_x + off_px[0], rp.active_y + off_px[1], (lab == c).astype(float),
                                       off_px, edges, do_rot, ca, sa))
    if ccds:
        stack = np.stack(cover)
        ccd_img = np.where(stack.max(axis=0) > 0, np.asarray(ccds)[np.argmax(stack, axis=0)], 0)
    else:
        ccd_img = np.zeros((len(edges) - 1, len(edges) - 1))
    out["CCDMAP"] = {code: ccd_img for _, _, code in bands}
    if opt.qpb_model_images == "on+alt":
        # the same simulated corner measurement as QPBEXP: only the FWC set (the spatial form) differs
        alt, _ = camera_qpb_maps(opt, inst, rp, ccf_dir, t_units, which="oldest")
        X, Y, per_band = qpb_expectation(rp, t_live, bands, qpb_scale, t_bkg, alt, off_px, band_gaps(opt, inst))
        out["QPBALT"] = {code: expectation_image(rp, X, Y, w, off_px, edges, do_rot, ca, sa, pn_rawy_cut=cut)
                         for code, w in per_band.items()}
    return {k: {c: np.asarray(v, dtype=np.int16 if k == "CCDMAP" else np.float32) for c, v in d.items()}
            for k, d in out.items()}


def catalogue_position_parameters(par: Any) -> dict[str, dict[str, Any]]:
    """The catalogue position a photon file's ``parameters`` group carries
    (:func:`tngxmm.sim.generate_photons.generate_photons`, :mod:`tngxmm.sim.catalogue_centre`), as header keys:
    ``{'SZ': {deg, SZRULE, SZTHETA, SZBEAM[, SZSNR, SZNOISE], CATPOSV, CPFLAG}}`` for an SZ position,
    ``{'XR': {deg, XRSMOOTH, CATPOSV, CPFLAG}}`` for an X-ray position (``deg`` the position in the frame of
    ``xsky``/``ysky``; ``CATPOSV`` the rule version, ``CPFLAG`` the position's flag); empty for a file without one (a
    view made before the rule, whose centre search then refuses it).  The SZ centroid of earlier files
    (``sz_centroid_deg``) is another rule and is not read."""
    import numpy as np

    def text(v):
        v = v[()]
        return v.decode() if isinstance(v, bytes) else str(v)

    common: dict[str, Any] = {}
    if 'catpos_version' in par:
        common = dict(CATPOSV=int(par['catpos_version'][()]), CPFLAG=text(par['catpos_flag']))
    out: dict[str, dict[str, Any]] = {}
    if 'sz_centre_deg' in par:
        out['SZ'] = dict(deg=np.asarray(par['sz_centre_deg'][()], dtype=np.float64), SZRULE=text(par['sz_rule']),
                         SZTHETA=float(par['sz_theta500_arcmin'][()]), SZBEAM=float(par['sz_beam_fwhm_arcmin'][()]),
                         **({'SZSNR': float(par['sz_snr'][()])} if 'sz_snr' in par else {}),
                         **({'SZNOISE': float(par['sz_noise_uk_arcmin'][()])} if 'sz_noise_uk_arcmin' in par else {}),
                         **common)
    if 'xray_centre_deg' in par:
        out['XR'] = dict(deg=np.asarray(par['xray_centre_deg'][()], dtype=np.float64),
                         XRSMOOTH=float(par['xray_smoothing_arcsec'][()]), **common)
    return out


def output_pixel(sky_deg, mask_pix_arcmin: float, dx_px: float, dy_px: float, ca: float, sa: float, wpix: float,
                 n4: int) -> tuple[float, float]:
    """0-indexed pixel of the merged output grid of a position ``sky_deg`` (x, y offsets [deg] in the frame of the
    photon list's ``xsky`` / ``ysky``), carried as the photons are: onto the mask grid (``mask_pix_arcmin`` per
    pixel), displaced by the placement (``dx_px``, ``dy_px``), rolled (cos, sin = ``ca``, ``sa``) and binned on the
    output grid (``wpix`` mask pixels per output pixel, ``n4`` pixels).  The offset (0, 0) is ``CLUSTX/CLUSTY``."""
    sdx = float(sky_deg[0]) * 60.0 / float(mask_pix_arcmin) + dx_px
    sdy = float(sky_deg[1]) * 60.0 / float(mask_pix_arcmin) + dy_px
    return (float((sdx * ca - sdy * sa) / wpix + n4 / 2.0 - 0.5),
            float((sdx * sa + sdy * ca) / wpix + n4 / 2.0 - 0.5))


def expectation_image(resp: CameraResponse, X, Y, w, off_px: tuple, edges, do_rot: bool, ca: float, sa: float,
                      pn_rawy_cut: bool = False, readout_tilt_deg: float = PN_READOUT_TILT_DEG):
    """An expectation carried to the camera's image grid like its events: the pixels outside the field of view
    (:func:`drop_out_of_fov`) and, for the pn, at RAWY <= PN_BAND1_RAWY_MIN dropped, the roll applied, weights ``w``
    histogrammed on ``edges``; each pixel split into 2 x 2 sub-positions (the events are spread +-0.5 px)."""
    import numpy as np

    idx = np.arange(len(X), dtype=np.float64)
    out = np.zeros((len(edges) - 1, len(edges) - 1))
    for dx in (-0.25, 0.25):
        for dy in (-0.25, 0.25):
            Xk, Yk, ik = drop_out_of_fov(resp, np.asarray(X, float) + dx, np.asarray(Y, float) + dy, idx, off_px)
            if pn_rawy_cut:
                keep = pn_rawy(resp, Xk, Yk, off_px, readout_tilt_deg) > PN_BAND1_RAWY_MIN
                Xk, Yk, ik = Xk[keep], Yk[keep], ik[keep]
            if do_rot:
                xr = resp.cx + (Xk - resp.cx) * ca - (Yk - resp.cy) * sa
                yr = resp.cy + (Xk - resp.cx) * sa + (Yk - resp.cy) * ca
            else:
                xr, yr = Xk, Yk
            h, _, _ = np.histogram2d(yr, xr, bins=[edges, edges], weights=np.asarray(w)[ik.astype(int)] / 4.0)
            out += h
    return out


@cites("xmm_sas_eexpmap")
def efficiency_thinning(resp: CameraResponse, X, Y, rng, off_px: tuple = (0.0, 0.0)):
    """Boolean keep-mask of the events at sky X, Y [mask px]: ``u < efficiency`` at the
    detector pixel ``round(sky - off)`` (0 off the grid).  One uniform draw per event."""
    import numpy as np

    eff = resp.efficiency
    xd = np.round(np.asarray(X, dtype=np.float64) - off_px[0]).astype(np.int64)
    yd = np.round(np.asarray(Y, dtype=np.float64) - off_px[1]).astype(np.int64)
    ok = (xd >= 0) & (xd < eff.shape[1]) & (yd >= 0) & (yd < eff.shape[0])
    p = np.zeros(len(xd))
    p[ok] = eff[yd[ok], xd[ok]]
    return rng.random(len(xd)) < p


# ── pn band 1: single-pixel events, RAWY > 12 ───────────────────────────────

@cites("xmm_mission", "epic_pn")
def pn_rawy(resp: CameraResponse, X, Y, off_px: tuple = (0.0, 0.0),
            readout_tilt_deg: float = PN_READOUT_TILT_DEG):
    """Row number RAWY (1 = the row next to the readout node of the event's CCD column) of
    events at sky positions X, Y [mask px]; ``inf`` where the column holds no active
    silicon.  Same readout-aligned frame as :func:`pn_out_of_time`: each half of the
    camera is read out through the node at the outer end of its columns, a row is
    ``PN_ROW_ARCSEC`` long."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ox, oy = off_px
    a = np.deg2rad(readout_tilt_deg)
    gx, gy = np.sin(a), np.cos(a)
    px_, py_ = np.cos(a), -np.sin(a)
    dx, dy = X - ox - resp.cx, Y - oy - resp.cy
    s = dx * gx + dy * gy
    q = dx * px_ + dy * py_
    j = np.clip(np.round(q + resp.cx).astype(np.int64), 0, resp.npix - 1)
    top = s >= resp.pn_half_s
    hi, lo = resp.pn_top_hi[j], resp.pn_bot_lo[j]
    edge = np.where(top, hi + 0.5, lo - 0.5).astype(np.float64) - resp.cy
    dist = np.where(top, edge - s, s - edge)
    row_px = (PN_ROW_ARCSEC / 60.0) / resp.pixsize
    rawy = np.floor(np.maximum(dist, 0.0) / row_px) + 1.0
    return np.where(np.where(top, hi, lo) >= 0, rawy, np.inf)


def check_camera_partition(p_cum: Any, t_gen: float, a_gen: float, cameras_s: dict[str, float]) -> None:
    """Raise ``ValueError`` when the camera-assignment probabilities of the parent photons sum to more than 1
    (``p_cum``: the cumulative sum over the cameras per photon, ARF_c / A_gen x t_c / t_gen).  Then the cameras
    would share photons (one photon kept by two cameras is two events from one) and a camera could be short of
    them.  The photon list is sized for the longest camera time of the view
    (:func:`tngxmm.sim.generate_view.draw_observing_conditions`), so this is a list built for shorter live times
    than the ones asked for, or an ``area_cm2`` below the summed effective areas."""
    import numpy as np

    p = np.asarray(p_cum, float)
    if p.size and float(p.max()) > 1.0 + P_CUM_TOL:
        raise ValueError(
            f'camera-assignment probability sum reaches {float(p.max()):.3f} > 1: the photon list '
            f'(exp_time {t_gen:g} s, area {a_gen:g} cm2) cannot supply the cameras\' live times '
            f'{ {k: float(v) for k, v in cameras_s.items()} } s')


@cites("xmm_ccf_sas", "epic_pn")
def pn_band1_single_pixel(resp: CameraResponse, X, Y, chan, rng, ccf_dir: str, pn_mode: str,
                          off_px: tuple = (0.0, 0.0),
                          readout_tilt_deg: float = PN_READOUT_TILT_DEG, rawy_cut: bool = True):
    """Boolean mask of the pn events that the PPS band-1 image keeps (``PATTERN == 0 &&
    RAWY > 12``), for the events (X, Y [sky mask px], ``chan`` [keV]) of the band.

    The mocks carry no event patterns: an event is kept with the CCF single-pixel fraction
    FRAC_S / FRAC_SD at its energy (the events already stand for PATTERN <= 4,
    :func:`~tngxmm.instrument.apply_pattern_loss.pn_single_pixel_keep`) and is dropped if its
    RAWY is <= ``PN_BAND1_RAWY_MIN`` (``rawy_cut``; with ``pn_edge_rows = 'footprint'`` the footprint already ends at
    the readout rows and no row is tested).  One uniform draw per event, in event order.
    Returns (keep, fraction of the events with RAWY > 12 that are single-pixel)."""
    import numpy as np

    from tngxmm.instrument.apply_pattern_loss import pn_single_pixel_keep

    n = len(chan)
    u = rng.random(n)
    p = pn_single_pixel_keep(ccf_dir, chan, pn_mode, sd_applied=resp.pattern_eff is not None)
    single = u < p
    far = (pn_rawy(resp, X, Y, off_px, readout_tilt_deg) > PN_BAND1_RAWY_MIN) if rawy_cut else np.ones(n, bool)
    frac = float(single[far].mean()) if far.any() else 1.0
    return single & far, frac


# ── top level: one view, all cameras ────────────────────────────────────────

def _band_mean_pattern(resp: CameraResponse, elo: float, ehi: float) -> float:
    import numpy as np

    if resp.pattern_eff is None:
        return 1.0
    m = (resp.box_ene_mean >= elo) & (resp.box_ene_mean < ehi)
    wid = np.asarray(resp.box_ene_wid, dtype=np.float64)
    if wid.ndim:                       # the RMF's grid (arf_model 'fit-linear') is not uniform
        return float(np.average(resp.pattern_eff[m], weights=wid[m]))
    return float(np.mean(resp.pattern_eff[m]))


def _xmmsim_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("xmm_simulator")
    except Exception:  # noqa: BLE001
        return None


@cites("xmm_simulator", "xmm_ccf_sas", "xmm_mission", "epic_pn", "epic_mos", "moretti2003")
def simulate_instrument(
    evt_file    : str | Path,
    out_dir     : str | Path,
    cameras_s   : dict[str, float],
    nh          : float,
    posangle_deg: float,
    rng         ,
    options     : InstrumentOptions,
    verbose     : bool = True,
    pn_oot_frac : float = PN_OOT_FRAC,
    per_camera_dir: str | Path | None = None,
    mockid      : str = '',
    dither_arcmin: float = 0.0,
    max_dither_arcmin: float | None = None,
    filters     : dict[str, str] | None = None,
    pn_mode     : str | None = None,
    context     : dict[str, Any] | None = None,
    core_offset_arcmin: tuple[float, float] | None = None,
    placement_info: dict[str, Any] | None = None,
    camera_submodes: dict[str, str] | None = None,
    camera_ccds: dict[str, Any] | None = None,
    donor_background: dict[str, dict[str, float]] | None = None,
    donor_obsid: str | None = None,
    nh_total_ratio: float | None = None,
    mos_soft_ccd: dict | None = None,
    donor_sp: dict | None = None,
    corner_qpb: dict | None = None,
    qpb_unit_factors: dict | None = None,
    donor_sky: dict | None = None,
    corner_counts: dict | None = None,
    sp_exclude_r500_arcmin: float | None = None,
) -> Path:
    """Produce ``image_merged.fits`` (integer counts, 4"/px) and
    ``image_merged_cov.fits`` from an ideal (absorbed) event file for the
    given per-camera live times; with ``per_camera_dir`` also the per-camera
    IMAGE/EXPMAP products (5 bands with the default ``product_bands='pps-1-5'``).

    ``filters`` (camera -> Thin1|Thin2|Medium|Thick) and ``pn_mode`` are the
    donor's filters and pn mode; ``context`` carries the view-level provenance
    keys of the headers (VIEW_ID, DONORID, ...).  ``core_offset_arcmin``: the
    donor's core offset (mask frame), required by the ``donor-offset``
    placement (no random draw); the offset is shrunk to ``max_dither_arcmin``.
    ``dither_arcmin`` is accepted for the callers' sake and not used.
    ``placement_info``, if a dict, receives the applied placement (rule,
    requested and applied offset [arcmin], cap flag).  ``camera_submodes``
    (camera -> donor SUBMODE; PN falls back to ``pn_mode``): the per-camera
    SUBMODE headers and the readout window each camera images
    (:mod:`tngxmm.instrument.readout_windows`).  ``camera_ccds`` (camera ->
    ``{'ccds': (...), 'source': ...}``, :func:`~tngxmm.instrument.active_ccds.
    donor_camera_ccds`): the CCDs each camera images
    (:mod:`tngxmm.instrument.active_ccds`), applied before the windows.
    ``donor_background`` (camera -> parameters, :func:`~tngxmm.instrument.donor_background.
    donor_background_parameters`): required by ``options.background == 'donor'`` and by ``live_time == 'donor-expmap'`` (where it replaces ``cameras_s``
    even under ``'nominal'``); otherwise ignored; the QPB level and the residual soft protons of each camera.  ``donor_obsid``:
    the donor whose PPS exposure maps give the detector footprint under
    ``options.detector_footprint = 'donor-pps'`` (:mod:`tngxmm.instrument.donor_footprint`)."""
    import h5py
    import numpy as np
    from astropy.io import fits

    from tngxmm.instrument import write_camera_products as W
    from tngxmm.instrument.apply_pattern_loss import PATTERN_SELECTION
    from tngxmm.instrument.load_responses import ensure_atomdb
    from tngxmm.instrument.active_ccds import apply_active_ccds, format_ccds
    from tngxmm.instrument.readout_windows import (
        PN_LW_ROWS,
        PN_PIXEL_ARCSEC,
        PN_SW_ROWS,
        apply_readout_windows,
        read_out_area_arcmin2,
    )
    from tngxmm.instrument.make_exposure_maps import (
        EXPFIX_TAG,
        band_vignetting_energies,
        make_exposure_maps,
    )
    from tngxmm.instrument.sas_headers import add_sas_keys, stringify_wide_ints
    from tngxmm.instrument.pn_oot_energy import OOT_ENERGY_RULES, cti_line
    from tngxmm.instrument.select_pn_mode import select_pn_mode

    ensure_atomdb()
    opt = options
    for name, value in _SUPPORTED.items():
        if getattr(opt, name) != value:
            raise ValueError(f"unsupported {name} {getattr(opt, name)!r}; the stage implements {value!r}")
    if opt.pn_edge_rows not in PN_EDGE_ROWS:
        raise ValueError(f"unknown pn_edge_rows {opt.pn_edge_rows!r}; known: {PN_EDGE_ROWS}")
    from tngxmm.instrument.make_exposure_maps import VIG_ENERGY_RULES
    if opt.vig_energy_rule not in VIG_ENERGY_RULES:
        raise ValueError(f"unknown vig_energy_rule {opt.vig_energy_rule!r}; known: {VIG_ENERGY_RULES}")
    ccf_dir = opt.ccf_dir
    offsets = opt.camera_offsets
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)  # noqa: E702

    with h5py.File(str(evt_file), 'r') as f:
        eobs  = f['data']['eobs'][()].astype(np.float64)
        xsky  = f['data']['xsky'][()].astype(np.float64)   # deg
        ysky  = f['data']['ysky'][()].astype(np.float64)
        a_gen = float(f['parameters']['area'][()])
        t_gen = float(f['parameters']['exp_time'][()])
        abs_tab = (f['parameters']['absorb_abund'][()] if 'absorb_abund' in f['parameters'] else b'lodd')
        # the view's catalogue position before its measurement error (sim.catalogue_centre) in the frame of xsky /
        # ysky [deg], with its rule: an SZ position (sz_*) or an X-ray position (xray_*)
        catpos = catalogue_position_parameters(f['parameters'])
    from tngxmm.instrument import galactic_column as GC
    if opt.galactic_nh not in GC.GALACTIC_NH:
        raise ValueError(f"unknown galactic_nh {opt.galactic_nh!r}; known: {GC.GALACTIC_NH}")
    nh_ratio = 1.0
    if opt.galactic_nh == "willingale13":
        if not (nh_total_ratio and nh_total_ratio >= 1.0):
            raise ValueError("galactic_nh 'willingale13' needs nh_total_ratio >= 1 (instrument.galactic_column)")
        nh_ratio = float(nh_total_ratio)
        # the missing molecular column on the stored (atomic-column absorbed) list; own stream, keyed by
        # the list size and column, so the stage's main random stream is not consumed
        gen = np.random.default_rng([len(eobs), int(round(float(nh) * 1e9)), 20130843])
        k_abs = GC.extra_absorption_keep(eobs, float(nh), nh_ratio, gen,
                                         abund_table=(abs_tab.decode() if isinstance(abs_tab, bytes) else str(abs_tab)))
        eobs, xsky, ysky = eobs[k_abs], xsky[k_abs], ysky[k_abs]
        nh = float(nh) * nh_ratio
        if verbose:
            print(f'    galactic N_H x {nh_ratio:.3f} (Willingale 2013): {int((~k_abs).sum()):,} cluster photons absorbed')

    insts = [i for i in _INSTRUMENTS if i in cameras_s]
    if not insts:
        raise ValueError(f'cameras_s has no known camera: {cameras_s}')

    from tngxmm.instrument.apply_pattern_loss import apply_pattern_loss
    pattern = apply_pattern_loss(ccf_dir, pn_mode=pn_mode or "PrimeFullWindow", rule=opt.pattern_rule)
    resp_filter: dict[str, str | None] = {}
    for i in insts:
        if not filters or i not in filters:
            raise ValueError(f"per-donor filter requested but no filter for {i}")
        if filters[i] not in DONOR_FILTERS:
            raise ValueError(f"unknown filter {filters[i]!r} for {i}; known: {DONOR_FILTERS}")
        resp_filter[i] = filters[i]
    # instrument.ccf_selection 'valid-at-date': the CCF files SAS selects for the conditions donor's date
    from tngxmm.instrument.ccf_select import rule_of as _ccf_rule
    ccf_kw: dict[str, Any] = {}
    if _ccf_rule(opt.ccf_selection) != "highest-issue":
        ccf_kw = dict(ccf_selection=opt.ccf_selection,
                      obs_date=((context or {}).get("_sas_donor") or {}).get("date_obs"))
    resps = {i: get_response(i, ccf_dir, filter=resp_filter[i], mos2_mirror=opt.mos2_mirror,
                             pattern=pattern, readout_tilt_deg=opt.pn_readout_tilt_deg,
                             energy_grid_max_kev=opt.arf_energy_max_kev,
                             **({} if opt.vignetting_beyond == "zero"
                                else {"vignetting_beyond": opt.vignetting_beyond}), **ccf_kw,
                             **({} if opt.arf_model == "xmm_simulator" else {"arf_model": opt.arf_model}))
             for i in insts}
    from tngxmm.instrument import obs_responses as OR
    if opt.responses not in OR.GENERATOR_RESPONSES:
        raise ValueError(f"unknown responses {opt.responses!r}; known: {OR.GENERATOR_RESPONSES}")
    if opt.responses != "packaged" and opt.pattern_rule != "sas":
        raise ValueError(f"responses {opt.responses!r} need pattern_efficiency 'sas': the donor's pn ARF is the "
                         "PATTERN<=4 ARF of arfgen, which carries FRAC_SD")
    resps = apply_vignetting_calibration(resps, opt)
    # donor submodes per camera (PN: pn_mode unless given explicitly)
    submodes: dict[str, str | None] = {i: None for i in insts}
    submodes.update({k: v for k, v in (camera_submodes or {}).items() if k in submodes})
    if 'PN' in submodes and submodes['PN'] is None:
        submodes['PN'] = pn_mode
    if opt.agn not in ("on", "off"):
        raise ValueError(f"unknown agn {opt.agn!r}; known: ('on', 'off')")
    if opt.background not in ("nominal", "donor"):
        raise ValueError(f"unknown background {opt.background!r}; known: ('nominal', 'donor')")
    if opt.qpb_spatial not in ("flat", "fwc"):
        raise ValueError(f"unknown qpb_spatial {opt.qpb_spatial!r}; known: ('flat', 'fwc')")
    if opt.live_time not in ("package", "donor-expmap"):
        raise ValueError(f"unknown live_time {opt.live_time!r}; known: ('package', 'donor-expmap')")
    if (opt.background == "donor" or opt.live_time == "donor-expmap") and not donor_background:
        raise ValueError(f"background {opt.background!r} / live_time {opt.live_time!r} need the donor's "
                         "background parameters (donor_background)")
    if opt.live_time == "donor-expmap":
        from tngxmm.instrument.donor_background import donor_live_times

        cameras_s = donor_live_times(cameras_s, donor_background)
    bkg_by_cam = {i: (dict(donor_background.get(i) or {'qpb_scale': 1.0, 'sp_rate': 0.0})
                      if opt.background == "donor" else None) for i in insts}
    from tngxmm.instrument import donor_inout_sp as SPM
    if opt.soft_protons not in SPM.SOFT_PROTONS:
        raise ValueError(f"unknown soft_protons {opt.soft_protons!r}; known: {SPM.SOFT_PROTONS}")
    if opt.sp_rate_unmeasured not in SPM.SP_RATE_UNMEASURED:
        raise ValueError(f"unknown sp_rate_unmeasured {opt.sp_rate_unmeasured!r}; known: {SPM.SP_RATE_UNMEASURED}")
    if opt.soft_protons != "off":
        if opt.background != "donor" or donor_sp is None:
            raise ValueError("soft_protons 'inout' needs background 'donor' and the donor's SP (donor_sp)")
        bkg_by_cam = SPM.apply_to_cameras(bkg_by_cam, donor_sp, insts)
    from tngxmm.instrument import donor_corner_qpb as CQ
    if opt.qpb_level not in CQ.QPB_LEVELS:
        raise ValueError(f"unknown qpb_level {opt.qpb_level!r}; known: {CQ.QPB_LEVELS}")
    if opt.background == "donor" and opt.qpb_level == "in-fov":
        raise ValueError("background 'donor' needs qpb_level 'corners': the in-FOV image-decomposition level of the "
                         "donor background table carried a replay calibration and is not a QPB level")
    qpb_source = {}
    if opt.qpb_level != "in-fov":
        if opt.background != "donor" or corner_qpb is None:
            raise ValueError("qpb_level 'corners' needs background 'donor' and the donor's corner scales (corner_qpb)")
        for i in insts:
            bkg_by_cam[i], qpb_source[i] = CQ.camera_qpb_level(bkg_by_cam[i], corner_qpb.get(i))
    check_qpb_model_options(opt, qpb_unit_factors, insts)
    if 'PN' in insts:
        check_pn_gap_level(opt, opt.background, corner_qpb)
    check_agn_switches(opt)
    check_psf_model(opt)
    if opt.sky_model not in SKY_MODELS:
        raise ValueError(f"unknown sky_model {opt.sky_model!r}; known: {SKY_MODELS}")
    if (opt.sky_model == "donor-rass") != (donor_sky is not None):
        raise ValueError("sky_model 'donor-rass' needs the donor's RASS sky (donor_sky) and no other model takes one")
    from tngxmm.instrument import point_sources as PS
    from tngxmm.instrument import donor_soft_ccd as SC
    if opt.mos_soft_ccd not in SC.MOS_SOFT_CCD:
        raise ValueError(f"unknown mos_soft_ccd {opt.mos_soft_ccd!r}; known: {SC.MOS_SOFT_CCD}")
    if opt.mos_soft_ccd != "off" and mos_soft_ccd is None:
        raise ValueError("mos_soft_ccd 'donor' needs the donor's factors (instrument.donor_soft_ccd)")
    agn_band = (None if opt.agn_lognlogs == "moretti03-hard"
                else PS.MORETTI03[PS.AGN_LOGNLOGS[opt.agn_lognlogs]]["band_kev"])
    bands_out = W.PRODUCT_BANDS[opt.product_bands]
    base_resps = dict(resps)
    camera_ccds = {k: (dict(v) if isinstance(v, dict) else {'ccds': tuple(v), 'source': 'caller'})
                   for k, v in (camera_ccds or {}).items()}
    resps = apply_active_ccds(resps, camera_ccds, opt.active_ccds)
    if verbose:
        for i in insts:
            print(f'    {i}: CCDs {format_ccds(camera_ccds[i]["ccds"])} '
                  f'({camera_ccds[i].get("source", "?")}; {int(resps[i].mask.sum()):,} px)')
    resps = apply_readout_windows(resps, submodes, opt.pn_readout_tilt_deg)
    footprint_src: dict[str, str] = {}
    if opt.detector_footprint != "mask":
        from tngxmm.instrument.donor_footprint import apply_detector_footprint

        resps, footprint_src = apply_detector_footprint(resps, base_resps, opt.detector_footprint,
                                                        donor_obsid=donor_obsid)
        if verbose:
            for i in insts:
                print(f'    {i}: footprint {opt.detector_footprint} ({footprint_src[i]}; '
                      f'{int(resps[i].mask.sum()):,} px, sum eff {float(resps[i].efficiency.sum()):,.0f})')
    if verbose:
        for i in insts:
            if resps[i].window is not None:
                print(f'    {i}: readout window {resps[i].window} '
                      f'({read_out_area_arcmin2(resps[i]):.1f} arcmin^2 read out)')
    # the conditions donor's own RMF (and on-axis ARF shape) instead of the packaged ones (instrument.obs_responses);
    # last in the chain, so the cached geometry of the steps above stays keyed by the packaged responses
    resp_obsid = donor_obsid or (context or {}).get("DONORID")
    resps, resp_used = OR.apply_observation_responses(resps, opt.responses, resp_obsid, _donor_expids(context),
                                                      opt.response_set or None)
    # a calibration-error arm (instrument.area_perturbation): the photons see the effective area times a published
    # difference; the exposure maps and the background model images stay those of the nominal response, as SAS makes
    # them from the calibration
    from tngxmm.instrument import area_perturbation as AP
    nominal = dict(resps)
    if AP.parse(opt.area_perturbation) is not None:
        resps = {i: AP.perturbed_response(r, opt.area_perturbation, ccf_dir) for i, r in resps.items()}
        if verbose:
            print(f'    effective area x {opt.area_perturbation} (exposure maps nominal)')
    r0    = resps[insts[0]]

    # ── camera partition of the parent photons ──────────────────────────────
    # p_i(photon) = ARF_i(theta, E)/A_gen x t_i/t_gen, theta from the
    # UNSCATTERED position (PSF << vignetting scale); one categorical draw per
    # photon; sum_i p_i <= 1 holds when the photon list is
    # sized for the longest camera time and A_gen >= sum_i max ARF; the stage raises otherwise.
    x_pix = xsky * 60.0 / r0.pixsize + r0.cx
    y_pix = ysky * 60.0 / r0.pixsize + r0.cy
    # Placement: the CLUSTER photons are moved off the detector centre by the
    # donor's own core offset (no random draw, no gap avoidance); QPB, AGN and
    # the mask stay in the detector frame.
    if core_offset_arcmin is None:
        raise ValueError("donor-offset placement needs the donor's core offset "
                         "(core_dx_mask_arcmin, core_dy_mask_arcmin)")
    core_req = (float(core_offset_arcmin[0]), float(core_offset_arcmin[1]))
    ox_am, oy_am, capped = donor_offset_displacement(core_req, max_dither_arcmin)
    if verbose:
        print(f'    donor-offset placement: ({ox_am:+.2f}\', {oy_am:+.2f}\')'
              + ('  (shrunk to the cap)' if capped else ''))
    dx_px = ox_am / r0.pixsize
    dy_px = oy_am / r0.pixsize
    x_pix = x_pix + dx_px
    y_pix = y_pix + dy_px
    # per-camera boresight offsets in mask pixels
    off_px = {i: (offsets[i][0] / r0.pixsize,
                  offsets[i][1] / r0.pixsize) for i in insts}
    # the cluster region that the donor's IN/OUT soft-proton measurement left out (beyond max(1.5 R500, 3') of the
    # cluster, donor_tables.mos_background): the flare maps are normalised over the same region
    sp_exclude = {}
    if sp_exclude_r500_arcmin is not None:
        from tngxmm.donor_tables.mos_background import MIN_CLUSTER_RADIUS_ARCMIN, R500_FACTOR

        r_exc = max(R500_FACTOR * float(sp_exclude_r500_arcmin), MIN_CLUSTER_RADIUS_ARCMIN)
        sp_exclude = {i: (r0.cx + dx_px - off_px[i][0], r0.cy + dy_px - off_px[i][1], r_exc) for i in insts}

    p_cum = np.zeros(len(eobs))
    edges_p = {}
    dth = r0.thetas[1] - r0.thetas[0]
    for inst in insts:
        rp = resps[inst]
        ox, oy = off_px[inst]
        vx, vy = _vig_axis(rp)
        th  = np.hypot(x_pix - rp.cx - ox - vx, y_pix - rp.cy - oy - vy) * rp.pixsize
        ith = np.clip(np.round(th / dth), 0, len(rp.thetas) - 1).astype(np.int64)
        k  = np.clip(np.searchsorted(rp.box_ene, eobs, side='right') - 1,
                     0, len(rp.box_ene_mean) - 1)
        p  = rp.arf_theta[ith, k] / a_gen * (float(cameras_s[inst]) / t_gen)
        if getattr(rp, "rga", None) is not None:
            p = p * rga_factor(rp, x_pix - rp.cx - ox - vx, y_pix - rp.cy - oy - vy, energy_index=k)
        lo = p_cum.copy()
        p_cum = p_cum + p
        edges_p[inst] = (lo, p_cum.copy())
    check_camera_partition(p_cum, t_gen, a_gen, cameras_s)
    u = rng.random(len(eobs))

    # incident photon spectrum on the ARF grid (count-weighted EXPMAP
    # vignetting energy; RNG-free)
    from tngxmm.instrument.make_exposure_maps import photon_spectrum_on_grid
    # per camera: under arf_model 'fit-linear' each camera has its RMF's grid
    phot_counts = {i: photon_spectrum_on_grid(resps[i], eobs) for i in insts}

    # ── AGN population: one sky, shared by all cameras ──────────────────────
    sources = generate_agn_sources(r0, rng, lognlogs=opt.agn_lognlogs, population=opt.agn_population)

    # ── per camera: events -> rotate -> accumulate into the 4" grid ─────────
    n4     = opt.merged_npix                                             # 648
    wpix   = opt.target_pix_arcsec / (r0.pixsize * 60.0)                 # 1.6
    edges  = (np.arange(n4 + 1) - n4 / 2.0) * wpix + r0.cx
    ang    = np.deg2rad(posangle_deg)
    ca, sa = np.cos(ang), np.sin(ang)

    counts   = np.zeros((n4, n4), dtype=np.int64)
    coverage = np.zeros((n4, n4), dtype=np.float64)
    do_rot   = abs(posangle_deg % 360.0) > 0.1

    # cluster pixel in the rotated 4"/px output grid (bin-edge coordinate):
    # the displaced cluster at mask (cx+dx, cy+dy) rotates about (cx,cy) and maps
    # to the output grid by the same edges transform used for the counts.
    crpix_x = (dx_px * ca - dy_px * sa) / wpix + n4 / 2.0
    crpix_y = (dx_px * sa + dy_px * ca) / wpix + n4 / 2.0
    clust_x = crpix_x - 0.5
    clust_y = crpix_y - 0.5
    sz_ctx: dict[str, Any] = {}
    for key, rec in catpos.items():
        px, py = output_pixel(rec.pop('deg'), r0.pixsize, dx_px, dy_px, ca, sa, wpix, n4)
        sz_ctx.update({f'{key}CENX': px, f'{key}CENY': py}, **rec)
    if per_camera_dir is not None:
        pcdir = Path(per_camera_dir); pcdir.mkdir(parents=True, exist_ok=True)  # noqa: E702

    mode = select_pn_mode(pn_mode)
    base_ctx: dict[str, Any] = dict(context or {})
    oot_line = None
    if opt.pn_oot_energy not in OOT_ENERGY_RULES:
        raise ValueError(f"unknown pn_oot_energy {opt.pn_oot_energy!r}; known: {OOT_ENERGY_RULES}")
    if opt.pn_oot_energy == "ccf-cti" and 'PN' in insts:
        date_obs = (base_ctx.get('_sas_donor') or {}).get('date_obs')
        if not date_obs:
            raise ValueError("pn_oot_energy 'ccf-cti' needs the donor's DATE-OBS (context['_sas_donor']['date_obs']): "
                             "the CTI correction depends on the epoch")
        oot_line = cti_line(mode.submode, date_obs)
    base_ctx.update(
        POSANGLE=float(posangle_deg),
        DITHERX=float(dx_px * (r0.pixsize * 60.0) / 60.0),
        DITHERY=float(dy_px * (r0.pixsize * 60.0) / 60.0),
        CLUSTX=float(clust_x), CLUSTY=float(clust_y), **sz_ctx,
        EXPFIX=EXPFIX_TAG[opt.expmap_method].replace("ss4", f"ss{opt.expmap_ss}"),
        XMMSIMV=_xmmsim_version(),
        PATTSEL=None,
        ARFEMAX=float(r0.box_ene[-1]),
        PSFMODEL=PSF_MODEL,
        PLACEMNT=opt.placement,
        OUTOFOV=opt.out_of_fov_events,
        RDWINDOW=opt.readout_windows,
    )
    base_ctx.update(CORDXREQ=core_req[0], CORDYREQ=core_req[1], CORCAPPD=bool(capped))
    base_ctx.setdefault('MOCKOBS', mockid)
    if opt.agn_lognlogs != "moretti03-hard":
        base_ctx.update(AGNLNLS=opt.agn_lognlogs)
    if opt.agn_psf != "king-onaxis":
        base_ctx.update(AGNPSF=opt.agn_psf.upper())
    if opt.psf_model != "king-onaxis":
        # the default keeps PSFMODEL = PSF_MODEL (unchanged products); another model is recorded, and the fit then
        # treats the image as it treats real data (massfit.fit_psf: on-axis King rows interpolated in energy)
        base_ctx.update(PSFMODEL=opt.psf_model.upper())
    if opt.agn_galactic_nh != "off":
        base_ctx.update(AGNGALNH=float(nh))
    if opt.galactic_nh != "hi4pi":
        base_ctx.update(GALNH=opt.galactic_nh.upper(), NHRATIO=nh_ratio)
    if opt.mos_soft_ccd != "off":
        base_ctx.update(SOFTCCD=";".join(f"{c}:{k}" for c in sorted(mos_soft_ccd)
                                         for k in sorted(mos_soft_ccd[c]))
                                         or ("unmeasured" if getattr(mos_soft_ccd, "unmeasured", False)
                                             else "none"))
    agn_by_cam: dict[str, dict] = {}

    for inst in insts:
        rp = resps[inst]
        t_live = float(cameras_s[inst])
        lo, hi = edges_p[inst]
        keep = (u >= lo) & (u < hi)
        if verbose:
            print(f'    fast-instrument {inst}: t={t_live:.0f}s  '
                  f'cluster share {int(keep.sum()):,}/{len(eobs):,}')
        stats: dict = {}
        oot_col = PN_COLUMN_ARCMIN
        if rp.window in ("PrimeLargeWindow", "PrimeSmallWindow"):
            # OoT events are assigned to the rows read out (the window)
            oot_col = (PN_LW_ROWS if rp.window == "PrimeLargeWindow" else PN_SW_ROWS) \
                * PN_PIXEL_ARCSEC / 60.0
        from tngxmm.instrument.donor_background import FWC_GENERATOR_HALF

        qmaps, sp_map = camera_qpb_maps(opt, inst, rp, ccf_dir, qpb_unit_factors, half=FWC_GENERATOR_HALF)
        if isinstance(sp_map, dict):
            from tngxmm.instrument.donor_background import FlareMaps

            sp_map = FlareMaps(sp_map, sp_exclude.get(inst))
        qpb_scale = None
        if inst == 'PN' and any(code == '1000' for _, _, code in bands_out):
            from tngxmm.instrument.apply_pattern_loss import pn_band1_qpb_scale

            qpb_scale = pn_band1_qpb_scale(ccf_dir, rp.eb_min, rp.eb_max, select_pn_mode(pn_mode).submode,
                                           sd_applied=rp.pattern_eff is not None)
        X, Y, chan = camera_events(rp, eobs[keep], x_pix[keep], y_pix[keep],
                                   t_live, nh, sources, rng,
                                   oot_frac=pn_oot_frac,
                                   off_px=off_px[inst],
                                   readout_tilt_deg=opt.pn_readout_tilt_deg,
                                   agn_stats=stats, oot_model=opt.pn_oot_model, agn_events=(opt.agn == "on"),
                                   agn_flux_band_kev=agn_band,
                                   agn_nh_gal=(float(nh) if opt.agn_galactic_nh == "view" else None),
                                   agn_psf_table=(None if opt.agn_psf == "king-onaxis" else
                                                  PS.ellbeta_table(str(ccf_dir).rstrip("/") + "/" + rp.sim.psf_file)),
                                   cluster_psf_table=(None if opt.psf_model == "king-onaxis" else
                                                      PS.ellbeta_table(str(ccf_dir).rstrip("/") + "/" + rp.sim.psf_file)
                                                      if opt.psf_model == "ellbeta" else
                                                      king_table(str(ccf_dir).rstrip("/") + "/" + rp.sim.psf_file)),
                                   oot_cti=(oot_line if inst == 'PN' else None), oot_column_arcmin=oot_col,
                                   **({} if qpb_scale is None else {'qpb_scale': qpb_scale}),
                                   **({} if bkg_by_cam[inst] is None else {'donor_bkg': bkg_by_cam[inst]}),
                                   **({} if qmaps is None else {'qpb_maps': qmaps}),
                                   **({} if sp_map is None else {'sp_map': sp_map}),
                                   **({} if donor_sky is None else {'sky': donor_sky}),
                                   **({} if (opt.mos_soft_ccd == "off" or not mos_soft_ccd.get(inst)) else
                                      {'soft_ccd': (_ccd_labels(inst, rp.mask.shape), mos_soft_ccd[inst])}))
        agn_by_cam[inst] = stats
        n_before = len(X)
        X, Y, chan = drop_out_of_fov(rp, X, Y, chan, off_px[inst])
        if verbose:
            print(f'    {inst}: {n_before - len(X):,} out-of-FOV events dropped')
        if inst == 'PN' and opt.pn_edge_rows == "drop":
            # releases up to 1.2: 12 rows inside the footprint's edge dropped, the EXPMAP left as it is; under
            # 'footprint' the footprint itself ends at the readout rows (events and EXPMAP alike)
            n_before = len(X)
            near = ~(pn_rawy(rp, X, Y, off_px[inst], opt.pn_readout_tilt_deg) > PN_BAND1_RAWY_MIN)
            X, Y, chan = X[~near], Y[~near], chan[~near]
            if verbose:
                print(f'    PN: {n_before - len(X):,} events at RAWY <= {PN_BAND1_RAWY_MIN} dropped')
        band = (chan >= opt.emin_image_kev) & (chan <= opt.emax_image_kev)
        Xf, Yf = X[band].astype(np.float64), Y[band].astype(np.float64)
        if do_rot:
            xr = rp.cx + (Xf - rp.cx) * ca - (Yf - rp.cy) * sa
            yr = rp.cy + (Xf - rp.cx) * sa + (Yf - rp.cy) * ca
        else:
            xr, yr = Xf, Yf
        h, _, _ = np.histogram2d(yr, xr, bins=[edges, edges])
        counts += h.astype(np.int64)

        vig_e = band_vignetting_energies(nominal[inst], bands_out, opt.vig_energy_rule, phot_counts[inst])
        expmaps, cov = make_exposure_maps(
            nominal[inst], t_live, bands_out if per_camera_dir is not None else [],
            off_px[inst], posangle_deg, n4, wpix, method=opt.expmap_method,
            ss=opt.expmap_ss, vig_energies=vig_e)

        # ── per-camera multi-band products (same events, split into bands) ──
        if per_camera_dir is not None:
            base = f'P{mockid}{W.CAM_FILECODE[inst]}'
            vx, vy = _vig_axis(rp)                 # the header axis is the vignetting centre
            ax_x, ax_y = off_px[inst][0] + vx, off_px[inst][1] + vy
            oxr = (ax_x * ca - ax_y * sa) if do_rot else ax_x
            oyr = (ax_x * sa + ax_y * ca) if do_rot else ax_y
            optx = float(oxr / wpix + n4 / 2.0 - 0.5)
            opty = float(oyr / wpix + n4 / 2.0 - 0.5)
            pnt: dict[str, float] = {}
            if (vx, vy) != (0.0, 0.0):            # RA_PNT stays the pointing (PPS convention)
                px0 = (off_px[inst][0] * ca - off_px[inst][1] * sa) if do_rot else off_px[inst][0]
                py0 = (off_px[inst][0] * sa + off_px[inst][1] * ca) if do_rot else off_px[inst][1]
                pnt = dict(_PNTX=float(px0 / wpix + n4 / 2.0 - 0.5), _PNTY=float(py0 / wpix + n4 / 2.0 - 0.5))
            cam_ctx = dict(base_ctx, **pnt, _inst=inst, OPTAXX=optx, OPTAXY=opty,
                           XCEN=optx + 1.0, YCEN=opty + 1.0,
                           SIMFILT=rp.filter, AREAFILE=rp.area_file,
                           PSFFILE=str(getattr(rp.sim, 'psf_file', '')) or None,
                           PATTSEL=('none' if rp.pattern_eff is None
                                    else PATTERN_SELECTION[inst]),
                           WINAREA=float(read_out_area_arcmin2(rp)))
            if opt.live_time != "package":
                cam_ctx.update(LIVESRC=opt.live_time)
            if opt.qpb_spatial != "flat":
                cam_ctx.update(QPBSPAT=opt.qpb_spatial)
            if opt.vig_axis != "pointing":
                cam_ctx.update(VIGAXIS=opt.vig_axis)
            if getattr(rp, "rga", None) is not None:
                cam_ctx.update(RGAMOD=opt.mos_rga_vignetting)
            if opt.soft_proton_map != "flat" and soft_protons_on(True, bkg_by_cam[inst]):
                cam_ctx.update(SPSPAT=opt.soft_proton_map)
            if bkg_by_cam[inst] is not None:
                cam_ctx.update(BKGMODEL=opt.background, QPBSCALE=float(bkg_by_cam[inst].get('qpb_scale', 1.0)),
                               SPRATE=float(bkg_by_cam[inst].get('sp_rate', 0.0)))
                if opt.soft_protons != "off":         # where SPRATE comes from: the donor's measurement or a substitute
                    cam_ctx.update(SPRFLAG=str(bkg_by_cam[inst].get('sp_source') or 'measured'),
                                   SPSUBST=str(bkg_by_cam[inst].get('sp_substitute') or 'none'))
                if opt.qpb_level != "in-fov":
                    cam_ctx.update(QPBLEVEL=qpb_source.get(inst, opt.qpb_level))
            if footprint_src:
                cam_ctx.update(FOOTPRNT=opt.detector_footprint, FOOTSRC=footprint_src[inst])
            if opt.responses != "packaged":
                cam_ctx.update(RESPSRC=opt.responses, RESPRMF=resp_used[inst][0],
                               RESPARF=resp_used[inst][1] if opt.responses == "donor" else None,
                               RESPSET=OR.set_digest(opt.response_set or None))
            if opt.area_perturbation:
                cam_ctx.update(AREAPERT=opt.area_perturbation)
            if getattr(rp, "ccf_files", None) is not None:
                cam_ctx.update(CCFSEL=opt.ccf_selection, QEFILE=dict(rp.ccf_files)["qe"])
            cam_ctx.update(CCDRULE=opt.active_ccds,
                           CCDLIST=format_ccds(camera_ccds[inst]["ccds"]),
                           CCDSRC=str(camera_ccds[inst].get("source", "")))
            if inst == 'PN':
                cam_ctx.update(PNEDGE=opt.pn_edge_rows)
                cam_ctx.update(OOTEVFR=float(pn_oot_frac), LIVEFRAC=float(mode.live_frac),
                               OOTFRAC=float(mode.live_frac * (1.0 - float(pn_oot_frac))),
                               OOTCORR=False, OOTMODEL=opt.pn_oot_model,
                               OOTENRG=(None if oot_line is None else opt.pn_oot_energy))
            cam_ctx.update(_pn_submode=mode.submode if inst == 'PN' else None,
                           _sas_ccds=camera_ccds[inst]["ccds"])
            sp_tmpl = sp_map
            if opt.soft_proton_template and opt.soft_proton_template != opt.soft_proton_map:
                # a study arm: the fit's soft-proton template is another map than the one the protons were drawn from
                _, sp_tmpl = camera_qpb_maps(dataclasses.replace(opt, soft_proton_map=opt.soft_proton_template), inst,
                                             rp, ccf_dir, qpb_unit_factors, half=FWC_GENERATOR_HALF)
            model_imgs = camera_model_images(opt, inst, nominal[inst], t_live, bands_out, qpb_scale, bkg_by_cam[inst], qmaps,
                                             sp_tmpl, off_px[inst], edges, do_rot, ca, sa, ccf_dir, qpb_unit_factors,
                                             corner_counts=corner_counts, rng=rng)
            gaps = band_gaps(opt, inst)
            for elo, ehi, code in bands_out:
                bsel = in_band(chan, elo, ehi, gaps.get(code))
                band_ctx: dict[str, Any] = {}
                if code in gaps:
                    band_ctx['PIGAP'] = f'{gaps[code][0]:g}-{gaps[code][1]:g}'
                if code == '1000' and inst == 'PN':
                    sp, sp_frac = pn_band1_single_pixel(
                        rp, X[bsel], Y[bsel], chan[bsel], rng, ccf_dir, mode.submode,
                        off_px[inst], opt.pn_readout_tilt_deg, rawy_cut=opt.pn_edge_rows == "drop")
                    bsel = np.flatnonzero(bsel)[sp]
                    rows = "modelled" if opt.pn_edge_rows == "drop" else "rows by the footprint"
                    band_ctx = dict(SINGLEFR=float(sp_frac), RAWYMIN=PN_BAND1_RAWY_MIN,
                                    PATTSEL=f'PATTERN==0 && RAWY>{PN_BAND1_RAWY_MIN} ({rows})')
                Xb = X[bsel].astype(np.float64); Yb = Y[bsel].astype(np.float64)  # noqa: E702
                if do_rot:
                    xb = rp.cx + (Xb - rp.cx) * ca - (Yb - rp.cy) * sa
                    yb = rp.cy + (Xb - rp.cx) * sa + (Yb - rp.cy) * ca
                else:
                    xb, yb = Xb, Yb
                img_b, _, _ = np.histogram2d(yb, xb, bins=[edges, edges])
                exp_b = expmaps[code]
                submode, sub_comment = _submode_header(inst, mode, submodes)
                hdr_b = W.percam_wcs_header(inst, crpix_x, crpix_y, opt.ra0_deg, opt.dec0_deg,
                                            t_live, elo, ehi, filter_label=rp.filter,
                                            submode_label=submode)
                if sub_comment:
                    hdr_b['SUBMODE'] = (submode, sub_comment)
                cam_ctx['PATEFF'] = _band_mean_pattern(rp, elo, ehi)
                cam_ctx['VIGENRG'] = float(vig_e[code][0])
                cam_ctx['VIGRULE'] = vig_e[code][1]
                bctx = dict(cam_ctx, **band_ctx)
                hdr_img = W.add_v4_keys(hdr_b.copy(), kind='IMAGE', context=bctx)
                hdr_exp = W.add_v4_keys(hdr_b.copy(), kind='EXPMAP', context=bctx)
                for hk, kd in ((hdr_img, 'IMAGE'), (hdr_exp, 'EXPMAP')):
                    add_sas_keys(hk, inst=inst, kind=kd, context=bctx)
                    stringify_wide_ints(hk)
                W.write_camera_products(pcdir, base, code, img_b, exp_b, hdr_img, hdr_exp,
                                        deterministic=True)
                for kind, by_band in model_imgs.items():
                    hdr_m = hdr_exp.copy()
                    hdr_m['MODELIMG'] = (kind, 'ESAS background model image (qpb_model_images)')
                    W.write_ftz(fits, by_band[code], hdr_m, Path(pcdir) / f'{base}{kind}_{code}.FTZ',
                                deterministic=True)

        coverage += cov * t_live

    # ── write merged image (integer counts) + coverage map ──────────────────
    filt = {i: resps[i].filter for i in insts}
    hdr = W.v4_merged_header(crpix_x, crpix_y, opt.ra0_deg, opt.dec0_deg,
                             opt.emin_image_kev, opt.emax_image_kev, cameras_s, insts,
                             filt, mode.submode if 'PN' in insts else None,
                             dict(base_ctx, OOTFRAC=None, OOTEVFR=None, LIVEFRAC=None, OOTMODEL=None))
    hdr_cov = W.v4_merged_header(crpix_x, crpix_y, opt.ra0_deg, opt.dec0_deg,
                                 opt.emin_image_kev, opt.emax_image_kev, cameras_s, insts,
                                 filt, mode.submode if 'PN' in insts else None,
                                 dict(base_ctx, OOTFRAC=None, OOTEVFR=None, LIVEFRAC=None, OOTMODEL=None),
                                 kind='COVERAGE')
    hdr_cov['EXPFIX'] = (base_ctx['EXPFIX'], 'coverage construction')
    stringify_wide_ints(hdr)
    stringify_wide_ints(hdr_cov)
    out_img = Path(out_dir) / 'image_merged.fits'
    fits.PrimaryHDU(data=counts.astype(np.int32), header=hdr).writeto(
        str(out_img), overwrite=True)
    fits.PrimaryHDU(data=coverage.astype(np.float32), header=hdr_cov).writeto(
        str(Path(out_dir) / 'image_merged_cov.fits'), overwrite=True)

    _write_sources_fits(sources, r0, out_dir, hdr, n4, wpix, ca, sa, do_rot, agn_by_cam, base_ctx)

    if verbose:
        print(f'    merged EPIC image -> {out_img}  '
              f'(total counts: {int(counts.sum())})')
    if placement_info is not None:
        placement_info.update(
            placement=opt.placement,
            core_dx_req_arcmin=core_req[0], core_dy_req_arcmin=core_req[1],
            offset_dx_arcmin=float(dx_px * r0.pixsize), offset_dy_arcmin=float(dy_px * r0.pixsize),
            offset_capped=bool(capped),
            clust_x=float(clust_x), clust_y=float(clust_y))
    return out_img


def _submode_header(inst: str, mode: Any,
                    submodes: dict[str, str | None]) -> tuple[str, str | None]:
    """(SUBMODE value, comment) of one camera's headers: the PN mode record
    (unknown -> PrimeFullWindow, as its OoT fraction) and the donor's MOS
    submode; a MOS camera whose mode the donor record lacks keeps the label
    with the comment that it is assumed."""
    from tngxmm.instrument import write_camera_products as W

    if inst == 'PN':
        return mode.submode, None
    sm = submodes.get(inst)
    if sm is None:
        return W.ASSUMED_SUBMODE_LABEL, 'MOS mode not in the donor record: assumed'
    return str(sm), 'donor submode'


def _write_sources_fits(sources, r0, out_dir, merged_hdr, n4, wpix, ca, sa, do_rot,
                        agn_by_cam, ctx) -> None:
    import numpy as np
    from astropy.wcs import WCS

    from tngxmm.instrument import write_camera_products as W

    sim = r0.sim
    x_mask = (sources[:, 1] - sim.boxshape1 / 2.0) * sim.pixsize_ori / r0.pixsize + r0.cx
    y_mask = (sources[:, 2] - sim.boxshape0 / 2.0) * sim.pixsize_ori / r0.pixsize + r0.cy
    if do_rot:
        xr = r0.cx + (x_mask - r0.cx) * ca - (y_mask - r0.cy) * sa
        yr = r0.cy + (x_mask - r0.cx) * sa + (y_mask - r0.cy) * ca
    else:
        xr, yr = x_mask, y_mask
    x_ima = (xr - r0.cx) / wpix + n4 / 2.0 - 0.5          # 0-indexed pixel centre
    y_ima = (yr - r0.cy) / wpix + n4 / 2.0 - 0.5
    w = WCS(merged_hdr, naxis=2)
    if len(x_ima):
        ra, dec = w.all_pix2world(x_ima, y_ima, 0)
    else:
        ra = dec = np.zeros(0)
    keys = {k: ctx.get(k) for k in ('VIEW_ID', 'MOCKOBS', 'VIEWSEED', 'SPECHASH', 'TNGXVER', 'AGNLNLS', 'AGNPSF',
                                    'AGNGALNH')}
    if ctx.get('AGNLNLS') == 'moretti03-soft':
        keys['FXBAND'] = '0.5-2 keV observed energy flux of the source model'
    keys = {k: (str(v) if isinstance(v, int) and not isinstance(v, bool) and abs(v) > 2**31 - 1
                else v) for k, v in keys.items()}
    W.write_source_list_fits(sources, Path(out_dir) / 'pts_sources.fits', x_ima=x_ima,
                             y_ima=y_ima, ra=ra, dec=dec, per_camera=agn_by_cam,
                             header_keys=keys)


__all__ = ["InstrumentOptions", "simulate_instrument", "camera_events",
           "pn_out_of_time", "PN_OOT_MODEL", "generate_agn_sources", "INSTRUMENTS"]

__cites__ = collect_cites(__name__)
