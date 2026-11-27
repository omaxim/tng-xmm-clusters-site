"""Write the per-camera IMAGE/EXPMAP products, the merged image, the coverage
map and the AGN source list of one mock view.

The headers are written from scratch (INTERFACES section 3) and say what was
simulated:

* no key is copied from the detection-mask header of a real observation;
* ``FILTER`` is the filter actually simulated (the donor's);
* ``SUBMODE`` is the donor's mode of that camera (PN; MOS when the donor
  record carries it, else ``PrimeFullWindow`` with the comment "assumed");
  ``RDWINDOW`` says whether the camera imaged through the readout window of
  that mode (``donor``) or on the full frame, ``WINAREA`` the detector area
  read out [arcmin^2]; ``CCDRULE``, ``CCDLIST`` (the CCDs the camera imaged,
  those the donor had) and ``CCDSRC`` (``pps-expmap``: measured on the donor's
  PPS exposure map; ``date-rule``: the epoch of the permanent MOS1 CCD
  losses); ``OOTEVFR`` is the PN out-of-time fraction applied, ``OOTMODEL``
  the model it was applied with (``full-column``: fraction of all X-ray
  events, placed anywhere on their own CCD column), ``OOTENRG`` how their
  energies were treated (``ccf-cti``: the CTI correction of the assigned row,
  :mod:`tngxmm.instrument.pn_oot_energy`; absent: unchanged), ``LIVEFRAC`` the live-time
  fraction of the mode and ``OOTFRAC`` the PPS-convention factor
  ``LIVEFRAC x (1 - OOTEVFR)``; ``OOTCORR = F``: the exposure map is NOT
  scaled by it (the donor-depth convention is an open decision);
* the WCS is exact (``CRPIX = edge coordinate + 0.5``,
  ``CLUSTX/CLUSTY = edge coordinate - 0.5``);
* provenance keys: ``MOCK``, ``VIEW_ID``, ``MOCKOBS``, ``DONORID``, ``SNAPNUM``,
  ``HALOID``, ``VIEWSEED``, ``SPECHASH``, ``TNGXVER``, ``GITHASH``, ``XMMSIMV``,
  ``AREAFILE``; the view's catalogue position before its measurement error
  (:mod:`tngxmm.sim.catalogue_centre`): ``SZCENX/Y`` with ``SZRULE`` (the
  catalogue), ``SZTHETA`` and ``SZBEAM`` (its filter), ``SZNOISE`` (ACT DR6: the
  white level of the filter's noise model) and ``SZSNR`` (ACT DR6:
  the entry's S/N), or ``XRCENX/Y`` with ``XRSMOOTH`` (an X-ray position), and
  ``CATPOSV`` (the rule version) and ``CPFLAG`` (the position's flag); geometry keys ``POSANGLE``, ``DITHERX/Y``, ``CLUSTX/Y``,
  ``OPTAXX/Y`` (camera optical axis = vignetting centre, 0-indexed pixels; ``VIGAXIS`` when the
  stage centres the vignetting on the telescope axis of the calibration files;
  on EXPMAPs also PPS ``XCEN/YCEN``, 1-indexed), ``PATEFF``;
  ``EXPFIX`` names the exposure-map construction, ``VIGENRG``/``VIGRULE``
  its vignetting energy; ``PSFMODEL``/``PSFFILE``, ``ARFEMAX`` (upper edge
  of the simulated ARF grid) and, when the sim stage provides them,
  ``APECVER``, ``EPHOTMAX``, ``TFLOORK``, ``GASLOAD``;
* with ``deterministic=True`` (the instrument stage passes it; the default is False) the gzip container has
  ``MTIME = 0`` and no file name, so equal content gives equal bytes.

References
----------
XMM-Newton SOC / SAS: PPS product naming and header conventions (Gabriel et
al. 2004); Jansen et al. 2001 (XMM-Newton).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

TARGET_PIX_ARCSEC = 4.0
#: PPS-like band codes (observed-frame keV) of the per-camera products
BANDS_OUT = [(0.5, 1.0, '2000'), (1.0, 2.0, '3000'),
             (2.0, 4.5, '4000'), (4.5, 12.0, '5000')]
#: the five PPS / 4XMM bands (``instrument.product_bands = 'pps-1-5'``): band 1000 = 0.2-0.5 keV
BANDS_OUT_PPS = [(0.2, 0.5, '1000')] + BANDS_OUT
PRODUCT_BANDS = {'pps-2-5': BANDS_OUT, 'pps-1-5': BANDS_OUT_PPS}
CAM_FILECODE = {'MOS1': 'M1S001', 'MOS2': 'M2S002', 'PN': 'PNS003'}
CAM_INSTRUME = {'MOS1': 'EMOS1', 'MOS2': 'EMOS2', 'PN': 'EPN'}
#: SUBMODE label of a camera whose mode the donor record lacks (the Full Frame mode)
ASSUMED_SUBMODE_LABEL = 'PrimeFullWindow'


@cites("xmm_ccf_sas")
def mock_obsid(snap: int, halo_id: int, view: int, style: str = "v4") -> str:
    """Obsid used in the per-camera file names of one mock view: fixed width
    ``{snap:03d}{halo:08d}v{view}`` (13 characters; halo ids reach 8 digits).
    ``style`` is accepted for the callers' sake; ``v4`` is the only style."""
    if style != "v4":
        raise ValueError(f"unknown obsid style {style!r}; known: 'v4'")
    if not (0 <= int(snap) <= 999 and 0 <= int(halo_id) <= 99_999_999 and 0 <= int(view) <= 9):
        raise ValueError(f"snap/halo/view out of range: {snap}, {halo_id}, {view}")
    return f'{int(snap):03d}{int(halo_id):08d}v{int(view)}'


@cites("xmm_ccf_sas")
def write_ftz(fits_mod: Any, data, hdr, path, deterministic: bool = False) -> None:
    """Write a gzipped FITS file (the real PPS ``.FTZ`` convention).

    ``deterministic=False`` uses ``gzip.open`` (the header stores the write time
    and the file name); ``True`` writes MTIME 0 and no file name, so equal FITS
    content gives equal bytes."""
    import gzip

    if not deterministic:
        with gzip.open(str(path), 'wb', compresslevel=6) as gz:
            fits_mod.PrimaryHDU(data, hdr).writeto(gz)
        return
    with open(str(path), 'wb') as raw, \
            gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=raw, mtime=0) as gz:
        fits_mod.PrimaryHDU(data, hdr).writeto(gz)


@cites("xmm_ccf_sas")
def percam_wcs_header(inst, crpix_x, crpix_y, ra0, dec0, t_live, elo, ehi,
                      filter_label: str, submode_label: str):
    """TAN WCS centred on the (dithered, rotated) cluster pixel, so the fitter
    locates the cluster from its catalogue RA/Dec regardless of the roll/dither.

    ``crpix_x/y`` are the cluster position in the continuous bin-edge
    coordinate of the 4" grid (integer = left edge of a pixel):
    ``CRPIX = crpix + 0.5`` (0-indexed centre ``crpix - 0.5``, FITS 1-indexed)."""
    from astropy.io import fits

    h = fits.Header()
    cd = TARGET_PIX_ARCSEC / 3600.0
    h['CRPIX1'], h['CRPIX2'] = crpix_x + 0.5, crpix_y + 0.5
    h['CRVAL1'], h['CRVAL2'] = ra0, dec0
    h['CDELT1'], h['CDELT2'] = -cd, cd
    h['CTYPE1'], h['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    h['CUNIT1'] = h['CUNIT2'] = 'deg'
    h['INSTRUME'] = CAM_INSTRUME[inst]
    h['FILTER'] = filter_label
    h['SUBMODE'] = submode_label
    h['EXPOSURE'] = float(t_live)
    h['EMIN'], h['EMAX'] = elo, ehi
    return h


@cites("xmm_ccf_sas")
def add_v4_keys(h, *, kind: str, context: dict[str, Any]) -> Any:
    """Append the geometry and provenance keys (``context`` values, see the
    module docstring); keys with value ``None`` are skipped."""
    h['RADESYS'] = 'FK5'
    h['EQUINOX'] = 2000.0
    if kind == 'IMAGE':
        h['CONTENT'] = ('EPIC IMAGE', 'mock counts image')
        h['HDUCLAS1'] = 'IMAGE'
        h['BUNIT'] = 'count'
    elif kind == 'EXPMAP':
        h['CONTENT'] = ('EXPOSURE MAP', 'vignetted, live-time weighted')
        h['HDUCLAS1'] = 'IMAGE'
        h['BUNIT'] = 's'
    elif kind == 'COVERAGE':
        h['CONTENT'] = ('COVERAGE', 'unvignetted, live-time weighted')
        h['BUNIT'] = 's'
    elif kind == 'MERGED':
        h['CONTENT'] = ('EPIC MERGED IMAGE', 'mock counts, all simulated cameras')
        h['BUNIT'] = 'count'
    h['MOCK'] = (True, 'simulated product (tngxmm)')
    order = [
        ('VIEW_ID', 'tngxmm view_id'), ('MOCKOBS', 'mock obsid in the file names'),
        ('DONORID', 'donor observation obsid'), ('SIMNAME', 'simulation'),
        ('SNAPNUM', 'snapshot'), ('SNAPMINI', 'mini snapshot (no GFM_Metals, GFM_CoolingRate)'),
        ('HALOID', 'FoF halo id'), ('ZOOMTGT', 'zoom_target_id'),
        ('Z_OBS', 'observed redshift'), ('NH_1E22', 'field N_H [1e22 cm^-2]'),
        ('POSANGLE', '[deg] roll applied to events and masks'),
        ('DITHERX', '[arcmin] applied dither X, pre-rotation'),
        ('DITHERY', '[arcmin] applied dither Y, pre-rotation'),
        ('PLACEMNT', 'cluster placement rule'),
        ('CORDXREQ', '[arcmin] donor core offset X, mask frame'),
        ('CORDYREQ', '[arcmin] donor core offset Y, mask frame'),
        ('CORCAPPD', 'donor offset shrunk to the dither cap'),
        ('CLUSTX', 'cluster centre X [0-indexed px]'),
        ('CLUSTY', 'cluster centre Y [0-indexed px]'),
        ('SZCENX', 'SZ position before its error X [0-indexed px]'),
        ('SZCENY', 'SZ position before its error Y [0-indexed px]'),
        ('SZRULE', 'catalogue whose SZ filter gave SZCENX/Y'),
        ('SZTHETA', '[arcmin] theta500 of the UPP filter of SZCENX/Y'),
        ('SZBEAM', '[arcmin] beam FWHM of the filter of SZCENX/Y'),
        ('SZSNR', 'entry S/N the filtered map is scaled to (ACT)'),
        ('SZNOISE', '[uK arcmin] white level of the CMB+white filter'),
        ('XRCENX', 'X-ray position before its error X [0-indexed px]'),
        ('XRCENY', 'X-ray position before its error Y [0-indexed px]'),
        ('XRSMOOTH', '[arcsec] Gaussian smoothing of XRCENX/Y'),
        ('CATPOSV', 'version of the catalogue-position rule'),
        ('CPFLAG', 'position flags: ok|global-elsewhere,merged,off-halo,edge'),
        ('OPTAXX', 'camera optical axis X [0-indexed px]'),
        ('OPTAXY', 'camera optical axis Y [0-indexed px]'),
        ('VIGAXIS', 'vignetting centre: absent pointing | ccf (telescope axis)'),
        ('RGAMOD', 'MOS RGA vignetting: ccf'),
        ('SIMFILT', 'filter transmission simulated'),
        ('OOTEVFR', 'PN out-of-time event fraction (OOTMODEL)'),
        ('OOTMODEL', 'PN out-of-time model'),
        ('OOTENRG', 'PN out-of-time energies: none|ccf-cti'),
        ('LIVEFRAC', 'PN live-time fraction of the mode'),
        ('OOTFRAC', 'PPS convention LIVEFRAC*(1-OOTEVFR)'),
        ('OOTCORR', 'EXPMAP scaled by OOTFRAC'),
        ('PATEFF', 'band-mean PATTERN efficiency applied'),
        ('PATTSEL', 'PATTERN selection modelled'),
        ('SINGLEFR', 'pn band 1: single-pixel fraction kept'),
        ('RAWYMIN', 'pn band 1: no event at RAWY <= this'),
        ('PIGAP', '[keV] PI range left out of the band (PPS pn)'),
        ('OUTOFOV', 'events outside the vignetted FOV: keep|drop'),
        ('PNEDGE', 'pn rows RAWY <= 12 (all bands): drop|footprint'),
        ('RDWINDOW', 'readout windows: full-frame|donor'),
        ('WINAREA', '[arcmin2] detector area read out'),
        ('LIVESRC', 'camera live time: package|donor-expmap'),
        ('QPBSPAT', 'QPB over the detector: flat|fwc'),
        ('SPSPAT', 'SP over the FOV: spdetmap|flare'),
        ('BKGMODEL', 'background model: nominal|donor'),
        ('QPBSCALE', 'QPB level / nominal FWC level (donor)'),
        ('SPRATE', '[ct/s/arcmin2] soft protons 0.5-12 keV, on axis'),
        ('SPRFLAG', 'SPRATE: measured (donor IN/OUT) | substitute'),
        ('SPSUBST', 'donor that lent SPRATE (none: measured)'),
        ('QPBLEVEL', 'QPB level: corners (out-of-FOV) | fwc (no corner level) | absent in-FOV'),
        ('FOOTPRNT', 'detector footprint: mask|donor-pps'),
        ('FOOTSRC', 'PPS EXPMAP(s) of the donor footprint'),
        ('CCDRULE', 'active CCDs: mask|donor'),
        ('CCDLIST', 'CCDs imaged (the donor active set)'),
        ('CCDSRC', 'source of CCDLIST: pps-expmap|date-rule'),
        ('AREAFILE', 'mirror effective-area CCF'),
        ('RESPSRC', 'photon response: donor-rmf|donor (absent packaged)'),
        ('RESPRMF', 'RMF of the conditions donor (obs_responses)'),
        ('RESPARF', 'on-axis ARF shape of the conditions donor'),
        ('RESPSET', 'digest of the per-observation response set'),
        ('AREAPERT', 'effective area x published difference (arm)'),
        ('CCFSEL', 'CCF selection: valid-at-date (absent: highest)'),
        ('QEFILE', 'quantum-efficiency CCF (with CCFSEL)'),
        ('PSFMODEL', 'PSF model of the simulation'),
        ('PSFFILE', 'PSF CCF'),
        ('AGNPSF', 'PSF of the AGN photons (absent: PSFMODEL)'),
        ('AGNLNLS', 'AGN number counts (absent: moretti03-hard)'),
        ('AGNGALNH', '[1e22 cm-2] Galactic N_H applied to the AGN'),
        ('GALNH', 'Galactic column: absent HI4PI | WILLINGALE13'),
        ('NHRATIO', 'N_H,tot / N_HI applied (Willingale 2013)'),
        ('SOFTCCD', 'MOS anomalous CCD soft excess: CAM:CCD;..'),
        ('ARFEMAX', '[keV] upper edge of the simulated ARF grid'),
        ('EXPFIX', 'EXPMAP construction'),
        ('XCEN', 'optical axis X [1-indexed px, PPS convention]'),
        ('YCEN', 'optical axis Y [1-indexed px, PPS convention]'),
        ('VIGENRG', '[keV] energy of the EXPMAP vignetting'),
        ('VIGRULE', 'rule giving VIGENRG'),
        ('APECVER', 'APEC/AtomDB version of the emission'),
        ('EPHOTMAX', '[keV] upper photon energy (rest frame)'),
        ('TFLOORK', '[K] gas temperature floor'),
        ('GASLOAD', 'gas selection'),
        ('PERIODIC', 'particles at their minimum image about GroupPos'),
        ('VIEWSEED', 'per-view instrument seed'), ('SPECHASH', 'CampaignSpec hash'),
        ('TNGXVER', 'tngxmm version'), ('GITHASH', 'tngxmm git commit'),
        ('XMMSIMV', 'xmm_simulator version'),
    ]
    for key, comment in order:
        val = context.get(key)
        if val is None:
            continue
        if key in ('EXPFIX', 'XCEN', 'YCEN', 'VIGENRG', 'VIGRULE') and kind != 'EXPMAP':
            continue
        if key in ('OOTFRAC', 'OOTEVFR', 'OOTMODEL', 'OOTENRG', 'LIVEFRAC', 'OOTCORR') and context.get('_inst') != 'PN' \
                and kind != 'MERGED':
            continue
        h[key] = (val, comment)
    return h


@cites("xmm_ccf_sas")
def v4_merged_header(crpix_x: float, crpix_y: float, ra0: float, dec0: float,
                     emin: float, emax: float, cameras_s: dict[str, float],
                     insts: list[str], filters: dict[str, str], submode: str | None,
                     context: dict[str, Any], kind: str = 'MERGED'):
    """Merged-image (or coverage) header built from scratch: the per-camera
    WCS, ``INSTRUME='EPIC-merged'``, ``EMSCEnnn``/``EMSCInnn``/``EMSFLnnn``."""
    h = percam_wcs_header('PN', crpix_x, crpix_y, ra0, dec0, 0.0, emin, emax,
                          filter_label='', submode_label='')
    for k in ('FILTER', 'SUBMODE', 'EXPOSURE'):
        del h[k]
    h['INSTRUME'] = 'EPIC-merged'
    if submode is not None:
        h['SUBMODE'] = (submode, 'PN submode of the donor')
    for k, inst in enumerate(insts, start=1):
        h[f'EMSCE{k:03d}'] = float(cameras_s[inst])
        h[f'EMSCI{k:03d}'] = inst
        h[f'EMSFL{k:03d}'] = filters.get(inst, '')
    ctx = dict(context)
    ctx.pop('OPTAXX', None)
    ctx.pop('OPTAXY', None)
    return add_v4_keys(h, kind=kind, context=ctx)


@cites("xmm_simulator", "moretti2003")
def write_source_list_fits(table, path: Path, *, x_ima, y_ima, ra, dec,
                           per_camera: dict[str, dict[str, Any]],
                           header_keys: dict[str, Any] | None = None) -> None:
    """``pts_sources.fits`` as a FITS BINTABLE: one row per drawn
    AGN with its sky position on the product grid, spectral parameters, the
    expected counts ``MU_<cam>`` and detected (on-silicon, all channels)
    counts ``N_<cam>`` per simulated camera."""
    import numpy as np
    from astropy.io import fits

    table = np.asarray(table, dtype=np.float64).reshape(-1, 9)
    cols = [
        fits.Column('SRC_ID', 'J', array=table[:, 0].astype(np.int32)),
        fits.Column('X_BOX', 'D', unit='pixel', array=table[:, 1]),
        fits.Column('Y_BOX', 'D', unit='pixel', array=table[:, 2]),
        fits.Column('X_IMA', 'D', unit='pixel', array=np.asarray(x_ima, dtype=np.float64)),
        fits.Column('Y_IMA', 'D', unit='pixel', array=np.asarray(y_ima, dtype=np.float64)),
        fits.Column('RA', 'D', unit='deg', array=np.asarray(ra, dtype=np.float64)),
        fits.Column('DEC', 'D', unit='deg', array=np.asarray(dec, dtype=np.float64)),
        fits.Column('FX', 'D', unit='erg/cm2/s', array=table[:, 3]),
        fits.Column('GAMMA', 'D', array=table[:, 4]),
        fits.Column('LOGNH', 'D', unit='log10(cm-2)', array=table[:, 5]),
        fits.Column('Z_SRC', 'D', array=table[:, 6]),
    ]
    for cam in sorted(per_camera):
        st = per_camera[cam]
        cols.append(fits.Column(f'MU_{cam}', 'D', unit='count', array=np.asarray(st['mu'])))
        cols.append(fits.Column(f'N_{cam}', 'J', unit='count',
                                array=np.asarray(st['n'], dtype=np.int32)))
    hdu = fits.BinTableHDU.from_columns(cols, name='SRCLIST')
    hdu.header['NSRC'] = (len(table), 'number of AGN drawn')
    hdu.header['FXBAND'] = ('see xmm_simulator pts/templates_pts_hard.dat',
                            'band of FX')
    for k, v in (header_keys or {}).items():
        if v is not None:
            hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(str(path), overwrite=True)


@cites("xmm_ccf_sas")
def write_camera_products(pcdir: Path, base: str, code: str, img, exp, hdr_img, hdr_exp,
                          deterministic: bool = False) -> tuple[Path, Path]:
    """Write ``<base>IMAGE_<code>.FTZ`` (int32) and ``<base>EXPMAP<code>.FTZ``
    (float32) into ``pcdir``."""
    import numpy as np
    from astropy.io import fits as _fits

    p_img = Path(pcdir) / f'{base}IMAGE_{code}.FTZ'
    p_exp = Path(pcdir) / f'{base}EXPMAP{code}.FTZ'
    write_ftz(_fits, np.asarray(img).astype(np.int32), hdr_img, p_img, deterministic)
    write_ftz(_fits, np.asarray(exp).astype(np.float32), hdr_exp, p_exp, deterministic)
    return p_img, p_exp


__cites__ = collect_cites(__name__)
