"""Per-camera and per-filter RMF/ARF.

Builds on-axis EPIC ARFs (one per camera x
filter) and stages the RMFs for mbproj2d / XSPEC.

Why on-axis: the PPS EXPMAP products are effective-exposure maps that already
carry the vignetting, so the spatial response must NOT be applied twice.  The
ARF handed to XSPEC is therefore the theta=0 curve and the exposure map
supplies vig(theta, E) * t_exp per pixel::

    counts(x, y) = rate[apec, rmf, arf_onaxis] * expmap(x, y)

The ARF model reproduces ``xmm_simulator.utils.calc_arf`` at theta=0 (where
the CCF vignetting factor is exactly 1.0), so the mocks and the observations
are fit through the same response model::

    ARF(E) = A_mirror(E) * corr(E) * T_filter(E) * QE(E) * [areacorr(E) if MOS]

``calc_arf`` itself cannot be reused: it calls ``scipy.interp2d`` (removed in
SciPy >= 1.14) and hardcodes the Medium filter, whereas the real observations
use Thin1 and Medium.

Outputs must land under a prefix the HEASoft container binds unchanged
(``TNGXMM_SITE_CONTAINER_BINDS``, environment/sites/<site>.env): a
path under ``$HOME`` would not resolve inside the container, and XSPEC is the
process that opens these files.

The output directory is an explicit argument (never the production response
directory); the CCF and RMF source directories are parameters whose defaults
are site paths (:mod:`tngxmm.core.site`: the XMM CCF ``TNGXMM_SITE_CCF`` and the
RMFs shipped with xmm_simulator).  The ARF header ``ORIGIN`` string is fixed, so
the files are byte-identical from run to run.

Which CCF file of each type an ARF reads (``rule``, the fit's ``massfit.ccf_selection``): ``'highest-issue'``
(the default) takes the highest number in the file name, the scan of xmm_simulator; ``'valid-at-date'`` takes
the constituent SAS cifbuild selects for the observation date (:mod:`tngxmm.instrument.ccf_select`).  The
selection then changes with the epoch, so ``build_responses(rule='valid-at-date')`` writes one ARF set per
interval of constant selection, ``ccf_<start>/<camera>_<filter>.arf``, next to the RMFs, and the index
``ccf_selection.json`` (:func:`ccf_epoch_dir` finds the set of an image's DATE-OBS).

References
----------
xmm_ccf_sas, sas_cifbuild, xmm_simulator
"""

from __future__ import annotations

import filecmp
import glob
import json
import shutil
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites
from tngxmm.core import site

#: CCF directory and xmm_simulator RMF directory (site defaults; read-only; a
#: non-existent ``$NAME/...`` placeholder when not configured; the RMFs: :func:`_rmf_src`)
LEGACY_CCF = site.path_or_placeholder('TNGXMM_SITE_CCF')


def _rmf_src() -> Path:
    """The xmm_simulator RMF directory: under the legacy tree where the site configures one, else the ``rmfs``
    directory of the installed xmm_simulator package (the same files at the pinned commit), else the placeholder."""
    legacy = site.path('TNGXMM_LEGACY_V3', 'xmm_simulator_src', 'xmm_simulator', 'rmfs')
    if legacy is not None:
        return legacy
    import importlib.util

    spec = importlib.util.find_spec('xmm_simulator')          # located, not imported
    if spec is not None and spec.origin and (Path(spec.origin).parent / 'rmfs').is_dir():
        return Path(spec.origin).parent / 'rmfs'
    return site.path_or_placeholder('TNGXMM_LEGACY_V3', 'xmm_simulator_src', 'xmm_simulator', 'rmfs')


LEGACY_RMF_SRC = _rmf_src()
#: production response directory (read-only reference)
LEGACY_RESP = site.path_or_placeholder('TNGXMM_SITE_MBPROJ', 'responses')

# xmm_simulator camera name -> mirror-module CCF prefix
XRT = {'PN': 'XRT3', 'MOS1': 'XRT1', 'MOS2': 'XRT2'}
# FILTER keyword as it appears in the PPS image headers -> CCF extension
FILTERS = {'Thin1': 'FILTER-THIN1', 'Medium': 'FILTER-MEDIUM', 'Thick': 'FILTER-THICK'}
CAMERAS = ('PN', 'MOS1', 'MOS2')
#: index of the per-epoch ARF sets in a response directory (build_responses rule 'valid-at-date')
CCF_INDEX = 'ccf_selection.json'


@cites("xmm_ccf_sas", "sas_cifbuild")
def latest_ccf(pattern: str, ccf_dir: str | Path = LEGACY_CCF, date: str | None = None,
               rule: str = 'highest-issue') -> Path:
    """The CCF ``<SCOPE>_<TYPEID>_*.CCF`` (``pattern``) of ``ccf_dir``: the highest number in the file name (rule
    'highest-issue', the rule of get_ccf_file_names) or the constituent cifbuild selects for ``date``
    ('valid-at-date', :func:`tngxmm.instrument.ccf_select.select`)."""
    from tngxmm.instrument import ccf_select as CS

    if CS.rule_of(rule) == 'valid-at-date':
        scope, typeid = Path(pattern).name.split('_*')[0].split('_', 1)
        return Path(ccf_dir) / CS.select(ccf_dir, scope, typeid, date, rule)
    files = glob.glob(str(Path(ccf_dir) / pattern))
    if not files:
        raise FileNotFoundError(f'no CCF matching {pattern}')
    return Path(sorted(files, key=lambda p: int(p.split('_')[-1].split('.')[0]))[-1])


def arf_ccf_types(camera: str) -> list[tuple[str, str]]:
    """(scope, type) of the CCF files an on-axis ARF of ``camera`` reads."""
    return [(XRT[camera], 'XAREAEF'), (f'E{camera}', 'QUANTUMEF'), (f'E{camera}', 'FILTERTRANSX')]


@cites("sas_cifbuild")
def selection_intervals(ccf_dir: str | Path = LEGACY_CCF, cameras: tuple[str, ...] = CAMERAS,
                        end: str | None = None) -> list[dict[str, Any]]:
    """The intervals of constant 'valid-at-date' selection of the ARF files, from the first date at which every file
    type has a valid constituent (2000-01-13 for the site CCF directory, the VALDATE of the mirror files) to ``end``
    (default the analysis date of the per-observation responses): ``[{'start': date, 'files': {camera: {'area',
    'qe', 'filter'}}}]``, consecutive intervals with the same files merged."""
    from tngxmm.instrument import ccf_select as CS

    types = [t for c in cameras for t in arf_ccf_types(c)]
    rows = CS.constituents(str(ccf_dir))
    firsts = {t: min((r[5] for r in rows if (r[1], r[2]) == t and r[5]), default=None) for t in types}
    if None in firsts.values():
        raise FileNotFoundError(f"no constituent of {[t for t, v in firsts.items() if v is None]} in {ccf_dir}")
    out: list[dict[str, Any]] = []
    for d in CS.validity_breaks(ccf_dir, types, max(firsts.values()), CS.default_analysis_date(end)):
        files = {c: {k: CS.select(ccf_dir, s, t, d, 'valid-at-date')
                     for k, (s, t) in zip(('area', 'qe', 'filter'), arf_ccf_types(c))} for c in cameras}
        if not out or out[-1]['files'] != files:
            out.append({'start': d, 'files': files})
    return out


def ccf_epoch(resp_dir: str | Path, date_obs: Any, ccf_dir: str | Path = LEGACY_CCF) -> dict[str, Any]:
    """The interval of :data:`CCF_INDEX` (``start``, ``dir``, ``files``) an image observed at ``date_obs`` takes in a
    response directory built with rule 'valid-at-date': the last one starting at or before the date.  The index must
    have been built from ``ccf_dir`` (the site CCF directory, which the instrument stage reads) with the analysis date of
    the per-observation responses (:data:`tngxmm.instrument.sas_detect.CCF_ANALYSIS_DATE`); ValueError otherwise, since
    the sets would then not hold the files the mocks were made with."""
    from tngxmm.instrument import ccf_select as CS

    rd = Path(resp_dir)
    if not (rd / CCF_INDEX).exists():
        raise FileNotFoundError(f"{rd / CCF_INDEX} missing (run build_responses --ccf-selection valid-at-date)")
    d = CS.iso_date(date_obs)
    if not d:
        raise ValueError("ccf_selection 'valid-at-date' needs the image's DATE-OBS")
    index = json.loads((rd / CCF_INDEX).read_text())
    if index.get("rule") != "valid-at-date":
        raise ValueError(f"{rd / CCF_INDEX}: rule {index.get('rule')!r}, not 'valid-at-date'")
    if Path(str(index.get("ccf_dir", ""))).resolve() != Path(ccf_dir).resolve():
        raise ValueError(f"{rd / CCF_INDEX} was built from the CCF directory {index.get('ccf_dir')!r}, not {ccf_dir}: "
                         "rebuild it (build_responses --ccf-selection valid-at-date)")
    if index.get("analysis_date") != CS.default_analysis_date():
        raise ValueError(f"{rd / CCF_INDEX} was built with the analysis date {index.get('analysis_date')!r}, not "
                         f"{CS.default_analysis_date()}: rebuild it")
    hits = [iv for iv in index["intervals"] if iv["start"] <= d]
    if not hits:
        raise ValueError(f"DATE-OBS {d} precedes the first ARF set of {rd} ({index['intervals'][0]['start']})")
    return hits[-1]


def ccf_epoch_dir(resp_dir: str | Path, date_obs: Any, ccf_dir: str | Path = LEGACY_CCF) -> Path:
    """The directory of the ARF set of :func:`ccf_epoch`."""
    return Path(resp_dir) / ccf_epoch(resp_dir, date_obs, ccf_dir)["dir"]


def _interp(x, xp, fp, zero_outside=False):
    """Linear interp; zero-fill (area) or edge-clamp (dimensionless factors)."""
    import numpy as np

    order = np.argsort(xp)
    xp, fp = np.asarray(xp)[order], np.asarray(fp)[order]
    if zero_outside:
        return np.interp(x, xp, fp, left=0.0, right=0.0)
    return np.interp(x, xp, fp)


@cites("xmm_simulator")
def rmf_grid(camera: str, rmf_src_dir: str | Path = LEGACY_RMF_SRC) -> tuple[Any, Any]:
    """(ENERG_LO, ENERG_HI) [keV] of the camera's RMF."""
    import numpy as np
    from astropy.io import fits

    with fits.open(Path(rmf_src_dir) / f'{camera}.rmf') as f:
        m = f['MATRIX'].data
        return np.asarray(m['ENERG_LO'], float), np.asarray(m['ENERG_HI'], float)


@cites("xmm_ccf_sas", "xmm_simulator")
def onaxis_arf(camera: str, filt: str, elo: Any, ehi: Any, ccf_dir: str | Path = LEGACY_CCF,
               rmf_src_dir: str | Path = LEGACY_RMF_SRC, date: str | None = None,
               rule: str = 'highest-issue') -> Any:
    """On-axis ARF [cm^2] on the RMF energy grid (``calc_arf`` at theta = 0); the CCF files by ``rule`` for the
    observation ``date`` (:func:`latest_ccf`)."""
    import numpy as np
    from astropy.io import fits

    emid = 0.5 * (elo + ehi)

    with fits.open(latest_ccf(f'{XRT[camera]}_XAREAEF_*.CCF', ccf_dir, date, rule)) as f:
        ene_on = f['ONAXISXAREAEF'].data['ENERGY'] / 1e3      # eV -> keV
        area_on = f['ONAXISXAREAEF'].data['AREA']
        # vignetting factor is identically 1.0 at theta=0 -> omitted by construction
        ene_corr = f['CORRAREA'].data['ENERGY'] / 1e3
        corr = f['CORRAREA'].data['FACTOR']

    with fits.open(latest_ccf(f'E{camera}_QUANTUMEF_*.CCF', ccf_dir, date, rule)) as f:
        ene_qe = np.asarray(f['EBINS_FRACTION'].data['ENERGY']).flatten() / 1e3
        qe = np.asarray(f['QE_TOTAL'].data['QE_TOTAL']).flatten()

    with fits.open(latest_ccf(f'E{camera}_FILTERTRANSX_*.CCF', ccf_dir, date, rule)) as f:
        ene_filt = np.asarray(f['EBINS'].data['ENERGY']).flatten() / 1e3
        trans = np.asarray(f[FILTERS[filt]].data['TRANSMISSION']).flatten()

    arf = (_interp(emid, ene_on, area_on, zero_outside=True)
           * _interp(emid, ene_corr, corr)
           * _interp(emid, ene_filt, trans)
           * _interp(emid, ene_qe, qe))

    if camera.startswith('MOS'):
        with fits.open(Path(rmf_src_dir) / 'mos_areacorr.fits') as f:
            d = f['AREACORR'].data
            arf = arf * _interp(emid, d['ENERGY'], d['AREACORR'])

    arf = np.nan_to_num(arf, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arf, 0.0, None)


@cites("xmm_ccf_sas")
def pn_pattern_fraction(e_kev: Any, ccf_dir: str | Path = LEGACY_CCF) -> Any:
    """FRAC_SD of the pn (Full Frame rows of ``EPN_QUANTUMEF``, the table of the mock's pattern efficiency)
    interpolated on ``e_kev``: the fraction of X-ray events a PATTERN<=4 image keeps."""
    import numpy as np

    from tngxmm.instrument.apply_pattern_loss import pattern_efficiency_table

    e, eff, _ = pattern_efficiency_table(str(ccf_dir), "PN", "PrimeFullWindow", rule="sas")
    return np.interp(np.asarray(e_kev, dtype=np.float64), e, eff)


def write_arf(path: Path, elo, ehi, arf, camera: str, filt: str) -> None:
    """Write an OGIP SPECRESP file."""
    from astropy.io import fits

    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='ENERG_LO', format='E', unit='keV', array=elo),
        fits.Column(name='ENERG_HI', format='E', unit='keV', array=ehi),
        fits.Column(name='SPECRESP', format='E', unit='cm**2', array=arf),
    ], name='SPECRESP')
    h = hdu.header
    h['TELESCOP'] = 'XMM'
    h['INSTRUME'] = {'PN': 'EPN', 'MOS1': 'EMOS1', 'MOS2': 'EMOS2'}[camera]
    h['FILTER'] = filt
    h['HDUCLASS'] = 'OGIP'
    h['HDUCLAS1'] = 'RESPONSE'
    h['HDUCLAS2'] = 'SPECRESP'
    h['HDUVERS'] = '1.1.0'
    h['ORIGIN'] = 'mass_fitting/build_responses.py'
    h['COMMENT'] = 'On-axis ARF (theta=0). Vignetting is carried by the PPS EXPMAP.'
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _write_arf_set(sub: Path, camera: str, filters: tuple[str, ...], elo: Any, ehi: Any, ccf_dir: str | Path,
                   rmf_src_dir: str | Path, written: dict, date: str | None = None,
                   rule: str = 'highest-issue') -> None:
    """``<camera>_<filter>.arf`` (and the pn ``_pat4`` ARFs) of one CCF selection into ``sub``."""
    import numpy as np

    for filt in filters:
        arf = onaxis_arf(camera, filt, elo, ehi, ccf_dir, rmf_src_dir, date, rule)
        out = sub / f'{camera}_{filt}.arf'
        write_arf(out, elo, ehi, arf, camera, filt)
        written[(camera, filt)] = out
        if camera == 'PN':
            # the PATTERN<=4 ARF of a PPS pn image (massfit.pn_arf 'pattern'): x FRAC_SD, the fraction SAS
            # arfgen applies (same table as the mock's instrument.pattern_efficiency 'sas')
            pat = sub / f'{camera}_{filt}_pat4.arf'
            write_arf(pat, elo, ehi, arf * pn_pattern_fraction(0.5 * (elo + ehi), ccf_dir), camera, filt)
            written[(camera, filt + '_pat4')] = pat
        emid = 0.5 * (elo + ehi)

        def at(e, emid=emid, arf=arf):
            return float(np.interp(e, emid, arf))

        print(f'{camera:5s} {filt:7s} nbin={len(arf):5d}  peak={arf.max():7.1f} cm2 '
              f'@{emid[arf.argmax()]:.2f} keV | A(1.5)={at(1.5):6.1f}  '
              f'A(3.0)={at(3.0):6.1f}  A(6.0)={at(6.0):6.1f}  A(9.0)={at(9.0):5.1f}')


@cites("xmm_ccf_sas", "sas_cifbuild", "xmm_simulator")
def build_responses(out_dir: str | Path, ccf_dir: str | Path = LEGACY_CCF,
                    rmf_src_dir: str | Path = LEGACY_RMF_SRC,
                    cameras: tuple[str, ...] = CAMERAS,
                    filters: tuple[str, ...] = tuple(FILTERS),
                    rule: str = 'highest-issue') -> dict[tuple[str, str], Path]:
    """Copy the RMFs and write ``<camera>_<filter>.arf`` into ``out_dir``; return the ARF paths (for pn also
    ``PN_<filter>_pat4.arf``, the PATTERN<=4 ARF times FRAC_SD, under the key ``(camera, filter + '_pat4')``).

    ``rule`` 'valid-at-date': one ARF set per interval of :func:`selection_intervals` in ``ccf_<start>/`` and the
    index :data:`CCF_INDEX`; returns the paths keyed ``(start, camera, filter)``.  The RMFs are copied when absent
    and must equal the source when present, so the sets can sit next to the ARFs of 'highest-issue'."""
    from tngxmm.instrument import ccf_select as CS

    OUT = Path(out_dir)
    if OUT.resolve() == LEGACY_RESP.resolve():
        raise ValueError(f'refusing to overwrite the production responses in {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'writing responses -> {OUT}\n')
    epochs = CS.rule_of(rule) == 'valid-at-date'

    written: dict[tuple[str, ...], Path] = {}
    grids = {}
    for camera in cameras:
        src, dst = Path(rmf_src_dir) / f'{camera}.rmf', OUT / f'{camera}.rmf'
        if not epochs or not dst.exists():
            shutil.copy(src, dst)
        elif not filecmp.cmp(src, dst, shallow=False):
            raise ValueError(f'{dst} differs from {src}: the ARF sets would sit next to another RMF')
        grids[camera] = rmf_grid(camera, rmf_src_dir)
        if not epochs:
            _write_arf_set(OUT, camera, filters, *grids[camera], ccf_dir, rmf_src_dir, written)
    if epochs:
        index: dict[str, Any] = dict(rule='valid-at-date', analysis_date=CS.default_analysis_date(),
                                     ccf_dir=str(ccf_dir), intervals=[])
        for iv in selection_intervals(ccf_dir, cameras):
            name = 'ccf_' + iv['start'].replace('-', '').replace(':', '')
            (OUT / name).mkdir(exist_ok=True)
            print(f"\n{name}: valid from {iv['start']}")
            sub: dict[tuple[str, str], Path] = {}
            for camera in cameras:
                _write_arf_set(OUT / name, camera, filters, *grids[camera], ccf_dir, rmf_src_dir, sub,
                               iv['start'], rule)
            written.update({(iv['start'],) + k: v for k, v in sub.items()})
            index['intervals'].append(dict(start=iv['start'], dir=name, files=iv['files']))
        (OUT / CCF_INDEX).write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    print(f'\nRMFs: {", ".join(sorted(p.name for p in OUT.glob("*.rmf")))}')
    return written


def main(argv: list[str] | None = None) -> int:
    """Command line: ``--out-dir`` (required), ``--ccf-dir``, ``--rmf-src-dir``, ``--ccf-selection``."""
    import argparse

    from tngxmm.instrument.ccf_select import CCF_RULES

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('--out-dir', required=True,
                    help='response directory (must be container-visible: under $TNGXMM_SITE_CONTAINER_BINDS)')
    ap.add_argument('--ccf-dir', default=str(LEGACY_CCF))
    ap.add_argument('--rmf-src-dir', default=str(LEGACY_RMF_SRC))
    ap.add_argument('--ccf-selection', default='highest-issue', choices=CCF_RULES,
                    help="CCF file of each type: highest file-name issue, or the file cifbuild selects per epoch "
                         "(massfit.ccf_selection; writes ccf_<start>/ ARF sets and ccf_selection.json)")
    a = ap.parse_args(argv)
    build_responses(a.out_dir, a.ccf_dir, a.rmf_src_dir, rule=a.ccf_selection)
    return 0


__cites__ = collect_cites(__name__)

if __name__ == "__main__":
    raise SystemExit(main())
