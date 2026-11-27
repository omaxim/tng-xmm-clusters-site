"""The astrophysical foreground of every donor field from the ROSAT All-Sky Survey (table ``donor_sky_rass.csv``).

The mock sky (:func:`tngxmm.instrument.simulate_instrument.skybkg_spectrum`) is the Local Hot Bubble (LHB, unabsorbed
APEC), the Galactic halo (absorbed APEC) and the unresolved cosmic X-ray background.  The foreground varies across the
sky by factors of several (Liu et al. 2017; Henley & Shelton 2013), so each donor carries its own LHB and halo levels,
measured as XMM-Newton cluster analyses measure them (Snowden et al. 2008, Sect. 3.3): the RASS diffuse-background
spectrum of an annulus about the field, from the HEASARC X-ray Background Tool (Sabol & Snowden; the maps of
Snowden et al. 1997), fitted with that model.

Per donor (cluster position of the observation catalogue):

* spectrum: the tool's rates spectrum (counts s^-1 arcmin^-2 in the seven RASS bands R1-R7, so the fitted
  normalisations are per arcmin^2) of the annulus :data:`ANNULUS_DEG` about the cluster, 1-2 degrees as in the
  Abell 1795 example of Snowden et al. (2008, Sect. 3.3), with the PSPC response ``pspcc_gain1_256.rsp``;
* model, in XSPEC (``abund angr``, the table of the stage's sky APEC and of Henley & Shelton 2013; AtomDB 3.1.3, the
  mocks' tables): ``apec + phabs * (apec + powerlaw)``.  Fixed: the LHB at kT = 0.111 keV (the median of Yeung et
  al. 2024 over |b| > 30 deg), the halo at kT = 0.191 keV (the median of Henley & Shelton 2013), both at solar
  abundance, the power law at photon index 1.41 (De Luca & Molendi 2004), the absorbing column the HI4PI column the
  tool reports for the annulus times the donor's N_H,tot / N_HI (Willingale et al. 2013; ``donor_nh_total.csv``),
  and the CXB power law at the Moretti et al. (2003) total 0.5-2 keV intensity
  (:func:`tngxmm.instrument.point_sources.cxb_total_norm`), the CXB the mocks carry as drawn AGN plus the unresolved
  remainder; fitted free, it came out at that level (median over the 638 fields 9.1e-7 against 9.2e-7 per
  arcmin^2).  The RASS maps lack the sources above the survey's removal limit, a few per cent of the intensity.
  Free: the LHB and halo normalisations.
* columns: ``obsid, ra_deg, dec_deg, nh_hi_annulus_1e22, nh_fit_1e22, lhb_norm, lhb_lo, lhb_hi, halo_norm, halo_lo,
  halo_hi, cxb_rass_norm`` (the fixed CXB), ``chi2, dof, status`` (norms in XSPEC units per arcmin^2, 68 per cent
  intervals).

The tool is reached from the login node (the compute nodes have no network): ``python -m
tngxmm.donor_tables.rass_sky <work_dir> [--obsids ...]``, then the table is copied to the data package.

References
----------
snowden2008, snowden1997, rass_xbt (method, maps, tool); yeung2024, henley2013, deluca2004 (fixed parameters);
willingale2013 (column); atomdb (APEC); heasoft_xspec (fit).
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from tngxmm.cites import cites, collect_cites

TOOL = "https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xraybg"
RESPONSE_URL = "https://heasarc.gsfc.nasa.gov/FTP/rosat/calib_data/pspc/cpf/matrices/pspcc_gain1_256.rsp"
RESPONSE = "pspcc_gain1_256.rsp"
#: annulus about the cluster [deg] (Snowden et al. 2008, Sect. 3.3, Abell 1795)
ANNULUS_DEG = (1.0, 2.0)
#: fixed temperatures [keV]: LHB (Yeung et al. 2024, median |b| > 30 deg), halo (Henley & Shelton 2013, median)
LHB_KT = 0.111
HALO_KT = 0.191
#: photon index of the CXB power law (De Luca & Molendi 2004)
CXB_INDEX = 1.41
TABLE = "donor_sky_rass.csv"
COLUMNS = ("obsid", "ra_deg", "dec_deg", "nh_hi_annulus_1e22", "nh_fit_1e22", "lhb_norm", "lhb_lo", "lhb_hi",
           "halo_norm", "halo_lo", "halo_hi", "cxb_rass_norm", "chi2", "dof", "status")


def tool_url(ra: float, dec: float) -> str:
    """The X-ray Background Tool query of the annulus about (ra, dec) [deg, J2000] with the spectrum created."""
    return (f"{TOOL}/xraybg.pl?Entry={ra:.5f}%2C{dec:.5f}&NR=GRB%2FSIMBAD%2BSesame%2FNED&CoordSys=J2000"
            f"&radius={ANNULUS_DEG[1]}&region=annulus&inner_radius={ANNULUS_DEG[0]}&results=integrate"
            "&spectrum=create")


def parse_page(html: str) -> dict[str, Any]:
    """The rates-spectrum id and the weighted HI4PI column [1e22 cm^-2] of the annulus from the tool's result page."""
    m = re.search(r"xraybgfits\.pl\?rates_spectrum=(\d+)", html)
    if m is None:
        raise ValueError("no rates spectrum on the tool's page")
    text = re.sub(r"<[^>]*>", " ", html)
    nh = re.search(r"HI4PI[^0-9]*\(using [0-9.]+\D+radius\)\s+([0-9.]+E[+-]?\d+)\s+([0-9.]+E[+-]?\d+)", text)
    return {"spectrum_id": m.group(1), "nh_hi_annulus_1e22": float(nh.group(2)) / 1e22 if nh else float("nan")}


def _get(url: str, timeout: float = 120.0, tries: int = 4) -> bytes:
    import urllib.request

    from tngxmm.donor_tables.archive import USER_AGENT

    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(10.0 * (k + 1))
    raise RuntimeError("unreachable")


@cites("rass_xbt", "snowden1997")
def fetch(obsid: str, ra: float, dec: float, work: str | Path) -> dict[str, Any]:
    """Download the rates spectrum of one donor's annulus into ``<work>/<obsid>/rates.fits`` (kept: a later run reads
    it again) and return the page's values."""
    import json

    d = Path(work) / obsid
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "page.json"
    if (d / "rates.fits").exists() and meta.exists():
        return json.loads(meta.read_text())
    info = parse_page(_get(tool_url(ra, dec)).decode("utf-8", "replace"))
    (d / "rates.fits").write_bytes(_get(f"{TOOL}/xraybgfits.pl?rates_spectrum={info['spectrum_id']}"))
    info.update(ra_deg=ra, dec_deg=dec)
    meta.write_text(json.dumps(info) + "\n")
    return info


def xcm(nh_1e22: float, apecroot: str, out: str = "result.txt", cxb_norm: float | None = None) -> str:
    """The XSPEC script of one fit (module docstring)."""
    return f"""query yes
chatter 10
abund angr
xsect bcmc
xset APECROOT {apecroot}
data rates.fits
response {RESPONSE}
ignore bad
model apec + phabs*(apec + powerlaw) & /*
newpar 1 {LHB_KT} -1
newpar 2 1 -1
newpar 4 1e-6
newpar 5 {nh_1e22:.6g} -1
newpar 6 {HALO_KT} -1
newpar 7 1 -1
newpar 9 1e-6
newpar 10 {CXB_INDEX} -1
newpar 11 {cxb_norm:.6g} -1
fit 500
set fo [open "{out}" w]
foreach p {{4 9 11}} {{ tclout param $p; puts $fo "P$p [lindex $xspec_tclout 0]" }}
tclout stat; puts $fo "STAT $xspec_tclout"
tclout dof; puts $fo "DOF [lindex $xspec_tclout 0]"
error stopat 10 0.01 maximum 1000.0 1.0 4 9
foreach p {{4 9}} {{ tclout error $p; puts $fo "E$p [lindex $xspec_tclout 0] [lindex $xspec_tclout 1]" }}
close $fo
exit
"""


@cites("snowden2008", "yeung2024", "henley2013", "deluca2004", "willingale2013", "heasoft_xspec", "atomdb")
def fit(obsid: str, work: str | Path, nh_1e22: float, sasrun: str, apecroot: str) -> dict[str, Any]:
    """Fit one donor's spectrum (``<work>/<obsid>/rates.fits``) in XSPEC through the site's HEASoft wrapper
    ``sasrun`` and return its table row values."""
    d = Path(work) / obsid
    resp = Path(work) / RESPONSE
    link = d / RESPONSE
    if not link.exists():
        link.symlink_to(resp)
    from tngxmm.instrument.point_sources import cxb_total_norm

    (d / "fit.xcm").write_text(xcm(nh_1e22, apecroot, cxb_norm=cxb_total_norm(CXB_INDEX)))
    res = d / "result.txt"
    res.unlink(missing_ok=True)
    with open(d / "xspec.log", "w") as log:
        subprocess.run([sasrun, f"cd {d} && xspec - fit.xcm < /dev/null"], stdout=log, stderr=subprocess.STDOUT,
                       timeout=600, check=False)
    if not res.exists():
        return {"status": "xspec-failed"}
    v: dict[str, list[str]] = {}
    for line in res.read_text().splitlines():
        k, *rest = line.split()
        v[k] = rest
    f = lambda k, i=0: float(v[k][i]) if k in v and len(v[k]) > i else float("nan")  # noqa: E731
    out = dict(lhb_norm=f("P4"), halo_norm=f("P9"), cxb_rass_norm=f("P11"), chi2=f("STAT"), dof=f("DOF"),
               lhb_lo=f("E4", 0), lhb_hi=f("E4", 1), halo_lo=f("E9", 0), halo_hi=f("E9", 1))
    ok = all(math.isfinite(out[k]) and out[k] > 0 for k in ("lhb_norm", "halo_norm"))
    out["status"] = "ok" if ok else "fit-failed"
    return out


def build_table(work: str | Path, observations: Any, nh_total: Any, sasrun: str, apecroot: str,
                echo: Any = print, workers: int = 1) -> Any:
    """The table for every observation (``observations``: obsid, ra_deg, dec_deg; ``nh_total``: the
    ``donor_nh_total.csv`` table): fetch (resumable), fit, one row each; ``workers`` donors at a time."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / RESPONSE).exists():
        (work / RESPONSE).write_bytes(_get(RESPONSE_URL))
    ratio = dict(zip(nh_total.obsid.astype(str).str.zfill(10), nh_total.nh_ratio.astype(float)))
    def one(r: Any) -> dict:
        obsid = str(r.obsid).zfill(10)
        row: dict[str, Any] = {"obsid": obsid, "ra_deg": float(r.ra_deg), "dec_deg": float(r.dec_deg)}
        try:
            info = fetch(obsid, float(r.ra_deg), float(r.dec_deg), work)
            nh = float(info["nh_hi_annulus_1e22"]) * float(ratio.get(obsid, float("nan")))
            row.update(nh_hi_annulus_1e22=info["nh_hi_annulus_1e22"], nh_fit_1e22=nh)
            row.update(fit(obsid, work, nh, sasrun, apecroot) if math.isfinite(nh) else {"status": "no-column"})
        except (OSError, ValueError) as e:
            row["status"] = f"error: {type(e).__name__}"
        return row

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        rows = list(ex.map(one, observations.itertuples()))
    echo(f"{len(rows)} donors fitted")
    return pd.DataFrame(rows).reindex(columns=list(COLUMNS))


def donor_sky(table: Any, obsid: Any) -> dict[str, float]:
    """The LHB and halo normalisations [per arcmin^2] of one donor (``KeyError`` when absent or not ``ok``)."""
    from tngxmm.instrument.donor_tables import donor_rows

    r = donor_rows(table, obsid, "RASS sky").iloc[0]
    if str(r["status"]) != "ok":
        raise KeyError(f"donor {obsid}: RASS sky fit status {r['status']!r}")
    return {"lhb_norm": float(r["lhb_norm"]), "halo_norm": float(r["halo_norm"])}


__cites__ = collect_cites(__name__)


def main(argv: list[str] | None = None) -> int:
    """``python -m tngxmm.donor_tables.rass_sky <work_dir> [--obsids ...] [--out CSV]``."""
    import argparse

    from tngxmm.core import site
    from tngxmm.instrument.donor_tables import read_table
    from tngxmm.massfit.build_realfit_targets import load_observations

    p = argparse.ArgumentParser(prog="python -m tngxmm.donor_tables.rass_sky")
    p.add_argument("work")
    p.add_argument("--obsids", nargs="*")
    p.add_argument("--out", default=None, help=f"default <work>/{TABLE}")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--allow-dirty", action="store_true", help="write into the data package from an uncommitted tree")
    a = p.parse_args(argv)
    from tngxmm.core.provenance import refuse_dirty_data_write

    out = Path(a.out or Path(a.work) / TABLE)
    refuse_dirty_data_write(out, a.allow_dirty)                    # before any work
    o = load_observations(centres="legacy")[["obsid", "ra_deg", "dec_deg"]].drop_duplicates("obsid")
    if a.obsids:
        o = o[o.obsid.isin([str(x).zfill(10) for x in a.obsids])]
    apecroot = str(site.get("TNGXMM_SITE_APEC_V313"))      # the prefix of the AtomDB 3.1.3 files
    t = build_table(a.work, o, read_table("nh_total"), str(site.get("TNGXMM_SITE_SASRUN")), apecroot,
                    workers=a.workers)
    t.to_csv(out, index=False, float_format="%.6g")
    # the provenance sidecar the instrument stage checks (instrument.donor_tables.check_table); in the data package
    # also the line of SHA256SUMS
    from tngxmm.core.hashing import update_sha256sums
    from tngxmm.core.paths import data_asset
    from tngxmm.core.provenance import build_provenance, write_table_provenance
    from tngxmm.instrument.donor_tables import observations_csv, table_path

    write_table_provenance(out, build_provenance(
        "tngxmm.donor_tables.rass_sky.build_table", {"observations": observations_csv(), "nh_total": table_path("nh_total")},
        work=str(a.work), apec_root=apecroot, n_donors=len(t), obsids=list(a.obsids or [])))
    if out.resolve().parent == data_asset("tables").resolve():
        update_sha256sums(out)
    print(f"wrote {out}: {len(t)} rows, status {t.status.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
