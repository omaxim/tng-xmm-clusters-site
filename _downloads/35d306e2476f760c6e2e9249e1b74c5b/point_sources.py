"""Field point sources of the instrument stage: the number counts the AGN are drawn from, the
band their flux refers to, and the off-axis PSF they are scattered with.

Three switches (``agn_lognlogs``, ``agn_galactic_nh``, ``agn_psf``) of :class:`~tngxmm.instrument.simulate_instrument.InstrumentOptions` select the
behaviour; their defaults are the xmm_simulator behaviour (hard, off, king-onaxis); the v1 presets set soft, view and ellbeta.

``agn_lognlogs``
    ``'moretti03-hard'`` (default; xmm_simulator 0.2.0 ``point_sources``): fluxes S drawn from
    Eq. 2 of Moretti et al. (2003) with the 2-10 keV best-fit parameters, between 1e-15 and
    1e-12 erg/s/cm^2; S is the 2-10 keV flux of the z = 0 absorbed template
    (``templates_pts_hard.dat``); log N_H uniform in [20.5, 23].
    ``'moretti03-soft'``: S drawn from the same Eq. 2 with the 0.5-2 keV best-fit parameters
    (the band of the images), between the same limits (1e-15 is also the soft-band flux limit
    below which the sky model's unresolved CXB, ``cxb_m13_unres``, holds the sources); S is the
    OBSERVED 0.5-2 keV energy flux of each source's own model (PhAbs(N_H, z) x power law),
    integrated per source instead of read from a z = 0 template; log N_H uniform in
    [20.5, :data:`SOFT_NH_MAX`].  Photon index and redshift as in the default.

    Why: the images, the realism features and the fit are 0.5-2 keV.  Drawn from the hard-band
    counts and converted with the population's spectra, the mock AGN have 1.4-1.7 times the
    Moretti soft-band counts at 1-3e-15 erg/s/cm^2 (the fluxes near an XMM detection limit) and
    0.55-0.8 times above 2e-14.  The N_H cap keeps the hard band of the soft-drawn population
    within 15 % of the Moretti hard counts above 3e-14 (a cap of 23 gives 2.5-5 times too many
    bright hard sources, because a soft-normalised absorbed source is very bright above 2 keV).

``agn_galactic_nh``
    ``'off'`` (default; xmm_simulator): the AGN spectra carry their intrinsic absorption only,
    while the cluster and the sky background are absorbed by the Galactic column of the view.
    ``'view'``: the AGN photon spectra are also absorbed by the view's Galactic N_H (PhAbs at
    z = 0), after the flux normalisation, so S stays the flux before Galactic absorption (the
    convention of number counts built from high-latitude fields, whose conversion factors are
    computed at each field's column, Moretti et al. 2003 Sect. 6).  At the median N_H of the
    195 donors (2.8e20) this removes 8 % of the 0.5-2 keV energy flux of a Gamma = 1.9 source
    (17 % at the 90th percentile, 6.4e20).

``agn_psf``
    ``'king-onaxis'`` (default): every AGN photon is scattered with the camera's on-axis King
    profile (row 0 of KING_PARAMS, 225 eV), as the cluster.
    ``'ellbeta'``: the SAS default 2-D PSF envelope (Read et al. 2011; ELLBETA_PARAMS of the
    XRT*_XPSF CCF): an elliptical King (beta2d) profile plus, for MOS, a Gaussian core, with
    the CCF core radius, slope, ellipticity, Gaussian FWHM and peak ratio at the photon's
    energy (nearest tabulated energy) and off-axis angle (linear in theta), the major axis
    tangential (the Wolter-I off-axis stretch).  The spokes and the large-scale azimuthal
    modulation of the full model are not drawn (they move no flux between radii).  The cluster
    keeps the on-axis King profile (the fit convolves with that one).

References: moretti2003 (Eq. 2, Sects. 7.1 and 7.2), read2011 (Sects. 2.1 and 3.1), xmm_ccf_sas
(XRT*_XPSF), xmm_simulator (the default generator).
"""

from __future__ import annotations

import math
from typing import Any

from tngxmm.cites import cites, collect_cites

#: Moretti et al. (2003) Eq. 2 best-fit parameters: soft (Sect. 7.1, 0.5-2 keV) and hard
#: (Sect. 7.2, 2-10 keV); N(>S) = N_S (2e-15)^a1 / (S^a1 + S0^(a1 - a2) S^a2) per deg^2
MORETTI03 = {
    "soft": dict(alpha1=1.82, alpha2=0.60, s0=1.48e-14, ns=6150.0, band_kev=(0.5, 2.0)),
    "hard": dict(alpha1=1.57, alpha2=0.44, s0=4.5e-15, ns=5300.0, band_kev=(2.0, 10.0)),
}
MORETTI03_PIVOT = 2e-15
#: values of InstrumentOptions.agn_lognlogs -> MORETTI03 band
AGN_LOGNLOGS = {"moretti03-hard": "hard", "moretti03-soft": "soft"}
#: values of InstrumentOptions.agn_psf
AGN_PSF = ("king-onaxis", "ellbeta")
#: values of InstrumentOptions.agn_galactic_nh ('view': PhAbs at z = 0 with the view's N_H)
AGN_GALACTIC_NH = ("off", "view")
#: normalisation of the AGN number counts: the published counts of Moretti et al. (2003), the same for every donor
AGN_NORM = ("moretti03",)
#: flux range of the drawn sources [erg/s/cm^2] (xmm_simulator's, both modes)
AGN_FLUX_RANGE = (1e-15, 1e-12)
#: upper end of the uniform log N_H [cm^-2] of the soft-drawn population (see the module docstring)
SOFT_NH_MAX = 22.0
#: ELLBETA PSF normalisation radius of the CCF (NORM_AT) [arcsec]: photons are drawn inside it
ELLBETA_RMAX_ARCSEC = 900.0
KEV_ERG = 1.602176634e-9


@cites("moretti2003")
def moretti03_n_above(s_cgs: Any, band: str) -> Any:
    """Integral counts N(>S) [deg^-2] of Moretti et al. (2003) Eq. 2 in ``band`` ('soft'|'hard')."""
    import numpy as np

    p = MORETTI03[band]
    s = np.asarray(s_cgs, dtype=float)
    return p["ns"] * MORETTI03_PIVOT ** p["alpha1"] / (
        s ** p["alpha1"] + p["s0"] ** (p["alpha1"] - p["alpha2"]) * s ** p["alpha2"])


#: total 0.5-2 keV CXB intensity of Moretti et al. (2003, Sect. 5) [erg s^-1 cm^-2 deg^-2]
MORETTI03_SOFT_TOTAL = 7.53e-12
#: photon index of the CXB power law (De Luca & Molendi 2004)
CXB_INDEX = 1.41


@cites("moretti2003")
def resolved_intensity(lo: float, hi: float, band: str = "soft") -> float:
    """Intensity [erg s^-1 cm^-2 deg^-2] of the sources with lo < S < hi of Eq. 2 in ``band``:
    the integral of S |dN/dS| on a fine log grid."""
    import numpy as np

    s = np.logspace(math.log10(lo), math.log10(hi), 20001)
    n = moretti03_n_above(s, band)
    # S |dN/dS| dS summed over the grid cells: -sum S_mid (N(S_i+1) - N(S_i))
    return float(-(np.sqrt(s[1:] * s[:-1]) * np.diff(n)).sum())


@cites("moretti2003", "deluca2004")
def cxb_total_norm(index: float = CXB_INDEX) -> float:
    """XSPEC power-law normalisation [photons keV^-1 cm^-2 s^-1 at 1 keV, per arcmin^2] of the whole Moretti et al.
    (2003) 0.5-2 keV CXB intensity at photon index ``index`` (the CXB of the RASS foreground fits)."""
    g = 2.0 - float(index)
    return float(MORETTI03_SOFT_TOTAL / 3600.0 / (KEV_ERG * (2.0 ** g - 0.5 ** g) / g))


@cites("moretti2003", "deluca2004")
def unresolved_cxb_norm(lo: float = AGN_FLUX_RANGE[0], hi: float = AGN_FLUX_RANGE[1],
                        index: float = CXB_INDEX) -> float:
    """XSPEC power-law normalisation [photons keV^-1 cm^-2 s^-1 at 1 keV, per arcmin^2] of the CXB that the drawn
    AGN do not carry: the Moretti et al. (2003) total 0.5-2 keV intensity minus that of the sources drawn
    (``lo`` < S < ``hi``, :func:`resolved_intensity`), spread as a power law of photon index ``index``.  At the
    drawn range 1e-15-1e-12 it is 22 per cent of the total."""
    rest = MORETTI03_SOFT_TOTAL - resolved_intensity(lo, hi, "soft")
    g = 2.0 - float(index)
    kev_per_unit = (2.0 ** g - 0.5 ** g) / g                 # integral of E E^-index over 0.5-2 keV
    return float(rest / 3600.0 / (KEV_ERG * kev_per_unit))


@cites("moretti2003")
def moretti03_draw(n: int, band: str, rng, lo: float = AGN_FLUX_RANGE[0],
                   hi: float = AGN_FLUX_RANGE[1]) -> Any:
    """``n`` fluxes [erg/s/cm^2] distributed as Eq. 2 of ``band`` on [lo, hi]: inverse of the
    normalised N(>S) on a fine log grid (monotonic, continuous; one uniform per source)."""
    import numpy as np

    grid = np.logspace(math.log10(lo), math.log10(hi), 4001)
    nlo, nhi = float(moretti03_n_above(lo, band)), float(moretti03_n_above(hi, band))
    cdf = (nlo - moretti03_n_above(grid, band)) / (nlo - nhi)          # 0 at lo, 1 at hi
    u = rng.random(int(n))
    return np.exp(np.interp(u, cdf, np.log(grid)))


@cites("moretti2003")
def moretti03_expected_number(band: str, area_deg2: float, lo: float = AGN_FLUX_RANGE[0],
                              hi: float = AGN_FLUX_RANGE[1]) -> float:
    """Expected number of sources with lo < S < hi on ``area_deg2``."""
    return float(area_deg2) * float(moretti03_n_above(lo, band) - moretti03_n_above(hi, band))


def band_energy_flux(photon_density, e1: float, e2: float, n: int = 400) -> float:
    """Energy flux [erg/s/cm^2] in [e1, e2] keV of a photon spectrum ``photon_density(E)``
    [ph/s/cm^2/keV] (vectorised callable): midpoint rule on ``n`` log-spaced bins."""
    import numpy as np

    edges = np.logspace(math.log10(e1), math.log10(e2), n + 1)
    mid = np.sqrt(edges[1:] * edges[:-1])
    return float((np.asarray(photon_density(mid), dtype=float) * mid * np.diff(edges)).sum() * KEV_ERG)


# ── AGN population (Ueda et al. 2014) ─────────────────────────────────────────

#: Ueda et al. (2014): the Compton-thin hard XLF (Eqs. 14-19, Table 4), the absorption function (Eqs. 2-6, Table 2)
#: and the photon index per absorption class (Eq. 10, Table 3); L the intrinsic rest-frame 2-10 keV luminosity
#: [erg/s], densities per Mpc^3 per dex at H0 = 70, Omega_m = 0.3 (their cosmology, used for the whole population).
#: psi_max = 0.84 is the value of p. 7 (the expression printed on p. 8 is the switch point between Eqs. 5 and 6).
UEDA14 = dict(A=2.91e-6, log_lstar=43.97, g1=0.96, g2=2.71, p1s=4.78, beta1=0.84, p2=-1.5, p3=-6.2, zc1s=1.86,
              log_la1=44.61, alpha1=0.29, zc2s=3.0, log_la2=44.0, alpha2=-0.1, psi0=0.43, beta=0.24, psi_min=0.2,
              psi_max=0.84, eps=1.7, a1=0.48, f_ctk=1.0, h0=70.0, om0=0.3,
              gamma_thin=(1.94, 0.09), gamma_thick=(1.84, 0.15))
#: values of InstrumentOptions.agn_population
AGN_POPULATIONS = ("xmm_simulator", "ueda14")
_UEDA_POP: dict = {}


@cites("ueda2014")
def ueda14_xlf(log_l: Any, z: Any) -> Any:
    """Compton-thin XLF dPhi/dlogL [Mpc^-3 dex^-1] of Ueda et al. (2014), Eqs. 14-19 (LDDE)."""
    import numpy as np

    u = UEDA14
    log_l, z = np.broadcast_arrays(np.asarray(log_l, float), np.asarray(z, float))
    x = 10.0 ** (log_l - u["log_lstar"])
    phi0 = u["A"] / (x ** u["g1"] + x ** u["g2"])
    p1 = u["p1s"] + u["beta1"] * (log_l - 44.0)
    zc1 = np.where(log_l <= u["log_la1"], u["zc1s"] * 10.0 ** (u["alpha1"] * (log_l - u["log_la1"])), u["zc1s"])
    zc2 = np.where(log_l <= u["log_la2"], u["zc2s"] * 10.0 ** (u["alpha2"] * (log_l - u["log_la2"])), u["zc2s"])
    e1 = (1.0 + z) ** p1
    e2 = (1.0 + zc1) ** p1 * ((1.0 + z) / (1.0 + zc1)) ** u["p2"]
    e3 = (1.0 + zc1) ** p1 * ((1.0 + zc2) / (1.0 + zc1)) ** u["p2"] * ((1.0 + z) / (1.0 + zc2)) ** u["p3"]
    e = np.where(z <= zc1, e1, np.where(z <= zc2, e2, e3))
    return phi0 * e


#: log N_H bins [cm^-2] of the absorption function and their edges
UEDA14_NH_BINS = ((20.0, 21.0), (21.0, 22.0), (22.0, 23.0), (23.0, 24.0), (24.0, 26.0))


@cites("ueda2014")
def ueda14_nh_function(log_l: Any, z: Any) -> Any:
    """Absorption function of Ueda et al. (2014), Eqs. 2-6: the density per dex in log N_H in each bin of
    :data:`UEDA14_NH_BINS` (last axis), normalised to 1 over 20-24 (Compton-thin); the 24-26 bin adds the Compton-thick
    AGN on top (f_CTK psi / 2 per dex)."""
    import numpy as np

    u = UEDA14
    log_l, z = np.broadcast_arrays(np.asarray(log_l, float), np.asarray(z, float))
    psi43 = u["psi0"] * (1.0 + np.minimum(z, 2.0)) ** u["a1"]
    psi = np.minimum(u["psi_max"], np.maximum(psi43 - u["beta"] * (log_l - 43.75), u["psi_min"]))
    eps = u["eps"]
    low = psi < (1.0 + eps) / (3.0 + eps)
    b20 = np.where(low, 1.0 - (2.0 + eps) / (1.0 + eps) * psi, 2.0 / 3.0 - (3.0 + 2.0 * eps) / (3.0 + 3.0 * eps) * psi)
    b21 = np.where(low, psi / (1.0 + eps), 1.0 / 3.0 - eps / (3.0 + 3.0 * eps) * psi)
    b22 = psi / (1.0 + eps)
    b23 = eps * psi / (1.0 + eps)
    b24 = u["f_ctk"] / 2.0 * psi
    return np.stack([b20, b21, b22, b23, b24], axis=-1)


def _phabs_unit_transmission(e_kev: Any) -> Any:
    """PhAbs transmission at N_H = 1e22 cm^-2 (the stage's absorber, :func:`agn_channel_cdfs`); any column is its
    power, exp(-sigma N_H)."""
    import numpy as np
    from threeML import PhAbs

    a = PhAbs()
    a.NH = 1.0
    a.redshift = 0.0
    return np.clip(np.asarray(a(np.asarray(e_kev, float)), dtype=float), 1e-300, 1.0)


@cites("ueda2014", "moretti2003")
def ueda14_population(band_kev: tuple = (0.5, 2.0), log_l=(41.0, 47.0, 0.05), z=(0.005, 5.0, 0.02),
                      dlog_nh: float = 0.1) -> dict:
    """The Ueda et al. (2014) AGN population on a (log L, z, log N_H) grid: per cell the number per deg^2
    (XLF x absorption function x comoving volume) and the observed energy flux in ``band_kev`` of the stage's source
    spectrum, PhAbs(N_H, z) x power law with the mean photon index of the cell's class, the intrinsic 2-10 keV
    luminosity L: S = L int_{e1(1+z)}^{e2(1+z)} E^(1-G) T(E) dE / int_2^10 E^(1-G) dE / (4 pi d_L^2).  Ueda's
    reflection and scattered components are left out, as in the stage's spectra.  Cached."""
    import numpy as np

    from tngxmm.core.cosmology import flat_lcdm

    key = (tuple(band_kev), tuple(log_l), tuple(z), dlog_nh)
    if key in _UEDA_POP:
        return _UEDA_POP[key]
    u = UEDA14
    cosmo = flat_lcdm(u["h0"], u["om0"])          # Ueda's own cosmology, in which the XLF is defined
    ll = np.arange(log_l[0] + log_l[2] / 2, log_l[1], log_l[2])
    zz = np.arange(z[0] + z[2] / 2, z[1], z[2])
    nh = np.arange(20.0 + dlog_nh / 2, 26.0, dlog_nh)
    dvdz = cosmo.differential_comoving_volume(zz).to("Mpc3/sr").value * (math.pi / 180.0) ** 2   # per deg^2
    dl_cm = cosmo.luminosity_distance(zz).to("cm").value
    fnh = ueda14_nh_function(ll[:, None], zz[None, :])                              # (L, z, 5)
    which = np.searchsorted([b[1] for b in UEDA14_NH_BINS], nh, side="right")       # bin of each N_H
    which = np.minimum(which, len(UEDA14_NH_BINS) - 1)
    w = (ueda14_xlf(ll[:, None], zz[None, :])[..., None] * fnh[..., which]
         * dvdz[None, :, None] * log_l[2] * z[2] * dlog_nh)                          # (L, z, NH) per deg^2
    # flux per unit L for each (z, N_H): rest-frame band [e1, e2](1 + z)
    eg = np.logspace(math.log10(band_kev[0]), math.log10(band_kev[1] * (1.0 + zz[-1])), 3000)
    t1 = _phabs_unit_transmission(eg)
    k = np.empty((len(zz), len(nh)))
    for i, zi in enumerate(zz):
        sel = (eg >= band_kev[0] * (1 + zi)) & (eg <= band_kev[1] * (1 + zi))
        e, de = eg[sel], np.gradient(eg)[sel]
        for j, nj in enumerate(nh):
            g = u["gamma_thin"][0] if nj < 22.0 else u["gamma_thick"][0]
            num = (e ** (1.0 - g) * t1[sel] ** (10.0 ** (nj - 22.0)) * de).sum()
            den = (10.0 ** (2.0 - g) - 2.0 ** (2.0 - g)) / (2.0 - g)
            k[i, j] = num / den / (4.0 * math.pi * dl_cm[i] ** 2)
    flux = 10.0 ** ll[:, None, None] * k[None, :, :]
    out = dict(log_l=ll, z=zz, log_nh=nh, weight=w, flux=flux, dz=z[2], dlog_nh=dlog_nh)
    _UEDA_POP[key] = out
    return out


@cites("ueda2014")
def ueda14_draw(flux: Any, rng: Any, dlog_s: float = 0.05) -> tuple:
    """(z, log N_H, photon index) of sources of observed soft flux ``flux`` [erg/s/cm^2] drawn from the Ueda et al.
    (2014) population at that flux (:func:`ueda14_population`): a cell of the flux bin of width ``dlog_s`` with
    probability proportional to its number, z and log N_H uniform within the cell, the photon index from the
    Gaussian of the cell's class (log N_H < 22 or >= 22)."""
    import numpy as np

    pop = ueda14_population()
    ls = np.log10(np.maximum(pop["flux"], 1e-300)).ravel()
    w = pop["weight"].ravel()
    nl, nz, nn = pop["weight"].shape
    idx_bin = np.floor(ls / dlog_s).astype(np.int64)
    order = np.argsort(idx_bin, kind="stable")
    sb, sw = idx_bin[order], w[order]
    out_z, out_nh, out_g = (np.empty(len(flux)) for _ in range(3))
    want = np.floor(np.log10(np.asarray(flux, float)) / dlog_s).astype(np.int64)
    for b in np.unique(want):
        lo, hi = np.searchsorted(sb, b), np.searchsorted(sb, b, side="right")
        if hi <= lo:
            raise ValueError(f"no Ueda et al. (2014) cell at log S = {b * dlog_s:.2f}")
        cw = np.cumsum(sw[lo:hi])
        m = want == b
        pick = order[lo + np.searchsorted(cw, rng.random(int(m.sum())) * cw[-1])]
        _, iz, inh = np.unravel_index(pick, (nl, nz, nn))
        out_z[m] = pop["z"][iz] + (rng.random(len(iz)) - 0.5) * pop["dz"]
        out_nh[m] = pop["log_nh"][inh] + (rng.random(len(inh)) - 0.5) * pop["dlog_nh"]
    thick = out_nh >= 22.0
    mu = np.where(thick, UEDA14["gamma_thick"][0], UEDA14["gamma_thin"][0])
    sd = np.where(thick, UEDA14["gamma_thick"][1], UEDA14["gamma_thin"][1])
    out_g[:] = mu + sd * rng.standard_normal(len(out_g))
    return out_z, out_nh, out_g


# ── ELLBETA PSF ──────────────────────────────────────────────────────────────

_ELLBETA: dict = {}


@cites("xmm_ccf_sas", "read2011")
def ellbeta_table(psf_ccf_path: str) -> dict[str, Any]:
    """ELLBETA_PARAMS of an XRT*_XPSF CCF: {'energy_kev': (nE,), 'theta_arcmin': (nT,),
    'params': (nE, nT, 5)} with params = (r0 [arcsec], alpha, ellipticity, Gaussian FWHM
    [arcsec], Gaussian-to-King peak ratio), the PHI = 0 rows (the only ones tabulated)."""
    import numpy as np
    from astropy.io import fits

    key = str(psf_ccf_path)
    if key in _ELLBETA:
        return _ELLBETA[key]
    with fits.open(key) as f:
        d = f["ELLBETA_PARAMS"].data
        e = np.asarray(d["ENERGY"], float) / 1000.0
        th = np.degrees(np.asarray(d["THETA"], float)) * 60.0
        ph = np.asarray(d["PHI"], float)
        par = np.asarray(d["PARAMS"], float)
    m = ph == 0
    tab = ellbeta_grid(e[m], th[m], par[m])
    _ELLBETA[key] = tab
    return tab


def ellbeta_grid(energy_kev, theta_arcmin, params) -> dict[str, Any]:
    """Rows (energy, theta, 5 params) -> the regular grid of :func:`ellbeta_table`."""
    import numpy as np

    es = np.unique(np.round(energy_kev, 6))
    ts = np.unique(np.round(theta_arcmin, 6))
    grid = np.full((len(es), len(ts), 5), np.nan)
    for e, t, p in zip(energy_kev, theta_arcmin, params):
        grid[np.searchsorted(es, round(float(e), 6)), np.searchsorted(ts, round(float(t), 6))] = p
    if np.isnan(grid).any():
        raise ValueError("ELLBETA_PARAMS: the (energy, theta) rows do not form a full grid")
    return {"energy_kev": es, "theta_arcmin": ts, "params": grid}


def ellbeta_params(table: dict[str, Any], energy_kev: float, theta_arcmin: float) -> Any:
    """The 5 ELLBETA parameters at the nearest tabulated energy, linear in off-axis angle
    (clipped to the tabulated range)."""
    import numpy as np

    ie = int(np.argmin(np.abs(table["energy_kev"] - float(energy_kev))))
    ts = table["theta_arcmin"]
    p = table["params"][ie]
    t = float(np.clip(theta_arcmin, ts[0], ts[-1]))
    return np.array([np.interp(t, ts, p[:, k]) for k in range(5)])


def _king_radius(u, r0: float, alpha: float):
    """Inverse of the circular King EEF 1 - (1 + (r/r0)^2)^(1 - alpha) at u."""
    import numpy as np

    return r0 * np.sqrt((1.0 - u) ** (1.0 / (1.0 - alpha)) - 1.0)


def ellbeta_offsets(params, n: int, rng, tangential_rad: float,
                    rmax_arcsec: float = ELLBETA_RMAX_ARCSEC) -> tuple[Any, Any]:
    """``n`` PSF offsets [arcsec] (dx, dy) from the ELLBETA envelope ``params``.

    Sherpa beta2d/gauss2d with ellipticity e: r^2 = x'^2 + y'^2 / (1 - e)^2, x' along the major
    axis.  Both components are integrated over the ellipse ((1 - e) times the circular area),
    so the Gaussian weight is N FWHM^2 / (4 ln 2) against r0^2 / (alpha - 1) for the King
    profile (King inside ``rmax_arcsec``).  A circular radius is drawn from the chosen
    component, compressed by (1 - e) along the minor axis and rotated so the major axis lies
    along ``tangential_rad`` (angle of the tangential direction in the output frame)."""
    import numpy as np

    r0, alpha, eps, fwhm, nrat = (float(v) for v in params)
    eps = min(max(eps, 0.0), 0.95)
    n = int(n)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    umax = 1.0 - (1.0 + (rmax_arcsec / r0) ** 2) ** (1.0 - alpha)        # King EEF at rmax
    w_k = r0 ** 2 / (alpha - 1.0) * umax
    w_g = nrat * fwhm ** 2 / (4.0 * math.log(2.0)) if (fwhm > 0 and nrat > 0) else 0.0
    gauss = rng.random(n) < (w_g / (w_k + w_g))
    r = np.empty(n)
    nk = int((~gauss).sum())
    r[~gauss] = _king_radius(rng.random(nk) * umax, r0, alpha)
    ng = n - nk
    if ng:
        sig = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        r[gauss] = sig * np.sqrt(-2.0 * np.log(1.0 - rng.random(ng)))
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    xp = r * np.cos(phi)
    yp = (1.0 - eps) * r * np.sin(phi)
    c, s = math.cos(tangential_rad), math.sin(tangential_rad)
    return xp * c - yp * s, xp * s + yp * c


__cites__ = collect_cites(__name__)
