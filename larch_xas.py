from __future__ import annotations

import numpy as np
from larch import Group
from larch.xafs import autobk, pre_edge, xftf


def _as_1d_float(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _average_duplicate_x(x, y):
    x_unique, inverse = np.unique(x, return_inverse=True)
    if len(x_unique) == len(x):
        return x, y

    y_sum = np.zeros_like(x_unique, dtype=float)
    counts = np.zeros_like(x_unique, dtype=float)
    np.add.at(y_sum, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return x_unique, y_sum / counts


def clean_transmission_arrays(energy, i0, i1):
    energy = _as_1d_float(energy)
    i0 = _as_1d_float(i0)
    i1 = _as_1d_float(i1)

    mask = np.isfinite(energy) & np.isfinite(i0) & np.isfinite(i1) & (i0 > 0) & (i1 > 0)
    energy = energy[mask]
    i0 = i0[mask]
    i1 = i1[mask]

    if energy.size < 5:
        raise ValueError("Not enough valid transmission points for Larch pre_edge().")

    order = np.argsort(energy)
    energy = energy[order]
    mu = np.log(i0[order] / i1[order])

    energy, mu = _average_duplicate_x(energy, mu)
    return energy, mu


def flatten_transmission(
    energy,
    i0,
    i1,
    *,
    e0=None,
    pre1=None,
    pre2=None,
    norm1=None,
    norm2=None,
    nnorm=None,
    nvict=0,
    make_flat=True,
    name="transmission",
):
    energy, mu = clean_transmission_arrays(energy, i0, i1)

    group = Group(name=name, energy=energy, mu=mu)
    pre_edge(
        group.energy,
        group.mu,
        group=group,
        e0=e0,
        pre1=pre1,
        pre2=pre2,
        norm1=norm1,
        norm2=norm2,
        nnorm=nnorm,
        nvict=nvict,
        make_flat=make_flat,
    )

    return {
        "group": group,
        "energy": group.energy,
        "mu": group.mu,
        "norm": group.norm,
        "flat": group.flat,
        "pre_edge": group.pre_edge,
        "post_edge": group.post_edge,
        "e0": group.e0,
        "edge_step": group.edge_step,
    }


def flatten_transmission_frame(
    frame,
    *,
    energy_col="energy_eV",
    i0_col="i0_nanoamps",
    i1_col="i1_nanoamps",
    **kws,
):
    return flatten_transmission(
        frame[energy_col],
        frame[i0_col],
        frame[i1_col],
        **kws,
    )


def extract_exafs_transmission(
    energy,
    i0,
    i1,
    *,
    e0=None,
    pre1=None,
    pre2=None,
    norm1=None,
    norm2=None,
    nnorm=None,
    nvict=0,
    make_flat=True,
    rbkg=1.0,
    bkg_kmin=0,
    bkg_kmax=None,
    bkg_kweight=1,
    bkg_dk=0.1,
    bkg_win="hanning",
    ft_kmin=2,
    ft_kmax=12,
    ft_kweight=2,
    ft_dk=2,
    ft_window="kaiser",
    rmax_out=6.0,
    with_phase=False,
    name="transmission",
):
    out = flatten_transmission(
        energy,
        i0,
        i1,
        e0=e0,
        pre1=pre1,
        pre2=pre2,
        norm1=norm1,
        norm2=norm2,
        nnorm=nnorm,
        nvict=nvict,
        make_flat=make_flat,
        name=name,
    )

    group = out["group"]

    autobk(
        group.energy,
        group.mu,
        group=group,
        rbkg=rbkg,
        ek0=group.e0,
        edge_step=group.edge_step,
        kmin=bkg_kmin,
        kmax=bkg_kmax,
        kweight=bkg_kweight,
        dk=bkg_dk,
        win=bkg_win,
    )
    xftf(
        group.k,
        group.chi,
        group=group,
        kmin=ft_kmin,
        kmax=ft_kmax,
        kweight=ft_kweight,
        dk=ft_dk,
        window=ft_window,
        rmax_out=rmax_out,
        with_phase=with_phase,
    )

    out.update(
        {
            "bkg": group.bkg,
            "chie": group.chie,
            "k": group.k,
            "chi": group.chi,
            "chi_kw": group.chi * np.power(group.k, ft_kweight),
            "kwin": group.kwin,
            "r": group.r,
            "chir": group.chir,
            "chir_mag": group.chir_mag,
            "chir_re": group.chir_re,
            "chir_im": group.chir_im,
            "rbkg": group.rbkg,
            "ek0": group.ek0,
        }
    )
    if with_phase and hasattr(group, "chir_pha"):
        out["chir_pha"] = group.chir_pha
    return out


def extract_exafs_transmission_frame(
    frame,
    *,
    energy_col="energy_eV",
    i0_col="i0_nanoamps",
    i1_col="i1_nanoamps",
    **kws,
):
    return extract_exafs_transmission(
        frame[energy_col],
        frame[i0_col],
        frame[i1_col],
        **kws,
    )
