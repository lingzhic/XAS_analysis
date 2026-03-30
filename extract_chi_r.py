#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import savgol_filter

from mda_reader import read_mda


EV_PER_ANGSTROM_SQUARED = 3.8099819442818976
DEFAULT_FLUOR_DETECTORS = (16, 17, 18, 19)
DEFAULT_ENERGY_DETECTOR = 0
DEFAULT_I0_DETECTOR = 2
DEFAULT_I1_DETECTOR = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read MEX MDA XAS data and extract chi(k) / chi(R) without matplotlib."
    )
    parser.add_argument("input", help="Path to an MDA file, for example raw_data/MEX1_84446.mda")
    parser.add_argument(
        "--output-dir",
        default="exafs_output",
        help="Directory for CSV and SVG outputs",
    )
    parser.add_argument(
        "--monitor",
        choices=("i0", "i1"),
        default="i1",
        help="Ion chamber used to normalize fluorescence",
    )
    parser.add_argument(
        "--fluor-detectors",
        default="16,17,18,19",
        help="Comma-separated detector indices to sum as fluorescence",
    )
    parser.add_argument(
        "--e0",
        type=float,
        default=None,
        help="Edge energy in eV. If omitted, estimate it from dmu/dE.",
    )
    parser.add_argument("--pre-start", type=float, default=-150.0, help="Pre-edge start relative to E0, in eV")
    parser.add_argument("--pre-end", type=float, default=-30.0, help="Pre-edge end relative to E0, in eV")
    parser.add_argument("--post-start", type=float, default=150.0, help="Post-edge start relative to E0, in eV")
    parser.add_argument("--post-end", type=float, default=700.0, help="Post-edge end relative to E0, in eV")
    parser.add_argument("--dk", type=float, default=0.05, help="Uniform k-grid spacing in A^-1")
    parser.add_argument("--kmin", type=float, default=2.0, help="FT lower k bound in A^-1")
    parser.add_argument("--kmax", type=float, default=12.0, help="FT upper k bound in A^-1")
    parser.add_argument("--kweight", type=float, default=2.0, help="k-weight used before FT")
    parser.add_argument(
        "--background-window",
        type=float,
        default=3.0,
        help="Smoothing span in k-space for the atomic background, in A^-1",
    )
    parser.add_argument("--rmax", type=float, default=6.0, help="Upper R bound shown in chi(R) plot")
    return parser.parse_args()


def flatten_numeric(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def load_mex_mda(
    path: Path,
    monitor_name: str = "i1",
    fluor_detectors: Sequence[int] = DEFAULT_FLUOR_DETECTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata, dim1, dim2 = read_mda(path)
    del metadata, dim1

    energy_ev = flatten_numeric(dim2.d[DEFAULT_ENERGY_DETECTOR].data) * 1000.0
    if monitor_name == "i0":
        monitor = flatten_numeric(dim2.d[DEFAULT_I0_DETECTOR].data)
    else:
        monitor = flatten_numeric(dim2.d[DEFAULT_I1_DETECTOR].data)

    fluor = np.zeros_like(energy_ev)
    for det_index in fluor_detectors:
        fluor += flatten_numeric(dim2.d[det_index].data)

    return energy_ev, monitor, fluor


def clean_and_sort(energy_ev: np.ndarray, monitor: np.ndarray, fluor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = fluor / np.where(monitor > 0, monitor, np.nan)
    mask = np.isfinite(energy_ev) & np.isfinite(mu)
    energy_ev = energy_ev[mask]
    mu = mu[mask]

    order = np.argsort(energy_ev)
    energy_ev = energy_ev[order]
    mu = mu[order]

    unique_energy, inverse = np.unique(energy_ev, return_inverse=True)
    if unique_energy.size != energy_ev.size:
        mu_avg = np.zeros_like(unique_energy)
        counts = np.zeros_like(unique_energy)
        np.add.at(mu_avg, inverse, mu)
        np.add.at(counts, inverse, 1.0)
        energy_ev = unique_energy
        mu = mu_avg / counts

    return energy_ev, mu


def odd_window_length(points: int, minimum: int, maximum: int) -> int:
    size = max(minimum, points)
    if size % 2 == 0:
        size += 1
    if size > maximum:
        size = maximum if maximum % 2 == 1 else maximum - 1
    if size < 5:
        raise ValueError("Not enough data points for smoothing.")
    return size


def estimate_e0(energy_ev: np.ndarray, mu: np.ndarray) -> float:
    if energy_ev.size < 11:
        raise ValueError("Energy points are too few to estimate E0.")

    median_step = float(np.nanmedian(np.diff(energy_ev)))
    smooth_len = odd_window_length(max(int(0.03 * energy_ev.size), 11), 11, energy_ev.size - 1)
    mu_smooth = savgol_filter(mu, window_length=smooth_len, polyorder=3, mode="interp")
    dmu_de = np.gradient(mu_smooth, median_step)

    lo = max(3, int(0.05 * energy_ev.size))
    hi = min(energy_ev.size - 3, int(0.95 * energy_ev.size))
    search_slice = slice(lo, hi)
    peak_index = lo + int(np.nanargmax(dmu_de[search_slice]))
    return float(energy_ev[peak_index])


def build_mask(energy_ev: np.ndarray, start: float, end: float) -> np.ndarray:
    if start > end:
        start, end = end, start
    return (energy_ev >= start) & (energy_ev <= end)


def normalize_mu(
    energy_ev: np.ndarray,
    mu: np.ndarray,
    e0: float,
    pre_start: float,
    pre_end: float,
    post_start: float,
    post_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pre_mask = build_mask(energy_ev, e0 + pre_start, e0 + pre_end)
    post_hi = min(e0 + post_end, float(energy_ev.max() - 5.0))
    post_mask = build_mask(energy_ev, e0 + post_start, post_hi)

    if pre_mask.sum() < 5:
        raise ValueError("Pre-edge points are insufficient. Widen --pre-start/--pre-end.")
    if post_mask.sum() < 5:
        raise ValueError("Post-edge points are insufficient. Widen --post-start/--post-end.")

    pre_coef = np.polyfit(energy_ev[pre_mask], mu[pre_mask], deg=1)
    pre_line = np.polyval(pre_coef, energy_ev)
    mu_pre = mu - pre_line

    edge_step = float(np.nanmedian(mu_pre[post_mask]))
    if not np.isfinite(edge_step) or edge_step <= 0:
        raise ValueError("Edge step is invalid after pre-edge subtraction.")

    mu_norm = mu_pre / edge_step
    return pre_line, mu_pre, mu_norm, edge_step


def energy_to_k(energy_ev: np.ndarray, e0: float) -> np.ndarray:
    delta_e = np.clip(energy_ev - e0, 0.0, None)
    return np.sqrt(delta_e / EV_PER_ANGSTROM_SQUARED)


def smooth_background(mu_uniform: np.ndarray, points_per_window: int) -> np.ndarray:
    window_length = odd_window_length(points_per_window, 21, mu_uniform.size - 1)
    polyorder = min(3, window_length - 2)
    return savgol_filter(mu_uniform, window_length=window_length, polyorder=polyorder, mode="interp")


def interpolate_to_uniform_k(
    energy_ev: np.ndarray,
    mu_norm: np.ndarray,
    e0: float,
    dk: float,
    background_window: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = energy_to_k(energy_ev, e0)
    mask = k > 0.0
    k = k[mask]
    mu_norm = mu_norm[mask]

    if k.size < 50:
        raise ValueError("Not enough post-edge points to build chi(k).")

    k_uniform = np.arange(max(dk, float(k.min())), float(k.max()) + 0.5 * dk, dk)
    mu_uniform = np.interp(k_uniform, k, mu_norm)
    points_per_window = int(round(background_window / dk))
    mu_background = smooth_background(mu_uniform, points_per_window)
    chi_k = mu_uniform - mu_background
    return k_uniform, chi_k, mu_background


def hanning_window(k: np.ndarray, kmin: float, kmax: float) -> np.ndarray:
    window = np.zeros_like(k)
    mask = (k >= kmin) & (k <= kmax)
    if not np.any(mask):
        return window
    phase = (k[mask] - kmin) / (kmax - kmin)
    window[mask] = np.sin(np.pi * phase) ** 2
    return window


def fourier_transform_chi(
    k_uniform: np.ndarray,
    chi_k: np.ndarray,
    kmin: float,
    kmax: float,
    kweight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kmax <= kmin:
        raise ValueError("--kmax must be larger than --kmin.")

    window = hanning_window(k_uniform, kmin, kmax)
    if not np.any(window > 0):
        raise ValueError("Selected k-range does not overlap the data.")

    dk = float(np.nanmedian(np.diff(k_uniform)))
    weighted = chi_k * np.power(k_uniform, kweight) * window

    nfft = 1
    target = max(2048, weighted.size * 4)
    while nfft < target:
        nfft *= 2

    padded = np.zeros(nfft, dtype=float)
    padded[: weighted.size] = weighted

    ft = np.fft.rfft(padded) * dk
    r = np.fft.rfftfreq(nfft, d=dk) * np.pi
    return r, ft.real, np.abs(ft)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def format_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def svg_line_plot(
    path: Path,
    series: Sequence[dict[str, object]],
    x_label: str,
    y_label: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    width = 960
    height = 640
    left = 90
    right = 30
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    y_values = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]

    if x_values.size == 0 or y_values.size == 0:
        raise ValueError(f"No finite data available for {path.name}.")

    xmin = float(np.nanmin(x_values)) if xlim is None else float(xlim[0])
    xmax = float(np.nanmax(x_values)) if xlim is None else float(xlim[1])
    ymin = float(np.nanmin(y_values)) if ylim is None else float(ylim[0])
    ymax = float(np.nanmax(y_values)) if ylim is None else float(ylim[1])

    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        pad = 1.0 if math.isclose(ymin, 0.0) else abs(ymin) * 0.1
        ymin -= pad
        ymax += pad

    ypad = 0.06 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def px(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_width

    def py(y: float) -> float:
        return top + plot_height - (y - ymin) / (ymax - ymin) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-size="24" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black" stroke-width="2"/>',
    ]

    if ymin < 0.0 < ymax:
        y0 = py(0.0)
        parts.append(
            f'<line x1="{left}" y1="{y0:.2f}" x2="{left + plot_width}" y2="{y0:.2f}" stroke="#999" stroke-dasharray="6,6" stroke-width="1"/>'
        )

    for tick in np.linspace(xmin, xmax, 6):
        xt = px(float(tick))
        parts.append(f'<line x1="{xt:.2f}" y1="{top + plot_height}" x2="{xt:.2f}" y2="{top + plot_height + 8}" stroke="black"/>')
        parts.append(
            f'<text x="{xt:.2f}" y="{top + plot_height + 30}" text-anchor="middle" font-size="16" font-family="Helvetica">{format_tick(float(tick))}</text>'
        )

    for tick in np.linspace(ymin + ypad, ymax - ypad, 6):
        yt = py(float(tick))
        parts.append(f'<line x1="{left - 8}" y1="{yt:.2f}" x2="{left}" y2="{yt:.2f}" stroke="black"/>')
        parts.append(
            f'<text x="{left - 14}" y="{yt + 5:.2f}" text-anchor="end" font-size="16" font-family="Helvetica">{format_tick(float(tick))}</text>'
        )

    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="20" font-family="Helvetica">{x_label}</text>'
    )
    parts.append(
        f'<text x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="20" font-family="Helvetica" transform="rotate(-90 24 {top + plot_height / 2:.1f})">{y_label}</text>'
    )

    legend_y = top + 18
    legend_x = left + plot_width - 180
    for item in series:
        x_data = np.asarray(item["x"], dtype=float)
        y_data = np.asarray(item["y"], dtype=float)
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[mask]
        y_data = y_data[mask]
        in_bounds = (x_data >= xmin) & (x_data <= xmax) & (y_data >= ymin) & (y_data <= ymax)
        x_data = x_data[in_bounds]
        y_data = y_data[in_bounds]
        if x_data.size == 0:
            continue

        points = " ".join(f"{px(float(x)):.2f},{py(float(y)):.2f}" for x, y in zip(x_data, y_data))
        color = str(item.get("color", "#004c6d"))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

        label = str(item.get("label", ""))
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 34}" y="{legend_y + 6}" font-size="16" font-family="Helvetica">{label}</text>'
        )
        legend_y += 24

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def analysis_summary(
    input_path: Path,
    output_dir: Path,
    e0: float,
    edge_step: float,
    kmin: float,
    kmax: float,
    kweight: float,
) -> str:
    return "\n".join(
        [
            f"input_file: {input_path}",
            f"output_dir: {output_dir}",
            f"e0_eV: {e0:.3f}",
            f"edge_step: {edge_step:.6g}",
            f"kmin_A^-1: {kmin:.3f}",
            f"kmax_A^-1: {kmax:.3f}",
            f"kweight: {kweight:.3f}",
        ]
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fluor_detectors = tuple(int(piece.strip()) for piece in args.fluor_detectors.split(",") if piece.strip())
    energy_ev, monitor, fluor = load_mex_mda(input_path, monitor_name=args.monitor, fluor_detectors=fluor_detectors)
    energy_ev, mu = clean_and_sort(energy_ev, monitor, fluor)

    e0 = float(args.e0) if args.e0 is not None else estimate_e0(energy_ev, mu)
    pre_line, mu_pre, mu_norm, edge_step = normalize_mu(
        energy_ev=energy_ev,
        mu=mu,
        e0=e0,
        pre_start=args.pre_start,
        pre_end=args.pre_end,
        post_start=args.post_start,
        post_end=args.post_end,
    )

    k_uniform, chi_k, mu_background = interpolate_to_uniform_k(
        energy_ev=energy_ev,
        mu_norm=mu_norm,
        e0=e0,
        dk=args.dk,
        background_window=args.background_window,
    )

    kmax = min(float(args.kmax), float(k_uniform.max()))
    r, chi_r_real, chi_r_mag = fourier_transform_chi(
        k_uniform=k_uniform,
        chi_k=chi_k,
        kmin=float(args.kmin),
        kmax=kmax,
        kweight=float(args.kweight),
    )

    energy_csv = output_dir / "mu_normalized.csv"
    chi_k_csv = output_dir / "chi_k.csv"
    chi_r_csv = output_dir / "chi_r.csv"
    summary_txt = output_dir / "summary.txt"
    mu_svg = output_dir / "mu_e.svg"
    chi_k_svg = output_dir / "chi_k.svg"
    chi_r_svg = output_dir / "chi_r.svg"

    write_csv(
        energy_csv,
        header=("energy_eV", "mu_raw", "pre_line", "mu_pre", "mu_norm"),
        rows=zip(energy_ev, mu, pre_line, mu_pre, mu_norm),
    )
    write_csv(
        chi_k_csv,
        header=("k_A^-1", "mu_background", "chi_k", f"k^{args.kweight:g}*chi_k"),
        rows=zip(k_uniform, mu_background, chi_k, chi_k * np.power(k_uniform, args.kweight)),
    )
    write_csv(
        chi_r_csv,
        header=("R_A", "Re_chi_R", "Abs_chi_R"),
        rows=zip(r, chi_r_real, chi_r_mag),
    )
    summary_txt.write_text(
        analysis_summary(
            input_path=input_path,
            output_dir=output_dir,
            e0=e0,
            edge_step=edge_step,
            kmin=float(args.kmin),
            kmax=kmax,
            kweight=float(args.kweight),
        ),
        encoding="utf-8",
    )

    svg_line_plot(
        mu_svg,
        series=(
            {"x": energy_ev, "y": mu_norm, "label": "mu_norm", "color": "#005f73"},
            {"x": energy_ev, "y": np.ones_like(energy_ev), "label": "edge step", "color": "#94d2bd"},
        ),
        x_label="Energy (eV)",
        y_label="Normalized mu(E)",
        title="Normalized XAS",
    )
    svg_line_plot(
        chi_k_svg,
        series=(
            {
                "x": k_uniform,
                "y": chi_k * np.power(k_uniform, args.kweight),
                "label": f"k^{args.kweight:g} chi(k)",
                "color": "#ca6702",
            },
        ),
        x_label="k (A^-1)",
        y_label=f"k^{args.kweight:g} chi(k)",
        title="EXAFS in k-space",
    )
    svg_line_plot(
        chi_r_svg,
        series=(
            {"x": r, "y": chi_r_mag, "label": "|chi(R)|", "color": "#0a9396"},
            {"x": r, "y": chi_r_real, "label": "Re[chi(R)]", "color": "#bb3e03"},
        ),
        x_label="R (A)",
        y_label="FT magnitude / real part",
        title="EXAFS Fourier Transform",
        xlim=(0.0, float(args.rmax)),
    )

    print(summary_txt.read_text(encoding="utf-8"))
    print(f"wrote: {mu_svg}")
    print(f"wrote: {chi_k_svg}")
    print(f"wrote: {chi_r_svg}")


if __name__ == "__main__":
    main()
