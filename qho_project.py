"""Integrated numerical study of the 1D quantum harmonic oscillator.

This is the primary runnable deliverable for Team 7. It consolidates the
Hamiltonian construction, sparse eigensolver, analytic comparisons, physical
sanity checks, grid-spacing convergence, box-size convergence, tables, and
figures behind one main function.

AI-use note: OpenAI Codex assisted Anderson Mao with integrating the team's
existing notebooks, refactoring repeated operations, and drafting validation
tests and documentation. The team should review all code and results before
submission. Scientific and software references are listed in CITATIONS.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.special import eval_hermite, gammaln


@dataclass(frozen=True)
class QHOConfig:
    """Physical and numerical parameters for one finite-box calculation."""

    half_width: float = 8.0
    grid_points: int = 1000
    num_states: int = 6
    hbar: float = 1.0
    mass: float = 1.0
    omega: float = 1.0

    def __post_init__(self) -> None:
        if self.half_width <= 0:
            raise ValueError("half_width must be positive")
        if self.grid_points < self.num_states + 3:
            raise ValueError("grid_points is too small for the requested states")
        if self.num_states < 1:
            raise ValueError("num_states must be positive")
        if min(self.hbar, self.mass, self.omega) <= 0:
            raise ValueError("hbar, mass, and omega must be positive")


@dataclass
class QHOSolution:
    """Grid, Hamiltonian, and normalized low-energy eigenpairs."""

    config: QHOConfig
    x_full: np.ndarray
    x: np.ndarray
    dx: float
    hamiltonian: csr_matrix
    energies: np.ndarray
    states: np.ndarray


@dataclass
class BaselineAnalysis:
    """Baseline solution and all comparison products derived from it."""

    solution: QHOSolution
    analytic_states: np.ndarray
    energy_table: pd.DataFrame
    state_table: pd.DataFrame
    checks: dict[str, float | int | bool]


def build_hamiltonian(
    config: QHOConfig,
) -> tuple[np.ndarray, np.ndarray, float, csr_matrix]:
    """Construct the sparse finite-difference Hamiltonian.

    The full grid includes both endpoints. Dirichlet conditions set the
    wavefunction to zero there, so the matrix acts only on interior values.
    """

    x_full = np.linspace(
        -config.half_width,
        config.half_width,
        config.grid_points,
        dtype=float,
    )
    dx = float(x_full[1] - x_full[0])
    x = x_full[1:-1]
    dimension = x.size

    kinetic_diagonal = config.hbar**2 / (config.mass * dx**2)
    kinetic_off_diagonal = -0.5 * config.hbar**2 / (config.mass * dx**2)
    potential = 0.5 * config.mass * config.omega**2 * x**2

    hamiltonian = diags(
        diagonals=[
            np.full(dimension - 1, kinetic_off_diagonal),
            kinetic_diagonal + potential,
            np.full(dimension - 1, kinetic_off_diagonal),
        ],
        offsets=[-1, 0, 1],
        shape=(dimension, dimension),
        format="csr",
    )
    return x_full, x, dx, hamiltonian


def validate_hamiltonian(
    config: QHOConfig,
    x: np.ndarray,
    dx: float,
    hamiltonian: csr_matrix,
) -> dict[str, float | int | bool]:
    """Check matrix dimensions, symmetry, sparsity, and coefficients."""

    dimension = config.grid_points - 2
    symmetry_residual = hamiltonian - hamiltonian.T
    symmetry_error = (
        float(np.max(np.abs(symmetry_residual.data)))
        if symmetry_residual.nnz
        else 0.0
    )

    expected_main = (
        config.hbar**2 / (config.mass * dx**2)
        + 0.5 * config.mass * config.omega**2 * x**2
    )
    expected_off_diagonal = np.full(
        dimension - 1,
        -0.5 * config.hbar**2 / (config.mass * dx**2),
    )
    main_diagonal_error = float(
        np.max(np.abs(hamiltonian.diagonal() - expected_main))
    )
    off_diagonal_error = float(
        np.max(np.abs(hamiltonian.diagonal(1) - expected_off_diagonal))
    )

    checks: dict[str, float | int | bool] = {
        "dimension": dimension,
        "stored_nonzeros": int(hamiltonian.nnz),
        "expected_nonzeros": 3 * dimension - 2,
        "symmetry_error": symmetry_error,
        "main_diagonal_error": main_diagonal_error,
        "off_diagonal_error": off_diagonal_error,
    }

    assert hamiltonian.shape == (dimension, dimension)
    assert hamiltonian.nnz == checks["expected_nonzeros"]
    assert symmetry_error < 1e-12
    assert main_diagonal_error < 1e-12
    assert off_diagonal_error < 1e-12
    return checks


def solve_qho(config: QHOConfig) -> QHOSolution:
    """Return the lowest states using sparse shift-invert Lanczos."""

    x_full, x, dx, hamiltonian = build_hamiltonian(config)
    deterministic_start = np.random.default_rng(2026).normal(size=x.size)
    energies, states = eigsh(
        hamiltonian.tocsc(),
        k=config.num_states,
        sigma=0.0,
        which="LM",
        v0=deterministic_start,
    )
    order = np.argsort(energies)
    energies = energies[order]
    states = states[:, order]

    # The endpoints are zero, so dx * sum(|psi_i|^2) is the trapezoidal
    # integral over the full grid.
    norms = np.sqrt(dx * np.sum(np.abs(states) ** 2, axis=0))
    states = states / norms
    return QHOSolution(
        config=config,
        x_full=x_full,
        x=x,
        dx=dx,
        hamiltonian=hamiltonian,
        energies=energies,
        states=states,
    )


def analytic_energies(config: QHOConfig) -> np.ndarray:
    """Return E_n = hbar * omega * (n + 1/2)."""

    quantum_numbers = np.arange(config.num_states)
    return config.hbar * config.omega * (quantum_numbers + 0.5)


def analytic_wavefunctions(
    config: QHOConfig,
    positions: np.ndarray,
) -> np.ndarray:
    """Evaluate the normalized analytic eigenfunctions on a grid."""

    alpha = config.mass * config.omega / config.hbar
    quantum_numbers = np.arange(config.num_states)
    log_normalizations = (
        0.25 * np.log(alpha / np.pi)
        - 0.5
        * (
            quantum_numbers * np.log(2.0)
            + gammaln(quantum_numbers + 1.0)
        )
    )
    gaussian = np.exp(-0.5 * alpha * positions**2)
    columns = tuple(
        np.exp(log_normalizations[state])
        * eval_hermite(state, np.sqrt(alpha) * positions)
        * gaussian
        for state in quantum_numbers
    )
    return np.column_stack(columns)


def align_eigenstate_signs(
    numerical_states: np.ndarray,
    analytic_states: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Remove the physically irrelevant overall sign difference."""

    overlaps = dx * np.sum(numerical_states * analytic_states, axis=0)
    signs = np.where(overlaps < 0.0, -1.0, 1.0)
    return numerical_states * signs


def count_resolved_nodes(state: np.ndarray) -> int:
    """Count sign changes while excluding numerical noise in the tails."""

    threshold = 1e-5 * np.max(np.abs(state))
    resolved = state[np.abs(state) > threshold]
    return int(np.count_nonzero(resolved[:-1] * resolved[1:] < 0.0))


def analyze_baseline(config: QHOConfig) -> BaselineAnalysis:
    """Solve and validate one baseline calculation."""

    solution = solve_qho(config)
    matrix_checks = validate_hamiltonian(
        config,
        solution.x,
        solution.dx,
        solution.hamiltonian,
    )
    quantum_numbers = np.arange(config.num_states)
    exact_energies = analytic_energies(config)
    absolute_energy_error = np.abs(solution.energies - exact_energies)
    relative_energy_error_percent = (
        100.0 * absolute_energy_error / exact_energies
    )

    energy_table = pd.DataFrame(
        {
            "n": quantum_numbers,
            "numerical_energy": solution.energies,
            "analytic_energy": exact_energies,
            "absolute_error": absolute_energy_error,
            "relative_error_percent": relative_energy_error_percent,
        }
    )

    reference_energies = eigh_tridiagonal(
        solution.hamiltonian.diagonal(),
        solution.hamiltonian.diagonal(1),
        eigvals_only=True,
        select="i",
        select_range=(0, config.num_states - 1),
    )
    solver_difference = float(
        np.max(np.abs(solution.energies - reference_energies))
    )

    exact_states = analytic_wavefunctions(config, solution.x)
    exact_norms = np.sqrt(
        solution.dx * np.sum(np.abs(exact_states) ** 2, axis=0)
    )
    exact_states = exact_states / exact_norms
    aligned_states = align_eigenstate_signs(
        solution.states,
        exact_states,
        solution.dx,
    )
    solution.states = aligned_states

    overlap_matrix = (
        solution.dx * aligned_states.T.conj() @ aligned_states
    )
    normalization_error = float(
        np.max(np.abs(np.diag(overlap_matrix) - 1.0))
    )
    off_diagonal_overlap = overlap_matrix - np.diag(
        np.diag(overlap_matrix)
    )
    orthogonality_error = float(np.max(np.abs(off_diagonal_overlap)))

    wavefunction_l2_error = np.sqrt(
        solution.dx * np.sum((aligned_states - exact_states) ** 2, axis=0)
    )
    numerical_density = np.abs(aligned_states) ** 2
    analytic_density = np.abs(exact_states) ** 2
    density_l1_error = solution.dx * np.sum(
        np.abs(numerical_density - analytic_density),
        axis=0,
    )
    parity = (-1.0) ** quantum_numbers
    parity_l2_error = np.sqrt(
        solution.dx
        * np.sum(
            (
                aligned_states
                - aligned_states[::-1, :] * parity[np.newaxis, :]
            )
            ** 2,
            axis=0,
        )
    )
    near_boundary_amplitude = np.maximum(
        np.abs(aligned_states[0, :]),
        np.abs(aligned_states[-1, :]),
    )
    node_counts = np.fromiter(
        (
            count_resolved_nodes(aligned_states[:, state])
            for state in quantum_numbers
        ),
        dtype=int,
        count=config.num_states,
    )

    state_table = pd.DataFrame(
        {
            "n": quantum_numbers,
            "wavefunction_l2_error": wavefunction_l2_error,
            "density_l1_error": density_l1_error,
            "parity_l2_error": parity_l2_error,
            "near_boundary_amplitude": near_boundary_amplitude,
            "nodes": node_counts,
            "expected_nodes": quantum_numbers,
        }
    )

    checks = {
        **matrix_checks,
        "solver_difference": solver_difference,
        "max_energy_error": float(np.max(absolute_energy_error)),
        "max_relative_energy_error_percent": float(
            np.max(relative_energy_error_percent)
        ),
        "normalization_error": normalization_error,
        "orthogonality_error": orthogonality_error,
        "max_wavefunction_l2_error": float(
            np.max(wavefunction_l2_error)
        ),
        "max_density_l1_error": float(np.max(density_l1_error)),
        "max_parity_l2_error": float(np.max(parity_l2_error)),
        "max_near_boundary_amplitude": float(
            np.max(near_boundary_amplitude)
        ),
        "node_counts_correct": bool(
            np.array_equal(node_counts, quantum_numbers)
        ),
    }

    assert solver_difference < 1e-9
    assert checks["max_energy_error"] < 5e-4
    assert normalization_error < 1e-10
    assert orthogonality_error < 1e-10
    assert checks["max_wavefunction_l2_error"] < 5e-4
    assert checks["max_density_l1_error"] < 5e-4
    assert checks["max_parity_l2_error"] < 1e-10
    assert checks["max_near_boundary_amplitude"] < 1e-8
    assert checks["node_counts_correct"]

    return BaselineAnalysis(
        solution=solution,
        analytic_states=exact_states,
        energy_table=energy_table,
        state_table=state_table,
        checks=checks,
    )


def grid_spacing_study(
    base_config: QHOConfig,
    grid_points: Iterable[int] = (200, 400, 800, 1600, 3200, 6400),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refine dx while keeping the physical boundaries fixed."""

    point_counts = np.asarray(tuple(grid_points), dtype=int)
    rows = point_counts.size
    energies = np.empty((rows, base_config.num_states))
    spacings = np.empty(rows)
    dimensions = np.empty(rows, dtype=int)
    nonzeros = np.empty(rows, dtype=int)
    sparse_megabytes = np.empty(rows)
    dense_megabytes = np.empty(rows)
    solve_seconds = np.empty(rows)

    for index, points in enumerate(point_counts):
        config = replace(base_config, grid_points=int(points))
        start = time.perf_counter()
        solution = solve_qho(config)
        solve_seconds[index] = time.perf_counter() - start
        energies[index, :] = solution.energies
        spacings[index] = solution.dx
        dimensions[index] = solution.x.size
        nonzeros[index] = solution.hamiltonian.nnz
        sparse_megabytes[index] = (
            solution.hamiltonian.data.nbytes
            + solution.hamiltonian.indices.nbytes
            + solution.hamiltonian.indptr.nbytes
        ) / 1e6
        dense_megabytes[index] = (
            solution.x.size**2 * np.dtype(float).itemsize / 1e6
        )

    errors = np.abs(energies - analytic_energies(base_config))
    data: dict[str, np.ndarray] = {
        "N": point_counts,
        "M": dimensions,
        "dx": spacings,
        "stored_nonzeros": nonzeros,
        "sparse_megabytes": sparse_megabytes,
        "dense_megabytes": dense_megabytes,
        "solve_seconds": solve_seconds,
    }
    data.update(
        {
            f"energy_{state}": energies[:, state]
            for state in range(base_config.num_states)
        }
    )
    data.update(
        {
            f"absolute_error_{state}": errors[:, state]
            for state in range(base_config.num_states)
        }
    )
    study = pd.DataFrame(data)

    fitted_orders = np.fromiter(
        (
            np.polyfit(np.log(spacings), np.log(errors[:, state]), 1)[0]
            for state in range(base_config.num_states)
        ),
        dtype=float,
        count=base_config.num_states,
    )
    order_table = pd.DataFrame(
        {
            "n": np.arange(base_config.num_states),
            "fitted_order": fitted_orders,
            "coarsest_error": errors[0, :],
            "finest_error": errors[-1, :],
            "error_reduction_factor": errors[0, :] / errors[-1, :],
        }
    )
    assert np.all((fitted_orders > 1.9) & (fitted_orders < 2.1))
    return study, order_table


def box_size_study(
    base_config: QHOConfig,
    half_widths: Iterable[float] = (
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ),
    target_dx: float = 0.02,
) -> pd.DataFrame:
    """Increase the box while holding dx fixed.

    For each half-width L, N is selected using N = round(2L/dx) + 1.
    The default values make dx exactly 0.02 for every row.
    """

    widths = np.asarray(tuple(half_widths), dtype=float)
    point_counts = np.rint(2.0 * widths / target_dx).astype(int) + 1
    rows = widths.size
    energies = np.empty((rows, base_config.num_states))
    edge_amplitudes = np.empty((rows, base_config.num_states))
    actual_spacings = np.empty(rows)
    dimensions = np.empty(rows, dtype=int)
    solve_seconds = np.empty(rows)

    for index, (width, points) in enumerate(
        zip(widths, point_counts, strict=True)
    ):
        config = replace(
            base_config,
            half_width=float(width),
            grid_points=int(points),
        )
        start = time.perf_counter()
        solution = solve_qho(config)
        solve_seconds[index] = time.perf_counter() - start
        energies[index, :] = solution.energies
        edge_amplitudes[index, :] = np.maximum(
            np.abs(solution.states[0, :]),
            np.abs(solution.states[-1, :]),
        )
        actual_spacings[index] = solution.dx
        dimensions[index] = solution.x.size

    exact_energies = analytic_energies(base_config)
    absolute_errors = np.abs(energies - exact_energies)
    changes_from_largest_box = np.abs(energies - energies[-1, :])
    data: dict[str, np.ndarray] = {
        "L": widths,
        "N": point_counts,
        "M": dimensions,
        "dx": actual_spacings,
        "solve_seconds": solve_seconds,
        "max_absolute_energy_error": np.max(absolute_errors, axis=1),
        "max_change_from_largest_box": np.max(
            changes_from_largest_box,
            axis=1,
        ),
        "max_near_boundary_amplitude": np.max(
            edge_amplitudes,
            axis=1,
        ),
    }
    data.update(
        {
            f"energy_{state}": energies[:, state]
            for state in range(base_config.num_states)
        }
    )
    data.update(
        {
            f"absolute_error_{state}": absolute_errors[:, state]
            for state in range(base_config.num_states)
        }
    )
    data.update(
        {
            f"change_from_largest_box_{state}":
                changes_from_largest_box[:, state]
            for state in range(base_config.num_states)
        }
    )
    data.update(
        {
            f"near_boundary_amplitude_{state}":
                edge_amplitudes[:, state]
            for state in range(base_config.num_states)
        }
    )
    study = pd.DataFrame(data)

    assert np.ptp(actual_spacings) < 1e-12
    assert np.max(np.abs(energies[-1, :] - energies[-2, :])) < 1e-8
    assert np.max(edge_amplitudes[-1, :]) < 1e-8
    return study


def _finish_figure(
    figure: plt.Figure,
    output_path: Path | None,
) -> plt.Figure:
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
    return figure


def plot_energy_comparison(
    analysis: BaselineAnalysis,
    output_path: Path | None = None,
) -> plt.Figure:
    table = analysis.energy_table
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(
        table["n"],
        table["analytic_energy"],
        "o-",
        label="Analytic",
    )
    axes[0].plot(
        table["n"],
        table["numerical_energy"],
        "x--",
        label="Numerical",
    )
    axes[0].set(
        xlabel="Quantum number n",
        ylabel="Energy",
        title="Lowest harmonic-oscillator energies",
    )
    axes[0].set_xticks(table["n"])
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(
        table["n"],
        table["absolute_error"],
        "s-",
    )
    axes[1].set(
        xlabel="Quantum number n",
        ylabel="Absolute energy error",
        title="Finite-difference energy error",
    )
    axes[1].set_xticks(table["n"])
    axes[1].grid(alpha=0.3, which="both")
    return _finish_figure(figure, output_path)


def plot_wavefunctions(
    analysis: BaselineAnalysis,
    output_path: Path | None = None,
) -> plt.Figure:
    states = analysis.solution.states
    exact_states = analysis.analytic_states
    positions = analysis.solution.x
    columns = 3
    rows = int(np.ceil(analysis.solution.config.num_states / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.2, 3.3 * rows),
        sharex=True,
        squeeze=False,
    )
    for state, axis in enumerate(axes.flat):
        if state >= analysis.solution.config.num_states:
            axis.set_visible(False)
            continue
        axis.plot(
            positions,
            states[:, state],
            linewidth=2,
            label="Numerical",
        )
        axis.plot(
            positions,
            exact_states[:, state],
            "--",
            linewidth=1.5,
            label="Analytic",
        )
        axis.set_title(f"n = {state}")
        axis.set_xlim(-6, 6)
        axis.grid(alpha=0.25)
        if state >= columns * (rows - 1):
            axis.set_xlabel("x")
        if state % columns == 0:
            axis.set_ylabel("Wavefunction")
    axes[0, 0].legend(fontsize=9)
    figure.suptitle(
        "Numerical and analytic harmonic-oscillator wavefunctions"
    )
    return _finish_figure(figure, output_path)


def plot_probability_densities(
    analysis: BaselineAnalysis,
    output_path: Path | None = None,
) -> plt.Figure:
    numerical_density = np.abs(analysis.solution.states) ** 2
    analytic_density = np.abs(analysis.analytic_states) ** 2
    positions = analysis.solution.x
    columns = 3
    rows = int(np.ceil(analysis.solution.config.num_states / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.2, 3.3 * rows),
        sharex=True,
        squeeze=False,
    )
    for state, axis in enumerate(axes.flat):
        if state >= analysis.solution.config.num_states:
            axis.set_visible(False)
            continue
        axis.plot(
            positions,
            numerical_density[:, state],
            linewidth=2,
            label="Numerical",
        )
        axis.plot(
            positions,
            analytic_density[:, state],
            "--",
            linewidth=1.5,
            label="Analytic",
        )
        axis.set_title(f"n = {state}")
        axis.set_xlim(-6, 6)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        if state >= columns * (rows - 1):
            axis.set_xlabel("x")
        if state % columns == 0:
            axis.set_ylabel("Probability density")
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("Numerical and analytic probability densities")
    return _finish_figure(figure, output_path)


def plot_grid_convergence(
    study: pd.DataFrame,
    num_states: int,
    output_path: Path | None = None,
) -> plt.Figure:
    order = np.argsort(study["dx"].to_numpy())
    spacings = study["dx"].to_numpy()[order]
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    for state in range(num_states):
        errors = study[f"absolute_error_{state}"].to_numpy()[order]
        axis.loglog(spacings, errors, "o-", label=f"n = {state}")
    reference = (
        study["absolute_error_0"].iloc[0]
        * (spacings / study["dx"].iloc[0]) ** 2
    )
    axis.loglog(
        spacings,
        reference,
        "k--",
        alpha=0.7,
        label="Second-order reference",
    )
    axis.set(
        xlabel="Grid spacing dx",
        ylabel="Absolute energy error",
        title="Grid refinement at fixed L = 8",
    )
    axis.grid(alpha=0.3, which="both")
    axis.set_xticks(spacings)
    axis.set_xticklabels([f"{spacing:.4g}" for spacing in spacings])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.legend(ncol=2, fontsize=8)
    return _finish_figure(figure, output_path)


def plot_box_convergence(
    study: pd.DataFrame,
    num_states: int,
    output_path: Path | None = None,
) -> plt.Figure:
    widths = study["L"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for state in range(num_states):
        changes = study[
            f"change_from_largest_box_{state}"
        ].to_numpy()
        positive_changes = np.where(changes > 0.0, changes, np.nan)
        axes[0].semilogy(
            widths,
            positive_changes,
            "o-",
            label=f"n = {state}",
        )
        axes[1].semilogy(
            widths,
            study[f"near_boundary_amplitude_{state}"],
            "o-",
            label=f"n = {state}",
        )

    axes[0].set(
        xlabel="Box half-width L",
        ylabel="|E(L) - E(L=8)|",
        title="Energy change at fixed dx = 0.02",
    )
    axes[1].set(
        xlabel="Box half-width L",
        ylabel="Near-boundary amplitude",
        title="Wavefunction decay near boundaries",
    )
    for axis in axes:
        axis.grid(alpha=0.3, which="both")
        axis.set_xlim(widths.min(), widths.max())
    axes[0].legend(ncol=2, fontsize=8)
    return _finish_figure(figure, output_path)


def plot_cost_scaling(
    study: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    dimensions = study["M"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].loglog(
        dimensions,
        study["solve_seconds"],
        "o-",
    )
    axes[0].set(
        xlabel="Matrix dimension M",
        ylabel="Solve time (s)",
        title="Sparse shift-invert runtime",
    )
    axes[1].loglog(
        dimensions,
        study["sparse_megabytes"],
        "o-",
        label="Sparse CSR",
    )
    axes[1].loglog(
        dimensions,
        study["dense_megabytes"],
        "s--",
        label="Dense estimate",
    )
    axes[1].set(
        xlabel="Matrix dimension M",
        ylabel="Storage (MB)",
        title="Sparse and dense storage",
    )
    for axis in axes:
        axis.grid(alpha=0.3, which="both")
        axis.set_xticks(dimensions)
        axis.set_xticklabels([f"{dimension:d}" for dimension in dimensions])
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend()
    return _finish_figure(figure, output_path)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_project(
    config: QHOConfig | None = None,
    output_dir: str | Path = "results",
) -> dict[str, object]:
    """Run every required analysis and write all final artifacts."""

    active_config = config or QHOConfig()
    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    baseline = analyze_baseline(active_config)
    grid_study, grid_orders = grid_spacing_study(active_config)
    box_study = box_size_study(active_config)

    baseline.energy_table.to_csv(
        output_path / "energy_comparison.csv",
        index=False,
        float_format="%.12g",
    )
    baseline.state_table.to_csv(
        output_path / "state_validation.csv",
        index=False,
        float_format="%.12g",
    )
    grid_study.to_csv(
        output_path / "grid_convergence.csv",
        index=False,
        float_format="%.12g",
    )
    grid_orders.to_csv(
        output_path / "grid_convergence_orders.csv",
        index=False,
        float_format="%.12g",
    )
    box_study.to_csv(
        output_path / "box_convergence.csv",
        index=False,
        float_format="%.12g",
    )

    figures = (
        plot_energy_comparison(
            baseline,
            figures_path / "energy_comparison.png",
        ),
        plot_wavefunctions(
            baseline,
            figures_path / "wavefunctions.png",
        ),
        plot_probability_densities(
            baseline,
            figures_path / "probability_densities.png",
        ),
        plot_grid_convergence(
            grid_study,
            active_config.num_states,
            figures_path / "grid_convergence.png",
        ),
        plot_box_convergence(
            box_study,
            active_config.num_states,
            figures_path / "box_convergence.png",
        ),
        plot_cost_scaling(
            grid_study,
            figures_path / "cost_scaling.png",
        ),
    )
    for figure in figures:
        plt.close(figure)

    box_energies = box_study.filter(regex=r"^energy_").to_numpy()
    box_widths = box_study["L"].to_numpy()
    index_l6 = int(np.flatnonzero(np.isclose(box_widths, 6.0))[0])
    index_l7 = int(np.flatnonzero(np.isclose(box_widths, 7.0))[0])
    index_l8 = int(np.flatnonzero(np.isclose(box_widths, 8.0))[0])
    max_l6_to_l8_change = float(
        np.max(np.abs(box_energies[index_l6] - box_energies[index_l8]))
    )
    max_l7_to_l8_change = float(
        np.max(np.abs(box_energies[index_l7] - box_energies[index_l8]))
    )

    summary: dict[str, object] = {
        "configuration": asdict(active_config),
        "baseline_checks": baseline.checks,
        "grid_convergence": {
            "minimum_fitted_order": float(
                grid_orders["fitted_order"].min()
            ),
            "maximum_fitted_order": float(
                grid_orders["fitted_order"].max()
            ),
        },
        "box_convergence": {
            "target_dx": 0.02,
            "largest_box_half_width": float(box_study["L"].iloc[-1]),
            "max_L6_to_L8_energy_change": max_l6_to_l8_change,
            "max_L7_to_L8_energy_change": max_l7_to_l8_change,
            "largest_box_boundary_amplitude": float(
                box_study["max_near_boundary_amplitude"].iloc[-1]
            ),
        },
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_ready(summary), file, indent=2)
        file.write("\n")

    print("Baseline energy comparison")
    print(baseline.energy_table.to_string(index=False))
    print()
    print("Baseline validation")
    print(
        f"solver difference: "
        f"{baseline.checks['solver_difference']:.3e}"
    )
    print(
        f"normalization error: "
        f"{baseline.checks['normalization_error']:.3e}"
    )
    print(
        f"orthogonality error: "
        f"{baseline.checks['orthogonality_error']:.3e}"
    )
    print(
        f"grid convergence orders: "
        f"{grid_orders['fitted_order'].min():.3f} to "
        f"{grid_orders['fitted_order'].max():.3f}"
    )
    print(
        f"L=6 to L=8 maximum energy change: "
        f"{summary['box_convergence']['max_L6_to_L8_energy_change']:.3e}"
    )
    print(
        f"L=7 to L=8 maximum energy change: "
        f"{summary['box_convergence']['max_L7_to_L8_energy_change']:.3e}"
    )
    print(f"Results written to {output_path.resolve()}")

    return {
        "baseline": baseline,
        "grid_study": grid_study,
        "grid_orders": grid_orders,
        "box_study": box_study,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point and complete usage demonstration."""

    parser = argparse.ArgumentParser(
        description=(
            "Solve and validate the 1D quantum harmonic oscillator."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV tables, JSON summary, and PNG figures.",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=8.0,
        help="Baseline box half-width L.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1000,
        help="Baseline number of full-grid points, including endpoints.",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=6,
        help="Number of low-energy states.",
    )
    arguments = parser.parse_args(argv)
    config = QHOConfig(
        half_width=arguments.half_width,
        grid_points=arguments.grid_points,
        num_states=arguments.states,
    )
    run_project(config=config, output_dir=arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
