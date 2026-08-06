import unittest

import numpy as np

from qho_project import (
    QHOConfig,
    analyze_baseline,
    box_size_study,
    build_hamiltonian,
    grid_spacing_study,
    validate_hamiltonian,
)


class QHOProjectTests(unittest.TestCase):
    def test_hamiltonian_structure(self) -> None:
        config = QHOConfig(half_width=5.0, grid_points=101, num_states=4)
        _, x, dx, hamiltonian = build_hamiltonian(config)
        checks = validate_hamiltonian(config, x, dx, hamiltonian)

        self.assertEqual(hamiltonian.shape, (99, 99))
        self.assertEqual(checks["stored_nonzeros"], 295)
        self.assertLess(checks["symmetry_error"], 1e-12)

    def test_baseline_physics(self) -> None:
        analysis = analyze_baseline(QHOConfig())
        checks = analysis.checks

        self.assertLess(checks["max_energy_error"], 5e-4)
        self.assertLess(checks["solver_difference"], 1e-9)
        self.assertLess(checks["normalization_error"], 1e-10)
        self.assertLess(checks["orthogonality_error"], 1e-10)
        self.assertTrue(checks["node_counts_correct"])

    def test_grid_refinement_is_second_order(self) -> None:
        config = QHOConfig(num_states=3)
        _, orders = grid_spacing_study(
            config,
            grid_points=(200, 400, 800, 1600),
        )
        self.assertTrue(
            np.allclose(orders["fitted_order"], 2.0, atol=0.02)
        )

    def test_box_study_holds_spacing_fixed_and_stabilizes(self) -> None:
        config = QHOConfig(num_states=3)
        study = box_size_study(
            config,
            half_widths=(3.0, 4.0, 6.0, 8.0),
            target_dx=0.02,
        )

        self.assertLess(np.ptp(study["dx"]), 1e-12)
        energies = study.filter(regex=r"^energy_").to_numpy()
        self.assertLess(np.max(np.abs(energies[-1] - energies[-2])), 1e-8)
        self.assertLess(
            study["max_near_boundary_amplitude"].iloc[-1],
            1e-8,
        )


if __name__ == "__main__":
    unittest.main()
