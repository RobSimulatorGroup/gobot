from __future__ import annotations

import importlib.metadata
import math
import os


if os.environ.get("GOBOT_RUN_WARP_IPC_GPU_TEST") != "1":
    raise SystemExit(77)

import numpy as np
import warp as wp

from gobot.ipc._warp_kernels import (
    FixedCgSolver,
    assemble_bsr33_from_triplets,
    barrier_kernel,
    edge_edge_ccd_kernel,
    edge_edge_distance_kernel,
    friction_kernel,
    point_triangle_ccd_kernel,
    point_triangle_distance_kernel,
    project_psd_3x3_kernel,
    tet_neo_hookean_kernel,
)
from gobot.ipc._runtime_kernels import (
    initialize_ccd_fraction_kernel,
    initialize_newton_kernel,
    point_triangle_line_search_kernel,
)


DEVICE = "cuda:0"


def _vec3_array(values: list[list[float]] | np.ndarray) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.float64), dtype=wp.vec3d, device=DEVICE)


def _scalar_output(count: int) -> wp.array:
    return wp.zeros(count, dtype=wp.float64, device=DEVICE)


def test_dependency_contract() -> None:
    assert importlib.metadata.version("warp-lang") == "1.15.0"
    assert importlib.metadata.version("newton") == "1.4.0"
    wp.init()
    assert wp.is_cuda_available(), "GOBOT_RUN_WARP_IPC_GPU_TEST=1 requires CUDA"
    assert wp.get_device(DEVICE).is_cuda


def test_distance_and_ccd_kernels() -> None:
    points = _vec3_array(
        [
            [0.25, 0.25, 1.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    triangle_a = _vec3_array([[0.0, 0.0, 0.0]] * 3)
    triangle_b = _vec3_array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    triangle_c = _vec3_array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    point_triangle_distance = _scalar_output(3)
    wp.launch(
        point_triangle_distance_kernel,
        dim=3,
        inputs=[points, triangle_a, triangle_b, triangle_c],
        outputs=[point_triangle_distance],
        device=DEVICE,
    )
    np.testing.assert_allclose(
        point_triangle_distance.numpy(), np.array([1.0, 1.0, 1.0]), atol=1.0e-12
    )

    edge_a0 = _vec3_array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])
    edge_a1 = _vec3_array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    edge_b0 = _vec3_array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    edge_b1 = _vec3_array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    edge_distance = _scalar_output(2)
    wp.launch(
        edge_edge_distance_kernel,
        dim=2,
        inputs=[edge_a0, edge_a1, edge_b0, edge_b1],
        outputs=[edge_distance],
        device=DEVICE,
    )
    np.testing.assert_allclose(edge_distance.numpy(), np.array([0.0, 1.0]), atol=1.0e-12)

    static_a = _vec3_array([[0.0, 0.0, 0.0]] * 2)
    static_b = _vec3_array([[1.0, 0.0, 0.0]] * 2)
    static_c = _vec3_array([[0.0, 1.0, 0.0]] * 2)
    point_start = _vec3_array([[0.25, 0.25, 1.0], [0.25, 0.25, 1.0]])
    point_end = _vec3_array([[0.25, 0.25, -1.0], [1.25, 0.25, 1.0]])
    point_fraction = _scalar_output(2)
    wp.launch(
        point_triangle_ccd_kernel,
        dim=2,
        inputs=[
            point_start,
            point_end,
            static_a,
            static_a,
            static_b,
            static_b,
            static_c,
            static_c,
            wp.float64(0.1),
            wp.float64(1.0e-9),
        ],
        outputs=[point_fraction],
        device=DEVICE,
    )
    np.testing.assert_allclose(point_fraction.numpy(), np.array([0.45, 1.0]), atol=1.0e-8)

    moving_a0_start = _vec3_array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    moving_a0_end = _vec3_array([[-1.0, 0.0, -1.0], [-1.0, 1.0, 1.0]])
    moving_a1_start = _vec3_array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    moving_a1_end = _vec3_array([[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    static_b0 = _vec3_array([[0.0, -1.0, 0.0]] * 2)
    static_b1 = _vec3_array([[0.0, 1.0, 0.0]] * 2)
    edge_fraction = _scalar_output(2)
    wp.launch(
        edge_edge_ccd_kernel,
        dim=2,
        inputs=[
            moving_a0_start,
            moving_a0_end,
            moving_a1_start,
            moving_a1_end,
            static_b0,
            static_b0,
            static_b1,
            static_b1,
            wp.float64(0.1),
            wp.float64(1.0e-9),
        ],
        outputs=[edge_fraction],
        device=DEVICE,
    )
    np.testing.assert_allclose(edge_fraction.numpy(), np.array([0.45, 1.0]), atol=1.0e-8)


def test_runtime_kinematic_scaling_and_near_contact_ccd() -> None:
    dt = 0.005
    iterate = _vec3_array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    predicted = _vec3_array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    target = _vec3_array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mass = wp.array(np.array([2.0, 3.0]), dtype=wp.float64, device=DEVICE)
    kinematic = wp.array(
        np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    gradient = wp.zeros(2, dtype=wp.vec3d, device=DEVICE)
    hessian = wp.zeros((2, 3, 3), dtype=wp.float64, device=DEVICE)
    contact_force = wp.zeros(2, dtype=wp.vec3d, device=DEVICE)
    wp.launch(
        initialize_newton_kernel,
        dim=2,
        inputs=[
            iterate,
            predicted,
            mass,
            kinematic,
            target,
            wp.float64(1.0e5),
            wp.float64(dt),
        ],
        outputs=[gradient, hessian, contact_force],
        device=DEVICE,
    )
    expected_kinematic = 2.0 * 1.0e5 / (dt * dt)
    expected_dynamic = 3.0 / (dt * dt)
    np.testing.assert_allclose(
        gradient.numpy(),
        np.array(
            [
                [0.01 * expected_kinematic, 0.0, 0.0],
                [0.01 * expected_dynamic, 0.0, 0.0],
            ]
        ),
        rtol=1.0e-13,
    )
    hessian_np = hessian.numpy()
    np.testing.assert_allclose(
        np.diagonal(hessian_np, axis1=1, axis2=2),
        np.array([[expected_kinematic] * 3, [expected_dynamic] * 3]),
        rtol=1.0e-13,
    )

    origin = _vec3_array(
        [
            [0.25, 0.25, 1.0e-7],
            [0.25, 0.25, 1.0e-7],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    candidate = _vec3_array(
        [
            [0.25, 0.25, -1.0e-7],
            [0.25, 0.25, 2.0e-7],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    point = wp.array(
        np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    triangle_a = wp.array(
        np.array([2, 2], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    triangle_b = wp.array(
        np.array([3, 3], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    triangle_c = wp.array(
        np.array([4, 4], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    environment = wp.array(
        np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    fraction = wp.zeros(2, dtype=wp.float64, device=DEVICE)
    wp.launch(
        initialize_ccd_fraction_kernel,
        dim=2,
        outputs=[fraction],
        device=DEVICE,
    )
    wp.launch(
        point_triangle_line_search_kernel,
        dim=2,
        inputs=[
            origin,
            candidate,
            point,
            triangle_a,
            triangle_b,
            triangle_c,
            environment,
            wp.float64(1.0e-6),
            wp.float64(1.0e-9),
        ],
        outputs=[fraction],
        device=DEVICE,
    )
    np.testing.assert_allclose(fraction.numpy(), np.array([0.25, 1.0]), atol=1.0e-12)


def _barrier_reference(
    distance_squared: float, activation_squared: float, stiffness: float
) -> float:
    if distance_squared >= activation_squared:
        return 0.0
    delta = distance_squared - activation_squared
    return -stiffness * delta * delta * math.log(distance_squared / activation_squared)


def _friction_reference(speed: float, smoothing_speed: float) -> float:
    if speed >= smoothing_speed:
        return speed - smoothing_speed / 3.0
    normalized = speed / smoothing_speed
    return speed * normalized - speed * normalized * normalized / 3.0


def test_barrier_and_friction_derivatives() -> None:
    distances = np.array([0.2, 0.5, 0.9, 1.0, 1.2], dtype=np.float64)
    distance_array = wp.array(distances, dtype=wp.float64, device=DEVICE)
    energy = _scalar_output(len(distances))
    gradient = _scalar_output(len(distances))
    hessian = _scalar_output(len(distances))
    wp.launch(
        barrier_kernel,
        dim=len(distances),
        inputs=[distance_array, wp.float64(1.0), wp.float64(2.5)],
        outputs=[energy, gradient, hessian],
        device=DEVICE,
    )
    barrier_energy = energy.numpy()
    barrier_gradient = gradient.numpy()
    barrier_hessian = hessian.numpy()
    expected_energy = np.array([_barrier_reference(value, 1.0, 2.5) for value in distances])
    np.testing.assert_allclose(barrier_energy, expected_energy, rtol=1.0e-13, atol=1.0e-13)
    for index in range(3):
        value = distances[index]
        step = 1.0e-5
        plus = _barrier_reference(value + step, 1.0, 2.5)
        center = _barrier_reference(value, 1.0, 2.5)
        minus = _barrier_reference(value - step, 1.0, 2.5)
        numeric_gradient = (plus - minus) / (2.0 * step)
        numeric_hessian = (plus - 2.0 * center + minus) / (step * step)
        np.testing.assert_allclose(barrier_gradient[index], numeric_gradient, rtol=2.0e-8)
        np.testing.assert_allclose(barrier_hessian[index], numeric_hessian, rtol=2.0e-5)
    assert np.all(barrier_hessian[:3] > 0.0)
    np.testing.assert_array_equal(barrier_gradient[3:], np.zeros(2))

    speeds = np.array([0.0, 0.2, 0.7, 1.0, 1.5], dtype=np.float64)
    speed_array = wp.array(speeds, dtype=wp.float64, device=DEVICE)
    wp.launch(
        friction_kernel,
        dim=len(speeds),
        inputs=[speed_array, wp.float64(1.0)],
        outputs=[energy, gradient, hessian],
        device=DEVICE,
    )
    friction_energy = energy.numpy()
    friction_gradient = gradient.numpy()
    friction_hessian = hessian.numpy()
    expected_friction = np.array([_friction_reference(value, 1.0) for value in speeds])
    np.testing.assert_allclose(friction_energy, expected_friction, atol=1.0e-13)
    for index in (1, 2):
        value = speeds[index]
        step = 1.0e-5
        plus = _friction_reference(value + step, 1.0)
        center = _friction_reference(value, 1.0)
        minus = _friction_reference(value - step, 1.0)
        np.testing.assert_allclose(
            friction_gradient[index], (plus - minus) / (2.0 * step), rtol=1.0e-9
        )
        np.testing.assert_allclose(
            friction_hessian[index],
            (plus - 2.0 * center + minus) / (step * step),
            rtol=2.0e-5,
        )
    assert friction_gradient[0] == 0.0
    np.testing.assert_allclose(friction_gradient[3:], np.ones(2), atol=1.0e-13)
    np.testing.assert_allclose(friction_hessian[3:], np.zeros(2), atol=1.0e-13)


def _tet_energy_reference(
    positions: np.ndarray,
    inverse_rest: np.ndarray,
    volume: float,
    mu: float,
    lam: float,
) -> float:
    edges = np.column_stack(
        (positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])
    )
    deformation_gradient = edges @ inverse_rest
    determinant = np.linalg.det(deformation_gradient)
    if determinant <= 0.0:
        return 1.0e300
    log_determinant = math.log(determinant)
    return volume * (
        0.5 * mu * (np.sum(deformation_gradient * deformation_gradient) - 3.0)
        - mu * log_determinant
        + 0.5 * lam * log_determinant * log_determinant
    )


def _tet_gradient_reference(
    positions: np.ndarray,
    inverse_rest: np.ndarray,
    volume: float,
    mu: float,
    lam: float,
) -> np.ndarray:
    edges = np.column_stack(
        (positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])
    )
    deformation_gradient = edges @ inverse_rest
    determinant = np.linalg.det(deformation_gradient)
    inverse_transpose = np.linalg.inv(deformation_gradient).T
    first_piola = (
        mu * (deformation_gradient - inverse_transpose)
        + lam * math.log(determinant) * inverse_transpose
    )
    edge_gradient = volume * first_piola @ inverse_rest.T
    result = np.empty((4, 3), dtype=np.float64)
    result[1:] = edge_gradient.T
    result[0] = -np.sum(result[1:], axis=0)
    return result


def test_tet_energy_gradient_and_hessian() -> None:
    rest = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    deformed = np.array(
        [[0.02, -0.01, 0.03], [1.08, 0.04, -0.02], [0.08, 0.94, 0.02], [-0.03, 0.05, 1.04]],
        dtype=np.float64,
    )
    inverted = rest.copy()
    inverted[[1, 2]] = inverted[[2, 1]]
    positions_np = np.stack((rest, deformed, inverted))
    inverse_rest_np = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)
    positions = wp.array(positions_np, dtype=wp.vec3d, device=DEVICE)
    inverse_rest = wp.array(inverse_rest_np, dtype=wp.mat33d, device=DEVICE)
    volumes = wp.array([1.0 / 6.0] * 3, dtype=wp.float64, device=DEVICE)
    shear = wp.array([10.0] * 3, dtype=wp.float64, device=DEVICE)
    lame = wp.array([20.0] * 3, dtype=wp.float64, device=DEVICE)
    energy = _scalar_output(3)
    gradient = wp.zeros((3, 4), dtype=wp.vec3d, device=DEVICE)
    hessian = wp.zeros((3, 12, 12), dtype=wp.float64, device=DEVICE)
    valid = wp.zeros(3, dtype=wp.int32, device=DEVICE)
    wp.launch(
        tet_neo_hookean_kernel,
        dim=3,
        inputs=[positions, inverse_rest, volumes, shear, lame],
        outputs=[energy, gradient, hessian, valid],
        device=DEVICE,
    )

    energy_np = energy.numpy()
    gradient_np = gradient.numpy()
    hessian_np = hessian.numpy()
    np.testing.assert_array_equal(valid.numpy(), np.array([1, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(energy_np[0], 0.0, atol=1.0e-13)
    np.testing.assert_allclose(gradient_np[0], np.zeros((4, 3)), atol=1.0e-13)
    assert energy_np[2] == 1.0e300
    np.testing.assert_allclose(
        energy_np[1], _tet_energy_reference(deformed, np.eye(3), 1.0 / 6.0, 10.0, 20.0)
    )
    expected_gradient = _tet_gradient_reference(deformed, np.eye(3), 1.0 / 6.0, 10.0, 20.0)
    np.testing.assert_allclose(gradient_np[1], expected_gradient, rtol=2.0e-12, atol=2.0e-12)

    finite_difference_hessian = np.empty((12, 12), dtype=np.float64)
    step = 1.0e-6
    for column in range(12):
        plus = deformed.copy()
        minus = deformed.copy()
        plus.reshape(-1)[column] += step
        minus.reshape(-1)[column] -= step
        plus_gradient = _tet_gradient_reference(plus, np.eye(3), 1.0 / 6.0, 10.0, 20.0)
        minus_gradient = _tet_gradient_reference(minus, np.eye(3), 1.0 / 6.0, 10.0, 20.0)
        finite_difference_hessian[:, column] = (
            plus_gradient.reshape(-1) - minus_gradient.reshape(-1)
        ) / (2.0 * step)
    np.testing.assert_allclose(
        hessian_np[1], finite_difference_hessian, rtol=2.0e-8, atol=2.0e-8
    )
    np.testing.assert_allclose(hessian_np[1], hessian_np[1].T, atol=2.0e-12)
    assert np.linalg.eigvalsh(hessian_np[0]).min() >= -1.0e-10


def test_psd_projection_bsr_assembly_and_cg() -> None:
    matrices_np = np.array(
        [
            [[2.0, 3.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 4.0]],
            [[-2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 3.0]],
        ],
        dtype=np.float64,
    )
    matrices = wp.array(matrices_np, dtype=wp.mat33d, device=DEVICE)
    projected = wp.zeros(2, dtype=wp.mat33d, device=DEVICE)
    wp.launch(
        project_psd_3x3_kernel,
        dim=2,
        inputs=[matrices, wp.float64(0.25)],
        outputs=[projected],
        device=DEVICE,
    )
    projected_np = projected.numpy()
    np.testing.assert_allclose(projected_np, np.swapaxes(projected_np, 1, 2), atol=1.0e-12)
    assert np.linalg.eigvalsh(projected_np).min() >= 0.25 - 1.0e-10

    identity = np.eye(3, dtype=np.float64)
    rows = wp.array([0, 0, 0, 1, 1, 1], dtype=wp.int32, device=DEVICE)
    columns = wp.array([0, 0, 1, 0, 1, 1], dtype=wp.int32, device=DEVICE)
    values = wp.array(
        np.stack(
            (
                2.0 * identity,
                2.0 * identity,
                -identity,
                -identity,
                1.5 * identity,
                1.5 * identity,
            )
        ),
        dtype=wp.mat33d,
        device=DEVICE,
    )
    matrix = assemble_bsr33_from_triplets(2, rows, columns, values)
    rhs_np = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]], dtype=np.float64)
    rhs = wp.array(rhs_np, dtype=wp.vec3d, device=DEVICE)
    solution = wp.zeros(2, dtype=wp.vec3d, device=DEVICE)
    solver = FixedCgSolver(
        matrix,
        rhs,
        solution,
        max_iterations=12,
        relative_tolerance=1.0e-12,
    )
    iterations, residual_squared, tolerance_squared, invalid = solver.solve()
    assert isinstance(iterations, wp.array)
    assert isinstance(residual_squared, wp.array)
    assert isinstance(tolerance_squared, wp.array)
    np.testing.assert_array_equal(invalid.numpy(), np.zeros(1, dtype=np.int32))

    dense = np.block(
        [[4.0 * identity, -identity], [-identity, 3.0 * identity]]
    )
    expected = np.linalg.solve(dense, rhs_np.reshape(-1)).reshape(2, 3)
    np.testing.assert_allclose(solution.numpy(), expected, rtol=2.0e-11, atol=2.0e-11)
    residual = dense @ solution.numpy().reshape(-1) - rhs_np.reshape(-1)
    assert np.linalg.norm(residual) <= 1.0e-10

    stable_buffers = (
        solver.diagonal,
        solver.inverse_diagonal,
        solver.residual,
        solver.preconditioned,
        solver.direction,
        solver.matrix_direction,
        solver.rho,
        solver.next_rho,
        solver.direction_dot_product,
        solver.residual_squared,
        solver.rhs_squared,
        solver.tolerance_squared,
        solver.alpha,
        solver.beta,
        solver.active,
        solver.invalid,
        solver.iteration_count,
        *solver._dot_workspace.partial,
    )
    storage_pointers = tuple(value.ptr for value in stable_buffers)
    solution.zero_()
    with wp.ScopedCapture(device=DEVICE) as capture:
        solver.solve()
    solution.zero_()
    wp.capture_launch(capture.graph)
    first_replay = solution.numpy().copy()
    solution.zero_()
    wp.capture_launch(capture.graph)
    second_replay = solution.numpy().copy()
    np.testing.assert_allclose(first_replay, expected, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_array_equal(second_replay, first_replay)
    assert tuple(value.ptr for value in stable_buffers) == storage_pointers

    block_count = 300
    diagonal_scale = np.linspace(1.0, 2.0, block_count, dtype=np.float64)
    large_rows = wp.array(np.arange(block_count, dtype=np.int32), dtype=wp.int32, device=DEVICE)
    large_columns = wp.array(
        np.arange(block_count, dtype=np.int32), dtype=wp.int32, device=DEVICE
    )
    large_values = wp.array(
        diagonal_scale[:, None, None] * identity[None, :, :],
        dtype=wp.mat33d,
        device=DEVICE,
    )
    large_matrix = assemble_bsr33_from_triplets(
        block_count, large_rows, large_columns, large_values
    )
    large_rhs_np = np.column_stack(
        (
            np.ones(block_count),
            np.linspace(-1.0, 1.0, block_count),
            np.linspace(2.0, 3.0, block_count),
        )
    )
    large_rhs = wp.array(large_rhs_np, dtype=wp.vec3d, device=DEVICE)
    large_solution = wp.zeros(block_count, dtype=wp.vec3d, device=DEVICE)
    large_solver = FixedCgSolver(
        large_matrix,
        large_rhs,
        large_solution,
        max_iterations=4,
        relative_tolerance=1.0e-12,
    )
    assert len(large_solver._dot_workspace.partial) == 2
    large_solver.solve()
    np.testing.assert_allclose(
        large_solution.numpy(),
        large_rhs_np / diagonal_scale[:, None],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


if __name__ == "__main__":
    test_dependency_contract()
    test_distance_and_ccd_kernels()
    test_runtime_kinematic_scaling_and_near_contact_ccd()
    test_barrier_and_friction_derivatives()
    test_tet_energy_gradient_and_hessian()
    test_psd_projection_bsr_assembly_and_cg()
