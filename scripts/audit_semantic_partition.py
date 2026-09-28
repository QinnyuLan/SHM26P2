#!/usr/bin/env python3
"""Independent, synthetic CPU numerical audit of a Gaussian semantic partition.

This is a verifier, not a training or performance experiment. Full execution needs
an externally frozen plan. It never opens scene, image, label, or camera assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm

SPEC = {
    "protocol": "semantic_partition_numerical_v1",
    "seed": 20260927,
    "projection_cases": 48,
    "modes": ["integrated", "point", "marginal"],
    "device": "cpu",
    "dtype": "float64",
    "quad_epsabs": 1e-11,
    "quad_epsrel": 1e-11,
    "quad_limit": 200,
    "analytic_absolute_tolerance": 1e-8,
    "gauss_hermite_order": 512,
    "mass_absolute_tolerance": 1e-8,
    "simplex_absolute_tolerance": 1e-12,
    "unit_conversion_absolute_tolerance": 1e-10,
    "screen_unit_factors": [0.5, 2.0],
    "world_unit_factors": [0.01, 100.0],
    "gradient_case_indices": [0, 1, 7, 13, 19, 25, 31, 37, 43, 44],
    "finite_difference_step": 1e-4,
    "finite_difference_stencil": "five_point_centered_independent_numpy_closed_form",
    "gradient_nonzero_threshold": 1e-7,
    "gradient_relative_tolerance": 1e-4,
    "gradient_zero_absolute_tolerance": 1e-8,
    "maximum_seconds": 180,
    "torch_threads": 4,
    "scope": "synthetic_normalized_single_primitive_affine_EWA_only",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x)


def generate_cases() -> list[dict]:
    """48 prespecified cases; randomness only supplies one deterministic rotation."""
    rng = np.random.default_rng(SPEC["seed"])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    front = np.array([[2., 0., 0.], [0., 1.5, 0.]])
    bases = [
        ("visible_normal", front, .09 * np.eye(2), unit([1, 0, 0])),
        ("ray_parallel_normal", front, .09 * np.eye(2), unit([0, 0, 1])),
        ("oblique", np.array([[1.6, .3, .4], [.1, 1.3, -.2]]),
         np.diag([.25, .08]), unit([.6, -.3, .7])),
        ("near_zero_conditional_variance", front, 1e-10 * np.eye(2), unit([1, 0, 1e-7])),
        ("near_prior_variance", front, .3 * np.eye(2), unit([1e-7, 0, 1])),
        ("AA_dominated", np.array([[1e-3, .2, 0.], [0., 1e-3, .15]]),
         .3 * np.eye(2), unit([.4, .5, .7])),
        ("rotated", front @ rotation, .09 * np.eye(2), rotation.T @ unit([.6, .2, .5])),
        ("anisotropic_correlated_AA", np.array([[4., .02, .3], [.02, .18, .01]]),
         np.array([[.16, .08], [.08, .25]]), unit([.2, .9, .1])),
    ]
    bands = [
        ("central", 0., .75, .25, [0., 0.]),
        ("offset", 1.2, .35, .35, [.25, -.75]),
        ("positive_tail", 8., .3, .45, [1., -.5]),
        ("negative_tail", -8., .3, .45, [-1., .5]),
        ("wide", .6, 4., .3, [.8, .3]),
        ("thin", .1, .05, .3, [2., -1.]),
    ]
    return [{"name": f"{name}/{band}", "P": P.copy(), "noise": noise.copy(),
                 "normal": n.copy(), "b": b, "w": w, "tau": tau, "delta": np.array(delta)}
            for name, P, noise, n in bases for band, b, w, tau, delta in bands]


def normal_interval(lo: float, hi: float) -> float:
    """Stable independent SciPy normal probability, including positive tails."""
    if lo > hi:
        raise ValueError("Reversed normal interval")
    if lo >= 0:
        return float(ndtr(-lo) - ndtr(-hi))
    return float(ndtr(hi) - ndtr(lo))


def conditional_scalar(case: dict) -> tuple[float, float]:
    """NumPy scalar conditional law, independently of the Torch module."""
    P, noise, n = case["P"], case["noise"], case["normal"]
    covariance = P @ P.T + noise
    cross = P @ n
    mean = float(cross @ np.linalg.solve(covariance, case["delta"]))
    variance = float(n @ n - cross @ np.linalg.solve(covariance, cross))
    if variance < -1e-12:
        raise ValueError("Negative scalar conditional variance")
    return mean, max(0., variance)


def mode_law(case: dict, mode: str) -> tuple[float, float]:
    mean, variance = conditional_scalar(case)
    if mode == "integrated":
        return mean, variance
    if mode == "point":
        return mean, 0.
    if mode == "marginal":
        return 0., 1.
    raise ValueError("Unknown mode")


def reference_quad(case: dict, mode: str) -> tuple[float, float]:
    """Integrate the *unintegrated* soft band over its scalar Gaussian law."""
    mean, variance = mode_law(case, mode)
    sigma = math.sqrt(variance)
    b, w, tau = case["b"], case["w"], case["tau"]

    def response(z):
        value = mean + sigma * z
        band = normal_interval((value - b - w) / tau, (value - b + w) / tau)
        return math.exp(-.5 * z * z) / math.sqrt(2 * math.pi) * band

    return tuple(float(x) for x in quad(response, -np.inf, np.inf,
                                       epsabs=SPEC["quad_epsabs"],
                                       epsrel=SPEC["quad_epsrel"], limit=SPEC["quad_limit"]))


def reference_closed(case: dict, mode: str) -> float:
    mean, variance = mode_law(case, mode)
    denominator = math.sqrt(case["tau"] ** 2 + variance)
    return normal_interval((mean - case["b"] - case["w"]) / denominator,
                           (mean - case["b"] + case["w"]) / denominator)


def tensor_case(core, case, mode, torch):
    values = [torch.as_tensor(case[key], dtype=torch.float64, device="cpu")
              for key in ("P", "noise", "normal", "b", "w", "tau")]
    return core.prepared_coefficients(*values, mode=mode)


def torch_gate(core, case, mode, torch):
    coeff = tensor_case(core, case, mode, torch)
    return core.evaluate_coefficients(torch.as_tensor(case["delta"], dtype=torch.float64), coeff)


def hermite_screen_mass(core, case, mode, torch, nodes_weights=None):
    """2-D screen integration; not 1-D analytic law reused as quadrature."""
    if nodes_weights is None:
        nodes_weights = roots_hermitenorm(SPEC["gauss_hermite_order"])
    nodes, weights = nodes_weights
    grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    chol = np.linalg.cholesky(case["P"] @ case["P"].T + case["noise"])
    delta = grid @ chol.T
    coeff = tensor_case(core, case, mode, torch)
    gates = core.evaluate_coefficients(torch.as_tensor(delta, dtype=torch.float64), coeff)
    gate_np = gates.detach().numpy()
    expectation = float((gate_np * np.outer(weights, weights).ravel()).sum() / (2 * math.pi))
    return expectation


def gradient_parameterization(case):
    """16 unconstrained chart coordinates; normal remains unit, noise remains PSD."""
    n = case["normal"]
    pivot = np.eye(3)[np.argmin(np.abs(n))]
    tangent1 = unit(np.cross(n, pivot))
    tangent2 = np.cross(n, tangent1)
    lower = np.linalg.cholesky(case["noise"])
    vector = np.concatenate((case["P"].ravel(), case["delta"],
                             [case["b"], case["w"], case["tau"], 0., 0.,
                              lower[0, 0], lower[1, 0], lower[1, 1]]))
    return vector, np.stack([tangent1, tangent2])


def numpy_chart(vector, case, tangents):
    out = dict(case)
    out["P"] = vector[:6].reshape(2, 3)
    out["delta"] = vector[6:8]
    out["b"], out["w"], out["tau"] = vector[8:11]
    out["normal"] = unit(case["normal"] + vector[11:13] @ tangents)
    lower = np.array([[vector[13], 0.], [vector[14], vector[15]]])
    out["noise"] = lower @ lower.T
    return out


def finite_difference(function, vector, step):
    derivative = np.empty_like(vector)
    for index in range(len(vector)):
        values = []
        for offset in (-2., -1., 1., 2.):
            shifted = vector.copy()
            shifted[index] += offset * step
            values.append(function(shifted))
        derivative[index] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
    return derivative


def gradient_comparison(analytic, numerical):
    analytic, numerical = np.asarray(analytic), np.asarray(numerical)
    signal = np.abs(analytic) > SPEC["gradient_nonzero_threshold"]
    error = np.abs(analytic - numerical)
    relative = np.divide(error, np.abs(analytic), out=np.zeros_like(error), where=signal)
    passed = np.where(signal, relative <= SPEC["gradient_relative_tolerance"],
                      error <= SPEC["gradient_zero_absolute_tolerance"])
    return {"passed": bool(passed.all()), "analytic": analytic.tolist(), "numerical": numerical.tolist(),
                "absolute_error": error.tolist(), "nonzero": signal.tolist(),
                "relative_error": [float(x) if use else None for x, use in zip(relative, signal)],
                "component_passed": passed.tolist()}


def check_gradients(core, case, mode, torch):
    vector, tangents = gradient_parameterization(case)
    x = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
    n = torch.tensor(case["normal"]) + x[11:13] @ torch.tensor(tangents)
    n = n / torch.linalg.vector_norm(n)
    zero = x.new_zeros(())
    lower = torch.stack((x[13], zero, x[14], x[15])).reshape(2, 2)
    coeff = core.prepared_coefficients(x[:6].reshape(2, 3), lower @ lower.T, n,
                                       x[8], x[9], x[10], mode=mode)
    value = core.evaluate_coefficients(x[6:8], coeff)
    analytic = torch.autograd.grad(value, x)[0].detach().numpy()
    numerical = finite_difference(lambda v: reference_closed(numpy_chart(v, case, tangents), mode),
                                  vector, SPEC["finite_difference_step"])
    return gradient_comparison(analytic, numerical)


def check_units(core, case, mode, torch):
    baseline = float(torch_gate(core, case, mode, torch))
    screen_errors, world_errors = [], []
    for factor in SPEC["screen_unit_factors"]:
        transformed = dict(case, P=case["P"] * factor, noise=case["noise"] * factor ** 2,
                           delta=case["delta"] * factor)
        screen_errors.append(abs(float(torch_gate(core, transformed, mode, torch)) - baseline))
    # P = J R S. Re-express world lengths by k: J -> J/k and S -> kS.
    local_scales = np.diag([.3, 1.7, .8])
    jacobian = case["P"] @ np.linalg.inv(local_scales)
    for factor in SPEC["world_unit_factors"]:
        transformed = dict(case, P=(jacobian / factor) @ (local_scales * factor))
        world_errors.append(abs(float(torch_gate(core, transformed, mode, torch)) - baseline))
    maximum = max(screen_errors + world_errors)
    return {"passed": bool(maximum <= SPEC["unit_conversion_absolute_tolerance"]),
                "screen_absolute_errors": screen_errors, "world_absolute_errors": world_errors}


def run_numerical(core, torch):
    cases = generate_cases()
    assert len(cases) == SPEC["projection_cases"]
    hermite = roots_hermitenorm(SPEC["gauss_hermite_order"])
    records = []
    q_in = torch.tensor([.1, .1, .6, .1, .1], dtype=torch.float64)
    q_out = torch.tensor([.6, .1, .1, .1, .1], dtype=torch.float64)
    for index, case in enumerate(cases):
        mean, variance = conditional_scalar(case)
        marginal = reference_closed(case, "marginal")
        for mode in SPEC["modes"]:
            value = torch_gate(core, case, mode, torch)
            quadrature, quad_error = reference_quad(case, mode)
            mass = hermite_screen_mass(core, case, mode, torch, hermite)
            # Point query is intentionally NOT asserted to conserve the prior mass.
            point_variance = max(0., 1. - variance)
            point_expected = normal_interval((-case["b"] - case["w"]) /
                                            math.sqrt(case["tau"] ** 2 + point_variance),
                                            (-case["b"] + case["w"]) /
                                            math.sqrt(case["tau"] ** 2 + point_variance))
            expected_mass = point_expected if mode == "point" else marginal
            probabilities = core.mixprob(value, q_in, q_out)
            identity = core.mixprob(value, q_in, q_in)
            simplex_error = abs(float(probabilities.sum()) - 1.)
            identity_error = float((identity - q_in).abs().max())
            grad = check_gradients(core, case, mode, torch) if index in SPEC["gradient_case_indices"] else None
            units = check_units(core, case, mode, torch)
            checks = {
                "quad": bool(abs(float(value) - quadrature) <= SPEC["analytic_absolute_tolerance"]),
                "mass_quadrature": bool(abs(mass - expected_mass) <= SPEC["mass_absolute_tolerance"]),
                "probability": bool(torch.isfinite(probabilities).all() and (probabilities >= 0).all()
                                 and (probabilities <= 1).all() and simplex_error <= SPEC["simplex_absolute_tolerance"]),
                "identity": bool(identity_error <= SPEC["simplex_absolute_tolerance"]),
                "units": units["passed"], "gradient": True if grad is None else grad["passed"],
            }
            records.append({"index": index, "case": case["name"], "mode": mode,
                                "conditional_mean": mean, "conditional_variance": variance,
                                "closed_value": float(value), "scipy_quad": quadrature, "scipy_quad_error_estimate": quad_error,
                                "absolute_quad_error": abs(float(value) - quadrature),
                                "normalized_screen_gate_mass": mass, "expected_screen_gate_mass": expected_mass,
                                "prior_gate_mass": marginal, "deviation_from_prior_mass": mass - marginal,
                                "mass_absolute_error": abs(mass - expected_mass),
                                "simplex_error": simplex_error, "identity_error": identity_error,
                                "gradient": grad, "units": units, "checks": checks})
    reference = check_reference_compositing(core, torch)
    passed = all(all(record["checks"].values()) for record in records) and reference["passed"]
    return {"status": "passed" if passed else "failed_numerical_contract", "specification": SPEC,
                "records": records, "fixed_weight_reference": reference,
                "cases": [{k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in case.items()} for case in cases],
                "scope_limitations": ["No scene, labels, RGB, training, CUDA, or performance evaluation.",
                                   "Exactness is conditional on the continuous affine Gaussian model.",
                                   "Normalized per-primitive mass is not absolute pixel mass or occluded class mass.",
                                   "Point-mode mass differences are descriptive, not a failure criterion.",
                                   "Finite differences audit local synthetic CPU64 derivatives, not a complete renderer."]}


def check_reference_compositing(core, torch):
    cases = generate_cases()
    selected = [cases[0], cases[12]]
    args = [torch.as_tensor(np.stack([case[key] for case in selected]), dtype=torch.float64)
            for key in ("P", "noise", "normal", "b", "w", "tau")]
    coeff = core.prepared_coefficients(*args, mode="integrated")
    delta = torch.tensor([[[0., 0.], [.3, .2]], [[1., -.3], [0., 0.]],
                          [[-.2, .7], [.8, -.4]]], dtype=torch.float64)
    weights = torch.tensor([[.2, .6], [.3, .1], [0., 0.]], dtype=torch.float64)
    background = 1. - weights.sum(dim=1)
    qin = torch.tensor([[.1, .1, .6, .1, .1], [.1, .5, .1, .2, .1]], dtype=torch.float64)
    qout = torch.tensor([[.6, .1, .1, .1, .1], [.7, .1, .1, .05, .05]], dtype=torch.float64)
    result = core.render_reference(weights, background, delta, coefficients=coeff, q_in=qin, q_out=qout)
    expected = np.zeros((3, 5), dtype=np.float64)
    for row in range(3):
        expected[row, 0] = float(background[row])
        for col in range(2):
            case = dict(selected[col], delta=delta[row, col].numpy())
            gate, _ = reference_quad(case, "integrated")
            expected[row] += float(weights[row, col]) * (qout[col].numpy() + gate * (qin[col] - qout[col]).numpy())
    error = float(np.max(np.abs(result["probabilities"].detach().numpy() - expected)))
    alpha_error = float((result["alpha"] - weights.sum(dim=1)).abs().max())
    zero_ray_exact = bool(torch.equal(result["probabilities"][2],
                                     torch.tensor([1., 0., 0., 0., 0.], dtype=torch.float64)))
    return {"passed": bool(error <= SPEC["analytic_absolute_tolerance"]
                           and alpha_error <= 1e-12 and zero_ray_exact),
                "probability_max_error": error, "alpha_max_error": alpha_error,
                "all_zero_weight_ray_background_exact": zero_ray_exact}


def validate_plan(plan_path, expected_sha, output):
    plan_path = Path(plan_path).resolve()
    if sha256(plan_path) != expected_sha:
        raise ValueError("Plan SHA mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("specification") != SPEC or plan.get("protocol") != SPEC["protocol"]:
        raise ValueError("Frozen specification mismatch")
    if Path(plan["output"]).resolve() != Path(output).resolve():
        raise ValueError("Output does not match frozen plan")
    sources = plan["source_hashes"]
    required = {str(Path(__file__).resolve()), str(Path(plan["source_root"]).resolve() / "bridge_rgs/semantic_partition.py")}
    if not required.issubset(sources):
        raise ValueError("Plan must bind this worker and its actual core module")
    for path, expected in sources.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen source mismatch: {path}")
    return plan


def execute(plan_path, expected_sha, output):
    plan = validate_plan(plan_path, expected_sha, output)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing non-empty audit output")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {"status": "running", "plan": str(Path(plan_path).resolve()), "plan_sha256": expected_sha,
                   "source_hashes": plan["source_hashes"], "device": "cpu", "data_reads": 0, "renders": 0, "optimizer_steps": 0}
    write_json(output / "execution_receipt.json", receipt)
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError("Frozen CPU numerical audit deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(SPEC["maximum_seconds"])
    try:
        sys.path.insert(0, str(Path(plan["source_root"]).resolve()))
        import torch
        core = importlib.import_module("bridge_rgs.semantic_partition")
        expected_module = Path(plan["source_root"]).resolve() / "bridge_rgs/semantic_partition.py"
        if Path(core.__file__).resolve() != expected_module:
            raise ValueError("Core module was not imported from frozen source")
        torch.set_num_threads(SPEC["torch_threads"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must remain uninitialized")
        report = run_numerical(core, torch)
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
        validate_plan(plan_path, expected_sha, output)
        report["provenance"] = {"plan_sha256": expected_sha, "core_path": str(expected_module),
                                    "worker_path": str(Path(__file__).resolve()),
                                    "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                                    "torch_version": torch.__version__}
        write_json(output / "audit.json", report)
        receipt.update(status="completed", numerical_status=report["status"],
                       audit_sha256=sha256(output / "audit.json"), cuda_initialized=False)
    except Exception as error:
        receipt.update(status="inconclusive_timeout" if isinstance(error, TimeoutError) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "execution_receipt.json", receipt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC, indent=2, allow_nan=False))
        return
    if not all((args.plan, args.expected_plan_sha256, args.output)):
        parser.error("--plan, --expected-plan-sha256, and --output are required")
    report = execute(args.plan, args.expected_plan_sha256, args.output)
    print(json.dumps({"status": report["status"], "projection_cases": SPEC["projection_cases"]}))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
