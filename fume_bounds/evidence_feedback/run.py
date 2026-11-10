"""Run the predeclared, known-model experiment; no NUS inputs or claims.

Candidate generation uses the same continuous FUME-loss training path for all
three policies. Confidence data are unique entity labels, never repeated epochs.
Oracle query labels/probabilities enter only evaluate_frozen_world(), after every
choice at every support size and initialization in that world has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.special import expit, xlogy
from scipy.stats import beta, t

from .confidence import LogisticConfidenceSequence
from .ranking import (binary_probability_box, class_ap_from_scores, cosine_scores,
                      same_model_ap_certificate)
from .student import AdamStudent, copy_parameters, design, embedding, initialize
from .uncertainty import information_radius


PACKAGE = Path(__file__).resolve().parent


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def encode(value):
    return json.dumps(native(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(native(value), ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n")


def append_jsonl(path, rows):
    with Path(path).open("a") as stream:
        for row in rows:
            stream.write(encode(row) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def array_hash(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        result.update(str(value.shape).encode())
        result.update(str(value.dtype).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def make_world(config, world):
    """Return public inputs and a separate evaluation-only object."""
    mean = np.asarray(config["dgp_class_mean"])
    transforms = [np.asarray(config["image_transform"]), np.asarray(config["text_transform"])]
    if config["dgp_class_prior"] != .5 or any(abs(np.linalg.det(a)) < 1e-6 for a in transforms):
        raise ValueError("This protocol requires equal priors and invertible modalities")
    sizes = [config["warmup_entities"], max(config["support_prefixes"]),
             config["gallery_entities"], config["queries"]]
    pools = []
    for stream, size in enumerate(sizes):
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], world, 100+stream]))
        labels = rng.integers(0, 2, size)
        canonical = (2*labels[:, None]-1)*mean + rng.normal(size=(size, 2))
        pools.append({"image": design(canonical @ transforms[0].T),
                      "text": design(canonical @ transforms[1].T),
                      "features": design(canonical), "labels": labels})
    warmup, support, gallery, query = pools
    parameter = np.r_[0., 2*mean]
    if np.linalg.norm(parameter) > config["known_parameter_norm_bound"]:
        raise ValueError("The declared data generator violates its ex-ante parameter bound")
    hidden = {"query_labels": query.pop("labels"),
              "query_probability": expit(query["features"] @ parameter),
              "parameter": parameter}
    public = {"warmup": warmup, "support": support, "gallery": gallery, "query": query}
    manifest = {
        "world": world, "pool_sizes": dict(zip(("warmup", "support", "gallery", "query"), sizes)),
        "entity_pools_disjoint_by_construction": True,
        "teacher_features_fixed_invertible_canonical_coordinates": True,
        "one_support_label_counted_once": True,
        "query_labels_available_to_selector": False,
        "public_pool_hashes": {name: array_hash(*(pool[key] for key in sorted(pool)))
                              for name, pool in public.items()},
    }
    return public, hidden, manifest


def ap_tables(parameters, public):
    q, g = public["query"], public["gallery"]
    qi, qt = embedding(q["image"], parameters["image"]), embedding(q["text"], parameters["text"])
    gi, gt = embedding(g["image"], parameters["image"]), embedding(g["text"], parameters["text"])
    i2t = class_ap_from_scores(cosine_scores(qi, gt), g["labels"], 2)
    t2i = class_ap_from_scores(cosine_scores(qt, gi), g["labels"], 2)
    return {"i2t": i2t, "t2i": t2i, "mean": (i2t+t2i)/2}


def choose_candidate(policy, incumbent, candidates, lower, upper, point_p1, config):
    """No true query labels, true parameter, or true probabilities in this API."""
    lo, hi = binary_probability_box(lower, upper)
    point = np.column_stack((1-point_p1, point_p1))
    candidate_records = []
    for candidate in candidates:
        difference = candidate["ap"]["mean"] - incumbent["ap"]["mean"]
        cert = same_model_ap_certificate(candidate["ap"]["mean"], incumbent["ap"]["mean"], lo, hi,
                                         numerical_slack=config["ap_numerical_slack"])
        plugin = float(np.mean(np.sum(point*difference, axis=1)))
        candidate_records.append({
            "candidate_id": candidate["id"],
            "lower_bound": cert["lower_bound"], "nominal_minimum": cert["nominal_minimum"],
            "plugin_expected_difference": plugin,
            "changed_ap_query_count": int(np.any(difference != 0, axis=1).sum()),
            "negative_query_lower_count": int((cert["per_query_lower"] < 0).sum()),
            "all_ap_vectors_unchanged": cert["all_ap_vectors_unchanged"],
            "_ap_difference": difference,
        })
    if policy == "ordinary_latest":
        index = len(candidates)-1
        accepted = True
    elif policy in ("point_guard", "certified_guard"):
        key = "plugin_expected_difference" if policy == "point_guard" else "lower_bound"
        index = max(range(len(candidates)), key=lambda i: candidate_records[i][key])
        accepted = candidate_records[index][key] > config["acceptance_margin"]
    else:
        raise ValueError("Unknown policy")
    chosen = candidates[index] if accepted else incumbent
    record = {
        "policy": policy, "before_id": incumbent["id"], "after_id": chosen["id"],
        "accepted_nonzero_update": bool(accepted and
            np.any(chosen["ap"]["mean"] != incumbent["ap"]["mean"])),
        "accepted_parameter_change": bool(accepted),
        "chosen_candidate_index": index if accepted else None,
        "selected_difference_lower_bound": candidate_records[index]["lower_bound"] if accepted else 0.,
        "proposed_best_certificate": max(r["lower_bound"] for r in candidate_records),
        "candidate_count": len(candidates),
        "all_queries_retained": True,
        "_before_ap": incumbent["ap"], "_after_ap": chosen["ap"],
    }
    return chosen, record, candidate_records


def oracle_ap(table, hidden):
    p = hidden["query_probability"]
    expected = float(np.mean((1-p)*table[:, 0] + p*table[:, 1]))
    realized = float(np.mean(table[np.arange(len(table)), hidden["query_labels"]]))
    return expected, realized


def evaluate_frozen_world(decisions, attempts, bounds, checkpoints, initial, hidden, public, config):
    """Only evaluation accesses the oracle. All policy decisions already exist."""
    parameter = hidden["parameter"]
    p = hidden["query_probability"]
    per_stage_coverage = {}
    running = True
    for cp in checkpoints:
        n = cp["n"]
        z = public["support"]["features"][:n] @ parameter
        y = public["support"]["labels"][:n]
        true_nll = float(np.logaddexp(0., (1-2*y)*z).sum())
        running = bool(running and np.linalg.norm(parameter) <= config["known_parameter_norm_bound"]
                       and true_nll <= cp["threshold"])
        cp.update(true_parameter_nll=true_nll, true_parameter_in_running_set=running,
                  oracle_evaluation_after_all_choices=True)
        per_stage_coverage[n] = running
    bound_failures = 0
    previous = None
    for record in bounds:
        lo, hi = np.asarray(record["lower"]), np.asarray(record["upper"])
        hits = (lo <= p) & (p <= hi)
        radius = np.asarray(record["information_radius"])
        record.update(true_probability=p.tolist(), true_query_interval_coverage=float(hits.mean()),
                      all_true_query_probabilities_covered=bool(hits.all()),
                      true_parameter_in_running_set=per_stage_coverage[record["n"]],
                      oracle_evaluation_after_all_choices=True)
        if record["true_parameter_in_running_set"] and not hits.all():
            bound_failures += int((~hits).sum())
        if previous is not None:
            record["maximum_lower_decrease"] = float(np.max(np.asarray(previous["lower"])-lo))
            record["maximum_upper_increase"] = float(np.max(hi-np.asarray(previous["upper"])))
            record["maximum_radius_increase"] = float(np.max(radius-np.asarray(previous["information_radius"])))
        previous = record
    for record in initial:
        for direction, table in record.pop("_ap").items():
            expected, realized = oracle_ap(table, hidden)
            record[direction+"_expected_ap"], record[direction+"_realized_ap"] = expected, realized
    initial_by_seed = {row["student_seed"]: row for row in initial}
    for row in decisions:
        before, after = row.pop("_before_ap"), row.pop("_after_ap")
        for direction in ("i2t", "t2i", "mean"):
            eb, rb = oracle_ap(before[direction], hidden)
            ea, ra = oracle_ap(after[direction], hidden)
            row.update({direction+"_expected_ap": ea, direction+"_realized_ap": ra,
                        direction+"_expected_increment": ea-eb, direction+"_realized_increment": ra-rb})
        baseline = initial_by_seed[row["student_seed"]]
        row["expected_gain_from_initial"] = row["mean_expected_ap"] - baseline["mean_expected_ap"]
        row["realized_gain_from_initial"] = row["mean_realized_ap"] - baseline["mean_realized_ap"]
        row["true_parameter_in_running_set"] = per_stage_coverage[row["n"]]
        row["oracle_evaluation_after_all_choices"] = True
        row["accepted_expected_decrease"] = bool(row["accepted_nonzero_update"]
                                                  and row["mean_expected_increment"] < -1e-12)
        if row["true_parameter_in_running_set"] and row["selected_difference_lower_bound"] > row["mean_expected_increment"]+1e-8:
            bound_failures += 1
    for row in attempts:
        difference = row.pop("_ap_difference")
        truth = float(np.mean((1-p)*difference[:, 0]+p*difference[:, 1]))
        row.update(true_expected_difference=truth, true_parameter_in_running_set=per_stage_coverage[row["n"]],
                   bound_exceeds_true_difference=bool(row["lower_bound"] > truth+1e-8),
                   oracle_evaluation_after_all_choices=True)
        if row["true_parameter_in_running_set"] and row["bound_exceeds_true_difference"]:
            bound_failures += 1
    entropy = -xlogy(p, p)-xlogy(1-p, 1-p)
    return {"simultaneous_parameter_coverage": bool(all(per_stage_coverage.values())),
            "simultaneous_query_probability_coverage": bool(all(row["all_true_query_probabilities_covered"] for row in bounds)),
            "conditional_mathematical_check_failures": bound_failures,
            "mean_true_aleatoric_entropy": float(entropy.mean()),
            "all_choices_frozen_before_oracle_evaluation": True}


def run_world(config, world):
    start = time.perf_counter()
    public, hidden, manifest = make_world(config, world)
    models, epoch_records, initial, states = [], [], [], {}
    for student_seed in config["student_seeds"]:
        seed = int(np.random.SeedSequence([config["seed"], world, student_seed, 200]).generate_state(1)[0])
        student = AdamStudent(initialize(seed), config["learning_rate"], config["weight_decay"], config["fume_alpha"])
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], world, student_seed, 201]))
        for epoch in range(config["warmup_epochs"]):
            record = student.train_epoch(public["warmup"]["image"], public["warmup"]["text"],
                                         public["warmup"]["labels"], config["batch_size"], rng)
            epoch_records.append({"world": world, "student_seed": student_seed, "n": 0,
                                  "epoch": epoch+1, **record})
        ident = f"w{world}_s{student_seed}_initial"
        baseline = {"id": ident, "parameters": copy_parameters(student.parameters),
                    "ap": ap_tables(student.parameters, public)}
        models.append({"world": world, "student_seed": student_seed, "id": ident, "n": 0,
                       "parameters": baseline["parameters"], "adam_steps": student.step_count})
        initial.append({"world": world, "student_seed": student_seed, "id": ident,
                        "_ap": baseline["ap"]})
        states[student_seed] = {"student": student, "rng": rng,
                                "incumbents": {policy: baseline for policy in config["policies"]}}
    sequence = LogisticConfidenceSequence(config["teacher_dimension"], config["confidence_delta"],
                                           config["regularization"], config["known_parameter_norm_bound"],
                                           config["projection_outward_tolerance"])
    checkpoints, bounds, decisions, attempts = [], [], [], []
    for n in config["support_prefixes"]:
        evidence_start = time.perf_counter()
        cp = sequence.update(public["support"]["features"][:n], public["support"]["labels"][:n],
                             observation_ids=[f"w{world}:support:{i}" for i in range(n)])
        checkpoints.append({"world": world, **cp})
        projection = sequence.project(public["query"]["features"])
        elapsed = time.perf_counter()-evidence_start
        lo, hi = projection["lower"], projection["upper"]
        diagnostic = projection["diagnostics"]
        sides = [side for item in diagnostic["endpoints"] for side in item.values()]
        bounds.append({
            "world": world, "n": n, "lower": lo, "upper": hi,
            "information_radius": information_radius(lo, hi),
            "mean_width": float(np.mean(hi-lo)),
            "mean_information_radius": float(np.mean(information_radius(lo, hi))),
            "evidence_and_projection_seconds": elapsed,
            "maximum_primal_dual_gap": diagnostic["maximum_primal_dual_gap"],
            "maximum_arithmetic_margin": max(side["arithmetic_margin"] for side in sides),
            "projection_solver_failures": sum(not side["solver_success"] for side in sides),
            "fallback_count": diagnostic["fallback_count"],
            "endpoint_count": diagnostic["endpoint_count"],
            "maximum_normalized_primal_violation": max(side["maximum_normalized_violation"] for side in sides),
            "cached_lower_reused": diagnostic.get("cached_lower_reused", 0),
            "cached_upper_reused": diagnostic.get("cached_upper_reused", 0),
        })
        point_p1 = expit(public["query"]["features"] @ sequence.map_estimate)
        train = {key: np.concatenate((public["warmup"][key], public["support"][key][:n]))
                 for key in ("image", "text", "labels")}
        for student_seed, state in states.items():
            candidates = []
            for epoch in range(config["epochs_per_prefix"]):
                record = state["student"].train_epoch(train["image"], train["text"], train["labels"],
                                                       config["batch_size"], state["rng"])
                epoch_records.append({"world": world, "student_seed": student_seed, "n": n,
                                      "epoch": epoch+1, **record})
                ident = f"w{world}_s{student_seed}_n{n}_e{epoch+1}"
                candidate = {"id": ident, "parameters": copy_parameters(state["student"].parameters),
                             "ap": ap_tables(state["student"].parameters, public)}
                candidates.append(candidate)
                models.append({"world": world, "student_seed": student_seed, "id": ident, "n": n,
                               "parameters": candidate["parameters"], "adam_steps": state["student"].step_count})
            for policy in config["policies"]:
                chosen, decision, proposal_records = choose_candidate(
                    policy, state["incumbents"][policy], candidates, lo, hi, point_p1, config)
                state["incumbents"][policy] = chosen
                decisions.append({"world": world, "student_seed": student_seed, "n": n, **decision})
                attempts.extend({"world": world, "student_seed": student_seed, "n": n,
                                 "policy": policy, "before_id": decision["before_id"], **row}
                                for row in proposal_records)
    audit = evaluate_frozen_world(decisions, attempts, bounds, checkpoints, initial, hidden, public, config)
    audit.update(world=world, elapsed_seconds=time.perf_counter()-start,
                 total_adam_updates=sum(state["student"].step_count for state in states.values()))
    return {"manifest": [manifest], "models": models, "training": epoch_records, "initial": initial,
            "checkpoints": checkpoints, "bounds": bounds, "decisions": decisions,
            "attempts": attempts, "world_audit": [audit]}


def mean_interval(values):
    a = np.asarray(values, dtype=float)
    center = float(a.mean())
    if len(a) < 2:
        return {"mean": center, "ci95": None, "independent_worlds": len(a)}
    half = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return {"mean": center, "ci95": [center-half, center+half], "independent_worlds": len(a)}


def summarize(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    decisions, bounds = load_jsonl(out/"decisions.jsonl"), load_jsonl(out/"bounds.jsonl")
    audits, attempts = load_jsonl(out/"world_audit.jsonl"), load_jsonl(out/"attempts.jsonl")
    initial = load_jsonl(out/"initial.jsonl")
    worlds = [r["world"] for r in audits]
    initial_metric = mean_interval([np.mean([r["mean_expected_ap"] for r in initial if r["world"] == w])
                                    for w in worlds])
    groups = []
    metrics = ("mean_expected_ap", "mean_realized_ap", "expected_gain_from_initial",
               "realized_gain_from_initial", "mean_expected_increment")
    for n in config["support_prefixes"]:
        for policy in config["policies"]:
            rows = [r for r in decisions if r["n"] == n and r["policy"] == policy]
            group = {"n": n, "policy": policy, "decisions": len(rows),
                     "accepted_nonzero_updates": sum(r["accepted_nonzero_update"] for r in rows),
                     "accepted_expected_decreases": sum(r["accepted_expected_decrease"] for r in rows),
                     "accepted_realized_decreases": sum(r["accepted_nonzero_update"] and r["mean_realized_increment"] < -1e-12 for r in rows)}
            for metric in metrics:
                group[metric] = mean_interval([np.mean([r[metric] for r in rows if r["world"] == w]) for w in worlds])
            groups.append(group)
    curve = []
    for n in config["support_prefixes"]:
        rows = [r for r in bounds if r["n"] == n]
        curve.append({"n": n, "mean_width": mean_interval([r["mean_width"] for r in rows]),
                      "mean_information_radius": mean_interval([r["mean_information_radius"] for r in rows]),
                      "projection_seconds": sum(r["evidence_and_projection_seconds"] for r in rows),
                      "maximum_primal_dual_gap": max(r["maximum_primal_dual_gap"] for r in rows),
                      "fallback_endpoints": sum(r["fallback_count"] for r in rows),
                      "solver_failure_endpoints": sum(r["projection_solver_failures"] for r in rows)})
    count = len(audits)
    hits = sum(r["simultaneous_parameter_coverage"] for r in audits)
    ci = [0. if hits == 0 else float(beta.ppf(.025, hits, count-hits+1)),
          1. if hits == count else float(beta.ppf(.975, hits+1, count-hits))]
    cert_attempts = [r for r in attempts if r["policy"] == "certified_guard"]
    final_n = config["support_prefixes"][-1]
    final = [r for r in decisions if r["n"] == final_n]
    comparisons = {}
    for policy in ("certified_guard", "point_guard"):
        by_world = []
        for world in worlds:
            own = [r["mean_expected_ap"] for r in final if r["world"] == world and r["policy"] == policy]
            ordinary = [r["mean_expected_ap"] for r in final if r["world"] == world and r["policy"] == "ordinary_latest"]
            by_world.append(float(np.mean(own)-np.mean(ordinary)))
        comparisons[policy+"_minus_ordinary_latest"] = mean_interval(by_world)
    result = {
        "scope": config["scope"], "real_nus_training": False,
        "original_full_network_training": False, "controlled_student_training": True,
        "independent_worlds": count, "student_initializations_per_world": len(config["student_seeds"]),
        "trained_student_paths": count*len(config["student_seeds"]),
        "total_adam_updates": sum(r["total_adam_updates"] for r in audits),
        "initial_expected_ap": initial_metric, "groups": groups, "interval_curve": curve,
        "simultaneous_parameter_coverage": {"covered": hits, "total": count, "estimate": hits/count,
                                           "clopper_pearson_two_sided_95": ci},
        "simultaneous_query_probability_coverage": {
            "covered": sum(r["simultaneous_query_probability_coverage"] for r in audits), "total": count},
        "conditional_mathematical_check_failures": sum(r["conditional_mathematical_check_failures"] for r in audits),
        "maximum_lower_decrease": max([r.get("maximum_lower_decrease", 0.) for r in bounds]),
        "maximum_upper_increase": max([r.get("maximum_upper_increase", 0.) for r in bounds]),
        "maximum_radius_increase": max([r.get("maximum_radius_increase", 0.) for r in bounds]),
        "certificate_candidate_checks": len(cert_attempts),
        "positive_expected_candidates_without_positive_certificate": sum(
            r["true_expected_difference"] > config["acceptance_margin"] and r["lower_bound"] <= config["acceptance_margin"]
            for r in cert_attempts),
        "true_positive_expected_candidates": sum(r["true_expected_difference"] > config["acceptance_margin"] for r in cert_attempts),
        "final_paired_comparisons": comparisons,
        "mean_true_aleatoric_entropy": float(np.mean([r["mean_true_aleatoric_entropy"] for r in audits])),
        "confidence_interval_unit": "independent_world; average three paired initializations inside each world",
        "floating_certificates": "analytic dual bounds with explicit float64 margins, not formal interval arithmetic",
    }
    result["status"] = "passed" if result["conditional_mathematical_check_failures"] == 0 else "failed"
    write_json(out/"summary.json", result)
    return result


def run(config_path, out, smoke=False):
    config = json.loads(Path(config_path).read_text())
    if smoke:
        config.update(worlds=2, student_seeds=[11], seed=config["seed"]+700000,
                      scope="synthetic_smoke_excluded_from_main")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config_digest = hashlib.sha256(encode(config).encode()).hexdigest()
    write_json(out/"config.json", config)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.glob("*.py")}
    state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "config_sha256": config_digest, "config_frozen_before_generation": True,
             "source_sha256": source_hashes, "python": platform.python_version(),
             "numpy": np.__version__, "scipy": scipy.__version__,
             "real_training": False, "controlled_student_training": True, "completed_worlds": 0}
    write_json(out/"run.json", state)
    started = time.perf_counter()
    try:
        for world in range(config["worlds"]):
            records = run_world(config, world)
            for name, rows in records.items():
                append_jsonl(out/(name+".jsonl"), rows)
            state.update(completed_worlds=world+1, elapsed_seconds=time.perf_counter()-started)
            write_json(out/"run.json", state)
            print(encode({"world_complete": world+1, "worlds": config["worlds"],
                          "elapsed_seconds": round(state["elapsed_seconds"], 2),
                          "coverage": records["world_audit"][0]["simultaneous_parameter_coverage"],
                          "mathematical_check_failures": records["world_audit"][0]["conditional_mathematical_check_failures"]}),
                  flush=True)
        result = summarize(out)
        state.update(status=result["status"], finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.perf_counter()-started)
        write_json(out/"run.json", state)
        return result
    except Exception as exc:
        state.update(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter()-started)
        write_json(out/"run.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--config", type=Path, default=PACKAGE/"pilot_config.json")
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--smoke", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.out, args.smoke) if args.command == "run" else summarize(args.out)
    print(encode({"status": result["status"], "independent_worlds": result["independent_worlds"],
                  "total_adam_updates": result["total_adam_updates"],
                  "real_nus_training": result["real_nus_training"]}), flush=True)


if __name__ == "__main__":
    main()
