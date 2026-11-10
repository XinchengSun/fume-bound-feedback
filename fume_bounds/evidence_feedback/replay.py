"""Independently audit saved controlled experiments without training or refitting.

Only deterministic data generation is reused from run.py. Cosine scoring, AP,
binary probability-box minima, selections, and accounting are recomputed here.
The audit does not re-solve confidence-set projections or certify directed
rounding of transcendental functions. The saved source hashes bind those
operations to the separately tested implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.special import expit, xlogy
from scipy.stats import t
from .run import make_world


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def independently_score(inputs, weights):
    """Preserve both declared normalization steps, including floating tie rules."""
    raw = np.asarray(inputs) @ np.asarray(weights, dtype=float).T
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(raw).all() or np.any(norm == 0):
        raise ValueError("Invalid saved embedding")
    unit = raw / norm
    scaled = unit / np.max(np.abs(unit), axis=1, keepdims=True)
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def independent_ap_table(scores, gallery_labels):
    """Vectorized cumsum AP; no calls to ranking.py."""
    gallery_labels = np.asarray(gallery_labels)
    positions = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    order = np.lexsort((positions, -scores), axis=1)
    ordered = gallery_labels[order]
    relevant = ordered[:, :, None] == np.arange(2)[None, None, :]
    precision = np.cumsum(relevant, axis=1) / np.arange(1, scores.shape[1]+1)[None, :, None]
    numerator = (relevant * precision).sum(axis=1)
    denominator = np.array([(gallery_labels == c).sum() for c in range(2)])
    return np.divide(numerator, denominator[None, :], out=np.zeros_like(numerator),
                     where=denominator[None, :] > 0)


def independent_tables(parameters, public):
    q, g = public["query"], public["gallery"]
    qi = independently_score(q["image"], parameters["image"])
    qt = independently_score(q["text"], parameters["text"])
    gi = independently_score(g["image"], parameters["image"])
    gt = independently_score(g["text"], parameters["text"])
    image = independent_ap_table(qi @ gt.T, g["labels"])
    text = independent_ap_table(qt @ gi.T, g["labels"])
    return {"i2t": image, "t2i": text, "mean": (image+text)/2}


def oracle(table, hidden):
    probability = hidden["query_probability"]
    expected = np.mean(table[:, 0] + probability * (table[:, 1]-table[:, 0]))
    realized = np.mean(table[np.arange(len(table)), hidden["query_labels"]])
    return float(expected), float(realized)


class Audit:
    def __init__(self):
        self.counts, self.errors, self.max_errors = {}, [], {}

    def condition(self, name, condition, detail=""):
        self.counts[name] = self.counts.get(name, 0)+1
        if not bool(condition):
            self.errors.append({"check": name, "detail": detail})

    def numeric(self, name, actual, expected, tolerance=1e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.condition(name+"_finite_shape", a.shape == b.shape and
                       np.isfinite(a).all() and np.isfinite(b).all())
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            return
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.max_errors[name] = max(self.max_errors.get(name, 0.), error)
        self.condition(name+"_tolerance", error <= tolerance, f"error={error}, tolerance={tolerance}")


def audit(out):
    out = Path(out)
    config = json.loads((out/"config.json").read_text())
    state = json.loads((out/"run.json").read_text())
    summary = json.loads((out/"summary.json").read_text())
    names = ("manifest", "models", "training", "initial", "checkpoints",
             "bounds", "decisions", "attempts", "world_audit")
    data = {name: load_rows(out/(name+".jsonl")) for name in names}
    check = Audit()
    worlds, seeds, prefixes = config["worlds"], config["student_seeds"], config["support_prefixes"]
    policies, epochs = config["policies"], config["epochs_per_prefix"]
    check.condition("completed_run", state["status"] == "passed" and state["completed_worlds"] == worlds)
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False).encode()).hexdigest()
    check.condition("configuration_hash", digest == state["config_sha256"])
    source_verified = {}
    for name, expected_hash in state["source_sha256"].items():
        source = Path(__file__).resolve().parent/name
        observed_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        source_verified[name] = observed_hash
        check.condition("frozen_source_hash", observed_hash == expected_hash, name)
    counts = {
        "manifest": worlds, "models": worlds*len(seeds)*(1+len(prefixes)*epochs),
        "training": worlds*len(seeds)*(config["warmup_epochs"]+len(prefixes)*epochs),
        "initial": worlds*len(seeds), "checkpoints": worlds*len(prefixes),
        "bounds": worlds*len(prefixes), "decisions": worlds*len(seeds)*len(prefixes)*len(policies),
        "attempts": worlds*len(seeds)*len(prefixes)*len(policies)*epochs, "world_audit": worlds,
    }
    for name, count in counts.items():
        check.condition("record_count", len(data[name]) == count, name)
    all_model_ids = [r["id"] for r in data["models"]]
    check.condition("unique_model_ids", len(set(all_model_ids)) == len(all_model_ids))
    model_by_id = {r["id"]: r for r in data["models"]}
    decisions_by_key = {(r["world"], r["student_seed"], r["n"], r["policy"]): r for r in data["decisions"]}
    attempts_by_key = {(r["world"], r["student_seed"], r["n"], r["policy"], r["candidate_id"]): r
                       for r in data["attempts"]}
    check.condition("unique_decision_keys", len(decisions_by_key) == len(data["decisions"]))
    check.condition("unique_attempt_keys", len(attempts_by_key) == len(data["attempts"]))
    total_steps, simultaneous_hits, selected_metrics = 0, 0, {}
    max_certificate_excess, max_covered_certificate_excess = -np.inf, -np.inf
    for world in range(worlds):
        public, hidden, regenerated_manifest = make_world(config, world)
        manifests = [r for r in data["manifest"] if r["world"] == world]
        check.condition("world_manifest", len(manifests) == 1 and manifests[0] == regenerated_manifest)
        canonical = np.concatenate([public[name]["features"] for name in ("warmup", "support", "gallery", "query")])
        check.condition("unique_canonical_entities", len(np.unique(canonical, axis=0)) == len(canonical))
        check.condition("hidden_query_labels_excluded", "labels" not in public["query"])
        models = [r for r in data["models"] if r["world"] == world]
        tables = {}
        for record in models:
            for modality in ("image", "text"):
                weights = np.asarray(record["parameters"][modality], dtype=float)
                check.condition("saved_weight_shape", weights.shape == (2, 3) and np.isfinite(weights).all())
            tables[record["id"]] = independent_tables(record["parameters"], public)
        checkpoints = {r["n"]: r for r in data["checkpoints"] if r["world"] == world}
        bounds = {r["n"]: r for r in data["bounds"] if r["world"] == world}
        check.condition("prefix_keys", set(checkpoints) == set(prefixes) == set(bounds))
        running_covered, old_bound = True, None
        point_probabilities, covered = {}, {}
        for prefix_index, n in enumerate(prefixes):
            cp, bound = checkpoints[n], bounds[n]
            X = public["support"]["features"][:n].astype(float)
            y = public["support"]["labels"][:n].astype(float)
            v = np.asarray(cp["map_estimate"])
            loss = float(np.logaddexp(0., (1-2*y)*(X@v)).sum())
            objective = loss + config["regularization"]*(v@v)/2
            sign, logdet = np.linalg.slogdet(np.eye(X.shape[1])+(X.T@X)/(4*config["regularization"]))
            raw = objective + logdet/2 + np.log(1/config["confidence_delta"])
            margin = config["projection_outward_tolerance"]*(1+abs(objective)+abs(logdet)+abs(np.log(config["confidence_delta"])))
            check.condition("threshold_positive_determinant", sign == 1)
            check.numeric("threshold_raw", cp["threshold_raw"], raw)
            check.numeric("threshold_margin", cp["threshold_margin"], margin)
            check.numeric("threshold", cp["threshold"], raw+margin)
            check.condition("checkpoint_prefix_hash", cp["prefix_sha256"] == array_digest(X, y))
            check.condition("checkpoint_protocol", cp["n"] == n and cp["checkpoint_sizes"] == prefixes[:prefix_index+1]
                            and cp["dimension"] == config["teacher_dimension"]
                            and cp["radius"] == config["known_parameter_norm_bound"]
                            and cp["delta"] == config["confidence_delta"]
                            and cp["regularization"] == config["regularization"])
            true_nll = float(np.logaddexp(0., (1-2*y)*(X@hidden["parameter"])).sum())
            running_covered = bool(running_covered and np.linalg.norm(hidden["parameter"]) <= cp["radius"]
                                   and true_nll <= cp["threshold"])
            covered[n] = running_covered
            check.numeric("true_parameter_nll", cp["true_parameter_nll"], true_nll)
            check.condition("running_parameter_coverage", cp["true_parameter_in_running_set"] == running_covered)
            lo, hi = np.asarray(bound["lower"]), np.asarray(bound["upper"])
            check.condition("probability_interval_valid", lo.shape == hi.shape == (config["queries"],)
                            and np.all((0 <= lo)&(lo <= hi)&(hi <= 1)))
            check.numeric("true_probability", bound["true_probability"], hidden["query_probability"])
            hits = (lo <= hidden["query_probability"]) & (hidden["query_probability"] <= hi)
            check.numeric("query_probability_coverage", bound["true_query_interval_coverage"], hits.mean())
            check.condition("query_simultaneous_coverage", bound["all_true_query_probabilities_covered"] == bool(hits.all()))
            check.numeric("mean_width", bound["mean_width"], np.mean(hi-lo))
            if old_bound is not None:
                old_lo, old_hi = np.asarray(old_bound["lower"]), np.asarray(old_bound["upper"])
                check.condition("nested_probability_bounds", np.all(lo >= old_lo) and np.all(hi <= old_hi))
                check.numeric("maximum_lower_decrease", bound["maximum_lower_decrease"], np.max(old_lo-lo))
                check.numeric("maximum_upper_increase", bound["maximum_upper_increase"], np.max(hi-old_hi))
            old_bound = bound
            point_probabilities[n] = expit(public["query"]["features"] @ v)
        simultaneous_hits += int(running_covered)

        world_steps = 0
        for seed in seeds:
            initial_id = f"w{world}_s{seed}_initial"
            initial_rows = [r for r in data["initial"] if r["world"] == world and r["student_seed"] == seed]
            check.condition("initial_identity", len(initial_rows) == 1 and initial_rows[0]["id"] == initial_id)
            initial = initial_rows[0]
            for direction, table in tables[initial_id].items():
                expected, realized = oracle(table, hidden)
                check.numeric("initial_expected_ap", initial[direction+"_expected_ap"], expected)
                check.numeric("initial_realized_ap", initial[direction+"_realized_ap"], realized)
            step = 0
            training = {(r["n"], r["epoch"]): r for r in data["training"]
                        if r["world"] == world and r["student_seed"] == seed}
            for n, epoch_count in [(0, config["warmup_epochs"])]+[(n, epochs) for n in prefixes]:
                entities = config["warmup_entities"]+n
                batches = (entities+config["batch_size"]-1)//config["batch_size"]
                if batches > 1 and entities % config["batch_size"] == 1:
                    batches -= 1
                for epoch in range(1, epoch_count+1):
                    row = training[(n, epoch)]
                    step += batches
                    check.condition("training_entity_count", row["entities"] == entities)
                    check.condition("training_batch_step_count", row["batches"] == batches
                                    and row["cumulative_adam_steps"] == step)
                    if n > 0 or epoch == epoch_count:
                        ident = initial_id if n == 0 else f"w{world}_s{seed}_n{n}_e{epoch}"
                        check.condition("saved_model_steps", model_by_id[ident]["adam_steps"] == step
                                        and model_by_id[ident]["n"] == n)
            world_steps += step
            for policy in policies:
                incumbent = initial_id
                for n in prefixes:
                    decision = decisions_by_key[(world, seed, n, policy)]
                    check.condition("decision_incumbent_id", decision["before_id"] == incumbent)
                    identifiers = [f"w{world}_s{seed}_n{n}_e{e}" for e in range(1, epochs+1)]
                    lo, hi = np.asarray(bounds[n]["lower"]), np.asarray(bounds[n]["upper"])
                    candidate_rows = []
                    for ident in identifiers:
                        row = attempts_by_key[(world, seed, n, policy, ident)]
                        check.condition("attempt_incumbent_id", row["before_id"] == incumbent)
                        difference = tables[ident]["mean"]-tables[incumbent]["mean"]
                        slope = difference[:, 1]-difference[:, 0]
                        nominal_per_query = difference[:, 0] + np.where(slope >= 0, lo, hi)*slope
                        nominal = float(nominal_per_query.mean())
                        plugin = float(np.mean(difference[:, 0]+point_probabilities[n]*slope))
                        truth = float(np.mean(difference[:, 0]+hidden["query_probability"]*slope))
                        unchanged_queries = np.all(difference == 0, axis=1)
                        # The exact dual also rounds arithmetic downward. Its
                        # residual beyond this closed form should be only ULPs.
                        guarded = nominal - config["ap_numerical_slack"]*np.mean(~unchanged_queries)
                        check.numeric("nominal_binary_certificate", row["nominal_minimum"], nominal)
                        check.numeric("guarded_binary_certificate", row["lower_bound"], guarded)
                        check.condition("stored_certificate_outward", row["lower_bound"] <= nominal+1e-12)
                        check.numeric("plugin_expected_difference", row["plugin_expected_difference"], plugin)
                        check.numeric("candidate_true_expected_difference", row["true_expected_difference"], truth)
                        excess = row["lower_bound"]-truth
                        max_certificate_excess = max(max_certificate_excess, excess)
                        if covered[n]:
                            max_covered_certificate_excess = max(max_covered_certificate_excess, excess)
                        check.condition("covered_candidate_certificate", not covered[n] or excess <= 1e-10)
                        check.condition("changed_query_count", row["changed_ap_query_count"] == int((~unchanged_queries).sum()))
                        check.condition("unchanged_ap_flag", row["all_ap_vectors_unchanged"] == bool(unchanged_queries.all()))
                        candidate_rows.append(row)
                    if policy == "ordinary_latest":
                        index, accepted = len(identifiers)-1, True
                    else:
                        field = "plugin_expected_difference" if policy == "point_guard" else "lower_bound"
                        index = max(range(len(candidate_rows)), key=lambda i: candidate_rows[i][field])
                        accepted = candidate_rows[index][field] > config["acceptance_margin"]
                    selected = identifiers[index] if accepted else incumbent
                    check.condition("selection_rule", decision["after_id"] == selected
                                    and decision["chosen_candidate_index"] == (index if accepted else None)
                                    and decision["accepted_parameter_change"] == accepted)
                    check.condition("all_queries_retained", decision["all_queries_retained"]
                                    and decision["candidate_count"] == epochs)
                    expected_bound = candidate_rows[index]["lower_bound"] if accepted else 0.
                    check.numeric("selected_certificate", decision["selected_difference_lower_bound"], expected_bound)
                    check.numeric("proposed_best_certificate", decision["proposed_best_certificate"],
                                  max(r["lower_bound"] for r in candidate_rows))
                    actual_changed = bool(accepted and np.any(tables[selected]["mean"] != tables[incumbent]["mean"]))
                    check.condition("accepted_nonzero_update", decision["accepted_nonzero_update"] == actual_changed)
                    for direction in ("i2t", "t2i", "mean"):
                        before = oracle(tables[incumbent][direction], hidden)
                        after = oracle(tables[selected][direction], hidden)
                        for metric, j in (("expected", 0), ("realized", 1)):
                            check.numeric("selected_"+metric+"_ap", decision[direction+"_"+metric+"_ap"], after[j])
                            check.numeric("selected_"+metric+"_increment", decision[direction+"_"+metric+"_increment"], after[j]-before[j])
                    after = oracle(tables[selected]["mean"], hidden)
                    baseline = oracle(tables[initial_id]["mean"], hidden)
                    check.numeric("expected_gain_from_initial", decision["expected_gain_from_initial"], after[0]-baseline[0])
                    check.numeric("realized_gain_from_initial", decision["realized_gain_from_initial"], after[1]-baseline[1])
                    selected_metrics[(world, seed, n, policy)] = after
                    incumbent = selected
        audit_rows = [r for r in data["world_audit"] if r["world"] == world]
        check.condition("world_adam_count", len(audit_rows) == 1 and audit_rows[0]["total_adam_updates"] == world_steps)
        check.condition("world_simultaneous_coverage", audit_rows[0]["simultaneous_parameter_coverage"] == running_covered)
        total_steps += world_steps

    check.condition("summary_total_steps", summary["total_adam_updates"] == total_steps)
    check.condition("summary_parameter_coverage", summary["simultaneous_parameter_coverage"]["covered"] == simultaneous_hits
                    and summary["simultaneous_parameter_coverage"]["total"] == worlds)
    for group in summary["groups"]:
        n, policy = group["n"], group["policy"]
        for metric, index in (("mean_expected_ap", 0), ("mean_realized_ap", 1)):
            values = np.array([np.mean([selected_metrics[(w, s, n, policy)][index] for s in seeds])
                               for w in range(worlds)])
            center = values.mean()
            check.numeric("summary_world_mean", group[metric]["mean"], center)
            if worlds > 1:
                half = t.ppf(.975, worlds-1)*values.std(ddof=1)/np.sqrt(worlds)
                check.numeric("summary_world_interval", group[metric]["ci95"], [center-half, center+half])
    return {
        "status": "passed" if not check.errors else "failed",
        "scope": "saved_record_replay_no_training_no_confidence_refit",
        "real_dataset_used": False, "training_performed": False,
        "independent_ap_implementation": "vectorized rank precision and cumsum with lexicographic stable ties",
        "independent_certificate_implementation": "binary affine endpoint minimum",
        "worlds": worlds, "student_paths": worlds*len(seeds),
        "record_counts": {name: len(rows) for name, rows in data.items()},
        "total_adam_updates_checked": total_steps, "checks": check.counts,
        "maximum_absolute_errors": check.max_errors,
        "maximum_stored_certificate_minus_true_difference": float(max_certificate_excess),
        "maximum_covered_certificate_minus_true_difference": (
            float(max_covered_certificate_excess) if np.isfinite(max_covered_certificate_excess) else None),
        "violation_count": len(check.errors), "violations": check.errors[:100],
        "source_sha256_verified": source_verified,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": digest,
        "limitations": [
            "Projection optimizers are not rerun; source hashes and separate tests cover that code.",
            "Arithmetic safeguards do not constitute formal interval arithmetic.",
            "Oracle access timing is checked through the frozen source and records, not a process sandbox.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.out)
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc), "training_performed": False,
                  "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    destination = args.out/"replay_verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                                     allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "violation_count": result.get("violation_count"), "error": result.get("error")},
                     ensure_ascii=False), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
