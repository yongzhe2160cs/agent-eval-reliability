"""Worked example: a full reliability + comparison pass on synthetic agent runs.

Run with:  uv run python examples/demo.py

Everything here is keyless and offline — `simulate_agent_runs` stands in for
real model runs. Swap it for `from_json` / `from_inspect_log` on your own eval
logs and the rest is identical.
"""

from __future__ import annotations

import agentrel as ar


def main() -> None:
    # --- 1. ingest -------------------------------------------------------
    # Two agent configs evaluated on the SAME 40 tasks, 10 runs each.
    # "baseline" is a weaker config; "candidate" is stronger and steadier.
    baseline = ar.simulate_agent_runs(
        n_tasks=40,
        runs_per_task=10,
        base_skill=0.45,
        skill_spread=0.22,
        luck=0.10,
        agent="baseline",
        seed=11,
    )
    candidate = ar.simulate_agent_runs(
        n_tasks=40,
        runs_per_task=10,
        base_skill=0.62,
        skill_spread=0.22,
        luck=0.04,
        agent="candidate",
        seed=11,
    )

    # --- 2. reliability report for the candidate -------------------------
    report = ar.reliability_report(candidate, ks=(1, 2, 5), seed=0)
    print("=" * 72)
    print(report.summary())

    # --- 3. how many runs would we need? ---------------------------------
    runs_needed = ar.min_runs_for_ci_width(0.1, p=report.pass_at_k[1].value)
    print("=" * 72)
    print(
        f"Power: to pin pass@1 to a +/-0.10 CI half-width at this pass rate, "
        f"you need ~{runs_needed} runs/task."
    )

    # --- 4. flakiest tasks -----------------------------------------------
    print("=" * 72)
    print("Flakiest tasks (widest pass-rate CI — trust these least):")
    for t in report.flakiness.flaky_tasks[:5]:
        print(
            f"  {t.task_id}: pass_rate={t.pass_rate:.2f} "
            f"CI=[{t.ci_low:.2f}, {t.ci_high:.2f}] width={t.ci_width:.2f} (n={t.n_runs})"
        )
    if not report.flakiness.flaky_tasks:
        print("  (none — every task's CI is within tolerance)")

    # --- 5. paired comparison: candidate vs baseline ---------------------
    cmp = ar.compare_agents(candidate, baseline, seed=0)
    print("=" * 72)
    print(f"Paired comparison: {cmp.agent_a} vs {cmp.agent_b} on {cmp.n_tasks} shared tasks")
    d = cmp.paired.delta
    print(f"  mean per-task delta (A-B): {d!r}")
    print(f"  paired t-test p={cmp.paired.t_pvalue:.2e}  significant={cmp.paired.significant}")
    print(
        f"  per-task discoveries: Holm(FWER)={cmp.n_sig_holm}  "
        f"BH(FDR)={cmp.n_sig_bh}  (of {cmp.n_tasks} tasks)"
    )

    # --- 6. determinism of the analysis pipeline -------------------------
    check = ar.determinism_check(lambda: [ar.reliability_report(candidate, seed=0).icc])
    print("=" * 72)
    print(f"Determinism check (seeded analysis): {check.detail}")


if __name__ == "__main__":
    main()
