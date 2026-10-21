"""Plain-text summaries of protocol results."""

from __future__ import annotations

from bidrig.collusion import IN_SCOPE


def fmt(v: dict) -> str:
    if "lo" not in v:
        return f"{v['point']:.3f}"
    return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def design_table(results: dict, design: str, sets) -> str:
    conds = [c for c in ("in", "size", "spread") if f"honest@{c}" in sets]
    head = "| mechanism | LOMO AUC (pooled) | " + " | ".join(f"AUC @{c}" for c in conds)
    head += " | prec@10% | base | TPR@5%FA |"
    lines = [head, "|" + "---|" * (5 + len(conds))]
    r = lambda k: results[f"{design}::{k}"]  # noqa: E731
    for m in IN_SCOPE:
        cells = [fmt(r(f"auc[{m}]"))] + [fmt(r(f"auc[{m}@{c}]")) for c in conds]
        cells += [fmt(r(f"prec10[{m}]")), fmt(r(f"base_rate[{m}]")), fmt(r(f"tpr[{m}]"))]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append(f"| **macro** | {fmt(r('macro_auc'))} | " + " | " * len(conds))
    lines[-1] += f"{fmt(r('macro_prec10'))} | | |"
    lines.append("")
    lines.append("| honest set | FA (all-mech model) | worst LOMO FA |")
    lines.append("|---|---|---|")
    for h in sorted(s for s in sets if s.startswith("honest@")):
        lines.append(f"| {h} | {fmt(r(f'fa[{h}]'))} | {fmt(r(f'fa_lomo_max[{h}]'))} |")
    if f"{design}::control_auc" in results:
        lines.append(
            f"\nUniform-markup control (out of scope): AUC {fmt(r('control_auc'))}, "
            f"alarm {fmt(r('control_alarm'))}"
        )
    return "\n".join(lines)


def comparison_table(results: dict, designs, baselines) -> str:
    lines = ["| design | macro LOMO AUC | macro prec@10% | min mech AUC |", "|---|---|---|---|"]
    for d in designs:
        mins = min(results[f"{d}::auc[{m}]"]["point"] for m in IN_SCOPE)
        lines.append(
            f"| {d} | {fmt(results[f'{d}::macro_auc'])} | {fmt(results[f'{d}::macro_prec10'])}"
            f" | {mins:.3f} |"
        )
    diffs = [k for k in results if "-" in k.split("::")[0] and k.endswith("::macro_auc")]
    if diffs:
        lines += ["", "| difference (macro AUC) | value [95% CI] |", "|---|---|"]
        for k in sorted(diffs):
            lines.append(f"| {k.split('::')[0]} | {fmt(results[k])} |")
    return "\n".join(lines)
