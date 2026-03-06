#!/usr/bin/env python3
"""
Generate ai_layer/REPORT.md from build logs.

Parses the model conversion log (Vela output) and pack layer generation log
to produce a structured Markdown report with:
  - Selected operators table (portable + quantized)
  - Vela conversion log (TOSA graphs, NPU perf, network summary)
  - Final exported program graph

Usage:
  python3 scripts/generate_report.py \
      --conversion-log ai_layer/logs/model_conversion_TIMESTAMP.log \
      --pack-log ai_layer/logs/generate_pack_layer_TIMESTAMP.log \
      -o ai_layer/REPORT.md

  # Auto-detect latest logs in a directory:
  python3 scripts/generate_report.py \
      --log-dir ai_layer/logs \
      -o ai_layer/REPORT.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Log Parsing
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_log(log_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent log file matching a prefix, sorted by timestamp."""
    candidates = sorted(log_dir.glob(f"{prefix}_*.log"), reverse=True)
    return candidates[0] if candidates else None


def parse_pack_log(log_path: Path) -> dict:
    """
    Parse the pack layer generation log for operator→component mappings.

    Returns dict with:
      portable:  list of (operator, component) tuples
      quantized: list of (operator, component) tuples
      total_ops: int
      n_portable: int
      n_quantized: int
    """
    portable = []
    quantized = []
    total_ops = 0

    with open(log_path) as f:
        for line in f:
            line = line.rstrip("\n")

            # Total unique operators
            m = re.search(r"Total unique operators:\s*(\d+)", line)
            if m:
                total_ops = int(m.group(1))

            # Mapping lines: "  aten::add → Operators Portable add"
            m = re.match(r"\s+(\S+)\s+→\s+Operators (Portable|Quantized)\s+(.+)", line)
            if m:
                op_name = m.group(1)
                kind = m.group(2)
                component = m.group(3).strip()
                if kind == "Portable":
                    portable.append((op_name, component))
                else:
                    quantized.append((op_name, component))

    return {
        "portable": portable,
        "quantized": quantized,
        "total_ops": total_ops,
        "n_portable": len(portable),
        "n_quantized": len(quantized),
    }


def parse_conversion_log(log_path: Path) -> dict:
    """
    Parse the model conversion log for Vela sections.

    Returns dict with:
      before_opt:    str  — "Before Graph Optimisation" block
      after_opt:     str  — "After Graph Optimization" block
      perf_table:    str  — NPU performance table
      network_summary: str — Network summary block
      final_graph:   str  — Final exported program graph
      system_config: str  — System/memory/architecture config
    """
    text = log_path.read_text()
    lines = text.splitlines()

    sections = {
        "before_opt": "",
        "after_opt": "",
        "perf_table": "",
        "network_summary": "",
        "final_graph": "",
        "system_config": "",
    }

    def extract_block(start_marker: str, end_markers: list[str],
                      include_start: bool = False) -> str:
        """Extract text between start_marker and the first end_marker found."""
        capturing = False
        result = []
        for line in lines:
            if start_marker in line:
                capturing = True
                if include_start:
                    result.append(line)
                continue
            if capturing:
                for em in end_markers:
                    if em in line:
                        return "\n".join(result).strip()
                result.append(line)
        if capturing:
            return "\n".join(result).strip()
        return ""

    # ── Before Graph Optimisation ──
    # There are multiple occurrences; we want the FIRST one
    first_before = extract_block(
        "[ Before Graph Optimisation ]",
        ["[ After Graph Optim", "[ Graph With Tensor"],
    )
    sections["before_opt"] = first_before

    # ── After Graph Optimization ──
    # Also multiple occurrences; we want the SECOND one (the optimised version)
    after_blocks = []
    capturing = False
    buf = []
    for line in lines:
        if "[ After Graph Optimization ]" in line:
            if buf and capturing:
                after_blocks.append("\n".join(buf).strip())
            capturing = True
            buf = []
            continue
        if capturing:
            if "[ Graph With Tensor" in line or "Schedule:" in line:
                after_blocks.append("\n".join(buf).strip())
                capturing = False
                buf = []
                continue
            buf.append(line)
    if capturing and buf:
        after_blocks.append("\n".join(buf).strip())
    # Use the second block if available (post-constant-fold), else first
    sections["after_opt"] = after_blocks[-1] if after_blocks else ""

    # ── NPU Performance Table ──
    # Starts with "Original Operator" header row, ends with blank line
    perf_lines = []
    capturing = False
    for line in lines:
        if line.strip().startswith("Original Operator") and "NNG Operator" in line:
            capturing = True
            perf_lines.append(line)
            continue
        if capturing:
            if line.strip() == "":
                break
            perf_lines.append(line)
    sections["perf_table"] = "\n".join(perf_lines).strip()

    # ── Network Summary ──
    # Starts with "Network summary for ..." and ends before "class GraphModule"
    summary_lines = []
    capturing = False
    for line in lines:
        if line.strip().startswith("Network summary for"):
            capturing = True
            continue
        if capturing:
            if "class GraphModule" in line:
                break
            summary_lines.append(line)
    sections["network_summary"] = "\n".join(summary_lines).strip()

    # ── Final Exported Program Graph ──
    # The LAST "class GraphModule" block in the file (after Vela output)
    graph_blocks = []
    capturing = False
    buf = []
    for line in lines:
        if "class GraphModule" in line:
            if buf and capturing:
                graph_blocks.append("\n".join(buf).strip())
            capturing = True
            buf = [line]
            continue
        if capturing:
            buf.append(line)
    if capturing and buf:
        graph_blocks.append("\n".join(buf).strip())
    sections["final_graph"] = graph_blocks[-1].strip() if graph_blocks else ""

    # ── System Configuration ──
    config_lines = []
    capturing = False
    for line in lines:
        if line.strip().startswith("Configuration files:"):
            capturing = True
            config_lines.append(line)
            continue
        if capturing:
            if line.strip().startswith("Original Operator") or line.strip() == "":
                if any("Architecture Settings:" in l for l in config_lines):
                    break
            config_lines.append(line)
            if "fast_storage_mem_area" in line:
                break
    sections["system_config"] = "\n".join(config_lines).strip()

    return sections


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(pack_info: dict, vela_info: dict) -> str:
    """Generate the REPORT.md content from parsed log data."""
    lines = []

    lines.append("# AI Layer Report")
    lines.append("")

    # ── Selected Operators ──
    lines.append("## Selected Operators")
    lines.append("")

    if pack_info["portable"]:
        lines.append("### Portable Operators (CPU)")
        lines.append("")
        lines.append("| Operator | Pack Component |")
        lines.append("|----------|---------------|")
        for op, comp in pack_info["portable"]:
            lines.append(
                f"| `{op}` | `Machine Learning:ExecuTorch:Operators Portable {comp}` |"
            )
        lines.append("")

    if pack_info["quantized"]:
        lines.append("### Quantized Operators (NPU wrapper)")
        lines.append("")
        lines.append("| Operator | Pack Component |")
        lines.append("|----------|---------------|")
        for op, comp in pack_info["quantized"]:
            lines.append(
                f"| `{op}` | `Machine Learning:ExecuTorch:Operators Quantized {comp}` |"
            )
        lines.append("")

    total = pack_info["n_portable"] + pack_info["n_quantized"]
    lines.append(
        f"**Total:** {pack_info['n_portable']} portable"
        f" + {pack_info['n_quantized']} quantized"
        f" = {total} operator components"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Vela Conversion Log ──
    lines.append("## Vela Conversion Log")
    lines.append("")

    if vela_info["before_opt"]:
        lines.append("### TOSA Graph — Before Optimisation")
        lines.append("")
        lines.append("```")
        lines.append(vela_info["before_opt"])
        lines.append("```")
        lines.append("")

    if vela_info["after_opt"]:
        lines.append("### TOSA Graph — After Optimisation")
        lines.append("")
        lines.append("```")
        lines.append(vela_info["after_opt"])
        lines.append("```")
        lines.append("")

    if vela_info["perf_table"]:
        lines.append("### NPU Performance Summary")
        lines.append("")
        lines.append("```")
        lines.append(vela_info["perf_table"])
        lines.append("```")
        lines.append("")

    if vela_info["network_summary"]:
        lines.append("### Network Summary")
        lines.append("")
        lines.append("```")
        lines.append(vela_info["network_summary"])
        lines.append("```")
        lines.append("")

    if vela_info["final_graph"]:
        lines.append("### Final Exported Program Graph")
        lines.append("")
        lines.append("```python")
        lines.append(vela_info["final_graph"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate ai_layer/REPORT.md from build logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--conversion-log", "-c",
        help="Path to model_conversion_*.log (Vela output)",
    )
    parser.add_argument(
        "--pack-log", "-p",
        help="Path to generate_pack_layer_*.log (operator mapping)",
    )
    parser.add_argument(
        "--log-dir", "-L",
        help="Auto-detect latest logs in this directory",
    )
    parser.add_argument(
        "--output", "-o",
        default="ai_layer/model/REPORT.md",
        help="Output path for REPORT.md (default: ai_layer/model/REPORT.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print report to stdout without writing file",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Resolve log paths
    conversion_log = None
    pack_log = None

    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"[ERROR] Log directory not found: {log_dir}", file=sys.stderr)
            sys.exit(1)
        conversion_log = find_latest_log(log_dir, "model_conversion")
        pack_log = find_latest_log(log_dir, "generate_pack_layer")
        if args.verbose:
            print(f"[INFO] Auto-detected conversion log: {conversion_log}")
            print(f"[INFO] Auto-detected pack log:       {pack_log}")

    if args.conversion_log:
        conversion_log = Path(args.conversion_log)
    if args.pack_log:
        pack_log = Path(args.pack_log)

    if not conversion_log or not conversion_log.exists():
        print(f"[ERROR] Model conversion log not found: {conversion_log}", file=sys.stderr)
        print("  Use --conversion-log or --log-dir to specify.", file=sys.stderr)
        sys.exit(1)

    if not pack_log or not pack_log.exists():
        print(f"[ERROR] Pack layer generation log not found: {pack_log}", file=sys.stderr)
        print("  Use --pack-log or --log-dir to specify.", file=sys.stderr)
        sys.exit(1)

    # Parse logs
    if args.verbose:
        print(f"[INFO] Parsing pack log: {pack_log}")
    pack_info = parse_pack_log(pack_log)
    if args.verbose:
        print(f"  Found {pack_info['n_portable']} portable, {pack_info['n_quantized']} quantized operators")

    if args.verbose:
        print(f"[INFO] Parsing conversion log: {conversion_log}")
    vela_info = parse_conversion_log(conversion_log)
    if args.verbose:
        for key, val in vela_info.items():
            status = "found" if val else "not found"
            print(f"  {key}: {status}")

    # Generate report
    report = generate_report(pack_info, vela_info)

    if args.dry_run:
        print(report)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"[OK] Generated {output_path}")
        print(f"     {pack_info['n_portable']} portable + {pack_info['n_quantized']} quantized operators")


if __name__ == "__main__":
    main()
