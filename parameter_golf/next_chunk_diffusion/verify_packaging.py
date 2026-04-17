from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path


def parse_log(log_path: Path, summary_path: Path, csv_path: Path, return_code: int) -> None:
    patterns = {
        "train": re.compile(
            r"step:(?P<step>\d+)/(?P<iters>\d+)\s+train_loss:(?P<train_loss>[0-9.]+)\s+"
            r"chunkdiff_loss:(?P<chunkdiff_loss>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms\s+"
            r"step_avg:(?P<step_avg_ms>[0-9.]+)ms\s+tok/s:(?P<tok_per_sec>[0-9.]+)"
        ),
        "val": re.compile(
            r"step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+"
            r"val_bpb:(?P<val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>[0-9.]+)ms"
        ),
        "final": re.compile(
            r"(?P<name>final_[^ ]+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
        ),
        "post_ema": re.compile(r"DIAGNOSTIC post_ema val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"),
        "size": re.compile(r"Total submission size [^:]+:\s*(?P<bytes>\d+)\s+bytes"),
        "model_bytes": re.compile(r"Serialized model(?: int6\+lzma)?:\s*(?P<bytes>\d+)\s+bytes"),
        "sparsity": re.compile(r"sparsity:(?P<kind>step:\d+|final)\s+target:(?P<target>[0-9.]+)\s+actual:(?P<actual>[0-9.]+)"),
    }

    rows: list[dict[str, str]] = []
    summary: dict[str, object] = {"return_code": return_code, "log_path": str(log_path)}
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            if m := patterns["train"].search(line):
                row = {"kind": "train", **m.groupdict()}
                rows.append(row)
                summary["last_train"] = row
            elif m := patterns["val"].search(line):
                row = {"kind": "val", **m.groupdict()}
                rows.append(row)
                summary["last_val"] = row
            elif m := patterns["final"].search(line):
                row = {"kind": "final", **m.groupdict()}
                rows.append(row)
                summary[m.group("name")] = row
            elif m := patterns["post_ema"].search(line):
                summary["post_ema"] = m.groupdict()
            elif m := patterns["size"].search(line):
                summary["total_submission_bytes"] = int(m.group("bytes"))
            elif m := patterns["model_bytes"].search(line):
                summary.setdefault("serialized_model_bytes", []).append(int(m.group("bytes")))
            elif m := patterns["sparsity"].search(line):
                summary["last_sparsity"] = m.groupdict()

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def zip_run_dir(run_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(run_dir.parent))


def main() -> None:
    if len(sys.argv) == 4:
        run_dir = Path(sys.argv[1])
        zip_path = Path(sys.argv[2])
        return_code = int(sys.argv[3])
    else:
        run_dir = Path("packaging_smoke")
        zip_path = Path("packaging_smoke.zip")
        return_code = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train.log").write_text(
            "\n".join(
                [
                    "step:50/20000 train_loss:3.2100 chunkdiff_loss:3.9000 train_time:12000ms step_avg:240.00ms tok/s:270000 progress:0.4% eta:49.1m",
                    "sparsity:final target:0.500 actual:0.501 include:mlp,attn",
                    "DIAGNOSTIC post_ema val_loss:2.0000 val_bpb:1.1800 eval_time:2227ms",
                    "Serialized model int6+lzma: 15249936 bytes",
                    "Total submission size int6+lzma: 15353950 bytes",
                    "final_int6_roundtrip_exact val_loss:1.92000000 val_bpb:1.13700000",
                    "final_int6_sliding_window_exact val_loss:1.88000000 val_bpb:1.11400000",
                ]
            )
            + "\n"
        )
        (run_dir / "final_model.int6.ptz").write_bytes(b"dummy")

    parse_log(run_dir / "train.log", run_dir / "metrics_summary.json", run_dir / "metrics.csv", return_code)
    zip_run_dir(run_dir, zip_path)
    names = set(zipfile.ZipFile(zip_path).namelist())
    required = {
        f"{run_dir.name}/train.log",
        f"{run_dir.name}/metrics_summary.json",
        f"{run_dir.name}/metrics.csv",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"zip missing: {missing}")
    print(zip_path)


if __name__ == "__main__":
    main()
