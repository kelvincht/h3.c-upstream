#!/usr/bin/env python3
"""Build the FL2VA/Ref2VA hybrid transformer: FL2VA weights everywhere except
blocks.N.adaln_proj.linear.* for N >= --from-block, which come from Ref2VA.

The two released checkpoints share layout and differ mostly in the per-block
AdaLN projections, which carry the reference pathway; FL2VA has the better raw
quality. The output clones the FL2VA shards (copy-on-write) and overwrites only
the selected tensors' bytes, so it loads as a Ref2VA transformer in h3.

Usage:
  python3 tools/hybrid_h3.py --fl2va MiniMax-H3/FL2VA/transformer \
      --ref2va MiniMax-H3/Ref2VA/transformer --from-block 25 --out hybrid/transformer
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

from fold_turbo_lora import clone, read_header


def tensor_bytes(path, info, base):
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        return f.read(end - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fl2va", required=True, type=Path)
    ap.add_argument("--ref2va", required=True, type=Path)
    ap.add_argument("--from-block", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pattern = re.compile(r"blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$")
    ref = {}
    for shard in sorted(args.ref2va.glob("model-*.safetensors")):
        hlen, hdr = read_header(shard)
        for key, info in hdr.items():
            m = pattern.match(key)
            if m and int(m.group(1)) >= args.from_block:
                ref[key] = (shard, info, 8 + hlen)

    args.out.mkdir(parents=True, exist_ok=True)
    for extra in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(args.fl2va / extra, args.out / extra)
    swapped = 0
    for shard in sorted(args.fl2va.glob("model-*.safetensors")):
        dst = args.out / shard.name
        clone(shard, dst)
        hlen, hdr = read_header(dst)
        todo = [k for k in hdr if k in ref]
        if not todo:
            continue
        with open(dst, "r+b") as f:
            for key in todo:
                info = hdr[key]
                src_shard, src_info, src_base = ref[key]
                if src_info["shape"] != info["shape"] or src_info["dtype"] != info["dtype"]:
                    raise SystemExit(f"{key}: layout differs between checkpoints")
                f.seek(8 + hlen + info["data_offsets"][0])
                f.write(tensor_bytes(src_shard, src_info, src_base))
                swapped += 1
    if swapped != len(ref):
        raise SystemExit(f"swapped {swapped} of {len(ref)} tensors")
    print(f"swapped {swapped} adaln tensors from block {args.from_block}")


if __name__ == "__main__":
    sys.exit(main())
