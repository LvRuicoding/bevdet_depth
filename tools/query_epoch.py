#!/usr/bin/env python3
"""查询训练日志中某个 epoch 的训练结果"""

import json
import sys
import argparse
from collections import defaultdict


def load_log(log_path):
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "mode" in obj:
                    entries.append(obj)
            except json.JSONDecodeError:
                continue
    return entries


def query_epoch(entries, epoch, mode="train"):
    rows = [e for e in entries if e.get("epoch") == epoch and e.get("mode") == mode]
    return rows


def avg_metrics(rows):
    sums = defaultdict(float)
    counts = defaultdict(int)
    skip = {"mode", "epoch", "iter", "lr", "memory", "time", "data_time"}
    for row in rows:
        for k, v in row.items():
            if k in skip:
                continue
            try:
                sums[k] += float(v)
                counts[k] += 1
            except (TypeError, ValueError):
                pass
    return {k: sums[k] / counts[k] for k in sums}


def print_epoch_summary(entries, epoch):
    train_rows = query_epoch(entries, epoch, "train")
    val_rows = query_epoch(entries, epoch, "val")

    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} 训练结果")
    print(f"{'='*60}")

    if not train_rows and not val_rows:
        print(f"  未找到 epoch {epoch} 的数据")
        available = sorted(set(e["epoch"] for e in entries if "epoch" in e))
        print(f"  可用 epoch: {available}")
        return

    if train_rows:
        print(f"\n[Train]  共 {len(train_rows)} 条记录")
        last = train_rows[-1]
        print(f"  最后 iter: {last.get('iter')}  lr: {last.get('lr')}")
        avgs = avg_metrics(train_rows)
        loss_keys = [k for k in avgs if "loss" in k]
        other_keys = [k for k in avgs if "loss" not in k]
        if loss_keys:
            print("\n  -- 损失 (epoch 平均) --")
            for k in sorted(loss_keys):
                print(f"    {k:<30s}: {avgs[k]:.5f}")
        if other_keys:
            print("\n  -- 其他指标 (epoch 平均) --")
            for k in sorted(other_keys):
                print(f"    {k:<30s}: {avgs[k]:.5f}")

    if val_rows:
        print(f"\n[Val]  共 {len(val_rows)} 条记录")
        for row in val_rows:
            skip = {"mode", "epoch", "iter", "time", "data_time"}
            for k, v in row.items():
                if k not in skip:
                    print(f"    {k:<30s}: {v}")
    elif train_rows:
        print("\n  (本 epoch 无验证结果)")

    print(f"{'='*60}\n")


def list_epochs(entries):
    epochs = sorted(set(e["epoch"] for e in entries if "epoch" in e))
    modes = {}
    for e in entries:
        ep = e.get("epoch")
        m = e.get("mode")
        if ep is not None:
            modes.setdefault(ep, set()).add(m)
    print(f"\n{'='*40}")
    print(f"  日志中包含的 epoch")
    print(f"{'='*40}")
    for ep in epochs:
        tag = "/".join(sorted(modes[ep]))
        print(f"  epoch {ep:>3d}  [{tag}]")
    print(f"{'='*40}\n")


def main():
    parser = argparse.ArgumentParser(description="查询训练日志中某个 epoch 的结果")
    parser.add_argument("epoch", nargs="?", type=int, help="要查询的 epoch 编号")
    parser.add_argument("-f", "--file",
                        default="/home/batchcom/lr/BEVDet/trainoutput/baseline/20260130_190618.log.json",
                        help="日志文件路径")
    parser.add_argument("-l", "--list", action="store_true", help="列出所有可用 epoch")
    args = parser.parse_args()

    entries = load_log(args.file)

    if args.list or args.epoch is None:
        list_epochs(entries)
        if args.epoch is None and not args.list:
            parser.print_help()
        return

    print_epoch_summary(entries, args.epoch)


if __name__ == "__main__":
    main()
