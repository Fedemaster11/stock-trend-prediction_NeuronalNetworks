#!/usr/bin/env python3
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to NPZ dataset file")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    for split in ["y_train", "y_val", "y_test"]:
        y = data[split]
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n[{split.upper()}] count = {len(y)}")
        for u, c in zip(unique, counts):
            print(f"  class {u}: {c} ({c/len(y):.2%})")

if __name__ == "__main__":
    main()
