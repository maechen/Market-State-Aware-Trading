from __future__ import annotations
import argparse, itertools, subprocess
from pathlib import Path

KS = [1.00, 1.25, 1.50]
HORIZONS = [6, 7, 8]

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument('--base-output', default='data/transformer_npy/event_grid_hscaled')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--max-folds', type=int, default=None)
    p.add_argument('--device', default='auto')
    return p.parse_args()

def main() -> None:
    args=parse_args()
    base=Path(args.base_output); base.mkdir(parents=True, exist_ok=True)
    for k,h in itertools.product(KS,HORIZONS):
        out_dir=base/f"k{k:.2f}_h{h}"
        cmd=["python","src/models/transformer/run.py","--variant","gating","--output-dir",str(out_dir),"--dir-label-mode","vol_threshold","--dir-vol-k",str(k),"--dir-n-forward",str(h),"--window-size","20","--epochs",str(args.epochs),"--batch-size","64","--lr","1e-4","--weight-decay","1e-4","--device",args.device]
        if args.max_folds is not None:
            cmd += ["--max-folds", str(args.max_folds)]
        print("Running:"," ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

if __name__=='__main__':
    main()
