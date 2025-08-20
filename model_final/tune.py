import argparse, itertools, os, json, subprocess, uuid, sys, random
from pathlib import Path

def run_one(csv, outdir, window, hidden, layers, batch, epochs, lr, seed=42, bidir=False):
    run_id = f"w{window}_h{hidden}_L{layers}_b{batch}_e{epochs}_lr{lr}_s{seed}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train_lstm_daily_seq.py",
        "--csv", str(csv),
        "--outdir", str(run_dir),
        "--window", str(window),
        "--hidden", str(hidden),
        "--layers", str(layers),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed),
    ]
    if bidir:
        cmd.append("--bidir")

    print(f"\n[RUN] {run_id}\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    metrics_path = run_dir / "metrics_daily.json"
    ckpt_path = run_dir / "model_daily_nextday.pth"
    if not metrics_path.exists() or not ckpt_path.exists():
        print(f"[WARN] missing outputs: {metrics_path.exists()=}, {ckpt_path.exists()=}")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    val_mse_seq = hist.get("val_mse", [])
    val_mae_seq = hist.get("val_mae", [])
    best_val_mse = float(min(val_mse_seq)) if val_mse_seq else float("inf")
    best_val_mae = float(min(val_mae_seq)) if val_mae_seq else float("inf")

    return {
        "run_id": run_id,
        "path": str(run_dir),
        "window": window,
        "hidden": hidden,
        "layers": layers,
        "batch": batch,
        "epochs": epochs,
        "lr": lr,
        "bidir": bidir,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="학습/검증에 사용할 CSV")
    ap.add_argument("--outdir", required=True, help="실험 결과 저장 폴더")
    ap.add_argument("--trials", type=int, default=0,
                    help="0이면 전체 그리드, >0이면 랜덤 샘플링 횟수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["slide_exact", "grid"], default="grid",
                    help="slide_exact: 슬라이드 표의 2개 구성만 재현 / grid: 전체 탐색")
    args = ap.parse_args()

    random.seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.mode == "slide_exact":
        # 슬라이드 표에 나온 2개 구성만
        candidates = [
            # baseline: 45d, hidden64, layer1, lr=0.001, batch64
            (45, 64, 1, 64, 12, False, 0.001),
            # tuned:    45d, hidden128, layer2, lr=0.005, batch32
            (45, 128, 2, 32, 12, False, 0.005),
        ]
    else:

        windows     = [45]                     
        hiddens     = [64, 128]
        layers_list = [1, 2]
        batches     = [32, 64]                 
        epochs_list = [12, 15]                
        bidir_list  = [False]                  
        lrs         = [0.001, 0.005]           

        grid = list(itertools.product(
            windows, hiddens, layers_list, batches, epochs_list, bidir_list, lrs
        ))

        if args.trials and args.trials > 0:
            candidates = random.sample(grid, k=min(args.trials, len(grid)))
        else:
            candidates = grid

    results = []
    for (W, H, L, B, E, BD, LR) in candidates:
        try:
            res = run_one(args.csv, args.outdir, W, H, L, B, E, LR, seed=args.seed, bidir=BD)
            if res: results.append(res)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] run failed: {e}")
        except Exception as e:
            print(f"[WARN] {type(e).__name__}: {e}")


    summary_path = Path(args.outdir) / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] saved: {summary_path}")


    if results:
        def key_fn(r):
            return (r["best_val_mse"], r["best_val_mae"])
        best = min(results, key=key_fn)
        print("\n[BEST CONFIG]")
        for k in ["run_id","window","hidden","layers","batch","epochs","lr","bidir","best_val_mse","best_val_mae","path"]:
            print(f"- {k}: {best[k]}")
    else:
        print("[ERROR] no successful runs")

if __name__ == "__main__":
    main()
