import os, glob, argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="per-user CSV들이 모여있는 폴더")
    ap.add_argument("--outdir", default="./eval_out", help="요약/플롯 저장 폴더")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = glob.glob(os.path.join(args.in_dir, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files in {args.in_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print("skip:", f, e)

    all_df = pd.concat(dfs, ignore_index=True)
    agg = (all_df
           .groupby("month", as_index=False)
           .agg({
              "pred_month_avg": "mean",
              "daily_mae": "mean",
              "monthly_mae": "mean",
              "coverage": "mean",
              "n_days": "max"
           })
           .sort_values("month"))

    agg.insert(0, "patient_id", "ALL")
    agg["note"] = ""
    out_csv = os.path.join(args.outdir, "eval_months_summary.csv")
    agg.to_csv(out_csv, index=False)
    print("saved:", out_csv)
    
    plot_df = agg.dropna(subset=["monthly_mae"]).copy()
    if not plot_df.empty:
        plot_df["ym_key"] = pd.to_datetime(plot_df["month"] + "-01")
        plot_df = plot_df.sort_values("ym_key")

        plt.figure(figsize=(8,5))
        plt.plot(plot_df["month"], plot_df["monthly_mae"], marker="o")
        plt.xticks(rotation=45)
        plt.ylabel("Monthly MAE")
        plt.title("Monthly MAE (ALL users)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "monthly_mae_trend.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(plot_df["month"], plot_df["daily_mae"], marker="o")
        plt.xticks(rotation=45)
        plt.ylabel("Daily MAE")
        plt.title("Daily MAE (ALL users)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "daily_mae_trend.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
