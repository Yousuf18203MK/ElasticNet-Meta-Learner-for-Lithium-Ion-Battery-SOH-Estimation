"""
generate_loo_figures.py — Publication figures for the LOO experiment.
Run from project root: python visualization/generate_loo_figures.py
"""
import sys, os, warnings, pickle
from pathlib import Path
_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as GS
from config import RESULTS_DIR, FIGURES_DIR, BATTERIES
from src.feature_engineering import extract_rich_features

CELLS = list(BATTERIES.keys()); SEEDS = [42, 123, 2024]
PAL   = {"B0005":"#2166ac","B0006":"#d6604d","B0007":"#1a9850","B0018":"#7b2d8b"}
MCLR  = {"GRU":"#4d9de0","LSTM":"#3bb273","XGB":"#d01c8b","Weighted":"#8073ac",
          "Stack_LR":"#e08214","Stack_Ridge":"#35978f","Stack_EN":"#bf5b17"}
plt.rcParams.update({"font.family":"serif","font.size":10.5,"axes.titlesize":11.5,
    "axes.labelsize":10.5,"xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.alpha":0.22,"grid.linestyle":"--","lines.linewidth":1.7})

LOO_DIR  = RESULTS_DIR/"loo"; FIGS_LOO = FIGURES_DIR/"loo"; FIGS_LOO.mkdir(exist_ok=True)
def save(fig,nm): fig.savefig(str(FIGS_LOO/nm),dpi=300); plt.close(fig); print(f"  {nm}")

def main():
    print("="*60+"\n  Generating LOO figures …\n"+"="*60)
    req = LOO_DIR/"loo_results_raw.csv"
    if not req.exists(): raise FileNotFoundError(f"{req}\nRun experiments/run_loo_experiment.py first.")

    loo_full = pd.read_csv(str(LOO_DIR/"loo_results_raw.csv"))
    loo_smry = pd.read_csv(str(LOO_DIR/"loo_summary_per_battery.csv"))
    loo_ovrl = pd.read_csv(str(LOO_DIR/"loo_overall.csv"))
    with open(str(LOO_DIR/"loo_preds_s42.pkl"),"rb") as f: preds=pickle.load(f)
    within   = pd.read_csv(str(RESULTS_DIR/"final_results_raw.csv"))

    rich = {}
    for bid in CELLS:
        rp = RESULTS_DIR/f"{bid}_rich.csv"
        rich[bid] = pd.read_csv(str(rp)) if rp.exists() else extract_rich_features(RESULTS_DIR.parent/"data"/BATTERIES[bid],bid)

    MODS_SHOW = ["GRU","LSTM","XGB","Weighted","Stack_Ridge","Stack_EN"]
    NICE_S    = ["GRU","LSTM","XGB","Weighted","Stack\n(Ridge)","Stack\n(EN)"]

    # FIG L1: actual vs predicted
    fig,axes=plt.subplots(2,2,figsize=(11,8)); axes=axes.flatten()
    for ax,bid in zip(axes,CELLS):
        p=preds[bid]; cyc=p.Cycle.values; act=p.Actual_SOH.values; en=p.Stack_EN.values
        ax.plot(rich[bid].Cycle,rich[bid].SOH,"-",color="lightgrey",lw=1,zorder=1,label="Full SOH")
        ax.plot(cyc,act,"k-",lw=2.4,label="Actual (LOO test)",zorder=5)
        ax.plot(cyc,p.GRU.values,"--",color="#2166ac",lw=1.3,alpha=.7,label="GRU")
        ax.plot(cyc,p.LSTM.values,"--",color="#4dac26",lw=1.3,alpha=.7,label="LSTM")
        ax.plot(cyc,en,"-",color="#bf5b17",lw=2.2,label="EN Stack")
        ax.set_title(f"{bid} — LOO RMSE={float(np.sqrt(np.mean((act-en)**2))):.4f}%",fontweight="bold")
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel("SOH (%)")
    axes[0].legend(fontsize=7.5,loc="lower left",framealpha=.85,ncol=3)
    fig.suptitle("Fig. L1 — LOO: Actual vs Predicted SOH (seed=42)\nModel trained on 3 cells, evaluated on 4th",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL01_loo_actual_vs_predicted.png")

    # FIG L2: per-battery RMSE bars
    x=np.arange(len(MODS_SHOW)); w=0.19
    fig,ax=plt.subplots(figsize=(12,5.5))
    for i,(bid,clr) in enumerate(PAL.items()):
        vals=[float(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model==m)].RMSE_mean.values[0])
              if len(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model==m)])>0 else 0 for m in MODS_SHOW]
        errs=[float(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model==m)].RMSE_std.values[0])
              if len(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model==m)])>0 else 0 for m in MODS_SHOW]
        ax.bar(x+i*w,vals,w,label=bid,color=clr,alpha=.85,yerr=errs,capsize=3,error_kw={"elinewidth":1.2})
    ax.set_xticks(x+1.5*w); ax.set_xticklabels(NICE_S,fontsize=10)
    ax.set_ylabel("RMSE (SOH %)"); ax.set_title("Fig. L2 — LOO RMSE per Held-Out Battery (mean±std, 3 seeds)",fontweight="bold")
    ax.legend(ncol=4,fontsize=9,loc="upper right",framealpha=.85)
    plt.tight_layout(); save(fig,"figL02_loo_rmse_per_battery.png")

    # FIG L3: within vs LOO comparison
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    wb_data={m:within[f"{m}_RMSE"].mean() for m in MODS_SHOW if f"{m}_RMSE" in within.columns}
    loo_data={m:float(loo_ovrl[loo_ovrl.Model==m].RMSE_mean.values[0]) for m in MODS_SHOW if len(loo_ovrl[loo_ovrl.Model==m])>0}
    x2=np.arange(len(MODS_SHOW)); w2=0.35
    axes[0].bar(x2-w2/2,[wb_data.get(m,0) for m in MODS_SHOW],w2,label="Within-Battery",color=[MCLR.get(m,"grey") for m in MODS_SHOW],alpha=.85)
    axes[0].bar(x2+w2/2,[loo_data.get(m,0) for m in MODS_SHOW],w2,label="LOO",color=[MCLR.get(m,"grey") for m in MODS_SHOW],alpha=.40,hatch="////",edgecolor="grey",lw=.5)
    axes[0].set_xticks(x2); axes[0].set_xticklabels(NICE_S,fontsize=9.5)
    axes[0].set_ylabel("Mean RMSE (SOH %)"); axes[0].set_title("Within-Battery vs LOO\n(mean, all cells, 3 seeds)",fontweight="bold"); axes[0].legend(fontsize=9.5)
    wb_per=within.groupby("Battery")["Stack_EN_RMSE"].mean()
    loo_per={bid:float(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model=="Stack_EN")].RMSE_mean.values[0])
             if len(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model=="Stack_EN")])>0 else 0 for bid in CELLS}
    x3=np.arange(4); w3=0.35
    axes[1].bar(x3-w3/2,[wb_per[b] for b in CELLS],w3,color=[PAL[b] for b in CELLS],alpha=.85,label="Within-Battery")
    axes[1].bar(x3+w3/2,[loo_per[b] for b in CELLS],w3,color=[PAL[b] for b in CELLS],alpha=.40,hatch="////",edgecolor="grey",lw=.5,label="LOO")
    axes[1].set_xticks(x3); axes[1].set_xticklabels(CELLS); axes[1].set_ylabel("RMSE (SOH %)")
    axes[1].set_title("Stack EN: Within-Battery vs LOO\n(mean, 3 seeds)",fontweight="bold"); axes[1].legend(fontsize=9.5)
    fig.suptitle("Fig. L3 — Within-Battery vs Leave-One-Out Performance Comparison",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL03_within_vs_loo.png")

    # FIG L4: seed stability
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    for ax,bid in zip(axes,CELLS):
        sub=loo_full[loo_full.Test_Battery==bid]; sv=sub.Stack_EN_RMSE.values
        ax.bar([str(s) for s in SEEDS],sv,color=PAL[bid],alpha=.85,edgecolor="white")
        ax.axhline(y=float(sv.mean()),color="k",lw=1.5,ls="--",label=f"μ={sv.mean():.3f}")
        ax.set_title(f"{bid}",fontweight="bold"); ax.set_xlabel("Seed"); ax.set_ylabel("EN RMSE" if bid==CELLS[0] else ""); ax.legend(fontsize=8.5)
    fig.suptitle("Fig. L4 — LOO Seed Stability: GRU+LSTM+XGB EN Stack",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL04_loo_seed_stability.png")

    # FIG L5: residual histograms
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    for ax,bid in zip(axes,CELLS):
        p=preds[bid]; res=p.Stack_EN.values-p.Actual_SOH.values
        ax.hist(res,bins=10,color=PAL[bid],alpha=.82,edgecolor="white")
        ax.axvline(x=0,color="k",lw=1.5,ls="--"); ax.axvline(x=float(res.mean()),color="red",lw=1.5,label=f"μ={res.mean():.3f}")
        ax.set_title(f"{bid}",fontweight="bold"); ax.set_xlabel("Residual (SOH %)")
        if bid==CELLS[0]: ax.set_ylabel("Count"); ax.legend(fontsize=8)
    fig.suptitle("Fig. L5 — LOO Residual Distributions (seed=42)",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL05_loo_residual_hist.png")

    # FIG L6: B0006 diagnostic
    fig=plt.figure(figsize=(12,8)); gs=GS.GridSpec(2,2,figure=fig,hspace=.4,wspace=.35)
    p6=preds["B0006"]; df6=rich["B0006"]
    ax0=fig.add_subplot(gs[0,:])
    ax0.plot(df6.Cycle,df6.SOH,"#d6604d",lw=2.0,alpha=.4,label="Full SOH")
    ax0.plot(p6.Cycle,p6.Actual_SOH,"#d6604d",lw=2.4,label="Actual (LOO)")
    ax0.plot(p6.Cycle,p6.Stack_EN,"#bf5b17",lw=2.2,label=f"EN Stack (RMSE={float(np.sqrt(np.mean((p6.Actual_SOH-p6.Stack_EN)**2))):.4f})")
    ax0.plot(p6.Cycle,p6.GRU,"--",color="#2166ac",lw=1.3,alpha=.7,label="GRU"); ax0.plot(p6.Cycle,p6.LSTM,"--",color="#4dac26",lw=1.3,alpha=.7,label="LSTM")
    ax0.set_xlabel("Discharge cycle"); ax0.set_ylabel("SOH (%)"); ax0.set_title("B0006 LOO (seed=42) — Training: B0005+B0007+B0018",fontweight="bold"); ax0.legend(fontsize=8,ncol=3)
    ax1=fig.add_subplot(gs[1,0]); sub6=loo_full[loo_full.Test_Battery=="B0006"]
    ax1.bar([str(s) for s in SEEDS],sub6.Stack_EN_RMSE.values,color="#d6604d",alpha=.85,edgecolor="white")
    ax1.axhline(y=sub6.Stack_EN_RMSE.mean(),color="k",lw=1.5,ls="--",label=f"μ={sub6.Stack_EN_RMSE.mean():.3f}"); ax1.set_xlabel("Seed"); ax1.set_ylabel("RMSE"); ax1.set_title("B0006 LOO Seed Stability",fontweight="bold"); ax1.legend(fontsize=9)
    ax2=fig.add_subplot(gs[1,1]); res6=p6.Stack_EN.values-p6.Actual_SOH.values
    ax2.scatter(p6.Actual_SOH,res6,color="#d6604d",s=35,alpha=.8); ax2.axhline(y=0,color="k",lw=1.2,ls="--")
    ax2.set_xlabel("Actual SOH (%)"); ax2.set_ylabel("Residual (SOH %)"); ax2.set_title("B0006 Residual vs Actual SOH",fontweight="bold")
    fig.suptitle("Fig. L6 — B0006 LOO Diagnostic (most challenging cell for cross-battery generalization)",fontsize=11,fontweight="bold"); save(fig,"figL06_b0006_loo_diagnostic.png")

    # FIG L7: B0018 diagnostic
    fig,axes=plt.subplots(1,2,figsize=(11,4.5)); p18=preds["B0018"]; df18=rich["B0018"]
    axes[0].plot(df18.Cycle,df18.SOH,"#7b2d8b",lw=2.0,alpha=.4,label="Full SOH")
    axes[0].plot(p18.Cycle,p18.Actual_SOH,"#7b2d8b",lw=2.4,label="Actual (LOO)")
    axes[0].plot(p18.Cycle,p18.Stack_EN,"#bf5b17",lw=2.2,label=f"EN Stack (RMSE={float(np.sqrt(np.mean((p18.Actual_SOH-p18.Stack_EN)**2))):.4f})")
    axes[0].plot(p18.Cycle,p18.GRU,"--",color="#2166ac",lw=1.3,alpha=.7,label="GRU")
    axes[0].set_xlabel("Discharge cycle"); axes[0].set_ylabel("SOH (%)"); axes[0].set_title("B0018 LOO (seed=42)\nTraining: B0005+B0006+B0007",fontweight="bold"); axes[0].legend(fontsize=8.5)
    sub18=loo_full[loo_full.Test_Battery=="B0018"]
    axes[1].bar([str(s) for s in SEEDS],sub18.Stack_EN_RMSE.values,color="#7b2d8b",alpha=.85,edgecolor="white")
    axes[1].axhline(y=sub18.Stack_EN_RMSE.mean(),color="k",lw=1.5,ls="--",label=f"μ={sub18.Stack_EN_RMSE.mean():.3f}")
    axes[1].set_xlabel("Seed"); axes[1].set_ylabel("EN Stack RMSE (LOO)"); axes[1].set_title("B0018 LOO Seed Stability",fontweight="bold"); axes[1].legend(fontsize=9)
    fig.suptitle("Fig. L7 — B0018 LOO Diagnostic (132 cycles; trained on 3 longer-life cells)",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL07_b0018_loo_diagnostic.png")

    # FIG L8: R² comparison
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    r2_wb={m:within[f"{m}_R2"].mean() for m in MODS_SHOW if f"{m}_R2" in within.columns}
    r2_loo={m:float(loo_ovrl[loo_ovrl.Model==m].R2_mean.values[0]) for m in MODS_SHOW if len(loo_ovrl[loo_ovrl.Model==m])>0}
    x2=np.arange(len(MODS_SHOW)); w2=0.35
    axes[0].bar(x2-w2/2,[r2_wb.get(m,0) for m in MODS_SHOW],w2,label="Within-Battery",color=[MCLR.get(m,"grey") for m in MODS_SHOW],alpha=.85)
    axes[0].bar(x2+w2/2,[r2_loo.get(m,0) for m in MODS_SHOW],w2,label="LOO",color=[MCLR.get(m,"grey") for m in MODS_SHOW],alpha=.40,hatch="////",edgecolor="grey",lw=.5)
    axes[0].set_xticks(x2); axes[0].set_xticklabels(NICE_S,fontsize=9.5); axes[0].set_ylabel("R²"); axes[0].set_title("R² Comparison: Within vs LOO",fontweight="bold"); axes[0].legend(fontsize=9.5)
    r2_loo_per={bid:float(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model=="Stack_EN")].R2_mean.values[0])
                if len(loo_smry[(loo_smry.Test_Battery==bid)&(loo_smry.Model=="Stack_EN")])>0 else 0 for bid in CELLS}
    axes[1].bar(CELLS,[r2_loo_per[b] for b in CELLS],color=[PAL[b] for b in CELLS],alpha=.85)
    axes[1].axhline(y=0,color="k",lw=1.2,ls="--")
    for i,(bid,v) in enumerate(r2_loo_per.items()): axes[1].text(i,v+.01,f"{v:.3f}",ha="center",va="bottom",fontsize=9)
    axes[1].set_ylabel("R² (LOO)"); axes[1].set_title("Stack EN R² — Per Held-Out Battery (LOO mean)",fontweight="bold")
    fig.suptitle("Fig. L8 — R² Comparison: Within-Battery vs LOO (GRU+LSTM+XGB models)",fontsize=11,fontweight="bold")
    plt.tight_layout(); save(fig,"figL08_r2_comparison.png")

    print(f"\n  All LOO figures saved to: {FIGS_LOO}")

if __name__ == "__main__":
    main()
