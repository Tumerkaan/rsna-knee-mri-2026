"""Rebuild publication figures from aggregate CSV only; no model or data access."""
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsna-publication-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)
INK, TEAL, BLUE, CORAL, GOLD = "#172c3b", "#007f79", "#3863ac", "#cf644c", "#aa7b18"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "text.color":INK,
    "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK,
    "axes.spines.top":False, "axes.spines.right":False, "axes.edgecolor":"#cad3d9",
    "figure.facecolor":"#fbfcfd", "axes.facecolor":"#fbfcfd", "savefig.facecolor":"#fbfcfd"})

def rows(name):
    with (ROOT / "results" / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(OUT / (name+".png"), dpi=180, bbox_inches="tight", metadata={"Software":"Matplotlib; aggregate tables"})
    plt.close(fig)

def fold_cv():
    data = rows("fold-cv.csv")
    fig, ax = plt.subplots(figsize=(10.5,5.2))
    styles = [("A",TEAL,"o"),("T",BLUE,"s"),("A + T",GOLD,"D"),
              ("F",CORAL,"^"),("A + T + F",INK,"P")]
    for family,color,marker in styles:
        subset = sorted([r for r in data if r["family"] == family],key=lambda r:int(r["fold"]))
        ax.plot([int(r["fold"]) for r in subset], [float(r["macro_auc"]) for r in subset],
                marker=marker,color=color,lw=2,ms=7,label=family)
    ax.set(xticks=range(5),xlabel="Held-out fold",ylabel="Macro AUC · 12 equally weighted targets",ylim=(.887,.925))
    ax.grid(axis="y",alpha=.2)
    ax.set_title("Complementarity is clearest on matched folds",loc="left",fontweight="bold",pad=22)
    ax.legend(ncol=3,loc="upper left",frameon=False,fontsize=9)
    fig.text(.12,-.015,"F and A + T + F: verified folds 0/1 only. Missing folds are not interpolated. E1–E2.",fontsize=9)
    save(fig,"fold-cv")

def heatmap():
    data = rows("target-family-auc.csv")
    targets = list(dict.fromkeys(r["target"] for r in data))
    families = ["A","T","F","A + T + F","Z"]
    values = np.array([[np.mean([float(r["auc"]) for r in data
          if r["target"]==t and r["family"]==f and int(r["fold"]) in (0,1)])
          for f in families] for t in targets])
    if not np.isfinite(values).all():
        raise ValueError("Missing aggregate heatmap cells")
    fig,ax=plt.subplots(figsize=(9,6.8))
    im=ax.imshow(values,cmap="YlGnBu",vmin=.75,vmax=1,aspect="auto")
    ax.set(xticks=range(5),xticklabels=["A","T","F","A + T + F","Z (excluded)"],
           yticks=range(len(targets)),yticklabels=targets)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j,i,f"{values[i,j]:.3f}",ha="center",va="center",color="white" if values[i,j]>.91 else INK,fontsize=9)
    ax.set_title("Target performance · common folds 0/1",loc="left",fontweight="bold",pad=20)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.04,label="Mean of fold-specific target AUCs")
    fig.text(.125,-.01,"Equal mean across folds, not pooled AUC. Z is an excluded research family. E1, E2, E4.",fontsize=9)
    save(fig,"target-heatmap")

def leaderboard():
    from textwrap import fill
    data = rows("lb-history.csv")
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, r in enumerate(data):
        final = bool(r["final_slot"])
        for column, marker, color, shift in [
            ("public_auc", "o", TEAL if final else BLUE, -.12),
            ("private_auc", "D", CORAL, .12),
        ]:
            if r[column]:
                value = float(r[column])
                ax.scatter(value, i + shift, marker=marker, color=color, s=65, zorder=3)
                ax.annotate(f"{value:.3f}", (value, i + shift),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9)
        if not r["public_auc"]:
            ax.text(.925, i, "ERROR — no score", color=CORAL, va="center", fontsize=9)
    labels = [r["date"][5:] + " · " + fill(r["model"], 32) for r in data]
    ax.set(yticks=range(len(data)), yticklabels=labels, xlim=(.924, .964),
           xlabel="Rounded leaderboard AUC")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.scatter([], [], marker="o", color=BLUE, label="Public (green = final selection)")
    ax.scatter([], [], marker="D", color=CORAL, label="Private (author supplied, 23 Oct)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Leaderboard history · public and private", loc="left",
                 fontweight="bold", pad=20)
    fig.text(.01, .01, "Dates are public observations, not runtime. Missing private scores remain missing.\n10-16 public labels denote the earlier Heliosli baseline; see ATTRIBUTION.md.",
             fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "lb-timeline")

def residual():
    data=rows("residual-spearman.csv")
    targets=list(dict.fromkeys(r["target"] for r in data))
    fig,ax=plt.subplots(figsize=(10.5,6))
    for pair,color,offset in [("F / A",CORAL,-.13),("G / A",TEAL,.13)]:
        v=[np.mean([float(r["spearman"]) for r in data if r["pair"]==pair and r["target"]==t]) for t in targets]
        ax.scatter(v,np.arange(len(targets))+offset,c=color,s=42,label=pair)
    ax.axvline(.91,color=INK,ls="--",lw=1,label="F diversity ceiling (mean criterion)")
    ax.set(yticks=range(len(targets)),yticklabels=targets,xlabel="Residual Spearman · mean across folds 0/1",xlim=(.65,1))
    ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    ax.legend(loc="lower left",frameon=False,fontsize=9)
    ax.set_title("A different backbone is not necessarily diverse",loc="left",fontweight="bold",pad=20)
    fig.text(.125,-.02,"F mean 0.92536: original diversity gate FAIL. G correlation is descriptive; its AUC gate also failed. E2–E3.",fontsize=9)
    save(fig,"residual-spearman")

def pipeline():
    fig,ax=plt.subplots(figsize=(12,5.2))
    ax.set(xlim=(0,12),ylim=(0,5));ax.axis("off")
    boxes=[(.15,2.1,2,1.6,"Authorized MRI\n+ CC0 consensus","Private inputs"),
           (2.65,2.1,2,1.6,"Geometry + mask\n96-slice cache","5-fold identity lock"),
           (5.15,2.1,2,1.6,"A · T · F\nfive folds / family","Equal family mean"),
           (7.65,2.1,2,1.6,"Bounded inference\nfinite + row checks","Own models only"),
           (9.85,.2,2,1.6,"public ensemble\n(goodpjw2008)\n+ 10% own ranks","Final blend")]
    for x,y,w,h,text,small in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04,rounding_size=.12",
                                  facecolor="#e9f2f1",edgecolor=TEAL,lw=1.2))
        ax.text(x+w/2,y+h*.63,text,ha="center",va="center",fontsize=10,fontweight="bold")
        ax.text(x+w/2,y+.23,small,ha="center",fontsize=8,color=TEAL)
    for x in (2.2,4.7,7.2):
        ax.add_patch(FancyArrowPatch((x,2.9),(x+.4,2.9),arrowstyle="-|>",mutation_scale=13,color=INK))
    ax.add_patch(FancyArrowPatch((8.7,2.05),(9.8,1.3),arrowstyle="-|>",mutation_scale=13,color=INK))
    ax.text(.1,4.45,"Two final submissions, one explicit provenance boundary",fontsize=16,fontweight="bold")
    ax.text(.1,.8,"Safety: original absolute deadline · preserve completed epochs\nF failure → A/T · base failure → frozen prevalence",fontsize=10)
    ax.text(.1,.15,"Conceptual diagram only. No competition image, individual record or model output is embedded.",fontsize=9,color="#526c7a")
    save(fig,"pipeline")

if __name__=="__main__":
    for fn in (fold_cv,heatmap,leaderboard,residual,pipeline):
        fn()
    print("5 aggregate figures generated")
