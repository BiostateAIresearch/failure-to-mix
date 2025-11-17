
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, re
from scipy.stats import binom

MODEL_COLORS={
 "gemini":"#1f77b4","openai":"#2ca02c","gpt":"#2ca02c",
 "claude":"#8c564b","anthropic":"#8c564b",
 "kimi":"#e377c2","moonshotai":"#e377c2","qwen":"#ffbf00",
}
EDGE_COLOR="#999b9e"
DATA_DIR="data"; OUT_DIR="figs"
os.makedirs(OUT_DIR,exist_ok=True)

def normalize_cols(df):
    df=df.copy()
    df.columns=[c.strip().lower() for c in df.columns]
    return df

def safe_name(x):return re.sub(r'[^A-Za-z0-9._-]+','_',str(x))
def color_for(model):
    m=str(model).lower()
    for k,v in MODEL_COLORS.items():
        if k in m:return v
    return "#555555"

def style(ax):
    ax.tick_params(axis="both",labelsize=12,color=EDGE_COLOR,labelcolor=EDGE_COLOR,
                   width=0.9,length=4,direction="out")
    for s in ax.spines.values():
        s.set_color(EDGE_COLOR); s.set_linewidth(0.9)

print("=== plot_from_inputs v2.6 ===")

# F2
f2=pd.read_csv(f"{DATA_DIR}/2C1.csv")
f2=normalize_cols(f2)
for m,sub in f2.groupby("model"):
    col=color_for(m)
    p=sub["p"].to_numpy(float)
    r_mean=sub["r_mean"].to_numpy(float)
    r1=sub["r1"].to_numpy(float); r2=sub["r2"].to_numpy(float)
    o=np.argsort(p); p,r_mean,r1,r2=p[o],r_mean[o],r1[o],r2[o]
    if p[0]>0: p=np.insert(p,0,0); r_mean=np.insert(r_mean,0,r_mean[0]); r1=np.insert(r1,0,r1[0]); r2=np.insert(r2,0,r2[0])
    if p[-1]<100: p=np.append(p,100); r_mean=np.append(r_mean,r_mean[-1]); r1=np.append(r1,r1[-1]); r2=np.append(r2,r2[-1])
    pp=p/100; rr=r_mean/100; dp=np.diff(pp)
    S=4*np.sum(dp*(np.abs(rr[:-1]-pp[:-1])+np.abs(rr[1:]-pp[1:]))/2)
    plt.figure(figsize=(7,4)); ax=plt.gca()
    ax.plot(p,r_mean,color=col,lw=2.5,marker='o',markerfacecolor=col,markeredgecolor="black")
    ax.plot(p,r1,color=col,lw=1.2,ls="--",alpha=0.7)
    ax.plot(p,r2,color=col,lw=1.2,ls=":",alpha=0.7)
    ax.plot(p,p,color=EDGE_COLOR,lw=1.2,ls="--")
    ax.fill_between(p,r_mean,p,alpha=0.20,color=col)
    for xv in [25,50,75]: ax.axvline(x=xv,color=EDGE_COLOR,lw=0.8,alpha=0.4)
    style(ax)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_xlabel("p (%)",fontsize=14,fontweight="bold",color=EDGE_COLOR)
    ax.set_ylabel("r (%)",fontsize=14,fontweight="bold",color=EDGE_COLOR)
    ax.text(0.02,0.95,f"{m}\nS={S:.3f}",transform=ax.transAxes,color=EDGE_COLOR)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/F2_{safe_name(m)}.png",dpi=300,bbox_inches="tight")
    plt.close()

print("F2 done.")
