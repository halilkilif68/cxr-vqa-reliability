"""Figures 3-5 of the paper (confusion matrices, black-image probabilities, per-disease AUC).

Input : results/{blip,vilt,clip}_test_predictions.csv.gz and results/per_disease_merged.csv
        (run analysis/paper_tables.py first)
Output: figures/fig3_confusion.png, fig4_black_image.png, fig5_per_disease_auc.png
Usage : python analysis/make_figures.py [results_dir] [output_dir]
"""
import sys, os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
R=(sys.argv[1] if len(sys.argv)>1 else "results").rstrip("/")+"/"
O=(sys.argv[2] if len(sys.argv)>2 else "figures").rstrip("/")+"/"
os.makedirs(O,exist_ok=True)
INK,INK2,GRID="#0b0b0b","#52514e","#e4e3df"
C={"BLIP":"#2a78d6","ViLT":"#eb6834","CLIP":"#1baf7a"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2})
names={"blip":"BLIP","vilt":"ViLT","clip":"CLIP"}
P={k:pd.read_csv(R+f"{k}_test_predictions.csv.gz") for k in names}
seq=LinearSegmentedColormap.from_list("s",["#f3f7fc","#a9c8ee","#2a78d6","#12396b"])

# ---- Fig 3: confusion matrices, threshold 0.5
fig,axs=plt.subplots(1,3,figsize=(9.6,3.9))
for ax,(k,n) in zip(axs,names.items()):
    p=P[k]; y=p.label.values.astype(int); pr=(p.prob_orig.values>=0.5).astype(int)
    m=np.array([[((y==0)&(pr==0)).sum(),((y==0)&(pr==1)).sum()],[((y==1)&(pr==0)).sum(),((y==1)&(pr==1)).sum()]])
    rate=m/m.sum(1,keepdims=True)
    ax.imshow(rate,cmap=seq,vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            col="white" if rate[i,j]>0.55 else INK
            ax.text(j,i-0.1,f"{m[i,j]:,}",ha="center",va="center",color=col,fontsize=10,fontweight="bold")
            ax.text(j,i+0.18,f"{rate[i,j]*100:.1f}% of row",ha="center",va="center",color=col,fontsize=8)
    ax.set_xticks([0,1],["No","Yes"]); ax.set_yticks([0,1],["No","Yes"])
    ax.set_xlabel("Predicted (threshold 0.5)"); ax.set_title(n,fontweight="bold",loc="left")
    if k=="blip": ax.set_ylabel("True label")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("Confusion matrices on the test grid (125,713 question–image pairs; 8.8% positive)",x=0.01,ha="left",fontsize=10,color=INK2)
fig.tight_layout(rect=(0,0.02,1,0.93)); fig.savefig(O+"fig3_confusion.png",dpi=300); plt.close(fig)

# ---- Fig 4: black-image probability per disease (constant input -> one value per disease)
B=pd.DataFrame({n:P[k].groupby("disease").prob_black.first() for k,n in names.items()})
B=B.sort_values("BLIP")
fig,ax=plt.subplots(figsize=(6.6,7.6))
yy=np.arange(len(B))
for yv in yy: ax.axhline(yv,color=GRID,lw=0.6,zorder=0)
ax.axvline(0.088,color=INK2,ls="--",lw=1,zorder=1,label="Positive rate of the test grid (8.8%)")
for n,c in C.items(): ax.scatter(B[n],yy,s=34,color=c,edgecolor="white",linewidth=0.8,label=n,zorder=3)
ax.set_yticks(yy,B.index); ax.set_xlim(0,1); ax.set_ylim(-0.7,len(B)+0.2)
ax.set_xlabel("Predicted probability for an all-black image"); 
h,l=ax.get_legend_handles_labels(); o=[l.index(k) for k in ["BLIP","ViLT","CLIP"]]+[i for i,k in enumerate(l) if k.startswith("Positive")]
ax.legend([h[i] for i in o],[l[i] for i in o],frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.07),ncol=2,handletextpad=0.3,columnspacing=1.4,fontsize=8.5)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.set_title("Output for an all-black image, by disease\n(constant input: one value per disease and model)",loc="left",fontsize=9,color=INK2)
fig.tight_layout(); fig.savefig(O+"fig4_black_image.png",dpi=300); plt.close(fig)

# ---- Fig 5: per-disease AUC heatmap
T=pd.read_csv(R+"per_disease_merged.csv")
T=T.sort_values(["reliable","n_pos"],ascending=[False,False]).reset_index(drop=True)
M=T[["blip","vilt","clip"]].values; nr=int(T.reliable.sum())
fig,ax=plt.subplots(figsize=(5.6,8.4))
ax.imshow(M,cmap=seq,vmin=0.70,vmax=1.0,aspect="auto")
for i in range(len(T)):
    for j in range(3):
        v=M[i,j]; rel=T.reliable[i]
        col="white" if (v-0.7)/0.3>0.55 else INK
        ax.text(j,i,f"{v:.3f}",ha="center",va="center",fontsize=8,color=col if rel else "#8a8985",fontstyle="normal" if rel else "italic")
        if not rel: ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=True,fc="#ffffff",alpha=0.65,lw=0,zorder=2))
ax.axhline(nr-0.5,color=INK,lw=1.2,zorder=3)
ax.set_xticks([0,1,2],["BLIP","ViLT","CLIP"]); ax.xaxis.tick_top()
ax.set_yticks(range(len(T)),[f"{d}  (n={int(n)})" for d,n in zip(T.disease,T.n_pos)])
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
for t,rel in zip(ax.get_yticklabels(),T.reliable): t.set_color(INK2 if rel else "#8a8985")
ax.text(1,len(T)+0.1,"Below the line: <30 positives; AUC unstable",ha="center",va="top",fontsize=7.5,color=INK2)
fig.tight_layout(); fig.savefig(O+"fig5_per_disease_auc.png",dpi=300); plt.close(fig)
print("ok")
