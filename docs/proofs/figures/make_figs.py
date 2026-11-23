import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":100})
INK="#222222"; GREY="#8c8c8c"; LIGHT="#dcdcdc"; BLUE="#2a6fb0"; PALEBLUE="#cfe0f1"; ORANGE="#d9731f"
import os; OUT=os.path.dirname(os.path.abspath(__file__))+"/"

def topf(x,f): return float(np.sort(x)[::-1][:f].sum())

# ---------------- Fig 1 ----------------
b=np.array([[3,1,4,2,1],[2,5,1,3,2],[1,2,2,6,3],[2,1,3,1,4]],float)
f=2; ell=b.max(0); Ri=[topf(r,f) for r in b]; R=topf(ell,f)
assert all(r<=R for r in Ri)
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1.15,1]})
nI,nS=b.shape
a1.set_xlim(-1.3,nS+2.1); a1.set_ylim(-1.8,nI+2.6); a1.invert_yaxis(); a1.axis("off")
for i in range(nI):
    for s in range(nS):
        ismax=b[i,s]==ell[s]
        a1.add_patch(Rectangle((s,i),1,1,fc=PALEBLUE if ismax else "white",ec=GREY,lw=.8))
        a1.text(s+.5,i+.5,f"{b[i,s]:.0f}",ha="center",va="center",color=INK,fontweight="bold" if ismax else "normal")
    a1.text(-.2,i+.5,f"row $i={i+1}$",ha="right",va="center")
    a1.add_patch(Rectangle((nS+.4,i),1.3,1,fc="white",ec=GREY,lw=.8))
    a1.text(nS+1.05,i+.5,f"{Ri[i]:.0f}",ha="center",va="center")
for s in range(nS):
    a1.text(s+.5,-.25,f"$s={s+1}$",ha="center",va="bottom")
    a1.add_patch(Rectangle((s,nI+.35),1,1,fc="white",ec=BLUE,lw=1.4))
    a1.text(s+.5,nI+.85,f"{ell[s]:.0f}",ha="center",va="center",color=BLUE,fontweight="bold")
a1.text(-.2,nI+.85,r"$\ell[s]$ = column max",ha="right",va="center",color=BLUE)
a1.text(nS+1.05,-.25,r"$R(i)$",ha="center",va="bottom")
a1.text(nS+1.05,nI+.85,f"$R={R:.0f}$",ha="center",va="center",color=BLUE,fontweight="bold")
a1.add_patch(Rectangle((-.08,1.92),nS+1.7+.12,1.16,fc="none",ec=ORANGE,lw=2))
a1.text(-.2,-1.05,r"(a) loads $b[i,s]$, $f=2$",ha="left",fontsize=12,fontweight="bold")
a1.text(-.2,nI+2.0,r"Shaded = the cell that sets $\ell[s]$. Every $R(i)$ is at most $R$.",ha="left",fontsize=10,color=INK)
# panel b
s=np.arange(1,nS+1); i3=2
a2.bar(s,ell,width=.78,fc="white",ec=BLUE,lw=1.6,label=r"$\ell[s]$ (worst-row load)")
cols=[ORANGE if k in np.argsort(b[i3])[::-1][:f] else GREY for k in range(nS)]
a2.bar(s,b[i3],width=.5,fc=cols,label=r"$b[3,s]$")
for k in range(nS):
    a2.text(s[k],ell[k]+.15,f"{ell[k]:.0f}",ha="center",color=BLUE,fontsize=10)
    a2.text(s[k],b[i3][k]/2,f"{b[i3][k]:.0f}",ha="center",va="center",color="white",fontsize=10,fontweight="bold")
a2.set_xticks(s); a2.set_xticklabels([f"$s={k}$" for k in s]); a2.set_ylim(0,8.6); a2.set_ylabel("load")
a2.set_title(r"(b) Row 3 sits below $\ell$ at every sender",loc="left",fontsize=12,fontweight="bold")
a2.text(0.5,8.35,r"$\mathrm{top}_2(b[3,\cdot])=6+3=9\ \ \leq\ \ \mathrm{top}_2(\ell)=6+5=11$",fontsize=11,va="top")
a2.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,.88),fontsize=10)
a2.text(5.5,7.2,"orange = row 3's two heaviest senders",ha="right",va="top",fontsize=9.5,color=ORANGE)
fig.tight_layout(); fig.savefig(OUT+"fig1_row_by_row.png",dpi=200); plt.close(fig)

# ---------------- Fig 2 ----------------
x=np.array([7,5,4,2,1.]); f=2; tau0=np.sort(x)[::-1][f-1]
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6))
s=np.arange(1,6)
a1.bar(s,np.minimum(x,tau0),width=.6,fc=LIGHT,ec=GREY)
a1.bar(s,np.maximum(x-tau0,0),bottom=np.minimum(x,tau0),width=.6,fc=BLUE,ec=BLUE)
a1.axhline(tau0,color=ORANGE,ls="--",lw=1.8)
a1.text(5.55,tau0+.15,r"$\tau=5$",color=ORANGE,ha="right",fontweight="bold")
a1.text(1,7.2,r"$r=2$",ha="center",color=BLUE,fontweight="bold")
for k in range(5): a1.text(s[k],x[k]-.35 if x[k]>2 else x[k]+.15,f"{x[k]:.0f}",ha="center",va="top" if x[k]>2 else "bottom",fontsize=10)
a1.set_xticks(s); a1.set_xticklabels([f"$s={k}$" for k in s]); a1.set_ylim(0,8.8); a1.set_ylabel(r"$x[s]$")
a1.set_title(r"(a) $\tau$ = the $f$-th largest entry, $r[s]=\max(x[s]-\tau,0)$",loc="left",fontsize=12,fontweight="bold")
a1.text(1.9,8.5,r"$f\tau+\sum_s r[s]=2\cdot5+2=12=7+5=\mathrm{top}_2(x)$",va="top",fontsize=11)
a1.text(5.6,6.3,"gray: bar up to $\\tau$    blue: excess $r[s]$",ha="right",fontsize=9.5)
t=np.linspace(0,8,801); g=f*t+np.maximum(x[None,:]-t[:,None],0).sum(1)
a2.plot(t,g,color=BLUE,lw=2.2)
a2.axhline(topf(x,f),color=GREY,ls=":",lw=1)
for tv,col in [(3,GREY),(5,ORANGE),(7,GREY)]:
    gv=f*tv+np.maximum(x-tv,0).sum()
    a2.plot([tv],[gv],"o",color=col,ms=8,mec="white",mew=1.5,zorder=5)
    a2.text(tv,gv+.7,f"{gv:.0f}",ha="center",color=INK if col==GREY else ORANGE,fontweight="bold")
a2.text(6.4,topf(x,f)-1.2,r"minimum = $\mathrm{top}_2(x)=12$"+"\n(any $\\tau\\in[4,5]$ attains it)",ha="center",va="top",color=ORANGE,fontsize=10.5)
a2.text(0.15,topf(x,f)-.35,"no choice of $\\tau$ goes below this line",fontsize=9.5,color=GREY,va="top")
a2.set_xlabel(r"threshold $\tau$"); a2.set_ylabel(r"$f\tau+\sum_s r[s]$")
a2.set_ylim(8,23); a2.set_xlim(0,8)
a2.set_title(r"(b) Cost vs. $\tau$: its minimum is the top-$f$ sum",loc="left",fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT+"fig2_topf_identity.png",dpi=200); plt.close(fig)

# ---------------- Fig 3 ----------------
n=6; f=2
def mk(kind):
    if kind=="sym":
        m=np.ones((n,n))
    elif kind=="two":
        rng=np.random.default_rng(5); m=rng.uniform(.1,.6,(n,n)); m[:,0]=1; m[:,1]=1
    else:
        rng=np.random.default_rng(int(kind)); m=rng.uniform(.1,1,(n,n))
    np.fill_diagonal(m,0); return m
def draw(m,title,axh,axb):
    ell=m.max(0); Rrow=np.array([topf(r,f) for r in m]); R=topf(ell,f); med=np.median(Rrow)
    axh.imshow(m,cmap="Blues",vmin=0,vmax=1.6,aspect="equal")
    for i in range(n):
        for s in range(n):
            if i==s: axh.text(s,i,"–",ha="center",va="center",color=GREY); continue
            axh.text(s,i,f"{m[i,s]:.1f}",ha="center",va="center",fontsize=8.5,color="white" if m[i,s]>.9 else INK)
            if m[i,s]==ell[s]: axh.add_patch(Rectangle((s-.5,i-.5),1,1,fill=False,ec=ORANGE,lw=2.2))
    axh.set_xticks(range(n)); axh.set_xticklabels([f"{k+1}" for k in range(n)]); axh.set_yticks(range(n)); axh.set_yticklabels([f"{k+1}" for k in range(n)])
    axh.set_xlabel("sender $s$"); axh.set_ylabel("row $i$")
    for sp in axh.spines.values(): sp.set_visible(False)
    axh.tick_params(length=0); axh.xaxis.set_label_position("top"); axh.xaxis.tick_top()
    axh.set_title(f"{title}: ratio $={R/med:.2f}$",loc="left",fontsize=11.5,fontweight="bold",pad=26)
    axb.barh(range(n),Rrow,height=.6,color=[BLUE if abs(r-R)<1e-9 else GREY for r in Rrow])
    axb.axvline(R,color=ORANGE,ls="--",lw=1.8); axb.text(R,-.9,f"$R={R:.2f}$",color=ORANGE,ha="center",fontsize=10)
    axb.set_ylim(n-.5,-.5); axb.set_yticks([]); axb.set_xlim(0,max(R*1.12,1)); axb.set_xlabel(r"row charge $R(i)$")
    axb.spines["left"].set_visible(False)
    return R/med, int((np.abs(Rrow-R)<1e-9).sum())
fig,ax=plt.subplots(3,2,figsize=(9.5,12.5),gridspec_kw={"width_ratios":[1.6,1],"hspace":.62})
res=[draw(mk("sym"),"(a) Symmetric city",ax[0,0],ax[0,1]),
     draw(mk("two"),"(b) Two broadcasters",ax[1,0],ax[1,1]),
     draw(mk("7"),"(c) Random city",ax[2,0],ax[2,1])]
print(res)
fig.text(.5,.008,"Orange box: the row that sets $\\ell[s]$ for that sender (a tie boxes every cell).\nBlue bar: a row whose charge reaches $R$. The ratio is 1 only if more than half the rows are blue.",ha="center",fontsize=9.5)
fig.subplots_adjust(left=.08,right=.97,top=.915,bottom=.095); fig.savefig(OUT+"fig3_ratio_equals_one.png",dpi=200); plt.close(fig)

# ---------------- Fig 4 ----------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1,1.15]})
z=np.linspace(0,1,50); a_,w=.5,1.
a1.plot(z,a_*w*z,color=BLUE,lw=2.2); a1.plot(z,((1+a_)-z)*w,color=ORANGE,lw=2.2)
a1.plot([0],[0],"o",color=BLUE,ms=8); a1.plot([0],[1.5],"o",color=ORANGE,ms=8)
a1.text(.98,.32,r"design-proportional attack: $a\,z\,w$",color=BLUE,fontsize=10.5,ha="right")
a1.text(.03,1.62,r"attacker switches link on: $((1+a)-z)\,w$",color=ORANGE,fontsize=10.5)
a1.text(.98,1.78,"drawn with $a=0.5$, $w=1$",ha="right",fontsize=9,color=GREY)
a1.text(.04,.27,"link off: no attack",fontsize=10,color=INK)
a1.text(.03,1.02,"link off, yet the attack\nstill costs $1.5\\,w$",fontsize=10,color=INK,va="top")
a1.set_xlabel(r"design's link intensity $z[e]$"); a1.set_ylabel("extra cost the attack adds")
a1.set_ylim(-.05,1.85); a1.set_xlim(-.02,1.02)
a1.set_title("(a) Switching a link off stops being a defence",loc="left",fontsize=12,fontweight="bold")
a2.set_xlim(.85,1.25); a2.set_ylim(-1,1.6); a2.axis("off")
a2.annotate("",xy=(1.25,0),xytext=(.85,0),arrowprops=dict(arrowstyle="->",color=INK,lw=1.2))
a2.text(1.25,-.18,r"spectral radius $\rho(K)$ under the attack",ha="right",va="top",fontsize=10.5)
a2.add_patch(Rectangle((.85,.02),.15,.45,fc=PALEBLUE,ec="none")); a2.text(.925,.25,"certifiable\n$\\rho<1$",ha="center",va="center",color=BLUE,fontsize=10.5)
a2.add_patch(Rectangle((1.083,.02),.167,.45,fc="#f6dcc5",ec="none")); a2.text(1.166,.25,"every other\ndesign",ha="center",va="center",color=ORANGE,fontsize=10.5)
a2.plot([1,1],[-.05,.6],color=INK,lw=1.5); a2.text(1,.68,r"$\rho=1$",ha="center")
a2.plot([1.083],[0],"o",color=ORANGE,ms=9,mec="white",mew=1.5,zorder=5)
a2.text(1.083,-.5,r"safest design: $\rho=1.083$"+"\n(backup everywhere, no links)",ha="center",va="top",fontsize=10)
a2.text(.86,1.3,"Same attack, so every other design's matrix is entrywise at least as large;\nby Perron–Frobenius monotonicity its $\\rho$ is at least 1.083.",fontsize=10,va="top")
a2.set_title("(b) No design can be certified",loc="left",fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT+"fig4_why_no_design_survives.png",dpi=200); plt.close(fig)
