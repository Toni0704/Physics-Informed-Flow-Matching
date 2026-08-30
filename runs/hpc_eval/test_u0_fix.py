"""Standalone A/B: cond_pcfm with u0=current-state (as shipped) vs u0=fixed noise seed.
Touches no repo code; replicates evaluate.py:pcfm_sample_with_physics_one exactly except for u0."""
import importlib.util, sys, torch, numpy as np
sys.path.insert(0, "/DATA/Sawan_projects/work/repos/Physics-Informed-Flow-Matching")
spec = importlib.util.spec_from_file_location(
    "ev", "/DATA/Sawan_projects/work/repos/Physics-Informed-Flow-Matching/NS_2D/experiments/evaluate.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from pcfm.pcfm_sampling import pcfm_2d_batched

REPO="/DATA/Sawan_projects/work/repos/Physics-Informed-Flow-Matching"
dev=torch.device("cuda")
DATA=f"{REPO}/ns_data/ns_nw10_nf100_s64_t50_mu0.001.h5"
N=2; NSTEP=200

w_gt,a_gt,f_gt,n_t,H,W = ev.load_ns_test_batch(DATA,N,dev)
model,w_scale = ev.load_conditioned_checkpoint(f"{REPO}/NS_2D/best_fm_conditioned.pt", n_t, dev)
cond_a=(a_gt/w_scale).unsqueeze(1); cond_f=f_gt.unsqueeze(1)
xg=torch.linspace(0,1.,H,device=dev); yg=torch.linspace(0,1.,W,device=dev); tg=torch.linspace(0,1.,n_t,device=dev)

def run(ca, cf, residual, fixed_u0, seed):
    torch.manual_seed(seed)
    xt = torch.randn(1, n_t, H, W, device=dev)
    u0_fixed = xt.permute(0,2,3,1).contiguous().clone()   # the ORIGINAL noise seed
    dt = 1.0/NSTEP
    def hfunc(u):
        return residual.full_residual_ns((u*w_scale).to(torch.float64)).to(torch.float32)
    for i in range(NSTEP):
        tv=i/NSTEP; t=torch.full((1,),tv,device=dev)
        with torch.no_grad(): vf=model(xt,t,ca,cf)
        ut=xt.permute(0,2,3,1).contiguous(); vfh=vf.permute(0,2,3,1).contiguous()
        with torch.enable_grad():
            pv=pcfm_2d_batched(ut=ut, vf=vfh, t=torch.tensor(tv,device=dev),
                               u0=(u0_fixed if fixed_u0 else ut), dt=dt,
                               hfunc=hfunc, mode="least_squares", newtonsteps=1,
                               guided_interpolation=False)
        xt = xt + pv.permute(0,3,1,2)*dt
    return xt[0]*w_scale

for label, fixed in [("SHIPPED  (u0 = current state)", False), ("FIXED    (u0 = noise seed)", True)]:
    preds=[]
    for i in range(N):
        res=ev.Residuals2D(data=w_gt[i].permute(1,2,0).unsqueeze(0),x=xg,y=yg,t_grid=tg,nx=H,ny=W,nt=n_t)
        preds.append(run(cond_a[i:i+1], cond_f[i:i+1], res, fixed, seed=1234+i))
        del res; torch.cuda.empty_cache()
    P=torch.stack(preds).detach().cpu().numpy()
    df=ev.rich_metrics(P, w_gt.cpu().numpy(), f_gt.cpu().numpy(), dev)
    print(f"\n===== {label} =====")
    print(df.to_string(index=False))
