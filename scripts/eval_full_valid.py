import json, sys, numpy as np, torch, torch.nn as nn, ase.io
from pathlib import Path
from collections import defaultdict
from e3nn import o3
REPO=Path('/workspace/group/shared/bza/symmetrix-polar')
sys.path.insert(0,str(REPO.parent/'external-repos/graph_electrostatics-0.4.0'))
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
D=REPO/'data_20261025_charges'; dev='cuda'
ck=torch.load(REPO/'bench/plan_d_bulk_2026-10-25/qmu_heads/qmu_heads.pt',map_location='cpu')
qhead=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1)); qhead.load_state_dict(ck['q_head']); qhead.to(dev).eval()
m0,s0=ck['mu0'].to(dev),ck['sd0'].to(dev)
rz=np.load(REPO/'bench/plan_d_bulk_2026-10-25/qmu_heads/mu_ridge.npz'); W=torch.tensor(rz['W_full'],dtype=torch.float32,device=dev); mm=torch.tensor(rz['mu'],dtype=torch.float32,device=dev); ms=torch.tensor(rz['sd'],dtype=torch.float32,device=dev)
ft=np.load(D/'features_valid.npz'); frames=ase.io.read(str(D/'valid_stride4_multipole.xyz'),index=':')
sigma=1.5; cutoff=1.5*gto_basis_kspace_cutoff([sigma],1)
module=GTOElectrostaticEnergy(density_max_l=1,density_smearing_width=sigma,kspace_cutoff=cutoff,include_self_interaction=True,include_pbc_corrections=True).to(dev).double()
def ewald(atoms,src):
    pos=torch.tensor(atoms.get_positions(),dtype=torch.float64,device=dev); cell=torch.tensor(np.asarray(atoms.cell),dtype=torch.float64,device=dev).unsqueeze(0)
    rc=2*np.pi*torch.linalg.inv(cell.mT); vol=torch.abs(torch.linalg.det(cell)); kv,kn,kb,k0=compute_k_vectors_flat(cutoff,cell,rc)
    batch=torch.zeros(len(atoms),dtype=torch.long,device=dev); pbc=torch.ones(1,3,dtype=torch.bool,device=dev)
    return float(module(kv,kn,kb,k0,torch.tensor(src,dtype=torch.float64,device=dev),pos,batch,vol,pbc).sum())
off=0; rows=[]
for i,a in enumerate(frames):
    n=len(a)
    x0=torch.tensor(ft['X0'][off:off+n],dtype=torch.float32,device=dev); x1=torch.tensor(ft['X1'][off:off+n],dtype=torch.float32,device=dev)
    with torch.no_grad():
        qp=qhead((x0-m0)/s0).squeeze(-1).double().cpu().numpy()/(4*np.pi)
        F=((x1-mm)/ms).transpose(1,2).reshape(n,-1); mup=(F@W).double().cpu().numpy()/(4*np.pi)
    qt=ft['yq'][off:off+n].astype(np.float64)/(4*np.pi); mut=ft['ymu'][off:off+n].astype(np.float64)/(4*np.pi)
    srcp=np.concatenate([qp[:,None],mup],axis=1); srct=np.concatenate([qt[:,None],mut],axis=1)
    ep=ewald(a,srcp); et=ewald(a,srct)
    sp=a.info['config_type'].split('_')[0]
    rows.append({'species':sp,'n':n,'q_mae':float(np.abs(qp-qt).mean()),'mu_mae':float(np.abs(mup-mut).mean()),'E_diff_meV_atom':(ep-et)/n*1e3})
    off+=n
    if (i+1)%100==0: print(i+1,flush=True)
g=defaultdict(list)
for r in rows: g[r['species']].append(r)
out={}
for s,rs in g.items():
    out[s]={'n':len(rs),'q_mae':float(np.mean([r['q_mae'] for r in rs])),'mu_mae':float(np.mean([r['mu_mae'] for r in rs])),'E_abs_diff_meV_atom':float(np.mean([abs(r['E_diff_meV_atom']) for r in rs]))}
out['ALL']={'n':len(rows),'q_mae':float(np.mean([r['q_mae'] for r in rows])),'mu_mae':float(np.mean([r['mu_mae'] for r in rows])),'E_abs_diff_meV_atom':float(np.mean([abs(r['E_diff_meV_atom']) for r in rows]))}
print(json.dumps(out,indent=1))
Path(REPO/'bench/plan_d_bulk_2026-10-25/results/full_valid_head_ewald.json').write_text(json.dumps(out,indent=1))
