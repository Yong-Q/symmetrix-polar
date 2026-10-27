import sys, numpy as np, ase.io
sys.path.insert(0,'/workspace/group/shared/bza/symmetrix-polar/scripts')
from charge_dipole_student import ChargeDipoleStudent
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pathlib import Path
REPO=Path('/workspace/group/shared/bza/symmetrix-polar')
# force rotation test on a 96-atom-like frame? use 1152 trajectory frame
a=ase.io.read(str(REPO/'bench/plan_d_md_1152_ase_2026-10-26/traj_50ps.xyz'),index=':1')[0]
s=ChargeDipoleStudent()
e1,f1=s.energy_forces(a)
rng=np.random.default_rng(3); A=rng.normal(size=(3,3)); Q,_=np.linalg.qr(A); Q*=np.sign(np.linalg.det(Q))
b=a.copy(); b.set_positions(a.get_positions()@Q.T); b.set_cell(a.cell@Q.T,scale_atoms=False)
e2,f2=s.energy_forces(b)
print('force rotation rel',np.linalg.norm(f2-f1@Q.T)/np.linalg.norm(f1),'dE',e1-e2)
# NVE 1000 steps at 1 fs
MaxwellBoltzmannDistribution(a,temperature_K=300)
a.calc=s.calc
dyn=VelocityVerlet(a,timestep=1.0*units.fs,logfile=None)
energies=[]
def rec(): energies.append(a.get_total_energy())
dyn.attach(rec,interval=10)
dyn.run(1000)
ea=np.array(energies); print('NVE steps',len(ea),'E drift meV/atom',(ea[-1]-ea[0])/len(a)*1e3,'std meV/atom',ea.std()/len(a)*1e3,'max-min', (ea.max()-ea.min())/len(a)*1e3)
