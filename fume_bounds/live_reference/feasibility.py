"""Phase I for the OLD fixed-family constraints. Does not modify thresholds.

Distinguish a stale infeasible starting point from an actually empty set.
Infeasibility requires a positive dual-side lower bound, not solver failure.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
from .convex_base import RiskSystem

@dataclass
class FeasibilityResult:
    status: str
    witness: np.ndarray
    max_violation: float
    minimax_lower: float
    minimax_upper: float
    start_violation: float
    message: str
    def as_dict(self):
        d=asdict(self); d['witness']=self.witness.tolist(); return d


def phase_one(risk: RiskSystem, thresholds, start=None, *, tol=2e-8,
              maxiter=1200):
    b=np.asarray(thresholds,dtype=float)
    m=risk.n_heads; j=risk.n_constraints
    if b.shape!=(j,) or not np.isfinite(b).all() or tol<0:
        raise ValueError('Invalid thresholds/tolerance')
    a=np.full(m,1/m) if start is None else np.asarray(start,dtype=float)
    if (a.shape!=(m,) or not np.isfinite(a).all() or (a<0).any()
            or not np.isclose(a.sum(),1,atol=1e-10,rtol=0)):
        raise ValueError('Invalid simplex start')
    residual=risk.risks(a)-b
    v0=float(residual.max())
    # min t subject to R_j(a)-b_j <= t; t may be negative (strict feasibility).
    z0=np.r_[a,v0+max(1e-5,abs(v0)*.01)]
    eq=lambda z: z[:-1].sum()-1
    ej=lambda z: np.r_[np.ones(m),0.]
    con=lambda z: z[-1]-(risk.risks(z[:-1])-b)
    cj=lambda z: np.column_stack([-risk.jacobian(z[:-1]), np.ones(j)])
    sol=minimize(lambda z:float(z[-1]),z0,jac=lambda z:np.r_[np.zeros(m),1.],
                 method='SLSQP', bounds=[(0,1)]*m+[(None,None)],
                 constraints=[dict(type='eq',fun=eq,jac=ej),dict(type='ineq',fun=con,jac=cj)],
                 options={'ftol':1e-13,'maxiter':maxiter})
    aa=np.maximum(sol.x[:-1],0); aa=aa/aa.sum()
    residual=risk.risks(aa)-b; upper=float(residual.max())
    active=np.flatnonzero(upper-residual<=1e-7)
    weights=[]
    if hasattr(sol,'multipliers'):
        w=np.maximum(np.asarray(sol.multipliers[1:1+j]),0)
        if w.shape==(j,) and np.isfinite(w).all() and w.sum()>0: weights.append(w/w.sum())
    w=np.zeros(j);w[active]=1/len(active);weights.append(w)
    w=np.zeros(j);w[int(np.argmax(residual))]=1;weights.append(w)
    lows=[]
    for w in weights:
        value=float(w@residual); grad=risk.jacobian(aa).T@w
        tangent=value+float(grad.min()-grad@aa)
        lows.append(tangent-2e-10*(1+abs(value)+np.linalg.norm(grad,1)))
    lower=float(max(lows))
    if lower>upper+max(1e-8,tol):
        raise RuntimeError('Inconsistent numerical primal/dual feasibility brackets')
    status=('feasible' if upper<=tol else
            'infeasible_certificate' if lower>tol else 'unresolved')
    return FeasibilityResult(status,aa,upper,lower,upper,v0,str(sol.message))
