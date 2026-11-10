"""Intercept-only split likelihood oracle. Fraction endpoints enclose the exact roots."""
from fractions import Fraction as Q
from math import comb
import argparse
import json
from pathlib import Path
from time import perf_counter
import mpmath as mp


def likelihood(p, n, k):
    return p**k * (1-p)**(n-k)


def coin_bounds(q, n, k, delta):
    if not isinstance(n, int) or not isinstance(k,int) or not 0<=k<=n:
        raise ValueError("invalid counts")
    if not (0<q<1 and 0<delta<1):
        raise ValueError("invalid fit probability or error budget")
    if n==0:
        return Q(0),Q(1)
    peak=Q(k,n)
    cutoff=delta*likelihood(q,n,k)
    def exterior(outside, inside):
        for _ in range(80):
            mid=(outside+inside)/2
            if likelihood(mid,n,k)>=cutoff:
                inside=mid
            else:
                outside=mid
        return outside
    return (exterior(Q(0),peak) if k else Q(0),
            exterior(Q(1),peak) if k<n else Q(1))


def project_coin(q,n,k,delta,direction):
    low,high=coin_bounds(q,n,k,delta)
    if direction==1:
        return low,high
    if direction==-1:
        return 1-high,1-low
    if direction==0:
        return Q(1,2),Q(1,2)
    raise ValueError("coin oracle supports directions -1, 0, 1")


def high_precision(q,n,k,delta):
    """Independent 80-decimal evaluation in log space, analytic at both extremes."""
    q=mp.mpf(q.numerator)/q.denominator
    delta=mp.mpf(delta.numerator)/delta.denominator
    if n==0:return mp.mpf(0),mp.mpf(1)
    if k==0:return mp.mpf(0),1-(1-q)*delta**(mp.mpf(1)/n)
    if k==n:return q*delta**(mp.mpf(1)/n),mp.mpf(1)
    target=k*mp.log(q)+(n-k)*mp.log1p(-q)+mp.log(delta)
    def solve(outside,inside):
        for _ in range(180):
            p=(outside+inside)/2
            if k*mp.log(p)+(n-k)*mp.log1p(-p)<target:
                outside=p
            else:
                inside=p
        return (outside+inside)/2
    peak=mp.mpf(k)/n
    return solve(0,peak),solve(1,peak)


def run():
    start=perf_counter()
    mp.mp.dps=80
    cases=0;grid=[];max_gap=mp.mpf(0)
    as_mp=lambda r:mp.mpf(r.numerator)/r.denominator
    for n in (0,1,8,24):
        for q in (Q(1,10),Q(1,2),Q(9,10)):
            previous={}
            for delta in (Q(1,10),Q(1,20)):
                for k in range(n+1):
                    low,high=coin_bounds(q,n,k,delta)
                    left,right=high_precision(q,n,k,delta)
                    gap=max(left-as_mp(low),as_mp(high)-right)
                    assert as_mp(low)<=left and right<=as_mp(high) and gap<mp.mpf('1e-20')
                    max_gap=max(max_gap,gap)
                    assert low<=q<=high
                    if k in previous:
                        assert low<=previous[k][0]<=previous[k][1]<=high
                    previous[k]=(low,high)
                    cases+=1
                for p in (Q(0),Q(1,100),Q(1,10),Q(3,10),Q(1,2),Q(9,10),Q(99,100),Q(1)):
                    def accepted(k):
                        return likelihood(p,n,k)>=delta*likelihood(q,n,k)
                    mass=sum(comb(n,k)*likelihood(p,n,k) for k in range(n+1) if accepted(k))
                    assert all(previous[k][0]<=p<=previous[k][1] for k in range(n+1) if accepted(k))
                    assert mass>=1-delta
                    grid.append(dict(n=n,q=str(q),p=str(p),delta=str(delta),coverage=str(mass)))
    # Catch inverted threshold, unscaled mean threshold and using q as the lower bound.
    n,k,q,d=8,3,Q(1,2),Q(1,20)
    f=lambda p:-mp.log(as_mp(likelihood(p,n,k)))
    budget=-mp.log(as_mp(d))
    p=Q(1,100)
    assert f(p)>f(q)+budget and f(p)/n < f(q)/n+budget
    assert mp.almosteq((f(q)+budget)-f(q),budget)
    assert mp.almosteq((f(p)-f(q))/n,f(p)/n-f(q)/n)
    lo,hi=coin_bounds(q,n,k,d)
    assert lo<Q(3,8)<q<hi
    elapsed=perf_counter()-start
    assert elapsed<300
    return dict(status="passed",endpoint_cases=cases,exact_coverage_grid=grid,
                maximum_outward_gap=str(max_gap),wall_seconds=elapsed,
                example=dict(n=n,k=k,q=str(q),delta=str(d),lower=float(lo),upper=float(hi)),
                scope="intercept-only; no real probability or high-dimensional projection certified")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.write_text(json.dumps(run(),indent=2)+"\n",encoding="utf-8")

