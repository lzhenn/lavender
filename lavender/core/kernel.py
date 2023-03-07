#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""

import numpy as np
import numba as nb 
from ..lib import const, math_func

# CONSTS
R2D=const.RAD2DEG
D2R=const.DEG2RAD
RE=const.R_EARTH
C1=1.0/RE*R2D
C2=R2D/RE




@nb.njit(
    nb.f4[:](
        nb.f4[:], nb.f4[:], nb.f4, nb.f4, nb.f4, nb.b1[:],
        nb.f4[:], nb.b1[:]),
    parallel=True,fastmath=True)   

def adv_z(
        pv, s, dt, np, z0, dflag, pt, eflag):
    s[-np:]=s[-np:]+dflag[-np:]*pv*dt
    s[-np:]=s[-np:]+eflag[-np:]*pv*pt[-np:]
    s[-np:]=math_func.non_negflag(z0+s[-np:])*s[-np:]
    return s

@nb.njit(
    nb.f4[:](
        nb.f4[:], nb.f4[:], nb.f4, nb.f4, nb.b1[:],
        nb.f4[:], nb.b1[:]),
    parallel=True,fastmath=True)   

def adv(
        pv, s, dt, np, dflag, pt, eflag):
    s[-np:]=s[-np:]+dflag[-np:]*pv*dt
    s[-np:]=s[-np:]+eflag[-np:]*pv*pt[-np:]
    return s



@nb.njit(
    nb.i4[:](nb.i4[:], nb.f4, nb.f4, nb.f4, nb.f4[:], nb.f4),
    parallel=True,fastmath=True)
def reloc_xy(
    mat_idx, org, EOdx, upbnd, dp, rDX):
    '''
    upbnd: upper bound of the index
    org: original position [idx]
    EOdp: EO vector (Emission --> Origin)
    dp: displacement
    rDX: 1/DX
    '''
    idx=org+((dp-EOdx)*rDX)
    idx=math_func.non_negflag(idx)*idx 
    idx=idx+math_func.non_negflag(idx-upbnd)*(upbnd-idx)
    return math_func.roundI4(idx,0,mat_idx)

@nb.njit(
    nb.i4[:](nb.i4[:], nb.f4, nb.f4, nb.f4[:], nb.f4, nb.f4, nb.f4),
    parallel=True,fastmath=True)
def reloc_z(mat_idz, z0, upbnd, dp, c0, rc0, lnc1):
    '''
    upbnd: upper bound of the index
    dp: displacement
    z0: original position
    rDX: 1/DX
    '''
    #for i in numba.prange(mat_idx.shape[0]):
    idz=(np.log(((z0+dp)+c0)*rc0))*lnc1
    idz=idz+math_func.non_negflag(idz-upbnd)*(upbnd-idz)
    return math_func.roundI4(idz, 0, mat_idz)

@nb.njit(parallel=True,fastmath=True)
def cpu_lambert_pos(
    ix, iy, dx, dy, sinw, cosw,
    EDdx, EDdy, xlat, xlon):
    '''
        ix,iy: index of pos for ptcls in 1D
        dx,dy: displacement of ptcls in 1D
        EDdx, EDdy: ED vectors of mesh 2D
        xlat,xlon: lat/lon of mesh 2D 
    '''
    napt=ix.shape[0] 
    DPx,DPy=np.zeros(napt),np.zeros(napt)
    dwe,dsn=np.zeros(napt),np.zeros(napt)
    plat,plon=np.zeros(napt),np.zeros(napt)
    for i in nb.prange(napt):
        py,px= iy[i],ix[i]
        sinw0,cosw0=sinw[py,px],cosw[py,px]
        Dlat,Dlon=xlat[py,px],xlon[py,px]

        DPx[i]=dx[i]-EDdx[py,px]
        DPy[i]=dy[i]-EDdy[py,px]
        # rotate coordinate 
        dwe[i]=DPx[i]*cosw0-DPy[i]*sinw0
        dsn[i]=DPx[i]*sinw0+DPy[i]*cosw0
        # update lat/lon
        plat[i]=Dlat+dsn[i]*C1
        plon[i]=Dlon+C2*dwe[i]/(np.cos(Dlat*D2R))
    return plat, plon

@nb.njit(parallel=True,fastmath=True)
def get_vel(v, idx, idy, idz, napt):
    '''
    get velocity of the parcel at the position [idx,idy,idz]
    of the particle
    '''
    pv=np.zeros(napt)
    for i in nb.prange(napt):
        # id of active particles
        id=-napt+i
        iz,iy,ix=idz[id],idy[id],idx[id]
        pv[i]=v[iz,iy,ix]

    return pv.astype(np.float32)

@nb.njit(parallel=True,fastmath=True)
def cal_zoter(pz, ter, idx, idy, napt):
    for i in nb.prange(napt):
        # id of active particles
        iy,ix=idy[i],idx[i]
        pz[i]=ter[iy,ix]+pz[i]
    return pz
 
@nb.njit(parallel=True,fastmath=True)
def turb_xy(idx, idy, idz, dt, napt, sigv, tauv):
    '''
    calculate turbulence effect on velocity
    du, dv
    '''
    np.random.seed(0)
    dv=np.zeros(napt)
    r=np.random.normal(0.0, 1.0, napt)
    for i in nb.prange(napt):
        # id of active particles
        id=-napt+i
        iz,iy,ix=idz[id],idy[id],idx[id]
        sgv=sigv[iz,iy,ix]
        tuv=tauv[iz,iy,ix]
        #dv[id]=pv[i]*r[i] # simplest works
        dv[i]=sgv*np.sqrt(2.0*dt/tuv)*r[i]
    return dv.astype(np.float32)

@nb.njit(parallel=True,fastmath=True)
def turb_z(w, idx, idy, idz, dt, napt, sigw, tauw):
    '''
    calculate turbulence effect on velocity
    dw
    '''
    np.random.seed(0)
    r=np.random.normal(0.0, 1.0, napt)
    pw=np.zeros(napt)
    for i in nb.prange(napt):
        # id of active particles
        id=-napt+i
        iz,iy,ix=idz[id],idy[id],idx[id]
        sgw=sigw[iz,iy,ix]
        tuw=tauw[iz,iy,ix]
        pw[i]=w[iz,iy,ix]
        pw[id]=pw[i]*(1.0+r[i]) # simplest works
        #pw[i]=pw[i]+sgw*np.sqrt(2.0*dt/tuw)*r[i]
    return pw.astype(np.float32)


def cpu_advection(
    u, v, w, pidx, pidy, pidz, pdx, pdy, pdz,
    pt, ix0, iy0, EOx, EOy, dt, DX, z0, zc0, zc1, **kw):
    """
    March the air parcel (single) in the UVW fields
    napt : number of active particles8
    """
    rDX=1.0/DX
    rc0=1.0/zc0
    rlnc1=1.0/np.log(zc1)
    nz,ny,nx=w.shape

    drift_flag=math_func.non_negflag(pt)
    pt=pt+dt
    # whether the particle is just emitted
    emit_flag=np.bitwise_xor(math_func.non_negflag(pt), drift_flag)
    # np number of active particles
    napt=np.sum(drift_flag)+np.sum(emit_flag)
    
    pu=get_vel(u, pidx, pidy, pidz, napt)
    pv=get_vel(v, pidx, pidy, pidz, napt)
    #pw=get_vel(w, pidx, pidy, pidz, napt)
    if not(kw =={}): # advection+turbulence
        # normal direction turbulent velocity
        du =turb_xy(
            pidx, pidy, pidz, dt, napt, kw['sigu'], kw['tauu'])
        # tangent direction turbulent velocity
        dv=turb_xy(
            pidx, pidy, pidz, dt, napt, kw['sigv'], kw['tauv'])
        
        du,dv=math_func.rotuv(du, dv, pu, pv)
        pu=pu+du
        pv=pv+dv
        
        #pw=get_vel(w, pw, pidx, pidy, pidz, napt)
        pw=turb_z(
            w, pidx, pidy, pidz, dt, napt, kw['sigw'], kw['tauw'])
        

    # calculate the displacement
    pdx=adv(pu, pdx, dt, napt, drift_flag, pt, emit_flag)
    pdy=adv(pv, pdy, dt, napt, drift_flag, pt, emit_flag)
    pdz=adv_z(pw, pdz, dt, napt, z0, drift_flag, pt, emit_flag)
    # update position
    #temp=np.empty_like(pidx, order='C')
    pidz=reloc_z(pidz, z0, nz, pdz, zc0, rc0, rlnc1)
    pidy=reloc_xy(pidy, iy0, EOy, ny, pdy, rDX)
    pidx=reloc_xy(pidx, ix0, EOx, nx, pdx, rDX)
    
    #Need optimization here (60-70% of the span)
    '''
    pidx=ip0[0]+(pdx*rDX)
    pidx=np.where(pidx<0,0,pidx)
    pidx=np.where(pidx>=nx,nx-1,pidx)
    pidx=pidx.astype(int)

    pidy=ip0[1]+(pdy*rDX)
    pidy=np.where(pidy<0,0,pidy)
    pidy=np.where(pidy>=ny,ny-1,pidy)
    pidy=pidy.astype(int)

    pidz=ip0[2]+(pdz*rsep_zs)
    pidz=np.where(pidz<0,0,pidz)
    pidz=np.where(pidz>=nz,nz-1,pidz)
    pidz=pidz.astype(int)
    '''    
    return pidx, pidy, pidz, pdx, pdy, pdz, pt, napt
    

def np_advection(
    u, v, w, pidx, pidy, pidz, pdx, pdy, pdz,
    pt, ip0, dt, DX, idzs, sep_zs):
    '''
    the most conventional implementation by numpy
    '''
    nz,ny,nx=w.shape
    # update position
    pt=pt+dt
    # whether the parcel is emitted
    emit_flag=np.sign(np.sign(pt)+1)
    pdx=pdx+emit_flag*u[pidz,pidy,pidx]*dt
    pdy=pdy+emit_flag*v[pidz,pidy,pidx]*dt
    pdz=pdz+emit_flag*w[pidz,pidy,pidx]*dt
    
    # update position
    pidx=ip0[0]+np.round(pdx/DX).astype(int)
    pidy=ip0[1]+np.round(pdy/DX).astype(int)
    pidz=ip0[2]+np.round(pdz/sep_zs[0]).astype(int)
    pidx=np.where(pidx<0,0,pidx)
    pidx=np.where(pidx>=nx,nx-1,pidx)
    pidy=np.where(pidy<0,0,pidy)
    pidy=np.where(pidy>=ny,ny-1,pidy)
    pidz=np.where(pidz<0,0,pidz)
    pidz=np.where(pidz>=nz,nz-1,pidz)
    return pidx, pidy, pidz, pdx, pdy, pdz, pt

if __name__ == "__main__":
    pass
