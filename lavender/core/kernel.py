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
        nb.f4[:,:,:], nb.f4[:], nb.i4[:], nb.i4[:], nb.i4[:], nb.f4, nb.b1[:]),
    parallel=True,fastmath=True)   
def adv(
    v, s, idx, idy, idz, dt, eflag):
    for i in nb.prange(eflag.shape[0]):
        s[i]=s[i]+eflag[i]*v[idz[i],idy[i],idx[i]]*dt
    return s

@nb.njit(
    nb.i4[:](nb.i4[:], nb.f4, nb.f4, nb.f4[:], nb.f4),
    parallel=True,fastmath=True)
def reloc_xy(
    mat_idx, org, upbnd, dp, rDX):
    '''
    upbnd: upper bound of the index
    dp: displacement
    org: original position
    rDX: 1/DX
    '''
    idx=org+(dp*rDX)
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
    org: original position
    rDX: 1/DX
    '''
    #for i in numba.prange(mat_idx.shape[0]):
    idz=(np.log((math_func.non_negflag(z0+dp)*(z0+dp)+c0)*rc0))*lnc1
    idz=idz+math_func.non_negflag(idz-upbnd)*(upbnd-idz)
    return math_func.roundI4(idz, 0, mat_idz)

@nb.njit(parallel=True,fastmath=True)
def cpu_lambert_pos(
    ix, iy, dx, dy, sinw, cosw,
    EDdx, EDdy, xlat, xlon):
    '''
        ix,iy,iz: index of pos for ptcls in 1D
        dx,dy,dz: displacement of ptcls in 1D
        EDdx, EDdy: ED vectors of mesh 2D
        xlat,xlon: lat/lon of mesh 2D 
    '''
    
    plat,plon=np.zeros(ix.shape[0]),np.zeros(ix.shape[0])
    
    for i in nb.prange(ix.shape[0]):
        
        sinw0,cosw0=sinw[iy[i],ix[i]],cosw[iy[i],ix[i]]
        Dlat,Dlon=xlat[iy[i],ix[i]],xlon[iy[i],ix[i]]

        DPx=dx[i]-EDdx[iy[i],ix[i]]
        DPy=dy[i]-EDdy[iy[i],ix[i]]
        # rotate coordinate 
        dwe=DPx*cosw0-DPy*sinw0
        dsn=DPx*sinw0+DPy*cosw0
        # update lat/lon
        plat[i]=Dlat+dsn*C1
        plon[i]=Dlon+C2*dwe/(np.cos(Dlat*D2R))
    return plat, plon

def cpu_advection(
    u, v, w, 
    pidx, pidy, pidz, pdx, pdy, pdz,
    pt, ip0, dt, DX, z0, zc0, zc1):
    """
    March the air parcel (single) in the UVW fields
    """
    rDX=1.0/DX
    rc0=1.0/zc0
    rlnc1=1.0/np.log(zc1)
    nz,ny,nx=w.shape

    # update position
    pt=pt+dt
    
    # whether the parcel is emitted
    emit_flag=math_func.non_negflag(pt)

    # calculate the displacement
    pdx=adv(u, pdx, pidx, pidy, pidz, dt, emit_flag)
    pdy=adv(v, pdy, pidx, pidy, pidz, dt, emit_flag)
    pdz=adv(w, pdz, pidx, pidy, pidz, dt, emit_flag)
    
    # update position
    #temp=np.empty_like(pidx, order='C')
    pidz=reloc_z(pidz, z0, nz, pdz, zc0, rc0, rlnc1)
    pidy=reloc_xy(pidy, ip0[1], ny, pdy, rDX)
    pidx=reloc_xy(pidx, ip0[0], nx, pdx, rDX)
    
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
    return pidx, pidy, pidz, pdx, pdy, pdz, pt
    

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
