#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""

import numpy as np
import numba 

@numba.njit(parallel=True,fastmath=True)   
def non_negflag(x):
    '''
    super optimized function to 
    return the non-negative flag of x
    val = (x>=0: 1, x<0: 0)
    '''
    return np.logical_not(np.signbit(x))

@numba.njit(parallel=True,fastmath=True)   
def adv(
    v, s, idx, idy, idz, dt, eflag):
    for i in numba.prange(eflag.shape[0]):
        s[i]=s[i]+eflag[i]*v[idz[i],idy[i],idx[i]]*dt
    return s

@numba.njit(parallel=True,fastmath=True)
def reloc_xy(
    mat_idx, org, upbnd, dp, rDX):
    '''
    upbnd: upper bound of the index
    dp: displacement
    org: original position
    rDX: 1/DX
    '''
    #for i in numba.prange(mat_idx.shape[0]):
    mat_idx=org+(dp*rDX)
    mat_idx=non_negflag(mat_idx)*mat_idx 
    mat_idx=mat_idx+non_negflag(mat_idx-upbnd)*(upbnd-mat_idx)
    #mat_idx=np.where(mat_idx>=upbnd,-1.0,mat_idx)
    return mat_idx

@numba.njit(parallel=True,fastmath=True)
def reloc_z(
    mat_idz, z0, upbnd, dp, c0, rc0, lnc1):
    '''
    upbnd: upper bound of the index
    dp: displacement
    org: original position
    rDX: 1/DX
    '''
    #for i in numba.prange(mat_idx.shape[0]):
    mat_idz=(np.log(non_negflag(z0+dp)*(z0+dp)+c0)*rc0)*lnc1
    mat_idz=mat_idz+non_negflag(mat_idz-upbnd)*(upbnd-mat_idz)
    #mat_idz=np.where(mat_idz>=upbnd,-1.0,mat_idz)
    return mat_idz


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
    emit_flag=non_negflag(pt)

    # calculate the displacement
    pdx=adv(u, pdx, pidx, pidy, pidz, dt, emit_flag)
    pdy=adv(v, pdy, pidx, pidy, pidz, dt, emit_flag)
    pdz=adv(w, pdz, pidx, pidy, pidz, dt, emit_flag)
    
    # update position
    pidz=pidz.astype(float)
    pidz=reloc_z(pidz, z0, nz, pdz, zc0, rc0, rlnc1)
    pidz=pidz.astype(int)

    pidx=pidx.astype(float)
    pidx=reloc_xy(pidx, ip0[0], nx, pdx, rDX)
    pidx=pidx.astype(int)
    
    pidy=pidy.astype(float)
    pidy=reloc_xy(pidy, ip0[1], ny, pdy, rDX)
    pidy=pidy.astype(int)
    
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
    
    

@numba.jit(nopython=True)   
def cpu_advection_all(
    u0, v0, w0, u1, v1, w1, 
    pidx, pidy, pidz, pt, dt, DX, zc0, zc1):
    """
    March the air parcel (single) in the UVW fields
    """
    # update position
    pt=pt+dt
    # whether the parcel is emitted
    emit_flag=np.sign(np.sign(pt)+1)
    
    # calculate the displacement
    ''' distributed loop '''
    for i in range(len(pidx)):
        idx0,idy0,idz0=pidx[i],pidy[i],pidz[i]
        dx=emit_flag[i]*u[idx0, idy0, idz0]*dt
        
    for i in range(len(pidx)):
        idx0,idy0,idz0=pidx[i],pidy[i],pidz[i]
        dy=emit_flag[i]*v[idx0, idy0, idz0]*dt
    
    for i in range(len(pidx)):
        idx0,idy0,idz0=pidx[i],pidy[i],pidz[i]
        dz=emit_flag[i]*w[idx0, idy0, idz0]*dt
        # update position
        
    for i in range(len(pidx)):
        pidx[i]=pidx[i]+np.round(dx/DX)
    for i in range(len(pidx)):
        pidy[i]=pidy[i]+np.round(dy/DX)
    for i in range(len(pidx)):
        pidz[i]=pidz[i]+np.round(dz/sep_zs[0])
    
    
    ''' ordinary implementation 
    for i in range(len(pidx)):
        idx0,idy0,idz0=pidx[i],pidy[i],pidz[i]
        dx=emit_flag[i]*u[idx0, idy0, idz0]*dt
        dy=emit_flag[i]*v[idx0, idy0, idz0]*dt
        dz=emit_flag[i]*w[idx0, idy0, idz0]*dt
        # update position
        pidx[i]=pidx[i]+np.round(dx/DX)
        pidy[i]=pidy[i]+np.round(dy/DX)
        pidz[i]=pidz[i]+np.round(dz/sep_zs[0])
    ''' 

    return pidx, pidy, pidz, pt


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
