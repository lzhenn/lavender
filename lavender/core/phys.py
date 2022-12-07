#/usr/bin/env python
"""Physical Schemes """

import numpy as np
import numba as nb 
from ..lib import const, math_func

def hanna(ol, pblh, ust, wst, z0, f, z, dt):
    z_3d=np.broadcast_to(z, (pblh.shape[0], pblh.shape[1], z.shape[0]))
    ol_3d, pblh_3d  = ol[:,:,np.newaxis], pblh[:,:,np.newaxis]
    ust_3d, wst_3d=ust[:,:,np.newaxis], wst[:,:,np.newaxis]
    z0_3d, f_3d=z0[:,:,np.newaxis], f[:,:,np.newaxis]
    zoh=z_3d/pblh_3d
    
    #Unstable conditions
    sigu_u=(12.0+pblh_3d/(2*abs(ust_3d)))**(1.0/3.0)
    sigv_u=sigu_u
    tauu_u=0.15*pblh_3d/sigu_u
    tauv_u=tauu_u
    sigw1=1.2*wst_3d**2*(1-0.9*zoh)*zoh**(2.0/3.0)
    sigw2=(1.8-1.4*zoh)*ust_3d**2
    sigw_u=np.sqrt(np.where(sigw1+sigw2<0, 1e-9, sigw1+sigw2))
    zosigw=z_3d/sigw_u
    # vectorized version
    tauw_u=np.where(
        np.bitwise_and(zoh<0.1, z-z0_3d>-ol_3d), 
        0.1*zosigw/(0.55-0.38*(z_3d-z0_3d)/ol_3d), 0)
    tauw_u=np.where(
        np.bitwise_and(zoh<0.1, z-z0_3d<-ol_3d), 
        0.59*zosigw, tauw_u)
    tauw_u=np.where(zoh>=0.1, 0.15*(pblh_3d/sigw_u)*(1-np.exp(-5*zoh)), tauw_u)

    #Stable conditions
    sigu_s=2.0*(1-zoh)*ust_3d
    sigv_s=1.3*(1-zoh)*ust_3d
    sigw_s=sigv_s
    tauu_s=0.15*(pblh_3d/sigu_s)*np.sqrt(zoh)
    tauv_s=tauu_s*(7/15.0)
    tauw_s=tauu_s*(1/1.5)

    # Neutral conditions
    zoust=z_3d/ust_3d
    sigu=2.0*np.exp(-3.0*f_3d*zoust)
    sigv=1.3*np.exp(-2.0*f_3d*zoust)
    sigw=sigv
    tauu=(0.5*(z_3d/(sigu+1e-9)))/(1+15*f_3d*zoust)
    tauv=tauu
    tauw=tauu

    # Combine
    unstable_mask=np.bitwise_and(pblh_3d>abs(ol_3d),ol_3d<0)
    stable_mask=np.bitwise_and(pblh_3d>abs(ol_3d),ol_3d>=0)
    sigu=np.where(unstable_mask,sigu_u,sigu)
    sigu=np.where(stable_mask,sigu_s,sigu)
    sigv=np.where(unstable_mask,sigv_u,sigv)
    sigv=np.where(stable_mask,sigv_s,sigv)
    sigw=np.where(unstable_mask,sigw_u,sigw)
    sigw=np.where(stable_mask,sigw_s,sigw)
    tauu=np.where(unstable_mask,tauu_u,tauu)
    tauu=np.where(stable_mask,tauu_s,tauu)
    tauv=np.where(unstable_mask,tauv_u,tauv)
    tauv=np.where(stable_mask,tauv_s,tauv)
    tauw=np.where(unstable_mask,tauw_u,tauw)
    tauw=np.where(stable_mask,tauw_s,tauw)


    # limit
    sigu=np.where(z_3d>pblh_3d,np.sqrt(const.diffh/dt),sigu)
    sigv=np.where(z_3d>pblh_3d,np.sqrt(const.diffh/dt),sigv)
    tauu=np.where(tauu<10.0,10.0,tauu)
    tauv=np.where(tauv<10.0,10.0,tauv)
    tauw=np.where(tauw<30.0,30.0,tauw)



    return sigu, sigv, sigw, tauu, tauv, tauw


if __name__ == "__main__":
    pass
