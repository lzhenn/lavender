#/usr/bin/env python
"""
    Build Air Particals Object
    Note this class is an array of air particals

"""

import numpy as np
from ..lib import utils, io, const
from . import kernel
import time
print_prefix='core.particals>>'

class Particles:

    '''
    Construct air parcel array

    Attributes

    part_id:        int, partical id
    itramem:        int, memorized release times (s) of the particles
    xlon, xlat:     float, longitude/latitude of the particles
    xmeter, ymeter: float, x, y displacement (m) of the particles
    ztra1:          float, height of the particles
    topo:           float, topography
    pvi:            float, pressure
    qvi:            float, specific humidity
    rhoi:           float, density
    hmixi:          float, mixing ratio
    tri:            float, temperature
    tti:            float, potential temperature
    xmass:          float*nspec, mass of each specie

    Methods
    -----------


    '''
    
    def __init__(self, nptcls=1, nspec=1):
        """ construct air parcel obj """  
        # initiate partical templates
        __ARFP32=np.zeros(nptcls,dtype=np.float32)
        __ARIP32=np.zeros(nptcls,dtype=np.int32)
        
        self.nptcls,self.nspec=nptcls,nspec
        self.part_id=np.arange(nptcls,dtype=np.int32)
        self.itramem=__ARFP32.copy()
        self.xlon,self.xlat=__ARFP32.copy(),__ARFP32.copy()
        self.ix, self.iy, self.iz=\
            __ARIP32.copy(),__ARIP32.copy(),__ARIP32.copy()
        self.dx, self.dy, self.dz=\
            __ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy()
        self.ztra1,self.topo=__ARFP32.copy(),__ARFP32.copy()
        self.du, self.dv, self.dw=\
            __ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy()
        self.pvi,self.qvi,self.rhoi,self.hmixi=\
            __ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy()
        self.tri,self.tti=__ARFP32.copy(),__ARFP32.copy()
        self.xmass=np.zeros((nptcls,nspec),dtype=np.float32)
    
        utils.write_log(
            print_prefix+'array with %d particals initiated!' % nptcls)

    def update(self, emis):
        """ update particle array by emission """ 
        self.itramem=np.linspace(
            -emis.emis_span,const.FP32_NEGZERO,self.nptcls,dtype=np.float32)  
        self.emis_span=emis.emis_span
        self.z0=emis.height
        self.ix, self.iy, self.iz=\
            self.ix+emis.ipos[0],\
            self.iy+emis.ipos[1],\
            self.iz+emis.ipos[2]
        self.ix0, self.iy0,self.iz0=emis.ipos[0],emis.ipos[1],emis.ipos[2]
        self.EOdx, self.EOdy=emis.EOdx, emis.EOdy
        # active particles number
        self.np=0

    def march(self,mesh):
        """ march particles """
        cfg=mesh.cfg
        if cfg['PHYS'].getboolean('turb_on'):
            (self.ix,self.iy,self.iz,
            self.dx,self.dy,self.dz,
            self.itramem, self.np)=kernel.cpu_advection(
                mesh.u, mesh.v, mesh.w,
                self.du, self.dv, self.dw, self.np,
                self.ix, self.iy, self.iz, 
                self.dx, self.dy, self.dz,
                self.itramem, self.ix0, self.iy0,
                self.EOdx, self.EOdy, 
                mesh.dt, mesh.dx, self.z0,
                mesh.z_c0,mesh.z_c1,
                sigu=mesh.sigu, sigv=mesh.sigv, sigw=mesh.sigw, 
                tauu=mesh.tauu, tauv=mesh.tauv, tauw=mesh.tauw
                )
        else:
            (self.ix,self.iy,self.iz,
            self.dx,self.dy,self.dz,
            self.itramem, self.np)=kernel.cpu_advection(
                mesh.u, mesh.v, mesh.w,
                self.du, self.dv, self.dw, self.np,
                self.ix, self.iy, self.iz, 
                self.dx, self.dy, self.dz,
                self.itramem, self.ix0, self.iy0,
                self.EOdx, self.EOdy, 
                mesh.dt, mesh.dx, self.z0,
                mesh.z_c0,mesh.z_c1) 
        
    def snapshot_pos(self, emis, inhdl, glb_time):
        """ snapshot particles position """
        # np number of active particles
        np=self.np
        # add back the original height
        pz=self.z0+self.dz[-np:]
        
        # add ter height to particle height
        pz=kernel.cal_zoter(
            pz, inhdl.ter.values, self.ix[-np:], self.iy[-np:], np)
         
        if inhdl.proj==1:
            plat,plon=kernel.cpu_lambert_pos(
                self.ix[-np:], self.iy[-np:],
                self.dx[-np:], self.dy[-np:],
                inhdl.sinw.values, inhdl.cosw.values,
                emis.EDdx, emis.EDdy,
                inhdl.XLAT.values, inhdl.XLONG.values) 
        else:
            utils.throw_error(print_prefix+'projection not supported yet!')
        
        self.ds=io.acc_ptcl_dump_ds(
            self.part_id[-np:], plat, plon, pz, 
            self.itramem[-np:], glb_time)
        


if __name__ == "__main__":
    pass
