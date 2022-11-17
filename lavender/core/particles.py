#/usr/bin/env python
"""
    Build Air Particals Object
    Note this class is an array of air particals

"""

import numpy as np
from ..lib import utils
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
        self.pvi,self.qvi,self.rhoi,self.hmixi=\
            __ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy()
        self.tri,self.tti=__ARFP32.copy(),__ARFP32.copy()
        self.xmass=np.zeros((nptcls,nspec),dtype=np.float32)
    
        utils.write_log(
            print_prefix+'array with %d particals initiated!' % nptcls)


    def update(self, emis):
        """ update partical array by emission """ 
        self.itramem=np.linspace(-emis.emis_span,0,self.nptcls)  
        self.z0=emis.height
        self.ix, self.iy, self.iz=\
            self.ix+emis.ipos[0],\
            self.iy+emis.ipos[1],\
            self.iz+emis.ipos[2]
        self.ipos0=emis.ipos.copy() 
    
    def march(self,mesh):
        """ march particals """
       
        (self.ix,self.iy,self.iz,
        self.dx,self.dy,self.dz,
        self.itramem)=kernel.cpu_advection(
            mesh.u, mesh.v, mesh.w,
            self.ix, self.iy, self.iz, 
            self.dx, self.dy, self.dz,
            self.itramem, self.ipos0, 
            mesh.dt, mesh.dx, self.z0,
            mesh.z_c0,mesh.z_c1) 

    def snapshot_pos(self, mesh):
        """ snapshot particals position """
        self.xlon,self.xlat=\
            mesh.xlon[self.ix,self.iy],\
            mesh.xlat[self.ix,self.iy]
        self.xmeter,self.ymeter=\
            mesh.xmeter[self.ix,self.iy],\
            mesh.ymeter[self.ix,self.iy]
        self.ztra1=mesh.z_c1[self.iz]
        self.topo=mesh.topo[self.ix,self.iy]


if __name__ == "__main__":
    pass