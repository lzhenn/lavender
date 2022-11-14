#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""
import datetime
import numpy as np
from scipy import interpolate
from ..lib import const, math_func
import wrf

print_prefix='core.mesh>>'

class Emission:
    '''
    Emission class
    '''
    def __init__(self, cfg, inhdl, mesh):
        
        if cfg['EMISSION']['strt_t']=='init':
            self.init_t=datetime.datetime.strptime(
                cfg['INPUT']['init_t'],const.YMDHM)
        else:
            self.init_t=datetime.datetime.strptime(
                cfg['EMISSION']['strt_t'],const.YMDHM)

        self.end_t=datetime.datetime.strptime(
            cfg['EMISSION']['end_t'],const.YMDHM)

        self.emis_span=(self.end_t-self.init_t).total_seconds()
        self.lat=float(cfg['EMISSION']['lat'])
        self.lon=float(cfg['EMISSION']['lon'])
        self.height=float(cfg['EMISSION']['height'])
        self.xmass=float(cfg['EMISSION']['total_mass'])
        self.xmass_per_sec=self.xmass/self.emis_span
        (iy,ix)=(math_func.get_closest_idxy(
            inhdl.XLAT.values, inhdl.XLONG.values, self.lat, self.lon))
        
        iz = math_func.get_closest_idx(mesh.z, self.height)
        
        self.ipos=np.array([ix,iy,iz]).astype(np.int32)
class Mesh:

    '''
    Construct model top driver 

    Attributes

    Methods
    -----------
    '''
    def __init__(self, inhdl):
        self.z=const.TEST_Z['layers']
        self.magic_idz=const.TEST_Z['magic_idz']
        self.sep_z=const.TEST_Z['sep_z']

        self.construct_frm_mesh(inhdl, 0)
        self.dx=inhdl.dx

    def construct_frm_mesh(self, inhdl, frm):
        '''
        interpolate mesh from wrf output
        '''

        inhdl.load_frame(frm)    
        f = interpolate.interp1d(
            inhdl.z.values, inhdl.U.values, axis=0,fill_value='extrapolate')
        self.u = f(self.z)
        f = interpolate.interp1d(
            inhdl.z.values, inhdl.V.values, axis=0,fill_value='extrapolate')
        self.v = f(self.z)
        f = interpolate.interp1d(
            inhdl.z_stag.values, inhdl.W.values, axis=0,fill_value='extrapolate')
        self.w = f(self.z)
