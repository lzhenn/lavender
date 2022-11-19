#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""
import datetime
import numpy as np
from scipy import interpolate
from ..lib import const, math_func, utils
import wrf

print_prefix='core.mesh>>'
R2D=const.RAD2DEG
D2R=const.DEG2RAD
RE=const.R_EARTH


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

        dlat=inhdl.XLAT.values[iy,ix]-self.lat
        dlon=inhdl.XLONG.values[iy,ix]-self.lon
        # vector naming convention:
        #   E --> Emission Pos, O --> Origin Grid, 
        #   D --> Destination Grid, P --> Particle Pos
        # vector from emission to nearest grid center (Mass mesh) dx & dy
        (EOdx, EOdy)=\
            dll2dxy(dlat, dlon, self.lat, self.lon, inhdl)
        
        n_sn, n_we = inhdl.n_sn, inhdl.n_we
        MatX=np.broadcast_to(np.arange(0,n_we), (n_sn,n_we))
        MatY=np.broadcast_to(np.arange(0,n_sn), (n_we, n_sn)).T
        
        (ODdx, ODdy)=(ix-MatX)*inhdl.dx, (iy-MatY)*inhdl.dx
        (self.EDdx, self.EDdy) = EOdx+ODdx, EOdy+ODdy


class Mesh:

    '''
    Construct model top driver 

    Attributes

    Methods
    -----------
    '''
    def __init__(self, inhdl):
        self.z=const.L53_LOG['layers']
        self.z_c0=const.L53_LOG['c0']
        self.z_c1=const.L53_LOG['c1']

        self.inhdl=inhdl

        (self.u0, self.v0, self.w0)=\
            self.construct_frm_mesh(inhdl, 0)
        
        (self.u1, self.v1, self.w1)=\
            self.construct_frm_mesh(inhdl, 1)
        
        self.u, self.v, self.w=self.u0, self.v0, self.w0
        self.dx=inhdl.dx

    def construct_frm_mesh(self, inhdl, frm):
        '''
        interpolate mesh from wrf output
        '''

        inhdl.load_frame(frm)    
        f = interpolate.interp1d(
            inhdl.z.values, inhdl.U.values, axis=0,fill_value='extrapolate')
        u = f(self.z).astype(np.float32)
        f = interpolate.interp1d(
            inhdl.z.values, inhdl.V.values, axis=0,fill_value='extrapolate')
        v = f(self.z).astype(np.float32)
        f = interpolate.interp1d(
            inhdl.z_stag.values, inhdl.W.values, axis=0,fill_value='extrapolate')
        w = f(self.z).astype(np.float32)
        return u,v,w

    def update_wind(self, iofrm, iofrac):
        if iofrac==0.0:
            self.u0, self.v0, self.w0 = self.u1, self.v1, self.w1
            (self.u1,self.v1, self.w1)=\
                self.construct_frm_mesh(self.inhdl, iofrm)
        self.u=self.u0*(1-iofrac)+self.u1*iofrac
        self.v=self.v0*(1-iofrac)+self.v1*iofrac
        self.w=self.w0*(1-iofrac)+self.w1*iofrac


def dxy2dll(dx, dy, lat0, lon0, inhdl):
    '''
    convert dx, dy to dlat, dlon
    '''
    if inhdl.proj==1:
        # lambert projection
        iy,ix=math_func.get_closest_idxy(
            inhdl.XLAT.values, inhdl.XLONG.values, lat0, lon0)
        sinw,cosw=inhdl.sinw[iy,ix].values, inhdl.cosw[iy,ix].values
        
        # rotate coordinate 
        dwe=dx*cosw-dy*sinw
        dsn=dx*sinw+dy*cosw
        
        dlat=dsn/RE*R2D
        dlon=R2D*dwe/(np.cos(lat0*D2R)*RE)

    elif inhdl.proj==3:
        # mercator projection
        dlat=dsn*R2D/RE
        dlon=R2D*dwe/(np.cos(lat0*D2R)*RE)
    else:
        utils.throw_error('projection not supported')
    griddlat=R2D*inhdl.dx/const.LATDIS
    if abs(dlat)> griddlat or abs(dlon)>griddlat:
        utils.write_log('dlat=%6.4fdeg, dlon=%6.4fdeg'%(dlat, dlon), 40)
        utils.throw_error(
            'dx or dy too large that exceed grid size: %10.1fm'%inhdl.dx)
    return dlat, dlon


def dll2dxy(dlat, dlon, lat0, lon0, inhdl):
    '''
    convert dlat, dlon to dx, dy
    '''
    if inhdl.proj==1:
        iy,ix=math_func.get_closest_idxy(
            inhdl.XLAT.values, inhdl.XLONG.values, lat0, lon0)
        sinw,cosw=inhdl.sinw[iy,ix].values, inhdl.cosw[iy,ix].values
        
        #Uearth = U*cosalpha - V*sinalpha
        #Vearth = V*cosalpha + U*sinalpha
        # in meter
        dwe=RE*np.cos(lat0*D2R)*dlon*D2R
        dsn=RE*dlat*D2R
        
        # lambert projection
        dx=dwe*cosw+dsn*sinw
        dy=-dwe*sinw+dsn*cosw

    elif inhdl.proj==3:
        # mercator projection
        dy=dlat*D2R*RE
        dx=dlon*D2R*RE*np.cos(lat0*D2R)
    else:
        utils.throw_error('projection not supported')

    if abs(dx)>inhdl.dx or abs(dy)>inhdl.dx:
        utils.write_log('dx=%10.1fm, dy=%10.1fm'%(dx, dy), 40)
        utils.throw_error('dx or dy too large that exceed grid size: %10.1fm'%inhdl.dx)
    return dx, dy