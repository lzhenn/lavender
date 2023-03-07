#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""
import datetime
import numpy as np
from scipy import interpolate
from ..lib import const, math_func, utils
from . import phys
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

        if int(cfg['EMISSION']['end_t'])<=int(cfg['INPUT']['end_t']):
            self.end_t=datetime.datetime.strptime(
                cfg['EMISSION']['end_t'],const.YMDHM)
        else:
            self.end_t=datetime.datetime.strptime(
                cfg['INPUT']['end_t'],const.YMDHM)
        
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

        EOdlat=inhdl.XLAT.values[iy,ix]-self.lat
        EOdlon=inhdl.XLONG.values[iy,ix]-self.lon
        # vector naming convention:
        #   E --> Emission Pos, O --> Origin Grid, 
        #   D --> Destination Grid, P --> Particle Pos
        # vector from emission to nearest grid center (Mass mesh) dx & dy
        (EOdx, EOdy)=\
            dll2dxy(EOdlat, EOdlon, self.lat, self.lon, inhdl)
        
        n_sn, n_we = inhdl.n_sn, inhdl.n_we
        MatX=np.broadcast_to(np.arange(0,n_we), (n_sn,n_we))
        MatY=np.broadcast_to(np.arange(0,n_sn), (n_we, n_sn)).T
        
        # all destination grids center
        (ODdx, ODdy)=(MatX-ix)*inhdl.dx, (MatY-iy)*inhdl.dx
        (self.EOdx, self.EOdy) = EOdx, EOdy
        (self.EDdx, self.EDdy) = EOdx+ODdx, EOdy+ODdy


class Mesh:

    '''
    Construct model top driver 

    Attributes

    Methods
    -----------
    '''
    def __init__(self, inhdl):
        utils.write_log(print_prefix+'init mesh...')
        self.z=const.L53_LOG['layers']
        self.z_c0=const.L53_LOG['c0']
        self.z_c1=const.L53_LOG['c1']

        self.inhdl=inhdl
        self.cfg=inhdl.cfg
        (self.u0, self.v0, self.w0)=\
            self.construct_frm_mesh(0)
        
        (self.u1, self.v1, self.w1)=\
            self.construct_frm_mesh(1)
        
        
        self.u, self.v, self.w=self.u0, self.v0, self.w0
        self.dx=inhdl.dx

    def construct_frm_mesh(self, frm):
        '''
        interpolate mesh from wrf output toward mesh z-levels
        '''
        inhdl=self.inhdl
        inhdl.load_frame(frm)    

        # convert vertical velocity to terrain-following
        w_ter=self.conv_vert_vel(inhdl)

        f = interpolate.interp1d(
            inhdl.z.values, inhdl.U.values, axis=0,fill_value='extrapolate')
        u = f(self.z).astype(np.float32)

        f = interpolate.interp1d(
            inhdl.z.values, inhdl.V.values, axis=0,fill_value='extrapolate')
        v = f(self.z).astype(np.float32)

        f = interpolate.interp1d(
            inhdl.z_stag.values, inhdl.W.values, 
            axis=0,fill_value='extrapolate')
        w = f(self.z).astype(np.float32)
        

        # no effective pbl paras in wrfout
        return u,v,w
    
    def conv_vert_vel(self, inhdl):
        '''
        convert vertical velocity to terrain-following  
        '''
        u,v,w=inhdl.U.values, inhdl.V.values, inhdl.W.values
        terdx_3d=np.broadcast_to(
            inhdl.terdx, (u.shape[0], u.shape[1], u.shape[2]))
        terdy_3d=np.broadcast_to(
            inhdl.terdy, (u.shape[0], u.shape[1], u.shape[2]))
        w_ter=w-u*terdx_3d-v*terdy_3d

        return w_ter

    def update_state(self, iofrm, iofrac):
        '''
        update wind field / turbulence state by temporal interpolation
        '''

        if iofrac==0.0:
            
            self.u0, self.v0, self.w0 = self.u1, self.v1, self.w1
            self.u1,self.v1, self.w1=\
                self.construct_frm_mesh(iofrm)
            if self.cfg['PHYS'].getboolean('turb_on'):
                utils.write_log(print_prefix+'update turbulence state...')
                self.cal_turb_paras()

        self.u=self.u0*(1-iofrac)+self.u1*iofrac
        self.v=self.v0*(1-iofrac)+self.v1*iofrac
        self.w=self.w0*(1-iofrac)+self.w1*iofrac

    
    def cal_turb_paras(self):
        '''
        calculate pbl parameters

        add attrs:
        pblh                        PBL height
        ol                          Obukhov length    
        wst                         convective velocity scale
        z0                          roughness length
        ust                         friction velocity
        sigu, sigv, sigw      turbulent velocity scale
        taou, taov, taow         Lagrangian time scale 
        '''
        inhdl=self.inhdl
        sf_idz=math_func.get_closest_idx(inhdl.z.values, const.SF_HEIGHT)
        ps, td2, t2= inhdl.PS.values, inhdl.TD2.values, inhdl.T2.values
        ust, hfx = inhdl.UST.values, inhdl.HFX.values

        u10,v10=inhdl.U10.values, inhdl.V10.values
        uv10=math_func.wind_speed(u10,v10)

        hfx=np.where(hfx>0, hfx, 0.0)
        # potential temp in surface layer
        pt_sf=inhdl.PT.sel(bottom_top=slice(0,sf_idz)).mean(dim='bottom_top')
        pt_sf=pt_sf.values
        
        vt_lv1=inhdl.VT.sel(bottom_top=0).values

        pblh=inhdl.PBLH.values

        # for Obukhov length
        e=math_func.ew(td2) 
        tv=t2*(1.+0.378*e/ps)               # virtual temperature
        rhoa=ps/(const.r_air*tv)                     
        thetastar=hfx/(rhoa*const.cp*ust)+const.FP32_ISIM           
        ol=pt_sf*ust**2/(const.k*const.G*thetastar)

        # for convective velocity scale
        wst=(hfx*const.G*pblh/(const.cp*vt_lv1))**(1.0/3.0)
        # for roughness length
        z0=10.0/np.exp(const.k*uv10/ust)

        # for turbulent velocity scale and Lagrangian time scale 
        self.sigu, self.sigv, self.sigw, self.tauu, self.tauv, self.tauw=\
            phys.hanna(
                ol, pblh, ust, wst, z0, 
                self.inhdl.F.values, self.z, self.dt
            )
        

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