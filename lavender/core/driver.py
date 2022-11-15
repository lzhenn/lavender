#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""
import datetime, time
from ..lib import io, cfgparser, utils, const
from . import mesh, particals

print_prefix='core.driver>>'


class Driver:

    '''
    Construct model top driver 

    Attributes

    Methods
    -----------
    '''
    def __init__(self, cfg):
        
        # init file handlers
        self.infhdl=io.InHandler(cfg)
        self.outfhdl=io.OutHandler(cfg)

        # init mesh and emissions        
        self.mesh=mesh.Mesh(self.infhdl)
        self.emission=mesh.Emission(
            cfg, self.infhdl, self.mesh)

        # init particals
        nptcls=int(
            utils.parse_bitunits(cfg['EMISSION']['nptcls']))
        nspecs=len(
            cfgparser.cfg_get_varlist(cfg,'EMISSION','specs'))

        self.ptcls=particals.Particals(nptcls, nspecs)
        self.ptcls.update(self.emission)

        # init timemanager
        self.tmgr=TimeManager(cfg, self.mesh)

        self.debug=cfg['RUNTIME'].getboolean('debug')
        utils.write_log(print_prefix+'model driver initiated!')
    
    def drive(self):
        '''
        drive the model!!!
        '''

        starttime=time.time()
        while self.tmgr.curr_t < self.tmgr.end_t:
            utils.write_log(print_prefix+'t=%s'% self.tmgr.curr_t)
            self.ptcls.march(self.mesh)
            if self.debug:
                self.debuginfo()
            self.tmgr.advance()
        
        endtime=time.time()
        utils.write_log(print_prefix+'advection finished in %f s' % (endtime-starttime))
        utils.write_log(print_prefix+'model driver finished!')
    
    def debuginfo(self):
            iz,iy,ix=self.ptcls.iz[-1],self.ptcls.iy[-1],self.ptcls.ix[-1]
            it=self.ptcls.itramem[-1] 
            dz=self.ptcls.dz[-1]
            utils.write_log(print_prefix+'ptcl0[iz,iy,ix]=(%04d,%04d,%04d),it=%10.1f,dz=%10.1f' % ( 
                iz,iy,ix,it,dz),lvl=10)
            utils.write_log(print_prefix+'u=%4.1f,v=%4.1f,w=%8.7f' % (
                self.mesh.u[iz,iy,ix],self.mesh.v[iz,iy,ix],self.mesh.w[iz,iy,ix]),lvl=10)
 
class TimeManager():
    '''
    Time manager class
    '''

    def __init__(self, cfg, mesh):
        self.init_t=datetime.datetime.strptime(
            cfg['INPUT']['init_t'],const.YMDHM)
        
        self.end_t=datetime.datetime.strptime(
            cfg['INPUT']['end_t'],const.YMDHM)
        
        self.total_span=(self.end_t-self.init_t).total_seconds()
        self.curr_t=self.init_t
        if cfg['RUNTIME']['dt']=='0':
            self.dt=mesh.dx/const.SCALE_VEL
        else:
            self.dt=int(cfg['RUNTIME']['dt'])
        utils.write_log(print_prefix+'dyn dt=%5.1f'%self.dt)
        mesh.dt=self.dt
        self.output_frq=cfg['OUTPUT']['output_frq']

    def advance(self):
        self.curr_t+=datetime.timedelta(seconds=self.dt)