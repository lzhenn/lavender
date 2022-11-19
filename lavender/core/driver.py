#/usr/bin/env python
"""CORE: March the Air Parcel by Lagrangian Approach"""
import datetime, time
from ..lib import io, cfgparser, utils, const
from . import mesh, particles

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

        self.ptcls=particles.Particles(nptcls, nspecs)
        self.ptcls.update(self.emission)

        # init timemanager
        self.tmgr=TimeManager(cfg, self.mesh, self.infhdl)

        self.debug=cfg['RUNTIME'].getboolean('debug')
        utils.write_log(print_prefix+'model driver initiated!')
    
    def drive(self):
        '''
        drive the model!!!
        '''

        starttime=time.time()
        # init from frm 1 as 0 all ptlcs are in emission pos
        utils.write_log(
            print_prefix+'t=%s start Lagrangian kernel...'% self.tmgr.curr_t)
        
        while self.tmgr.curr_t < self.tmgr.end_t:    
            utils.write_log(
                '%smarch @ %s' % (print_prefix, self.tmgr.curr_t))
            self.ptcls.march(self.mesh)
            if self.debug:
                self.debuginfo()
            self.tmgr.advance()
            # output frame
            if self.tmgr.curr_t in self.outfhdl.tfrms:
                utils.write_log(
                    '%sparticle dump @ %s' % (print_prefix, self.tmgr.curr_t))
                self.ptcls.snapshot_pos(self.emission, self.infhdl)
 
            self.mesh.update_wind(
                self.tmgr.iofrm, self.tmgr.iofrac)
        
        endtime=time.time()
        utils.write_log(
            print_prefix+'advection finished in %f s' % (endtime-starttime))
        utils.write_log(print_prefix+'model driver completed!')
    
    def debuginfo(self):
        iz,iy,ix=self.ptcls.iz[-1],self.ptcls.iy[-1],self.ptcls.ix[-1]
        dx,dy,dz=self.ptcls.dx[-1],self.ptcls.dy[-1],self.ptcls.dz[-1]
        it=self.ptcls.itramem[-1] 
        
        utils.write_log('%sptcl0[iz,iy,ix]=(%04d,%04d,%04d),it=%10.1f' % (
            print_prefix,iz,iy,ix,it),lvl=10)
            
        utils.write_log('%sdz=%10.1f,dx=%10.1f,dy=%10.1f' % ( 
            print_prefix,dz,dx,dy),lvl=10)
        
        utils.write_log(print_prefix+'u=%4.1f,v=%4.1f,w=%8.7f' % (
            self.mesh.u[iz,iy,ix],self.mesh.v[iz,iy,ix],self.mesh.w[iz,iy,ix]),
            lvl=10)

class TimeManager():
    '''
    Time manager class
    '''

    def __init__(self, cfg, mesh, infhdl):
        self.init_t=datetime.datetime.strptime(
            cfg['INPUT']['init_t'],const.YMDHM)
        
        self.end_t=datetime.datetime.strptime(
            cfg['INPUT']['end_t'],const.YMDHM)
        
        self.curr_t=self.init_t

        iodt=(infhdl.tfrms[1]-infhdl.tfrms[0]).total_seconds()
        self.total_span=(self.end_t-self.init_t).total_seconds()
        

        if cfg['RUNTIME']['dt']=='0':
            self.dt=mesh.dx/const.SCALE_VEL # upper bound
            if self.dt>iodt:
                self.dt=iodt
        else:
            self.dt=int(cfg['RUNTIME']['dt'])
        
        if self.dt>const.MAX_DT:
            self.dt=const.MAX_DT
        self.dt=float(self.dt)
        res=iodt % self.dt
        if res>0:
            utils.throw_error(
                '%s[INPUT][feed_frq]=%d is not a multiple of [RUNTIME][dt]=%d'\
                % (print_prefix, iodt, self.dt))

        utils.write_log(print_prefix+'dyn dt=%5.1f'%self.dt)
        mesh.dt=self.dt

        self.glb_step=0
        self.NSTEPS_PER_IO=iodt/self.dt
        self.iofrac=0.0
        self.iofrm=1

        self.output_frq=cfg['OUTPUT']['output_frq']

    def advance(self):
        self.curr_t+=datetime.timedelta(seconds=self.dt)
        self.glb_step+=1
        self.iofrm=1+int(self.glb_step/self.NSTEPS_PER_IO)
        self.iofrac=(self.glb_step % self.NSTEPS_PER_IO)/self.NSTEPS_PER_IO
        if self.curr_t==self.end_t:
            self.iofrac=1.0