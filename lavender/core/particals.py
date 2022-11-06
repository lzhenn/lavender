#/usr/bin/env python
"""
    Build Air Particals Object
    Note this class is an array of air particals

"""

import numpy as np
from ..lib import utils
print_prefix='core.particals>>'

class Particals:

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
    
    def __init__(self, nptcls, nspec):
        """ construct air parcel obj """  
        # initiate partical templates
        __ARFP32=np.zeros(nptcls,dtype=np.float32)
        __ARIP32=np.zeros(nptcls,dtype=np.int32)
        self.part_id=np.arange(nptcls,dtype=np.int32)
        self.itramem=__ARIP32.copy()
        self.xlon,self.xlat=__ARFP32.copy(),__ARFP32.copy()
        self.xmeter,self.ymeter=__ARFP32.copy(),__ARFP32.copy()
        self.ztra1,self.topo=__ARFP32.copy(),__ARFP32.copy()
        self.pvi,self.qvi,self.rhoi,self.hmixi=\
            __ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy(),__ARFP32.copy()
        self.tri,self.tti=__ARFP32.copy(),__ARFP32.copy()
        self.xmass=np.zeros((nptcls,nspec),dtype=np.float32)
        
        utils.write_log(
            print_prefix+'array with %d particals initiated!' % nptcls)


    def update(self, lat_new, lon_new, height_new, time_new):
        """ update air parcel position """
        
        self.lat.append(lat_new)
        self.lon.append(lon_new)
        self.h.append(height_new)
        self.t.append(time_new)
    
    def output(self, cfg):
        """ output air parcel records according to configurations"""
        
        
        out_fn='./output/P%06d.I%s.E%s' % (int(self.idx), self.t0.strftime("%Y%m%d%H%M%S"),self.t[-1].strftime("%Y%m%d%H%M%S"))
        outfrq_per_dt=int(int(cfg['OUTPUT']['out_frq'])/int(cfg['CORE']['time_step']))
        out_data={'lat':self.lat[::outfrq_per_dt], 'lon':self.lon[::outfrq_per_dt], 'h':self.h[::outfrq_per_dt]}
        df=pd.DataFrame(out_data, index=self.t[::outfrq_per_dt])
        df.to_csv(out_fn)

def acc_output(airp_lst, num_acc, cfg, prefix=''):
    """ 
    output air parcel records according to configurations (accumulated)
    """
    outfrq_per_dt=int(int(cfg['OUTPUT']['out_frq'])/int(cfg['CORE']['time_step']))
    
    ipos=0
    acc_t=[]
    acc_lat=[]
    acc_lon=[]
    acc_h=[]
    
    for airp in airp_lst:
        ipos=ipos+1
        acc_t.extend(airp.t[::outfrq_per_dt]) 
        acc_lat.extend(airp.lat[::outfrq_per_dt]) 
        acc_lon.extend(airp.lon[::outfrq_per_dt]) 
        acc_h.extend(airp.h[::outfrq_per_dt]) 
        if ipos % num_acc ==0:
            out_fn='./output/%sP%06d.I%s.E%s' % (prefix, int(airp.idx), airp.t0.strftime("%Y%m%d%H%M%S"), airp.t[-1].strftime("%Y%m%d%H%M%S"))
            with open(out_fn, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for row in zip(acc_t, acc_lat, acc_lon, acc_h):
                    spamwriter.writerow(row)
            acc_t=[]
            acc_lat=[]
            acc_lon=[]
            acc_h=[]

    out_fn='./output/P%06d.I%s.E%s' % (int(airp_lst[-1].idx), airp_lst[-1].t0.strftime("%Y%m%d%H%M%S"), airp_lst[-1].t[-1].strftime("%Y%m%d%H%M%S"))
    with open(out_fn, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in zip(acc_t, acc_lat, acc_lon, acc_h):
            spamwriter.writerow(row)


    
if __name__ == "__main__":
    pass
