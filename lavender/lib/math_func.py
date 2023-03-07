
import numpy as np
import numba as nb
from . import const

@nb.jit(nb.i4[:](nb.float32[:], nb.int64, nb.i4[:]),nopython=True)
def roundI4(x, decimals, out):
    '''
    In: fp32 array, number of decimals, output array
    OUT: output array Int32
    '''
    return np.round_(x, decimals, out)

@nb.njit(nb.b1[:](nb.f4[:],),parallel=True,fastmath=True)   
def non_negflag(x):
    '''
    super optimized function to 
    return the non-negative flag of x
    val = (x>=0: 1, x<0: 0)
    '''
    return np.invert(np.signbit(x))

def rotuv(nu, tv, mu, mv):
    '''
    rotate normal and tangent dircetion of uv 
    to WRF projection coordinate
    nu: normal u, tv: tangent v
    mu: mean u on WRF mesh, mv: mean v on WRF mesh
    '''
    
    ffinv=1./wind_speed(mu, mv)
    sinphi=mv*ffinv
    vy1=sinphi*nu
    cosphi=mu*ffinv
    ux1=cosphi*nu
    
    ux2=-sinphi*tv
    vy2=cosphi*tv

    # Add contributions from along and cross wind components
    du=ux1+ux2
    dv=vy1+vy2    

    return du, dv

def ew(td):
    """
    calculate vapor pressure from dew point temperature 
    purely based on FLEXPART code
    """
    ew=0.
    y=373.16/td
    a=-7.90298*(y-1.)
    a=a+(5.02808*0.43429*np.log(y))
    c=(1.-(1./y))*11.344
    c=-1.+(10.**c)
    c=-1.3816*c/(10.**7)
    d=(1.-y)*3.49149
    d=-1.+(10.**d)
    d=8.1328*d/(10.**3)
    y=a+c+d
    ew=101324.6*(10.**y)
    
    return ew

def great_cir_dis_2d(lat0, lon0, lat1, lon1):
    """ Haversine formula to calculate great circle distance"""  
    D2R=const.DEG2RAD
    R=const.R_EARTH

    lat0_rad, lon0_rad = lat0*D2R, lon0*D2R
    lat1_rad, lon1_rad=lat1*D2R, lon1*D2R
    
    A=np.power(np.sin((lat1_rad-lat0_rad)/2),2)
    B=np.cos(lat0_rad)*np.cos(lat1_rad)*\
        np.power(np.sin((lon1_rad-lon0_rad)/2),2)

    return 2*R*np.arcsin(np.sqrt(A+B))

def wswd2uv(ws, wd):
    """ convert wind component to UV """
    WD_DIC={'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
            'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
            'W':270.0, 'WNW':292.5, 'NW':315.0, 'NNW':337.5}
    
    # below test valid wind direction input
    wd_error=False
    try: 
        wd_int=int(wd)
        if wd_int>=0.0 and wd_int<=360.0:
            wd_rad=wd_int*DEG2RAD
        else:
            wd_error=True
    except ValueError:
        try:
            wd_rad=WD_DIC[wd]*DEG2RAD
        except KeyError:
            wd_error=True
    
    
    if wd_error:
        throw_error('utils.wswd2uv>>','Invalid wind direction input! '
                +'with record: wind_dir='+wd)

    u=-np.sin(wd_rad)*ws
    v=-np.cos(wd_rad)*ws
    
    return (u,v)

def wind_speed(u, v):
    """ calculate wind speed according to U and V """
    ws=np.sqrt(u*u+v*v)
    ws=np.where(ws==0.0, const.FP32_ISIM, ws)
    return ws

def temp_prof(temp0, h0, p0, tgt_h, l):
    """ 
    calculate temperature at tgt_h according to
    temp0 at h0 and lapse rate l

    result converted to WRF perturbated potential T
    """
    TZERO=273.15
    BASE_T=300.0
    REF_P=100000.0 # 1000 hPa
    R=287.43
    CP=1005.0

    
    # change according lapse rate in Kelvin
    k=TZERO + temp0-l*(tgt_h-h0)/100.0
    
    # transfer to potential temp
    theta=k*np.power((REF_P/p0),R/CP)
    
    return theta - BASE_T

def wind_prof(ws0, h0, tgt_h, p):
    """ 
    calculate wind speed at tgt_h according to
    ws0 at h0 and exponent value p
    """
    return ws0*pow((tgt_h/h0), p)

def wind_prof_2d(ws0_2d, h0, tgt_h, p):
    """ 
    calculate wind speed at tgt_h according to
    ws0 at h0 and exponent value p, all data in 2D
    """
    return ws0_2d*(np.power((tgt_h/h0), p)).values


def get_closest_idx(a1d, val):
    """
        Find the nearest idx in 1-D array (a1d) according to a given val
    """
    
    dis=abs(val-a1d)
    return np.argwhere(dis==dis.min())[0][0]

def get_closest_idxy(lat2d, lon2d, lat0, lon0):
    """
        Find the nearest idx, idy in lat2d and lon2d for lat0 and lon0
    """
    dis_lat2d=lat2d-lat0
    dis_lon2d=lon2d-lon0
    dis=abs(dis_lat2d)+abs(dis_lon2d)
    idx=np.argwhere(dis==dis.min())[0].tolist() # x, y position
    return idx[0], idx[1]


def div_2d(uwnd, vwnd, dx, dy):
    """ 
        Calculate divergence on the rightmost 2 dims of uwnd and vwnd
        given dx and dy (in SI), in staggered mesh
    """
    
    (nz,ny,nx)=uwnd.shape
    nx=nx-1

    div=np.zeros((nz,ny,nx))
    div=(uwnd[:,:,1:nx+1]-uwnd[:,:,0:nx])/dx+(vwnd[:,1:ny+1,:]-vwnd[:,0:ny,:])/dy
    return div




