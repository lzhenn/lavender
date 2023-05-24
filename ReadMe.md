# LAVENDER 
![](https://github.com/Novarizark/lavender/blob/master/lavender.png?raw=true)
Lavender is a Lagrangian dispersion model suitable for the simulation of atmospheric transport processes within O(1-100km) spatial scales.

**NOTE:** For the 3D Renderer, it has been tested that python 3.10 with matplotlib 3.7.1 is much slower (8X) than python 3.8 with matplotlib 3.4.3.

### Input Files

#### config.ini
`./conf/config.ini`: Configure file for the model. You may set IO options and interpolation details in this file.


#### obv.csv
`./input/obv.csv`: This file prescribes the in-situ observations. File name can be assigned in `config.ini`.

File format:
```
yyyymmddhhMM,lat,lon,height,wind_speed,wind_dir,temp,rh,pres,attr1,attr2
```

#### power_coef_wind.csv
`./db/power_coef_wind.csv`: Table for near surface layer wind exponent coefficient determined by roughness length - stability level relationship.

### Module Files

#### run.py
`./run.py`: Main script to run Aeolus. 

#### lib

* `./lib/cfgparser.py`: Module file containing read/write funcs of the `config.ini`

* `./lib/preprocess_wrfinp.py`: Class template to construct the field_hdl obj, which contains template grid fields data such as lat, lon, U, V for interpolation

* `./lib/obv_constructor.py`: Class template to construct in-situ observation obj

* `./lib/time_manager.py`: Class template to construct time manager obj

#### core 
`./core/aeolus.py`: Core module, Aeolus interpolator, interpolate in-situ obvs onto wrf mesh

#### utils
`./utils/utils.py`: Commonly used utilities, such as wind convert funcs, grid locationing, and Haversine formula to calculate great circle distance. 

#### doc
Documents related to the model.

#### post_process
Scratch script for visualization.

**Any question, please contact Zhenning LI (zhenningli91@gmail.com)**
