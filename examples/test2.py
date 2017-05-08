from __future__ import print_function, division
from refnx.dataset.reflectdataset import ReflectDataset
import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter
import numpy as np

DATASET_NAME = 'c_PLP0011859_q.txt'
USE_DIFFERENTIAL_EVOLUTION = True

#load the data
dataset = ReflectDataset()
with open(DATASET_NAME) as f:
    dataset.load(f)

xdata, ydata, dydata, dxdata = dataset.data

# create an array to hold the parameters, also create limits
layers = np.array([[0,   2.07, 0, 0],     # fronting medium
                   [30,  3.47, 0, 3],     # 1st layer
                   [250, 2.00, 0, 3],     # 2nd layer
                   [0,   6.36, 0, 3]])     # backing medium

lowlim = np.array([[0,   2.07, 0, 0],     # fronting medium
                   [15,  3.47, 0, 1],     # 1st layer
                   [200, 0.10, 0, 1],     # 2nd layer
                   [0,   6.36, 0, 1]])     # backing medium

hilim = np.array([[0,   2.07, 0, 0],     # fronting medium
                  [50,  3.47, 0, 9],     # 1st layer
                  [300, 3.00, 0, 9],     # 2nd layer
                  [0,   6.36, 0, 9]])     # backing medium

# create a linear array of the parameters
# coefs[1] is the scale factor
# coefs[6] is the background
# these will both be 1 and 0 respectively to start off with
coefs = reflect.layer_to_coefs(layers)
lowlim = reflect.layer_to_coefs(lowlim)
hilim = reflect.layer_to_coefs(hilim)

coefs[1] = 1.0
coefs[6] = 3.e-6

lowlim[1] = 0.6
hilim[1] = 1.2

lowlim[6] = 0.
hilim[6] = 9e-6

bounds = zip(lowlim, hilim)

# create a parameter instance
parameters = curvefitter.to_parameters(coefs, bounds=bounds, varies=[False] * 16)

# which parameters do you want to allow to vary
fitted_parameters = np.array([1, 6, 7, 8, 11, 12, 13, 15])
for fit in fitted_parameters:
    parameters['p%d' % fit].vary = True

# use resolution smearing and fit on a logR scale (transform the data as well)
fitter = reflect.ReflectivityFitFunction(xdata, ydata, parameters, edata=dydata,
                                    fcn_kws = {'dqvals': dxdata})

fitter.model(parameters)
