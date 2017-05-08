# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:45:43 2017

@author: igres

"It would be good to write some code that can generate a spline profile and slice it up into the array needed for refnx.
I image we have would add “class BrushSpline(Brush)” that could take an argument to select the type of spline used
(e.g. monotone vs. B-spline with end point control)."
"""

from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt


class Brush_SLD:
    """
    
    """
    
    def __init__(self, number_knots, layer_thickness):
        
        #Adds knots to either side of the range in order to ensure that the spline is
        #definined over the entire range. This has been tested up to 100,000 splines.
        self.number_knots_used = number_knots + 4 #Add two waste knots to each end to ensure spline is defined
        self.number_knots_total = self.number_knots_used + 4 #Length of knot array must be n - k - 1 (k is set to 3 by default)
        self.layer_thickness = layer_thickness
        self.knot_spacing = layer_thickness/number_knots
        self.knots = np.arange(-4*self.knot_spacing, self.layer_thickness + 4*self.knot_spacing, self.knot_spacing)
        self.number_coefs = self.number_knots_used
        self.coefs = np.zeros(self.number_coefs)
        self.knot_mobile = np.ones(self.number_knots_used).astype(bool)
        
        self.tail = 0
        
        print("Initialised Brush_SLD object")
    
    def profile (self, z_axis):
        SLD = np.zeros(len(z_axis))
        
        #Spline Section
        spline_mask = np.logical_and(z_axis < self.layer_thickness, z_axis >= 0)
        spl = BSpline(self.knots, self.coefs, 3, extrapolate=False) 
        SLD[spline_mask] = spl(z_axis[spline_mask])

        #Tail:
        tail_mask = z_axis >= self.layer_thickness
        SLD[tail_mask] = self.tail(z_axis[tail_mask]-self.layer_thickness, SLD[spline_mask][-1])
              
        return SLD
    
    def define_coefs (self, coefs=1):
        if type(coefs) == int:
            self.coefs = np.ones(self.number_coefs)
        elif type(coefs) == np.ndarray:
            coefs_l = len(coefs) 
            if coefs_l == self.number_coefs:
                self.coefs = np.array(coefs)
            else:
                coefs = np.interp(np.arange(0, coefs_l, coefs_l/self.number_coefs), np.arange(0, coefs_l), coefs)
                assert len(coefs) == self.number_coefs , "Function define_coefs in class Brush_SLD Failed to scale inputted spline coefficent array to desired size (number of used knots)"
            self.coefs[self.knot_mobile] = coefs[self.knot_mobile]

    
    def set_backing_SLD(self, backing_SLD, deriv_zero=False):
        if deriv_zero == False:
            self.knot_mobile[0:3] = False
            self.coefs[0:3] = backing_SLD
        else:
            self.knot_mobile[0:4] = False
            self.coefs[0:4] = backing_SLD
                  
    def set_fronting_SLD(self, fronting_SLD, deriv_zero=False):
        if deriv_zero == False:
            self.knot_mobile[-2:] = False
            self.coefs[-2:] = fronting_SLD
        else:
            self.knot_mobile[-3:] = False
            self.coefs[-3:] = fronting_SLD
                      
    def add_gaussian_tail(self, decay_to, sigma=1):
        self.tail = lambda x, decay_from: decay_to-(decay_to - decay_from)*(np.exp(-(x)**2/(2*sigma**2))) 
                      
                     
    


num_splines = 15
z_start = 0
z_end = 60
domain = [z_start, z_end]
z_range = z_end - z_start
z_axis = np.arange(z_start, z_end, 0.1)  

         
BSLD = Brush_SLD(num_splines, 40)
BSLD.set_backing_SLD(2.07, True)
#BSLD.set_fronting_SLD(4, True)
BSLD.define_coefs(np.array([3,3,3,1.5,1.5]))
BSLD.add_gaussian_tail(6.4, 3)

SLD = BSLD.profile(z_axis)

plt.plot(z_axis, SLD)

plt.show()


        










