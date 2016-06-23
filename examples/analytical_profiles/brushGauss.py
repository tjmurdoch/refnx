"""
Various Analytic profiles for studying lipid membranes at an interface
"""

import numpy as np
from refnx.analysis import AnalyticalReflectivityFunction
from scipy.integrate import simps
from numpy.testing import assert_


class BrushGauss(AnalyticalReflectivityFunction):
    """

    """

    def __init__(self, sld_poly, n_interior, *args, n_slices=50,vol_cut = 0.005, **kwds):
        """
        Parameters
        ----------
\        """
        super(BrushGauss, self).__init__(*args, **kwds)
        self.n_interior = n_interior
        self.n_slices = n_slices
        self.sld_poly = sld_poly
        self.vol_cut = vol_cut

    def to_slab(self, params):
        """
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        slab_model: np.ndarray
            Parameters for a slab-model reflectivity calculation
        """
        # with this model there are 6 layers, and the reflectivity calculation
        # needs to be done with 4*N + 8 = 32 variables
        lmfit_values = params.valuesdict()

        # Interior Slabs, Analytical Slices and SiO2
        n_layers = self.n_interior + self.n_slices + 1
        n_par = 4*n_layers + 8

        slab_model = np.zeros((n_par,), float)
        slab_model[0] = n_layers
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_backing']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_SiO2']
        slab_model[9] = lmfit_values['SLD_SiO2']
        slab_model[11] = lmfit_values['roughness_SiO2']

        overall_SLD = lambda vf1: vf1 * self.sld_poly + (1 - vf1) * lmfit_values['SLD_sub']

        # Interior Layers
        for i in range(self.n_interior):
            slab_model[12 + 4*i] = lmfit_values['thickness_{}'.format(i + 1)]
            slab_model[13 + 4*i] = overall_SLD(lmfit_values['phi_{}'.format(i + 1)])
            slab_model[15 + 4*i] = lmfit_values['roughness_{}'.format(i + 1)]

        # Gaussian Layers
        cutoff_thickness = np.sqrt(-lmfit_values['tail_thickness']**2 * np.log(self.vol_cut / lmfit_values['phi_init']))
        slab_thick = cutoff_thickness / self.n_slices
        for i in range(self.n_slices):
            distance = (i + 0.5) * slab_thick
            phi = lmfit_values['phi_init'] *  np.exp(-(distance / lmfit_values['tail_thickness'])**2)
            slab_model[12 + 4 * (self.n_interior + i)] = slab_thick
            slab_model[13 + 4 * (self.n_interior + i)] = overall_SLD(phi)

            # If first iteration of loop, set roughness between slab and tail,
            # else apply an arbitrary smoothing roughness
            if not i:
                slab_model[15 + 4 * (self.n_interior + i)] = lmfit_values['roughness_tail2int']
            else:
                slab_model[15 + 4 * (self.n_interior + i)] = slab_thick / 3
        return slab_model

    def parameter_names(self, nparams=None):
        gen_par = ['scale', 'bkg',
                 'SLD_super', 'SLD_sub', 'thickness_SiO2', 'SLD_SiO2', 'roughness_SiO2',
                 'SLD_poly', 'adsorbed_amount', 'roughness_backing']

        tail_par = ['phi_init','tail_thickness','roughness_tail2int']

        int_par = ",".join(['thickness_%d,phi_%d,roughness_%d' % (i+1, i+1, i+1) for i in range(self.n_interior)]).split(',')

        return gen_par + tail_par + int_par

    def params_test(self,params):

        lmfit_values = params.valuesdict()
        test = lmfit_values['roughness_SiO2']

        test2 = 2
        lmfit_values['roughness_SiO2'] = test2
        return test, test2

    def vol_fraction(self, params):
        """
        Calculates SLD profile and sets boundary between SiO2 and polymer to z = 0. Then calculates volume fraction
        profile from additive mixing of SLD
        Some guff here
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        z: z value for volume fraction profile
        profile: phi values for volume fraction profile

        """
        lmfit_values = params.valuesdict()
        roughness = lmfit_values['roughness_1']
        low_number = 500
        # Don't know how to change the actual values
        lmfit_values['roughness_1'] = low_number
        print(lmfit_values['roughness_1'])
        z,profile = self.sld_profile(params)
        end = max(z)
        # range = np.linspace(lmfit_values['thickness_SiO2'],end, num=101)
        range = np.linspace(lmfit_values['thickness_SiO2'], end, num=101)
        z,profile = self.sld_profile(params, points = range)
        profile = (profile - lmfit_values['SLD_sub']) / (lmfit_values['SLD_poly'] - lmfit_values['SLD_sub'])
        z = z - lmfit_values['thickness_SiO2']
        # lmfit_values['roughness_1'] = roughness

        return z, profile



    def adsorbed_amount(self, params):
        """

        Parameters
        ----------
        params

        Returns
        -------

        """
        points, profile = self.vol_fraction(params)
        area = simps(profile, points)
        return area