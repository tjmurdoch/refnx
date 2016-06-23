"""
Various Analytic profiles for studying lipid membranes at an interface
"""

import numpy as np
from refnx.analysis import AnalyticalReflectivityFunction, Transform
from scipy.integrate import simps, trapz
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from lmfit import Parameter, Parameters
import os.path


class BrushPara(AnalyticalReflectivityFunction):
    """

    """

    def __init__(self, sld_poly, n_interior, *args,
                 n_slices=50, **kwds):
        """
        Parameters
        ----------
        """
        super(BrushPara, self).__init__(*args, **kwds)
        self.n_interior = n_interior
        self.n_slices = n_slices
        self.sld_poly = sld_poly

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

        # Parabola Layers
        slab_thick = lmfit_values['tail_thickness'] / self.n_slices
        for i in range(self.n_slices):
            distance = (i + 0.5) * slab_thick
            phi = lmfit_values['phi_init'] *  (1 - (distance / lmfit_values['tail_thickness'])**2)
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
                   'SLD_super', 'SLD_sub', 'thickness_SiO2', 'SLD_SiO2',
                   'roughness_SiO2', 'roughness_backing']

        tail_par = ['phi_init','tail_thickness','roughness_tail2int']

        int_par = ",".join(['thickness_%d,phi_%d,roughness_%d' %
                           (i+1, i+1, i+1) for i in range(self.n_interior)]).split(',')

        return gen_par + tail_par + int_par

    # def params_test(self,params):
    #
    #     lmfit_values = params.valuesdict()
    #     test = lmfit_values['roughness_SiO2']
    #
    #     test2 = 2
    #     lmfit_values['roughness_SiO2'] = test2
    #     return test, test2

    def vol_fraction(self, params):
        """
        Calculates SLD profile and sets boundary between SiO2 and polymer to
        z = 0. Then calculates volume fraction profile from additive mixing of
        SLD Some guff here
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

        params['roughness_1'].val = 0
        # print(lmfit_values['roughness_1'])

        z, profile = self.sld_profile(params)
        end = max(z)

        points = np.linspace(lmfit_values['thickness_SiO2'], end, num=1001)
        z, profile = self.sld_profile(params, points=points)

        profile = (profile - lmfit_values['SLD_sub']) / (self.sld_poly - lmfit_values['SLD_sub'])
        z = z - lmfit_values['thickness_SiO2']
        params['roughness_1'].val = roughness

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

    def moment(self, params, moment=1):
        """
        Calculates the n'th moment of the volume fraction profile

        Parameters
        ----------
        params
        moment

        Returns
        -------

        """
        points, profile = self.vol_fraction(params)
        profile *= points**moment
        val = simps(profile, points)
        area = self.adsorbed_amount(params)
        return val / area


if __name__ == "__main__":
    # load in some previously calculated data (from IGOR) for a test
    path = os.path.dirname(os.path.abspath(__file__))
    igor_r, igor_q = np.hsplit(np.loadtxt(os.path.join(path, 'brush_para.txt')), 2)

    transform = Transform('logY').transform
    brush = BrushPara(0.46, 3, transform=transform, dq=0)

    names = brush.parameter_names()
    # these are the parameters we used in our IGOR analysis
    vals = [1, 1e-7, 2.07, 6.36, 8.8, 3.47, 3.5, 10, 0.1, 1000, 4,
            28, 0.95, 2, 50, 0.85, 2, 100, 0.2, 20]

    # igor_params = {'scale': 1, 'bkg': 1e-7}

    P = Parameters()
    for name, val in zip(names, vals):
        P.add(name, val, True)

    ref = brush.model(igor_q, P)
    assert_almost_equal(ref,igor_r)

    area = brush.adsorbed_amount(P)
    assert_allclose(area, 155.40491, atol=1e-2)

    first_moment = brush.moment(P)
    assert_allclose(2 * first_moment, 542.9683913)
