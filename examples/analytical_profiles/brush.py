"""
Various Analytic profiles for studying lipid membranes at an interface
"""

import numpy as np
from refnx.analysis import AnalyticalReflectivityFunction
from numpy.testing import assert_


class BrushPara(AnalyticalReflectivityFunction):
    """
    Order of layers:

    superphase (Si?)
    native SiO2
    sticker layer (could be permalloy, could be Cr)
    Au layer
    inner heads
    chain region, assumed to be symmetric
    outer heads
    subphase

    This model assumes that the Area per Molecule is identical in the inner
    and outer leaflets. This model can easily be changed to remove that
    assumption.
    """

    def __init__(self, sld_poly, n_interior, *args, n_slices=50, **kwds):
        """
        Parameters
        ----------
\        """
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

            # If first iteration loop, set roughness between slab and tail, else apply an arbitrary smoothing roughness
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

    def vol_fraction(self, params):
        pass

    def adsorbed_amount(self, params):
        pass