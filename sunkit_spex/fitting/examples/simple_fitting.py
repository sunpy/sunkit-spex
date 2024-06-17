"""
This is a file to show...

Things assumed:
* Square response where the count and photon energy axes are identical
"""

from astropy.modeling import fitting
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sunkit_spex.fitting.models.data.gaussian import GaussianCountModel
from sunkit_spex.fitting.models.matrix.matrix import MatrixModel
from sunkit_spex.fitting.objective_functions.optimising_functions import minimize_func
from sunkit_spex.fitting.optimizer_tools.minimizer_tools import scipy_minimize
from sunkit_spex.fitting.models.physical.gaussian import GaussianPhotonModel 
from sunkit_spex.fitting.models.physical.straight_line import StraightLinePhotonModel
from sunkit_spex.fitting.statistics.gaussian import chi_squared


# this should keep "random" stuff is the same ech run
np.random.seed(seed=10)

# get some fake photon energies
def photon_energies(start, stop, inc):
    """ Get a `ndarray` of energies. """
    return np.arange(start, stop, inc)

def plot_fake_photon_spectrum(axis, 
                              photon_energies, 
                              photon_values, 
                              title="Fake Photon Spectrum"):

    axis.plot(photon_energies, photon_values)
    axis.set_xlabel("Energy [keV]")
    axis.set_ylabel("ph s$^{-1}$ cm$^{-2}$ keV$^{-1}$")
    axis.set_title(title)

def response_matrix(photon_energies):
    # fake SRM
    fake_srm = np.identity(photon_energies.size)

    # add some off-diagonal terms
    for c, r in enumerate(fake_srm):
        # add some features into the fake SRM
        off_diag = np.random.rand(c)*0.005
        
        # add a diagonal feature
        _x = 50
        if c>=_x:
            off_diag[-_x] = np.random.rand(1)
        
        # add a vertical feature in 
        _y = 200
        __y = 30
        if c>_y+100:
            off_diag[_y-__y//2:_y+__y//2] = np.arange(2*(__y//2))*np.random.rand(2*(__y//2))*5e-4
            
        # put these features in the fake_srm row and normalize
        r[:off_diag.size] = off_diag
        r /= np.sum(r)

    return fake_srm

def plot_fake_srm(axis, fake_srm, photon_energies, title="Fake SRM"):

    axis.imshow(fake_srm, 
                origin="lower", 
                extent=[photon_energies[0], 
                        photon_energies[-1], 
                        photon_energies[0], 
                        photon_energies[-1]],
                norm=LogNorm())
    axis.set_ylabel("Photon Energies [keV]")
    axis.set_xlabel("Count Energies [keV]")
    axis.set_title(title)

def plot_fake_count_spectrum(axis, 
                             photon_energies, 
                             photon_model_features, 
                             gaussian_feature, 
                             total_count_spectrum, 
                             total_count_spectrum_wnoise,
                             title="Fake Count Spectrum"):

    axis.plot(photon_energies, photon_model_features, label="photon model features")
    axis.plot(photon_energies, gaussian_feature, label="gaussian feature")
    axis.plot(photon_energies, total_count_spectrum, label="total fake spectrum")
    axis.plot(photon_energies, total_count_spectrum_wnoise, label="total fake spectrum + noise")
    axis.set_xlabel("Energy [keV]")
    axis.set_ylabel("cts s$^{-1}$ keV$^{-1}$")
    axis.set_title(title)
    axis.legend()

def plot_fake_count_spectrum_fit(axis, 
                                 total_count_spectrum_wnoise,
                                 fitted_count_spectrum,
                                 title="Fake Count Spectrum Fit"):

    axis.plot(photon_energies, total_count_spectrum_wnoise, label="total fake spectrum + noise")
    axis.plot(ph_energies, fitted_count_spectrum, ls=":", label="model fit")
    axis.set_xlabel("Energy [keV]")
    axis.set_ylabel("cts s$^{-1}$ keV$^{-1}$")
    axis.set_title(title)
    axis.legend()

def plot_table_of_results(ax, row_labels, column_labels, cell_text):
    ax.axis("off")
    ax.table(cellText=cell_text, 
             cellColours=None, 
             cellLoc='center', 
             rowLabels=row_labels, 
             rowColours=None, 
             colLabels=column_labels, 
             colColours=None, 
             colLoc='center', 
             bbox=[0,0,1,1])

if __name__=="__main__":
    #*******************************************************************
    # Start by creating fake data and instrument.
    # This would all be provided by a given observation.
    #*******************************************************************

    # define the photon energies
    start, inc = 1.6, 0.04
    stop = 80+inc/2
    ph_energies = photon_energies(start, stop, inc)

    # let's start making a fake photon spectrum
    fake_cont = {"m":-1, "c":100}
    fake_line = {"a":100, "b":30, "c":2}
    # use a straight line model for a continuum, Gaussian for a line
    ph_model = StraightLinePhotonModel(**fake_cont) + \
        GaussianPhotonModel(**fake_line)

    # now want a response matrix
    srm = response_matrix(ph_energies)
    srm_model = MatrixModel(matrix=srm)

    # now work on a count model
    fake_gauss = {"a":70, "b":40, "c":2}
    # the brackets are very necessary
    ct_model = (ph_model | srm_model) + GaussianCountModel(**fake_gauss)

    # generate fake count data to (almost) fit
    fake_count_model = ct_model(ph_energies)
    # add some noise
    fake_count_model_wn = fake_count_model + \
        (2*np.random.rand(fake_count_model.size)-1)*np.sqrt(fake_count_model)
    
    #*******************************************************************
    # Now we have the fake data, let's start setting up to fit it
    #*******************************************************************

    # get some initial guesses that are off from the fake data above
    guess_cont = {"m":-0.5, "c":80} # original {"m":-1, "c":100}
    guess_line = {"a":150, "b":32, "c":5} # original {"a":100, "b":30, "c":2}
    guess_gauss = {"a":350, "b":39, "c":0.5} # original {"a":70, "b":40, "c":2}

    # define a new model since we have a rough idea of the mode we should use
    ph_mod_4fit = StraightLinePhotonModel(**guess_cont) + \
        GaussianPhotonModel(**guess_line)
    count_model_4fit = (ph_mod_4fit | srm_model) + \
        GaussianCountModel(**guess_gauss)

    # let's fit the fake data
    opt_res = scipy_minimize(minimize_func, 
                             count_model_4fit.parameters, 
                             (fake_count_model_wn, 
                              ph_energies,
                              count_model_4fit,
                              chi_squared))
    
    #*******************************************************************
    # Now try and fit with Astropy native fitting infrastructure
    #*******************************************************************

    # try and ensure we start fresh with new model definitions
    ph_mod_4astropyfit = StraightLinePhotonModel(**guess_cont) + \
        GaussianPhotonModel(**guess_line)
    count_model_4astropyfit = (ph_mod_4fit | srm_model) + \
        GaussianCountModel(**guess_gauss)

    astropy_fit = fitting.LevMarLSQFitter()

    astropy_fitted_result = astropy_fit(count_model_4astropyfit, 
                                        ph_energies, 
                                        fake_count_model_wn)
    
    #*******************************************************************
    # Plot the results
    #*******************************************************************

    fig = plt.figure(layout="constrained")

    gs = GridSpec(2, 3, figure=fig)

    # plto the fake photon spectrum to be converted to count-space
    ax1 = fig.add_subplot(gs[0, 0])
    plot_fake_photon_spectrum(ax1, 
                              photon_energies, 
                              ph_model(ph_energies), 
                              title="Fake Photon Spectrum")
    
    # the fake response
    ax2 = fig.add_subplot(gs[0, 1])
    plot_fake_srm(ax2, srm, photon_energies, title="Fake SRM")

    # the count spectrum to fir with it's components highlighted
    ax3 = fig.add_subplot(gs[0, 2])
    plot_fake_count_spectrum(ax3, 
                             photon_energies, 
                             (ph_model | srm_model)(ph_energies), 
                             GaussianCountModel(**fake_gauss)(ph_energies), 
                             fake_count_model, 
                             fake_count_model_wn,
                             title="Fake Count Spectrum")
    
    # the count spectrum fitted with Scipy
    ax4 = fig.add_subplot(gs[1, 0])
    plot_fake_count_spectrum_fit(ax4, 
                                 fake_count_model_wn,
                                 count_model_4fit.evaluate(ph_energies, *opt_res.x),
                                 title="Fake Count Spectrum Fit with Scipy")
    
    # the count spectrum fitted with Astropy
    ax5 = fig.add_subplot(gs[1, 1])
    plot_fake_count_spectrum_fit(ax5, 
                                 fake_count_model_wn,
                                 astropy_fitted_result(ph_energies),
                                 title="Fake Count Spectrum Fit with Astropy")
    
    # the fitted value result compared to true values
    ax6 = fig.add_subplot(gs[1, 2])
    row_labels = (tuple(fake_cont)+tuple(fake_line)+tuple(fake_gauss))
    column_labels = ("True Values", "Guess Values", "Scipy Fit", "Astropy Fit")
    true_vals = np.array(tuple(fake_cont.values())+\
                         tuple(fake_line.values())+\
                         tuple(fake_gauss.values()))
    guess_vals = np.array(tuple(guess_cont.values())+\
                          tuple(guess_line.values())+\
                          tuple(guess_gauss.values()))
    scipy_vals = opt_res.x
    astropy_vals = astropy_fitted_result.parameters
    cell_vals = np.vstack((true_vals,guess_vals,scipy_vals,astropy_vals)).T
    cell_text = np.round(np.vstack((true_vals,guess_vals,scipy_vals,astropy_vals)).T, 2).astype(str)
    plot_table_of_results(ax6, row_labels, column_labels, cell_text)

    plt.show()
