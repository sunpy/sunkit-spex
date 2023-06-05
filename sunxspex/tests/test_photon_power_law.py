import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np

from sunxspex import photon_power_law as ppl


''' i know this isn't really a test but it shows that things are working '''


def basic_plot_test():
    paramz = dict()
    paramz['energy_edges'] = (edges := np.logspace(0, 2, num=40) << u.keV)
    paramz['reference_energy'] = 50 << u.keV
    paramz['reference_flux'] = 10 << ppl.PHOTON_RATE_UNIT
    paramz['break_energy'] = 20 << u.keV
    paramz['lower_index'] = 1
    paramz['upper_index'] = 6

    flux = ppl.broken_power_law_binned_flux(**paramz)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stairs(flux.value, edges.value)
    ax.set(
        xlabel=edges.unit,
        ylabel=flux.unit,
        xscale='log',
        yscale='log',
        title='Broken power law'
    )
    plt.show()


if __name__ == '__main__':
    basic_plot_test()
