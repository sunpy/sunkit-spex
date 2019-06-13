import numpy as np 
from acgaunt import Acgaunt
from astropy import units as u

@u.quantity_input(energy=u.keV, temperature=u.keV)
def brem_49(energy, temperature):
	"""
	The function calculates the optically thin continuum thermal bremmstrahlung
	photon flux incident on the Earth from an isothermal plasma on the Sun.
	Normalization is for an emission measure on the sun of 1.e49cm^-3

	Catagory Spectra
	
	Parameters
	----------
	energy : `~astropy.Quantity`
		energy array in keV
	temperature : `~astropy.Quantity` 
		 plasma temperature in keV

	Returns
	-------
	Differential photon flux in units of photons/(cm2 s keV) per (1e49 cm-3 emission measure)

	Examples
	--------
	>>> flux = brem_49(energy, kt)

	Notes
	-----
	Calls acgaunt.py

	"""

	acgaunt = Acgaunt(energy.to(u.angstrom, equivalencies=u.spectral()), temperature.to(u.Kelvin, equivalencies=u.temperature_energy()))
	exponential_values = (energy/temperature)[energy/temperature < 50]

	result = (1.e8/9.26) * acgaunt.acgaunt() * np.exp(-exponential_values) / energy / temperature ** 0.5


	return result.T