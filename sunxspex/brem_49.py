import numpy as np 
import acgaunt

def brem_49(kt, E):
	"""
	The function calculates the optically thin continuum thermal bremmstrahlung
	photon flux incident on the Earth from an isothermal plasma on the Sun.
	Normalization is for an emission measure on the sun of 1.e49cm^-3

	Catagory Spectra
	
	Parameters
	----------
	Energy : energy vector in keV
	kt 	   : plasma temperature in keV

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

	kt0 = (kt[0] > 0.1)
	result  = (1.e8/9.26) * float(acgaunt(12.3985/E, kt0/0.08617)) * np.exp(-(E/kt0 < 50)) / E / kt0**0.5

	return result