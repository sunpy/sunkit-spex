"""
Example: fitting thermal + thick target model to STIX spectrum

Before running this script, it is neccessary to run the following commands in a terminal:

export HEADAS=/home/stix_public/Documents/heasoft-6.29/x86_64-pc-linux-gnu-libc2.31
. $HEADAS/headas-init.sh


run this script from the sunxspex/examples/example_data directory of https://github.com/elastufka/sunxspex

eg:
%%bash
conda activate py37 #scientific python environment
cd ~/Solar/sunxspex/examples/example_data
python ../pyxspec_example.py

"""
from astropy.io import fits
from sunxspex import sun_xspec
import time
import xspec
#import logging

#logging.basicConfig(filename='pyxspec_example.log', level=logging.DEBUG)

t0=time.time()
thick=sun_xspec.ThickTargetModel()
thick.print_ParInfo()

xspec.AllModels.addPyMod(thick.model, thick.ParInfo, 'add')
xspec.AllData.clear()

xspec.AllData("1:1 stx_spectrum_20210908_1712.fits{13}")
xspec.Xset.abund="file feld92a_coronal0.txt"

fit1start=2.0 #keV
fit1end=10.0 #keV
print(f"Fitting with single thermal model initially over range {fit1start}-{fit1end} keV")

m1=xspec.Model('apec')
m1.show()

xspec.AllData.ignore(f"0.-{fit1start} {fit1end}-**")
xspec.Fit.statMethod = "chi"
xspec.Fit.nIterations=100
tstart=time.time()
xspec.Fit.perform()
print(f"Single thermal fit took {time.time()-tstart:.3f} seconds")

apec_kt=m1.apec.kT.values
apec_norm=m1.apec.norm.values
m2=xspec.Model('apec+thick2')
m2.setPars({1:f"{apec_kt[0]:.2f} -.1,,,"})
m2.setPars({4:f"{apec_norm[0]:.2f} -.1,,,"})
m2.setPars({9:"3.4 -.1,,,10"}) #set so it cannot exceed eebrk and cause error
m2.setPars({7:"16.7 -.1,,,18"})
for p in ['p','eebrk','q','eelow','a0']:
    pp=getattr(m2.thick2,p)
    pp.frozen=False
m2.show()

fit2start=10.0 #keV
fit2end=30.0 #keV
print(f"New moodel: single thermal fit with previous fit parameters (kT={apec_kt[0]:.2f}, norm={apec_norm[0]:.2f}) plus thick target model initially over range {fit2start}-{fit2end} keV")

xspec.AllData.notice("3.0-50.0") #could just use this instead of new ignore command
xspec.AllData.ignore(f"0.-{fit2start} {fit2end}-**")
tstart=time.time()
xspec.Fit.perform()
print(f"Thick target fit took {time.time()-tstart:.3f} seconds")
print(f"Chisq: {xspec.Fit.statistic:.3f}")

pp=getattr(m2.apec,'kT')
pp.frozen=False
pp=getattr(m2.apec,'norm')
pp.frozen=False

m2.show()

xspec.AllData.notice("3.0-50.0")
fit3start=3.0 #keV
fit3end=30.0 #keV
print(f"Single thermal plus thick target model fit over range {fit3start}-{fit3end} keV, all parameters free")

xspec.AllData.ignore(f"0.-{fit2start} {fit2end}-**")
tstart=time.time()
xspec.Fit.perform()
print(f"Thermal + thick target fit took {time.time()-tstart:.3f} seconds")
print(f"Chisq: {xspec.Fit.statistic:.3f}")
print()

#plot in matplotlib...eventually

print(f"Script run time: {time.time()-tstart:.3f} seconds")
