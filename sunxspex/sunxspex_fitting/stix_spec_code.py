"""
The following code is used to make SRM/counts data in consistent units from STIX spectral data.
"""

import astropy.io.fits as fits
import numpy as np
import warnings

from astropy import units as u
from astropy.time import Time, TimeDelta

from .common_spec_code import SpecFileInfo, SrmFileInfo

__all__ = ["load_spec", "load_srm"]


def load_spec(spec_file: str, time_base: str) -> SpecFileInfo:
    """ Return all STIX data needed for fitting.

    Parameters
    ----------
    spec_file : str
            String for the STIX spectral file under investigation.
    time_base : str
        utc or spacecraft time

    Returns
    -------
    SpecFileInfo object
    """
    with fits.open(spec_file) as sf:
        # get units from the fits columns rather than hard-coding
        slice_unit = lambda hdu, c: u.Unit(sf[hdu].columns[c].unit)

        time_mids = sf['data'].data['time'] << slice_unit('data', 'time')
        dt = sf['data'].data['timedel'] << slice_unit('data', 'timedel')
        # if odd `t` in seconds then edges are [midt-floor(bin_width/2), midt+ceil(bin_width/2)]
        # e.g., mid_t=19, del_t=13 then edges would be [19-6, 19+7]=[13,26]
        t_lo = time_mids - (np.floor(dt.value / 2) << dt.unit)
        t_hi = time_mids + (np.ceil(dt.value / 2) << dt.unit)

        start_date = Time(sf['primary'].header['date-beg'], format='isot', scale='utc')
        if time_base == 'utc':
            solo_earth_light_timedelta = sf['primary'].header['ear_tdel'] << u.s
            start_date += solo_earth_light_timedelta
        elif time_base != 'spacecraft':
            raise ValueError(
                "Please select a STIX timebase from the following: "
                "utc, spacecraft")

        time_bins = np.column_stack((start_date + t_lo, start_date + t_hi))

        channel_bins = np.column_stack((
            sf['energies'].data['e_low'].astype(float),
            sf['energies'].data['e_high'].astype(float)
        )) << slice_unit('energies', 'e_low')
        last_bin = channel_bins[-1, -1]
        if np.isnan(last_bin):
            channel_bins[-1, -1] = 1 << u.MeV
        # TODO re-incorporate bin edge mask if needed
        # seems like it's not needed as of L1 data release v1?

        # Livetime estimation is based on real & estimated count rate
        counts = sf['data'].data['counts'] << slice_unit('data', 'counts')
        triggers = sf['data'].data['triggers']
        trig_comp_err = sf['data'].data['triggers_comp_err']
        num_dets = sf['control'].data['detector_masks'].sum()

        nice_dt = []
        for td in np.diff(time_bins, axis=1).flatten():
            nice_dt.append(td.to(u.s))
        nice_dt = u.Quantity(nice_dt)
        livetime_dat = compute_livetimes(
            triggers=triggers,
            trig_comp_err=trig_comp_err,
            num_dets=num_dets,
            counts=counts,
            time_bins=nice_dt
        )

        counts_error = np.sqrt(
            (counts.value << u.ct**2) +
            (sf['data'].data['counts_comp_err'] << slice_unit('data', 'counts_comp_err'))**2
        )
        warnings.warn(
            "Not incorporating energy edge locations into counts error (STIX loader).\n"
            "Need to add that!"
        )

        de = np.diff(channel_bins, axis=1).flatten()
        # last overflow bin is >150 keV --> make it big
        de[-1] = 1 << u.MeV
        count_rate = (counts.T / (livetime_dat['value'] * nice_dt)).T / de

        livetime_prop_err = livetime_dat['error'] / livetime_dat['value']
        count_rate_error = np.sqrt(
            (count_rate * counts_error / counts)**2 +
            (count_rate.T * livetime_prop_err).T**2
        )

    return SpecFileInfo(
        channel_bins_2d=channel_bins.to_value(u.keV),
        time_bins=time_bins,
        livetime=livetime_dat['value'],
        counts=counts.to_value(u.ct),
        counts_error=counts_error.to_value(u.ct),
        count_rate=count_rate.to_value(u.ct / u.keV / u.s),
        count_rate_error=count_rate_error.to_value(u.ct / u.keV / u.s)
    )


def load_srm(srm_file: str, srm_choice='unattenuated') -> SrmFileInfo:
    """ Return all STIX SRM data needed for fitting.

    SRM units returned as [ct . cm2 / ph].

    Parameters
    ----------
    srm_file : str
            String for the STIX SRM spectral file under investigation.

    Returns
    -------
    SrmFileInfo object
    """
    ATT_OPTS = ('attenuated', 'unattenuated')
    if srm_choice not in ATT_OPTS:
        raise ValueError(f"Select a STIX attenuator state from: {', '.join(ATT_OPTS)}")

    with fits.open(srm_file) as sf:
        # get units from the fits columns rather than hard-coding
        slice_unit = lambda hdu, c: u.Unit(sf[hdu].columns[c].unit)

        # changes for attenuator vs no attenuator
        srm_idx = (1 if srm_choice == 'unattenuated' else 4)
        photon_bins = np.column_stack((
            sf[srm_idx].data['energ_lo'],
            sf[srm_idx].data['energ_hi']
        )) << slice_unit(srm_idx, 'energ_lo')

        # the 1/keV part is from count bins
        srm = sf[srm_idx].data['matrix'] << (u.ct / u.keV / u.ph)
        count_bins = np.column_stack(
            (sf['ebounds'].data['e_min'], sf['ebounds'].data['e_max'])
        ) << slice_unit('ebounds', 'e_min')

        # ...
        area = sf['specresp matrix'].header['geoarea'] << u.cm**2

    return SrmFileInfo(
        photon_bin_edges=photon_bins.to_value(u.keV),
        count_bin_edges=count_bins.to_value(u.keV),
        srm=(srm * np.diff(count_bins, axis=1).flatten() * area).to_value(u.cm**2 * u.ct / u.ph)
    )


def estimate_livetime(triggers, counts_arr, time_bins, num_detectors):
    '''
    Functional form pulled from stixdcpy.
    Updated time constants from correspondence with Hannah Collier.
    '''
    # TODO: adjust for pixel data when we need to
    time_bins = (time_bins << u.s).value

    BETA = 0.94
    FPGA_TAU = 10.1e-6
    # Updated ASIC_TAU from Hannah Collier (July 2023)
    ASIC_TAU = 1.1e-6
    TRIG_TAU = FPGA_TAU + ASIC_TAU
    # STIX detector parameters
    tau_conv_const = 1e-6

    photons_in = triggers / (
        time_bins * num_detectors - TRIG_TAU * triggers
    )
    # photon rate approximated using triggers

    count_rate = counts_arr / time_bins[:, None]
    nin = photons_in

    live_ratio = np.exp(-BETA * nin * ASIC_TAU * tau_conv_const)
    live_ratio /= (1 + nin * TRIG_TAU)
    corrected_rate = count_rate / live_ratio[:, None]
    return {
        'corrected_rate': corrected_rate,
        'count_rate': count_rate,
        'photons_in': photons_in,
        'live_ratio': live_ratio
    }


@u.quantity_input
def compute_livetimes(
    triggers: u.one,
    trig_comp_err: u.one,
    num_dets: int,
    counts: u.ct,
    time_bins: u.s
):
    '''
    Estimate livetime and its error from triggers and counts.
    '''

    lt_high = estimate_livetime(
        triggers=triggers + trig_comp_err,
        counts_arr=counts,
        time_bins=time_bins,
        num_detectors=num_dets,
    )

    lt_low = estimate_livetime(
        triggers=triggers - trig_comp_err,
        counts_arr=counts,
        time_bins=time_bins,
        num_detectors=num_dets
    )

    lt_err = np.abs(lt_low['live_ratio'] - lt_high['live_ratio']) / 2
    lt = estimate_livetime(
        triggers=triggers,
        counts_arr=counts,
        time_bins=time_bins,
        num_detectors=num_dets)

    return {'value': lt['live_ratio'], 'error': lt_err}
