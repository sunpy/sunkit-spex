"""
This module contains functions to visualise the results obtained via fitting
"""

import astropy.units as u
from astropy.modeling import fitting
from astropy.modeling import models
from matplotlib import pyplot as plt

import numpy as np

from sunkit_spex.models.physical.thermal import ThermalEmission
from sunkit_spex.models.physical.nonthermal import ThickTarget
from sunkit_spex.models.physical.albedo import Albedo
from sunkit_spex.models.scaling import InverseSquareFluxScaling
from sunkit_spex.models.instrument_response import MatrixModel

__all__ = ["plot"]


def plot(count_edges,photon_edges,observed_counts,observed_counts_err,model,save_name,fit_times,cov,spec_obj):

    count_centers = count_edges[:-1] + 0.5*np.diff(count_edges)

    compound_model_evaluation = model(photon_edges)
    
    count_bin_widths = 0.5*np.diff(count_edges)

    model_names = list(model.submodel_names)

    norm = np.diff(spec_obj.spectral_axis._bin_edges) * spec_obj.meta['exposure_time'].value

    x_unit = count_edges.unit
    # y_unit = observed_counts.unit
    y_unit = r'$\mathrm{ct \: s^{-1} \: keV^{-1}}$'

    left, bottom, width, height = 0,0,1,1
    spacing = 0.005

    w,h=14,10

    plt.rcParams.update({
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16 # This sets both x and y axis label sizes
    })

    tfs = 20

    colour_tot = 'k'
    colour_thermal = 'C3'
    colour_nonthermal = 'C0'
    colour_albedo = 'C2'

    fig = plt.figure(figsize=(w,h))

    rect_dat = [left,bottom+(1/4)*height+1*spacing,width,(3/4)*height]
    rect_rat = [left,bottom,width,(1/4)*height]

    ax_dat = plt.axes(rect_dat)
    ax_rat = plt.axes(rect_rat,sharex=ax_dat)

    ax_dat.errorbar(count_centers,observed_counts/norm,label='Observed Data',marker='None',ms=5,linestyle='None',yerr=observed_counts_err/norm,xerr=count_bin_widths, color='grey',elinewidth=2)
    ax_dat.stairs(compound_model_evaluation.value/norm,count_edges.value,baseline=None, label='Total', linewidth=2,alpha=0.75,zorder=10000,color='k')

    dict = {}

    if 'ThermalEmission' in model_names and 'ThickTarget' in model_names:

        eval_noalbedo = ((((model['ThermalEmission'] + model['ThickTarget']) * model['InverseSquareFluxScaling'] ) ) | model['SRM'])(photon_edges)
        eval_albedo = ((((model['ThermalEmission'] + model['ThickTarget']) * model['InverseSquareFluxScaling'] ) | model['Albedo'] ) | model['SRM'])(photon_edges)
        eval_thermal = (((model['ThermalEmission'] * model['InverseSquareFluxScaling'] ) ) | model['SRM'])(photon_edges)
        eval_nonthermal = (((model['ThickTarget'] * model['InverseSquareFluxScaling'] )  ) | model['SRM'] )(photon_edges)

        albedo = eval_albedo - eval_noalbedo

        ax_dat.stairs(eval_thermal.value/norm,count_edges.value,label='ThermalEmission',baseline=None,alpha=0.9,zorder=1000000, color = colour_thermal,linewidth=1.5)
        ax_dat.stairs(eval_nonthermal.value/norm,count_edges.value,label='ThickTarget',baseline=None,alpha=0.9,zorder=100000, color = colour_nonthermal,linewidth=1.5)
        ax_dat.stairs(albedo.value/norm,count_edges.value,label='Albedo',baseline=None,alpha=0.9,zorder=100000,color=colour_albedo,linewidth=1)

    # Predefine unit strings for clarity
    em_unit = r"$\mathrm{\times \: 10^{49} \: cm^{-3}}$"
    eflux_unit = r"$\mathrm{\times \: 10^{35} \: s^{-1}}$"

    err = np.sqrt(np.diag(cov))

    t_err, em_err = err[0],err[1]
    p_err, ec_err, ef_err =  err[2],err[3],err[4]

    x_loc = 0.67
    y_loc = 0.75 

    # Thermal parameters
    ax_dat.text(
        x_loc, y_loc,
        rf"T = {np.round(model.temperature_0.value, 1)} $\mathrm{{\pm}}$ {np.round(t_err,1)} {model.temperature_0.unit}",
        transform=ax_dat.transAxes,
        fontsize=tfs,
        color=colour_thermal
    )

    ax_dat.text(
        x_loc, y_loc-0.05,
        f"EM = {np.round(model.emission_measure_0.value, 3)} $\mathrm{{\pm}}$ {np.round(em_err,2)} {em_unit}",
        transform=ax_dat.transAxes,
        fontsize=tfs,
        color=colour_thermal
    )

    # Nonthermal parameters
    ax_dat.text(
        x_loc, y_loc-(2*0.05),
        rf"$\mathrm{{\delta}}$ = {np.round(model.p_1.value, 1)}  $\mathrm{{\pm}}$ {np.round(p_err,2)} ",
        transform=ax_dat.transAxes,
        fontsize=tfs,
        color=colour_nonthermal
    )

    ax_dat.text(
        x_loc, y_loc-(3*0.05),
        rf"$\mathrm{{E_{{cutoff}}}}$  = {np.round(model.low_e_cutoff_1.value, 1)}  $\mathrm{{\pm}}$ {np.round(ec_err,1)}  {model.low_e_cutoff_1.unit}",
        transform=ax_dat.transAxes,
        fontsize=tfs,
        color=colour_nonthermal
    )

    ax_dat.text(
        x_loc, y_loc-(4*0.05),
        rf"$\mathrm{{e_{{flux}}}}$  = {np.round(model.total_eflux_1.value, 1)}  $\mathrm{{\pm}}$ {np.round(ef_err,1)}  {eflux_unit}",
        transform=ax_dat.transAxes,
        fontsize=tfs,
        color=colour_nonthermal
    )

    # If only ThermalEmission
    if 'ThermalEmission' in model_names and 'ThickTarget' not in model_names:

        eval_noalbedo = ((model['ThermalEmission'] * model['InverseSquareFluxScaling']) | model['SRM'])(photon_edges)
        eval_albedo = ((model['ThermalEmission'] * model['InverseSquareFluxScaling'] | model['Albedo']) | model['SRM'])(photon_edges)
        eval_thermal = ((model['ThermalEmission'] * model['InverseSquareFluxScaling']) | model['SRM'])(photon_edges)

        albedo = eval_albedo - eval_noalbedo

        ax_dat.stairs(
            eval_thermal.value/norm,
            count_edges.value,
            label='ThermalEmission',
            baseline=None,
            linewidth=2,
            alpha=0.9,
            zorder=10000,
            color=colour_thermal
        )
        ax_dat.stairs(
            albedo.value/norm,
            count_edges.value,
            label='Albedo',
            baseline=None,
            linewidth=2,
            alpha=0.9,
            zorder=10000,
            color=colour_albedo
        )

        # Thermal text
        ax_dat.text(
            0.77, 0.7,
            f"T = {np.round(model.temperature_0.value, 1)} {model.temperature_0.unit}",
            transform=ax_dat.transAxes,
            fontsize=tfs,
            color=colour_thermal
        )

        ax_dat.text(
            0.77, 0.65,
            f"EM = {np.round(model.emission_measure_0.value, 3)} {em_unit}",
            transform=ax_dat.transAxes,
            fontsize=tfs,
            color=colour_thermal
        )

    
    ax_dat.set_ylim(0.6*np.min(observed_counts.value/norm),2*np.max(observed_counts.value/norm))
    ax_dat.legend(frameon=False,fontsize=14, ncol=2)
    ax_dat.loglog()

    params_fixed_free = model.fixed

    params_free = {k: v for k, v in params_fixed_free.items() if v is False}


    dof = len(observed_counts) - len(params_free)

    delchi = (observed_counts - compound_model_evaluation) / observed_counts_err
    chi = np.sum((observed_counts - compound_model_evaluation)**2 / (observed_counts_err**2))
    chi_red = np.round(chi / dof,1).value

    ax_rat.stairs(delchi.value, count_edges.value,baseline=None, linewidth=2,color='k')
    ax_rat.axhline(0,linestyle='--',linewidth=2, color='k')

    ax_dat.set_ylabel(f'{y_unit}')
    ax_rat.set_xlabel(f'Energy ({x_unit})')

    ax_rat.set_ylabel(r'$\mathrm{(D - M) / \sigma}$')
    ax_rat.text(0.9,0.85,r'$\mathrm{\chi^{2}_{red} = }$'+str(chi_red),transform=ax_rat.transAxes,fontsize=18)

    for ax in fig.axes:
        ax.tick_params(axis='both', which='both',top=True, bottom=True, left=True, right=True,  direction='in', length=6)
        ax.tick_params(axis='both', which='minor',top=True, bottom=True, left=True, right=True,  length=3)
        ax.minorticks_on()

    fig.suptitle(fit_times,fontsize=14,x=0.5,y=1.035)

    if not save_name:
        pass
    else:
        fig.savefig(str(save_name),bbox_inches='tight')        

    plt.show()