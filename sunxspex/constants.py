"""
Class and code to easily switch between astropy constants and values taken directly from SSW for
comparision.
"""

from astropy import constants as const

CONSTANTS = {
    'idl': {
            'mc2': 510.98,  # electron rest mass keV
            'clight': 2.9979e10,  # cm s^-1
            'au': 1.496e13,  # cm
            'r0': 2.8179e-13,  # classical electron radius cm
            'alpha': 7.29735308e-3,  # alpha
            'twoar02': 1.15893512e-27  # 2 * alpha * r0^2
        },

    'astropy': {
            'mc2': (const.m_e * const.c**2).to('keV').value,
            'clight': const.c.cgs.value,
            'au': const.au.cgs.value,
            'r0': (const.a0*const.alpha**2).cgs.value,
            'alpha': const.alpha.cgs.value,
            'twoar02': 2.0 * const.alpha.cgs.value * (const.a0*const.alpha**2).cgs.value**2
        }
    }


class Singleton(type):
    """
    A singleton metaclass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Constants(metaclass=Singleton):
    """
    Centralised constant representation
    """
    def __init__(self, ref='idl'):
        """

        Parameters
        ----------
        ref
        """
        self.ref = ref

    def get_constant(self, name):
        """
        Return the value of the constant.
        Parameters
        ----------
        name : str
            Name of constant

        Returns
        -------
        float
            Value of constant
        """
        if self.ref not in ['astropy', 'idl']:
            raise ValueError(f'Valid values for ref are astropy or idl not {self.ref}.')

        if name not in CONSTANTS[self.ref].keys():
            raise ValueError(f'Valid names are {CONSTANTS[self.ref].keys()} or idl not {name}.')

        return CONSTANTS[self.ref][name]