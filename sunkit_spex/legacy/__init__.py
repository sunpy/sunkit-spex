import warnings

from sunpy.util.exceptions import SunpyDeprecationWarning

warnings.warn(
    "The legacy module has been deprecated since version 0.4 and will be removed in a future version.",
    SunpyDeprecationWarning,
    stacklevel=2,
)
