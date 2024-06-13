import logging


def get_logger(name, level=logging.WARNING):
    """
    Return a configured logger instance.

    Parameters
    ----------
    name : `str`
        Name of the logger
    level : `int` or level, optional
        Level of the logger e.g `logging.DEBUG`

    Returns
    -------
    `logging.Logger`
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(lineno)s: %(message)s',
                                  datefmt='%Y-%m-%dT%H:%M:%SZ')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
