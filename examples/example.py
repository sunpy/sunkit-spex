"""
=======
Example
=======

This example demonstrates how to search and download Solar Orbiter data using ``sunpy.net.Fido``.
"""

import numpy as np
from matplotlib import pyplot as plt

#####################################################
# Finally we can download the data.
x = np.linspace(0, 2 * 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
