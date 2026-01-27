.. _installation:

**********************
Installing sunkit-spex
**********************

Installing the development version
----------------------------------
Sunkit-spex is still under development, and no stable version has been released. However, users can install the the development version.
Detailed instructions for setting up a development environment, as well as a discussion on how to contribute code to any SunPy package, can be found `here <https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html#setting-up-a-development-environment>`__.
We highly encourage users to read this, especially if considering contributing to sunkit-spex (which we welcome enthusiastically!)
For brevity though, the key installation steps are as follows.

First, open a terminal and navigate to the directory where you want the sunkit-spex repo to live on your computer.
Then, clone the sunkit-spex repo:

.. code-block:: console

    $ git clone https://github.com/sunpy/sunkit-spex.git

Change into the sunkit-spex repo, then install sunkit-spex:

.. code-block:: console

	$ cd sunkit-spex
	$ pip install -e .

This will install the development version of sunkit-spex. Please see the `sunpy installation guide for more general installation help <https://docs.sunpy.org/en/stable/installation.html>`__.
