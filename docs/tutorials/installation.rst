.. _installation:

**********************
Installing sunkit-spex
**********************

Installing the latest stable release
------------------------------------

To install the latest stable release of ``sunkit-spex`` from PyPI with pip use:

.. code-block::

   $ pip install sunkit-spex

This will install the latest stable release of sunkit-spex and its required dependencies.
To additionally install the optional legacy dependencies, needed for the legacy fitting code and examples, use:

.. code-block::

   $ pip install sunkit-spex[legacy]

Installing the development version
----------------------------------
``sunkit-spex`` is still under development but users can also install the development version from the latest source.
Detailed instructions for setting up a development environment, as well as a discussion on how to contribute code to any SunPy package, can be found in the `Developer's Guide <https://docs.sunpy.org/en/latest/dev_guide/index.html#developer-s-guide>`__.
We highly encourage users to read this, especially if considering contributing to sunkit-spex (which we welcome enthusiastically!)
For brevity though, the key installation steps are as follows.

First, open a terminal and navigate to the directory where you want the sunkit-spex repo to live on your computer.
Then, clone the sunkit-spex repository:

.. code-block:: console

   $ git clone https://github.com/sunpy/sunkit-spex.git

Change into the sunkit-spex repository, then install sunkit-spex:

.. code-block:: console

   $ cd sunkit-spex
   $ pip install -e .

or to install the legacy dependencies

.. code-block:: console

   $ pip install -e .[legacy]

This will install the development version of sunkit-spex. Please see the :ref:`sunpy-tutorial-installing` guide.
