.. _installation:

**********************
Installing sunkit-spex
**********************

Installing the release version
------------------------------

To install the current release version of ``sunkit-spex`` use pip

.. code-block::

    $ pip install sunkit-spex

or with the legacy optional dependency which are needed to use the legacy fitting code and examples

.. code-block::

    $ pip install sunkit-spex[legacy]

Installing the development version
----------------------------------
``sunkit-spex`` is still under development, and no stable version has been released. However, users can install the the development version.
Detailed instructions for setting up a development environment, as well as a discussion on how to contribute code to any SunPy package, can be found in the `Developer's Guide <https://docs.sunpy.org/en/latest/dev_guide/index.html#developer-s-guide>`__.
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

or to install the legacy dependencies

	$ pip install -e .[legacy]

This will install the development version of sunkit-spex. Please see the :ref:`sunpy-tutorial-installing` guide.
