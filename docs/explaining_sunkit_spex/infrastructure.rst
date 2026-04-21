**************************
Sunkit-spex Infrastructure
**************************

This section will detail the workflow design of Sunkit-spex. Here we 
will provide details on the flexible package structure while providing 
relavent links to examples and code.

Overview
========

Sunkit-spex takes a modular design perspective when it comes to working 
with X-ray spectral data. Broadly, four categories exist: 1. the spectral 
data; 2. the models; 3. the statistical approach; and 4. the model-to-data 
comparison. The diagram below shows how the former three categories come 
together, enabling the model-to-data comparison and also indicates how the 
three categories are dependent on each other issuing the need for checks.

.. raw:: html
    :file: ./basic-pipeline.drawio.svg

1. For data provided from a given instrument team, a user can use **Spectrum 
   containers** to store the all information relavent for fitting each 
   specific spectrum.

2. **Core Model Objects** are provided although users can create their own.
   These borrow their use from Astropy somewhat and can be combined into 
   composite models for more complicated cases.

3. The final piece of the puzzle before model-to-data comparison can take
   place is for the **Objective Function** or comparison statistics to be
   defined. These will be used to govern the goodness of comparison and 
   their choice can influence validity of the model-to-data comparison 
   analysis.

4. Providing the information contained above, a given model can be compared 
   to the user's data. All three components above are all in some way dependent 
   on each other; therefore, some **Checks (4a)** need to be performed for 
   validility. Post-checks, the models can be converted to the same unit-space 
   as the data using the observation & instrument information from the 
   **Spectrum Container** for direct comparison in the **Model-to-Data Comparison 
   Object (4b)**. Comparison will take place using the determined **Objective 
   Function** that will be consistent with the data error type from the 
   **Spectrum Container**.
   
Spectral Data
=============

Instrument Data Container
-------------------------

Spectrum Container
------------------

Core Model Object
=================

Objective Function
==================

Fitter
======

Model-to-Data Comparison Object
-------------------------------
