# Dual state/rainfall correction project - data assimilation of SMAP for VIC and GPM

This repository stores all code development for the dual correction project that assimilates SMAP soil moisture measurement data to correct model states in the Variable Infiltration Capacity (VIC) model and to correct the GPM precipitation data.

## Overview of the dual correction system
The dual correction system consists of two main parts:
1) State update part
In this part, the SMAP soil moisture measurements are assimilated into the VIC model to update VIC soil moisture states sequentially over time. The ensemble Kalman filter (EnKF) method is used.
2) Rainfall correction part (SMART)
The SMAP soil moisture measurements are assimilated into a simple antecedent precipitation index (API) model sequentially over time, and the API increment at each timestep is then related to the amount of rainfall to correct. This so-called “SMART” method was developed by Crow et al. [2009] and Crow et al. [2011] and is further updated in this project.
The state update part and the rainfall correction part are then brought together to produce ensemble streamflow analysis. Synthetic experiments in addition to real SMAP data assimilation can be performed in the system

## Code for this project
The code in this repository performs data preprocessing, synthetic data generation, dual correction system operation, result analyzing and plotting. See the documentation for steps of using these utilities.


## Publications
Mao, Y., W. T. Crow, and B. Nijssen (2018), A framework for diagnosing factors degrading the streamflow performance of a soil moisture data assimilation system, in preparation.

## References:
Crow, W. T., G. F. Huffman, R. Bindlish, and T. J. Jackson (2009), Improving satellite rainfall accumulation estimates using spaceborne soil moisture retrievals, J. Hydrometeorol., 10, 199-212, doi: 10.1175/2008JHM986.1.
Crow, W. T., M. J. van Den Berg, G. F. Huffman, and T. Pellarin (2011), Correcting rainfall using satellite-based surface soil moisture retrievals: The soil moisture analysis rainfall tool (SMART), Water Resour. Res., 47, W08521, doi:10.1029/2011WR010576.

