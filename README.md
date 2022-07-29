# maxRPA

## Python code to scan maxRPA controller designs for a given controlled network

### This is the code accompanying the manuscript titled "Universal structural requirements for maximal robust perfect adaptation in biomolecular networks" by Ankit Gupta and Mustafa Khammash. It contains the following files:


1. main.py: This is the main python script to scan through the maxRPA designs for a specific controlled network (with specific actuation reactions), and with a fixed number of controller (or Internal model (IM)) species. The top k maxRPA designs in each category (deterministic homothetic, deterministic antithetic and stochastic antithetic) are stored, and visualised. It also plots the perturbation analysis results in both deterministic and stochastic settings (if desired).

  
2. findMaxRpaNetworks.py: The file contains all the helper functions to construct possible maxRPA designs by exploiting the algebraic characterisation of such networks provided in the manuscript. The details on how this is done can be found in the Supplement of the manuscript.

3. networkAnalysisClass.py: Contains a class and functions to analyse and visualise each candidate maxRPA design. Here one can set:
    1. The simulation time-interval for computation of the maxRPA score. 
    2. The parameter to be disturbed, along with disturbance times and the relative disturbance amounts.
    3. The threshold for instability and the tolerance for detecting zero. These are used to check if the dynamics becomes unbounded (maximum of the state      components breaches the instability threshold) or if the dynamics is stable (average variation in state-components in some cut-off time-interval at the end is below the tolerance). The tolerance parameter is also used in the check of whether the fixed-point is at the boundary, i.e. some components are close to zero.


## Dependencies: 
numpy, itertools, scipy.integrate, networkx, matplotlib.pyplot, seaborn

## Questions?:
If you have any questions regarding the code, please contact Ankit Gupta at ankit.gupta@bsse.ethz.ch. 
