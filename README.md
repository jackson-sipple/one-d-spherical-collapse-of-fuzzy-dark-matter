# 1D Spherical Collapse of Fuzzy Dark Matter and Cold Dark Matter
Numerical simulation framework for modeling spherical shell collapse dynamics, developed as part of my PhD dissertation. See Chapter 5: "Towards the Modified Spherical Collapse Model of Fuzzy Dark Matter" of [Probing Cosmic Reionization and Fuzzy Dark Matter with the First Light and the First Galaxies](https://www.proquest.com/docview/3298663609?accountid=14707&sourcetype=Dissertations%20&%20Theses) Solves the equations of motion for N concentric shells under gravity, gas pressure, angular momentum, quantum pressure, and other configurable physics. Strategy/factory patterns allow for plug-and-play swapping of integrators, force laws, timestep criteria, boundary conditions, etc.

Key files:

collapse.py — Main simulation class

simulation_strategies.py — Strategy implementations

utils.py — HDF5 I/O 

Status: Active research code; functional but not polished for release.
