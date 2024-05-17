<p align="center">
  <img src=T4-logo.png width="200">
</p>

## Introduction

This repository contains code and results for the simulations described in the paper *Competitive advantages of T-even phage lysis inhibition in response to secondary infection*.

## Model explanation

The code contains three separate types of simulations:

M: Well-mixed system (a broth culture). Here there is no spatial component.\
MP: Plaque experiment. Phages are allowed to diffuse through space, but bacteria are immobile.\
MS: Experiment in swimming medium. Bacteria and phages are inoculated together in the center of the plate. Phages diffuse and bacteria swim towards the nutrient gradients that arise once they start consuming nutrients locally. This simulation is not included in the paper.\

The codes for these three systems are contained in Broth_model.py, Plaque_model.py and Swimming_model.py, respectively. \

For all simulations, default parameters and initial values are initialized from Initial_values.py, as object methods. \

The variables are:

B: Uninfected bacteria \
I: Wild-type-infected bacteria \
L: Lysis-inhibited (superinfected) bacteria \
P: Wild-type phages \
R: Wild-type phages \
Ir: r-mutant-infected bacteria\
n: Nutrients

The broth model is integrated with scipy's solve_ivp (4th order Runge Kutta), while the spatial models are integrated manually (forward Euler) with a fixed time step. The spatial simulations assume radial symmetry and just display a radial cross-section of the system.

## Requirements

Numpy, scipy, matplotlib, tqdm

## Showcase notebook

The jpgs and gifs in the repository show example simulations. Most figures in the paper are output by the notebook Figs_2-4+S1.ipynb. Only Figure 5, which is a parameterscan performed in parallel, is done in a separate file, Fig5.py. That file outputs csv files that can be plotted in the notebook Fig5_plotter.ipynb. \

The default parameter values are imported as objects, and parameters can be altered by redefining methods of those parameters. Refer to the comments in "Initial_values.py" for the names - as well as brief explanations - of the parameters. Note that some parameters must be passed as arguments to the class, because they couple to other parameters.

Note, the function GifGenerator saves a gif, but it does not play inside the notebook.

One technical comment: The default spatial resolution is dr = 20. However, the spatial model MS, which is the most complex, requires a significantly finer resolution. I have left the default value in the notebook, because it reproduces roughly the right behavior, and because a more precise simulation requires an impractical computation time.

Enjoy!