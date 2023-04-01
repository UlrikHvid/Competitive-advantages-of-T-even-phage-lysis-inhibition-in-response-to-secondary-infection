<p align="center">
  <img src=T4-logo.png width="200">
</p>

## :rocket: Welcome

This repository contains simulations of experiments involving bacteriophage T4 in interaction with cultures of e. coli. The purpose of these models is to explore the evolutionary significance of lysis-inhibition which is an important aspect of phage T4's infection strategy. 

The models have been developed by me with guidance from my supervisor Namiko Mitarai at the Center for Models of Life, NBI. She is currently working on performing the experiements that I simulate.

## :test_tube:  Model explanation

I have written three separate types of simulations:

M: Well-mixed system (a broth culture). Here there is no spatial component.\
MP: Plaque experiment. Phages are allowed to diffuse through space, but bacteria are immobile.\
MS: Experiment in swimming medium. Bacteria and phages are inoculated together in the center of the plate. Phages diffuse and bacteria swim towards the nutrient gradients that arise once they start consuming nutrients locally.

The variables are:

B: Uninfected bacteria \
L: Infected bacteria \
LI: Lysis-inhibited (superinfected) bacteria \
P: Phages \
n: Nutrients\
Lr: r-mutant infected bacteria (competition version)\
Pr: r-mutant phages (competition version) 

M is integrated with scipy's solve_ivp (4th order Runge Kutta), while MP and MS are integrated manually (forward Euler) with a fixed time step.

The spatial simulations, MP and MS, assume radial symmetry and can therefore run fairly efficiently.

The model names get the suffixes 0, 1 or 1C. Here, 0 corresponds to a simulation of a so-called T4 r-mutant, which cannot perform lysis inhibition. 1 designates the wild-type T4 which can. 1C means that the simulation includes both strains in competition with each other. Eg. M1C means a simulation of competition in broth culture, and MP0 means an r-mutant plaque.

## :gear:  Requirements

Nothing fancy. Numpy, scipy, matplotlib, tqdm

## :ringed_planet:  Showcase notebook

The gifs in the repository show example simulations of the spatial models, MP and MS. In addition, I include a notebook that can run each of the simulations. The default parameter values are attributes of the classes V (for model M) or VS (for MP and MS). Refer to the comments in "Initial_values.py" for the names - as well as brief explanations - of the parameters. Feel free to play around with the model by changing those values. Note, however, that some parameters must be passed as arguments to the class, because they couple to other parameters.

Note, the function GifGenerator saves a gif, but it does not play inside the notebook.

One technical comment: The default spatial resolution is dr = 20. However, the spatial model MS, which is the most complex, requires a significantly finer resolution. I have left the default value in the notebook, because it reproduces roughly the right behavior, and because a more precise simulation requires an impractical computation time.

Enjoy!