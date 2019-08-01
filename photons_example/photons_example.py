'''
Example code for PCFS data analysis.
'''

import numpy as py
import os
from matplotlib import pyplot as plt
import photons as ph
import PCFS


file = 'ExampleT2.stream'
example_photons = ph.photons(file) # create photons class
example_photons.get_photon_records() # create .photons

# all functions can be used...
