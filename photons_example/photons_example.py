'''
Example code for PCFS data analysis.
'''

import photons as ph

if __name__ == '__main__':
    file = 'ExampleT2.stream'
    example_photons = ph.photons(file) # create photons class
    example_photons.get_photon_records() # create .photons

    # all functions can be used...
