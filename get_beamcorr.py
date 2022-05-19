"""
Get the beam correction from Alan's code outputs. We really just need to average it here.
"""
from pathlib import Path
import numpy as np
if __name__ == '__main__':
    ALAN_CODE_PATH = Path('~/data4/Projects/radio/EOR/Edges/alans-pipeline').expanduser()
    h2case = ALAN_CODE_PATH / 'scripts' / 'H2Case'

    all_beamcorr_files = h2case.glob("beamcorr_*.txt")

    beamcorrs = []
    for fl in all_beamcorr_files:
        data = np.genfromtxt(fl)
        freq = data[:, 0]
        beamcorr = data[:, -1]
        beamcorrs.append(beamcorr)

    mean_bc = np.mean(beamcorrs, axis=0)

    np.savetxt("alan-data/beamcorr.txt", np.array([freq, mean_bc]).T, header='freq (MHz), bmcorr')