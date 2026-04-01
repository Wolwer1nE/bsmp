#!/usr/bin/env python3
"""
petsc_to_mtx.py

Convert PETSc binary files (Mat or Vec saved with PetscViewerBinary) into
Matrix Market (.mtx) files using petsc4py + scipy.

Usage:
  ./petsc_to_mtx.py matrix <input_petsc_mat.bin> <output.mtx>
  ./petsc_to_mtx.py vector <input_petsc_vec.bin> <output.mtx>

The matrix conversion writes a Matrix Market coordinate (sparse) file.
The vector conversion writes a Matrix Market dense column vector (m x 1).

Requires: petsc4py, scipy, numpy

Exit codes:
  0 - success
  1 - usage / argument error
  2 - import error (missing packages)
  3 - IO / conversion error
"""

from __future__ import annotations
import sys
import os

def die(msg: str, code: int = 3):
    sys.stderr.write(msg + "\n")
    sys.exit(code)

def main(argv):
    if len(argv) < 4:
        sys.stderr.write(__doc__)
        return 1

    mode = argv[1].lower()
    infile = argv[2]
    outfile = argv[3]

    try:
        from petsc4py import PETSc
        import numpy as np
        import scipy.io
        import scipy.sparse as sps
    except Exception as e:
        sys.stderr.write("Failed to import required Python packages: %s\n" % e)
        return 2

    if not os.path.exists(infile):
        die(f"Input file does not exist: {infile}", 3)

    try:
        viewer = PETSc.Viewer().createBinary(infile, 'r')
    except Exception as e:
        die(f"Failed to open PETSc binary file '{infile}': {e}", 3)

    try:
        if mode == 'matrix' or mode == 'mat' or mode == 'm':
            A = PETSc.Mat().load(viewer)
            # getValuesCSR returns ia, ja, a such that csr = csr_matrix((a, ja, ia))
            ia, ja, a = A.getValuesCSR()
            m, n = A.getSize()
            ia = np.array(ia, dtype=np.int64)
            ja = np.array(ja, dtype=np.int64)
            a = np.array(a, dtype=np.float64)
            csr = sps.csr_matrix((a, ja, ia), shape=(m, n))
            scipy.io.mmwrite(outfile, csr, comment='Converted from PETSc binary by petsc_to_mtx.py')
            print(f"Wrote Matrix Market file: {outfile} ({m}x{n}, nnz={csr.nnz})")
            return 0
        elif mode == 'vector' or mode == 'vec' or mode == 'v':
            x = PETSc.Vec().load(viewer)
            # Convert to dense numpy array
            arr = x.getArray()
            arr = np.array(arr, dtype=np.float64).reshape(-1, 1)  # column vector
            scipy.io.mmwrite(outfile, arr, comment='Converted from PETSc binary Vec by petsc_to_mtx.py')
            print(f"Wrote Matrix Market vector file: {outfile} ({arr.shape[0]} entries)")
            return 0
        else:
            sys.stderr.write("Unknown mode: %s\n" % mode)
            sys.stderr.write("Usage: petsc_to_mtx.py [matrix|vector] <in> <out>\n")
            return 1
    except Exception as e:
        die(f"Conversion failed: {e}", 3)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
