"""
Collection of helper functions
"""
import numpy as np

from csb.bio.io import StructureParser

def make_title(string, n=10):
    """
    Make separating title
    """
    return '\n{0} {1} {0}\n'.format('=' * n, string)

def numerical_gradient(score, R, eps=1e-8):
    """
    Calculate numerical gradient of scoring function with
    respect to rotation matrix
    """
    num = R * 0
    val = score(R)

    for i in range(3):
        for j in range(3):
            R[i,j]  += eps
            num[i,j] = (score(R)-val) / eps
            R[i,j]  -= eps
            
    return num

def relerr(a, b):
    """
    Relative error
    """
    return np.fabs(a-b).max() / np.fabs(a).max()

def corr(a, b):
    """
    Correlation coefficient in percentage
    """
    return 100 * np.corrcoef(a.flatten(),b.flatten())[0,1]

def compare_grad(a, b):
    """
    Returns relative error and correlation coefficient
    """
    return relerr(a,b), corr(a, b)

def random_walk(n):
    """
    Generate a random walk in 3D
    """
    bonds  = np.random.standard_normal((int(n), 3))
    bonds  = (bonds.T / np.linalg.norm(bonds, axis=1)).T
    coords = np.add.accumulate(bonds, axis=0)

    return coords

def load_coords(codes, center_coords=True, path='./data/{0}.pdb'):
    """
    Load CA coordinates from PDB files.
    """    
    structs = [StructureParser(path.format(code)).parse()
               for code in codes]
    coords  = [struct.get_coordinates(['CA']) for struct in structs]

    if center_coords:
        coords  = [xyz - xyz.mean(0) for xyz in coords]

    return coords
