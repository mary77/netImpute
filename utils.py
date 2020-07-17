"""
@author: maryam <maryam.zand@utsa.edu>
# C (2019) Ruan Lab (Bioinformatics Research and Development Group (BRDG))
"""

import numbers
import numpy as np
import pandas as pd
from scipy import sparse

def check_positive(**params):
    """Check that parameters are positive as expected
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] <= 0:
            raise ValueError(
                "Expected {} > 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError("Expected {} integer, got {}".format(p, params[p]))
            
def convertDataFormat(data, rowNames, colNames):
    data = pd.DataFrame(data, index = rowNames, columns = colNames)
    return data

def checkQuality(data):
    if np.sum(np.sum(data,1) == 0) > 0:
        raise ValueError("input data has unexpressed genes, please remove before imputation")
        
        
def load_data(filename, delimiter=',',
              gene_names='True', cell_names='True',
              gene_dir='row', **kwargs):
    '''
    load read counts matrix
    Parameters
    ---------
    filename : str
        the name of the read counts matrix
    delimiter : str, optional (default=',')
        use other delimiter if needed
    gene_names : 'bool', or None (default = 'True')
        if True, it means gene names are in the first row/col 
    cell_names : 'bool', or None (default = 'True')
        if True, it means cell IDs are in the first row/col
    gene_dir : {'row', 'col'}, optional (default: 'row')
        if columns represents the genes use 
        gene_dir='col'
        
    Returns
    --------
    data : pd.DataFrame data.shape = [n_genes, n_cells]
    '''
    if gene_dir not in ['row','col']:
        raise ValueError("gene axis not recognized. use 'row' or 'col'")
    if gene_dir == 'row':
        if gene_names == 'True' and cell_names == 'True':
            header = 'True'
            index_col = 0
        elif gene_names == 'True' and cell_names == 'False':
            header = None
            index_col = 0
        elif gene_names == 'False' and cell_names == 'True':
            header = True
            index_col = None
        else:
            header = None
            index_col = None
            
    else:
        if gene_names == 'True' and cell_names == 'True':
            header = 'True'
            index_col = 0
        elif gene_names == 'True' and cell_names == 'False':
            header = True
            index_col = None
        elif gene_names == 'False' and cell_names == 'True':
            header = None
            index_col = 0
        else:
            header = None
            index_col = None
    if header:
        data = pd.read_csv(filename , delimiter=delimiter, index_col=index_col) 
    else: 
        data = pd.read_csv(filename , delimiter=delimiter, header=header, index_col=index_col)
    return data