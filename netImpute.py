"""
@author: maryam <maryam.zand@utsa.edu>
# C (2019) Ruan Lab (Bioinformatics Research and Development Group (BRDG))
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import spdiags
import scipy
import sys
import utils


class netImpute():
    '''
    Network-based imputation method to recover the scRNAseq real signal
    
    '''
    def __init__(self):
        return
        
    @property
    def check_params(self):
        '''
        check netImpute paameter and raise a valueError if needed
        '''
        utils.check_positive(k=self.k)
        
        
        
    # Normalization
    def colNorm(self,df):
        df = df / (df.sum(0) + sys.float_info.epsilon)
        df[np.isnan(df)] = 0
        return df
    
    def rowNorm(self, df):
        df = df / (df.sum(1) + sys.float_info.epsilon)
        return df
    
    # Library size normalization (all cells will have same sum counts)
    def libSizeNorm(self, df, offset = 1):
        libSize =  np.array(df.sum(0))
        df = df / libSize * np.median(libSize)
        return np.log2(df + offset)


    # gene filtering: filter the genes which has been expressed in less than m cell
    def geneFilteringIndex(self, df , m):
        # m: each gene should have been expressed in at least m cells
        geneFilterIndex = np.sum(df != 0 , 1) <= m
        return geneFilterIndex
    
    # data preProcessing removes genes that have not been expressed 
    def preProcessing(self,data, m=0):
        # m : keep features have been expressed in more than m cells
        index = np.sum(data != 0 , 1) > m
        if isinstance(data,pd.DataFrame):
            data = data.iloc[index.values,:]
        else:
            data = data[index,:]
        print(np.sum(~index) , ' genes have been removed')
        return data
    
    
    # calculate different similarity metrics
    def eucliSim(self, data):
        dist = squareform(pdist(data, 'euclidean'))
        sim = 1 / ( 1 + dist)
        np.fill_diagonal(sim, 0)
        return sim
    
    
    def cosineSim(self, data):
        sim = 1 - squareform(pdist(data, metric='cosine'))
        np.fill_diagonal(sim , 0)
        return sim
    
    def cosineSimFast(self, data):
        similarity = np.dot(data, data.T)
        # squared magnitude of preference vectors (number of occurrences)
        squareMag = np.diag(similarity)
        # inverse squared magnitude
        invSquareMag = 1 / squareMag
        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        invSquareMag[np.isinf(invSquareMag)] = 0
        # inverse of the magnitude
        invMag = np.sqrt(invSquareMag)
        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * invMag
        sim = cosine.T * invMag
        np.fill_diagonal(sim, 0)
        return sim
    
    def corrSim(self, data , method = 'pearson'):
        if method == 'pearson':
            sim = np.corrcoef(data)
        else:
            sim = scipy.stats.spearmanr(data)
        np.fill_diagonal(sim, 0)
        return sim
    
    # construct gene-gene network by getting ED similarity matrix 
    def simToNetARank(self,sim, k=10):
        print('start calculating aknn ...')
        # sim is similarity matrix and k is number of neighbors
        # return A graph which two vertices are connected if one of them are in k negbor of the other one
        np.fill_diagonal(sim, 0)
        I = np.argsort(sim, axis = 0) + 1
        I2 = (np.argsort(I, axis = 0) + 1)
        net = I2 > (len(sim) - k)
        net = np.logical_or(net, net.T)
        np.fill_diagonal(net, False)  
        net = net*1
        return net
    
    # approximation for knn
    def dataToNet(self, data , k=10 , simMetric = 'euclidean'):
        net = kneighbors_graph(data, n_neighbors=k, mode='connectivity', metric = simMetric)
        return net
    
    def simToNetMRank(self, sim , k = 10):
        print('start calculating mknn net ...')
        np.fill_diagonal(sim, 0)
        # sim is similarity matrix and k is num of neigbors
        # return A graph which two vertices are connected if both of them are in k neighbor of the  other
     
        I = np.argsort(sim, axis = 0) + 1
        I = np.argsort(I, axis = 0) + 1
        net = I >  (sim.shape[0] - k)
        net = np.logical_and(net, net.T)
        np.fill_diagonal(net, False)  
        net = net*1
        return net
    
    def simToNetCRank(self, sim , k = 10):
        print('start calculating knn net ...')
        # combination rank : mutual rank + asym rank to prevent disconnected nodes
        # if a node will be discoonected by mutual rank then add a edge by asym rank rules
        np.fill_diagonal(sim, 0)
        I = np.argsort(sim, axis = 0) + 1
        I = np.argsort(I, axis = 0) + 1
        net = I >  (len(sim) - k)
        net2 = np.logical_and(net, net.T)
        net[np.any(net2 , axis = 1)] = 0
        net2 = net + net2
        net2 = np.logical_or(net2, net2.T)
        np.fill_diagonal(net2, False)  
        net2 = net2*1
        #net=scipy.sparse.csr_matrix(net)
        return net2
        
    
    # here 1-alpha is the restart probability
    def RWR(self, A, nSteps = 500, alpha = 0.5, p0 = None):
        A = np.array(A)
        n = A.shape[0]
        if p0 == None:
            p0 = np.eye(n)
        #W = A * spdiags(sum(A)'.^(-1), 0, n, n);
        #W = spdiags(np.power(sum(np.float64(A)) , -1).T  , 0, n, n).toarray()
        W = A.dot( spdiags(np.power(sum(np.float64(A)) , -1)[np.newaxis],0, n, n).toarray() )
        p = p0
        pl2norm = np.inf
        unchanged = 0
        for i in range(1, nSteps+1):
            if i % 100 == 0:
                print('      done rwr ' + str(i-1) )
                
            pnew = (1 - alpha) * W.dot(p) + (alpha) * p0
            l2norm = max(np.sqrt(sum((pnew - p) ** 2) ))
            p = pnew
            if l2norm < np.finfo(float).eps:
                break
            else:
                if l2norm == pl2norm:
                    unchanged = unchanged +1
                    if unchanged > 10:
                        break
                else:
                    unchanged = 0
                    pl2norm = l2norm
        return p
    
    # here 1 - alpha is the restart probability
    def RWR2(self, geneEGraph, alpha):
        geneEGraph = self.colNorm(geneEGraph)
        b =  np.matrix(np.eye(geneEGraph.shape[0]) - ((1-alpha)*geneEGraph))  
        kernel  = (alpha) * (b.I)
        
        colSum = np.sum(kernel, 0)
        modify = colSum == alpha
        modify = (1-alpha) * modify
        np.fill_diagonal(kernel, modify+np.diag(kernel))
        return kernel
    
    def diffuseGeneEByCol(self,kernel , geneE):
        netSmooth = np.dot( kernel.T , geneE)
        return netSmooth
    
    def smoothing(self, geneE , geneEGraph, alpha):
        kernel = self.RWR2(geneEGraph , alpha)
        geneESmoothed = self.diffuseGeneEByCol(kernel , geneE)
        return geneESmoothed
    
    def smoothingByCellSim(self, geneE , geneEGraph, alpha):
        kernel = self.RWR2(geneEGraph , alpha)
        geneESmoothed = self.diffuseGeneEByCol(kernel , geneE.T).T
        return geneESmoothed
    
    
    def smoothingWithPPI(self, geneE , ppi, alpha):
        ppi = self.colNorm(ppi)
        geneEOnNet = self.projectOnPPI(geneE , ppi.index)
        geneEOnNetSmooth = self.randomWalkCol(geneEOnNet , ppi , alpha)
        geneEOnNetSmooth = pd.DataFrame(geneEOnNetSmooth, index = geneEOnNet.index , columns = geneEOnNet.columns)
        geneESmooth = self.projectFromPPI(geneE, geneEOnNetSmooth)
        return geneESmooth

    def projectOnPPI(self, geneExp , ppiGenes):
        geneEOnPPI = pd.DataFrame(0 , index = ppiGenes , columns  = geneExp.columns)
        geneListE = set(geneExp.index)
        geneListPPI = set(ppiGenes)
        genesInBoth = geneListE.intersection(geneListPPI)
        genesInPPIOnly = geneListPPI - geneListE
        geneEOnPPI.loc[genesInBoth] = geneExp.loc[genesInBoth]
        geneEOnPPI.loc[genesInPPIOnly] = 0
        return geneEOnPPI
    
    def projectFromPPI(self, geneE , geneESmooth):
        geneENew = geneE.copy()
        g1 = set(geneE.index)
        g2 = set(geneESmooth.index)
        genesInBoth = g1.intersection(g2)
        geneENew.loc[genesInBoth] = geneESmooth.loc[genesInBoth]
        return geneENew
    
    
    
    def impute(self, data , alpha = 0.5, similarity = 'correlation', k=10, knn_method = 'mknn'):
        # remove genes that have not been expressed
        
        utils.checkQuality(data)
        print('number of genes:', data.shape[0])
        print('number of cells:', data.shape[1])
        df = False
        if isinstance(data, pd.DataFrame):
            genes = data.index
            cells = data.columns
            data = data.values
            df = True

        # calcuate similarity matrix and create gene-gene net
        if similarity=='correlation':
            print('start calculating ' +similarity+ ' sim ...')
            sim = self.corrSim(data)
        if similarity == 'cosine':
            print('start calculating ' +similarity+ ' sim ...')
            sim = self.cosineSim(data)
        if similarity == 'euclidean':
            print('start calculating ' +similarity+ ' sim ...')
            sim = self.eucliSim(data)
        print('start calculating gene-gene graph ...')
        if knn_method == 'mknn':
            net = self.simToNetMRank(sim , k) 
        elif knn_method == 'aknn':
            net = self.simToNetARank(sim , k) 
        else:
            net = self.simToNetCRank(sim , k) 
        print('start imputing ...')
        dataImputed = self.smoothing(data , net , alpha)
        if df:
            dataImputed = utils.convertDataFormat(dataImputed, genes, cells)
        return dataImputed
    
    def impute_ppi(self, data, ppi, alpha = 0.5, k=64):
        # remove genes that have not been expressed
        #data, index = self.preProssesing(data)
        # calcuate similarity matrix and create gene-gene net
        utils.checkQuality(data)
        df = False
        if isinstance(data, pd.DataFrame):
            genes = data.index
            cells = data.columns
            data = data.values
            df = True
        print('start imputing ...')
        dataImputed = self.smoothingWithPPI(data , ppi , alpha)
        if df:
            dataImputed = utils.convertDataFormat(dataImputed, genes, cells)
        return dataImputed
    
