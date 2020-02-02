# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
from ..utils.vector2function import vector2Function
import numpy as np
import os

abspath = os.path.dirname( os.path.abspath(__file__) )
source_directory = os.path.join(abspath,"cpp_multivector")
with open(os.path.join(source_directory,"multivector.cpp"), "r") as cpp_file:
    cpp_code    = cpp_file.read()


include_dirs = [".", source_directory]
cpp_module = dl.compile_cpp_code(cpp_code, include_dirs=include_dirs)


class MultiVector(cpp_module.MultiVector):
    def dot_v(self, v):
        return self.dot(v)
    
    def dot_mv(self,mv):
        shape = (self.nvec(),mv.nvec())
        m = self.dot(mv)
        return m.reshape(shape, order='C')
        
    def Borthogonalize(self,B):
        """ 
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic

        .. math::
            R \\,= \\,Z, && (1),

            Q^*BQ\\, = \\, I, && (2),

            Q^*BZ \\, = \\,R, && (3),

            ZR^{-1} \\, = \\, Q, && (4). 
        
        Returns:

            :code:`Bq` of type :code:`MultiVector` -> :code:`B`:math:`^{-1}`-orthogonal vectors
            :code:`r` of type :code:`ndarray` -> The :math:`r` of the QR decomposition.
        
        .. note:: :code:`self` is overwritten by :math:`Q`.    
        """
        return self._mgs_stable(B)
    
    def orthogonalize(self):
        """ 
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic
        
            .. math::
                QR \\, = \\, Z, && (1),
                
                Q^*Q \\, = \\, I, && (2),
                
                Q^*Z \\, = \\, R, && (3),
                
                ZR^{-1} \\, = \\, Q, && (4).
        
        Returns:

            :code:`r` of type :code:`ndarray` -> The `r` of the QR decomposition
        
        .. note:: :code:`self` is overwritten by :math:`Q`.    
        """
        return self._mgs_reortho()
    
    def _mgs_stable(self, B):
        """ 
        Returns :math:`QR` decomposition of self, which satisfies conditions (1)--(4).
        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the :math:`B`-orthogonal :math:`QR` factorization.
        
        References:
            1. `A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized \
            Hermitian Eigenvalue Problems with application to computing \
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885`
            2. `W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980`
        
        https://github.com/arvindks/kle
        
        """
        n = self.nvec()
        Bq = MultiVector(self[0], n)
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            B.mult(self[k], Bq[k])
            t = np.sqrt( Bq[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = Bq[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                B.mult(self[k], Bq[k])
                tt = np.sqrt(Bq[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            Bq.scale(k, tt)
            
        return Bq, r 
    
    def _mgs_reortho(self):
        n = self.nvec()
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            t = np.sqrt( self[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = self[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                tt = np.sqrt(self[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            
        return r
    
    def export(self, Vh, filename, varname = "mv", normalize=False):
        """
        Export in paraview this multivector.

        Inputs:

        - :code:`Vh`:        the parameter finite element space.
        - :code:`filename`:  the name of the paraview output file.
        - :code:`varname`:   the name of the paraview variable.
        - :code:`normalize`: if :code:`True` the vector is rescaled such that :math:`\\| u \\|_{\\infty} = 1.` 
        """
        if '.xdmf' in filename:
            self._exportXDMF(Vh, filename, varname, normalize)
        else:
            self._exportFile(Vh, filename, varname, normalize)
            
    def _exportXDMF(self, Vh, filename, varname, normalize):
        """
        Specialization of export using dl.File
        """
        fid = dl.XDMFFile(Vh.mesh().mpi_comm(), filename)
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        
        fun = dl.Function(Vh, name = varname)
        
        if not normalize:
            for i in range(self.nvec()):
                fun.vector().zero()
                fun.vector().axpy(1., self[i])
                fid.write(fun,i)
        else:
            for i in range(self.nvec()):
                s = self[i].norm("linf")
                fun.vector().zero()
                fun.vector().axpy(1./s, self[i])
                fid.write(fun,i)
        
    def _exportFile(self, Vh, filename, varname, normalize):
        """
        Specialization of export using dl.File
        """
        fid = dl.File(filename)
        fun = dl.Function(Vh, name = varname)
        if not normalize:
            for i in range(self.nvec()):
                fun.vector().zero()
                fun.vector().axpy(1., self[i])
                fid << fun
        else:
            for i in range(self.nvec()):
                s = self[i].norm("linf")
                fun.vector().zero()
                fun.vector().axpy(1./s, self[i])
                fid << fun
            
    
def MatMvMult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    for i in range(x.nvec()):
        A.mult(x[i], y[i])

def MatMvTranspmult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    assert hasattr(A,'transpmult'), "A does not have transpmult method implemented"
    for i in range(x.nvec()):
        A.transpmult(x[i], y[i])
        
def MvDSmatMult(X, A, Y):
    assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
    assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        Y[j].zero()
        X.reduce(Y[j], A[:,j].flatten())
