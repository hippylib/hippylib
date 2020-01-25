# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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
import numpy as np
from mpi4py import MPI

class NullCollective:
    """
    No-overhead "Parallel" reduction utilities when a serial system of PDEs is solved on 1 process.
    """
    def __init__(self):
        pass
    
    def size(self):
        return 1
    
    def rank(self):
        return 0
    
    def allReduce(self, v, op):
        
        if op.lower() not in ["sum", "avg"]:
            err_msg = "Unknown operation *{0}* in NullCollective.allReduce".format(op)
            raise NotImplementedError(err_msg)
        
        return v
    
class MultipleSerialPDEsCollective:
    """
    Parallel reduction utilities when several serial systems of PDEs (one per process) are solved concurrently.
    """
    def __init__(self, comm):
        """
        :code:`comm` is :code:`mpi4py.MPI` comm
        """
        self.comm = comm
    
    def size(self):
        return self.comm.Get_size()
    
    def rank(self):
        return self.comm.Get_rank()
    
    def allReduce(self, v, op):
        """
        Case handled:
        - :code:`v` is a scalar (:code:`float`, :code:`int`);
        - :code:`v` is a numpy array (NOTE: :code:`v` will be overwritten)
        - :code:`v` is a  :code:`dolfin.Vector` (NOTE: :code:`v` will be overwritten)
        Operation: :code:`op = "Sum"` or `"Avg"` (case insentive).
        """
        op = op.lower()
        err_msg = "Unknown operation *{0}* in MultipleSerialPDEsCollective.allReduce".format(op)
        
        if type(v) in [float, np.float64]:
            send = np.array([v], dtype=np.float64)
            receive = np.zeros_like(send)
            self.comm.Allreduce([send, MPI.DOUBLE], [receive, MPI.DOUBLE], op = MPI.SUM)
            if op == "sum":
                return receive[0]
            elif op == "avg":
                return receive[0]/float(self.size())
            else:
                raise NotImplementedError(err_msg)
                
        if type(v) in [int, np.int, np.int32]:
            send = np.array([v], dtype=np.int32)
            receive = np.zeros_like(send)
            self.comm.Allreduce([send, MPI.INT], [receive, MPI.INT], op = MPI.SUM)
            if op == "sum":
                return receive[0]
            elif op == "avg":
                return receive[0]//self.size()
            else:
                raise NotImplementedError(err_msg)
        
        if (type(v) is np.array) or (type(v) is np.ndarray):
            receive = np.zeros_like(v)
            self.comm.Allreduce([v, MPI.DOUBLE], [receive, MPI.DOUBLE], op = MPI.SUM)
            if op == "sum":
                v[:] = receive
            elif op == "avg":
                v[:] == (1./float(self.size()))*receive
            else:
                raise NotImplementedError(err_msg)
                
            return v
              
        elif hasattr(v, "mpi_comm") and hasattr(v, "get_local"):
            # v is most likely a dl.Vector
            assert v.mpi_comm().Get_size() == 1
            send = v.get_local()
            receive = np.zeros_like(send)
        
            self.comm.Allreduce([send, MPI.DOUBLE], [receive, MPI.DOUBLE], op = MPI.SUM)
            if op == "sum":
                pass
            elif op == "avg":
                receive *= (1./float(self.size()))
            else:
                raise NotImplementedError(err_msg) 
             
            v.set_local(receive)
            v.apply("")
            
            return v
        else:
            msg = "MultipleSerialPDEsCollective.allReduce not implement for v of type {0}".format(type(v))
            raise NotImplementedError(msg) 
