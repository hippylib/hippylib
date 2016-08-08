# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as cls
import dolfin as dl
import numpy as np
from matplotlib import animation

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def mplot_cellfunction(cellfn):
    C = cellfn.array()
    tri = mesh2triang(cellfn.mesh())
    return plt.tripcolor(tri, facecolors=C)

def mplot_function(f, vmin, vmax, logscale):
    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        C = f.vector().array()
        if logscale:
            return plt.tripcolor(mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm() )
        else:
            return plt.tripcolor(mesh2triang(mesh), C, vmin=vmin, vmax=vmax)
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        C = f.compute_vertex_values(mesh)
        if logscale:
            return plt.tripcolor(mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm() )
        else:
            return plt.tripcolor(mesh2triang(mesh), C, shading='gouraud', vmin=vmin, vmax=vmax)
    # Vector function, interpolated to vertices
    elif f.value_rank() == 1:
        w0 = f.compute_vertex_values(mesh)
        if (len(w0) != 2*mesh.num_vertices()):
            raise AttributeError('Vector field must be 2D')
        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = w0[:mesh.num_vertices()]
        V = w0[mesh.num_vertices():]
        C = np.sqrt(U*U+V*V)
        return plt.quiver(X,Y,U,V, C, units='x', headaxislength=7, headwidth=7, headlength=7, scale=4, pivot='middle')

# Plot a generic dolfin object (if supported)
def plot(obj, colorbar=True, subplot_loc=None, mytitle=None, show_axis='off', vmin=None, vmax=None, logscale=False):
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
#    plt.gca().set_aspect('equal')
    if isinstance(obj, dl.Function):
        pp = mplot_function(obj, vmin, vmax, logscale)
    elif isinstance(obj, dl.CellFunctionSizet):
        pp = mplot_cellfunction(obj)
    elif isinstance(obj, dl.CellFunctionDouble):
        pp = mplot_cellfunction(obj)
    elif isinstance(obj, dl.CellFunctionInt):
        pp = mplot_cellfunction(obj)
    elif isinstance(obj, dl.Mesh):
        if (obj.geometry().dim() != 2):
            raise AttributeError('Mesh must be 2D')
        pp = plt.triplot(mesh2triang(obj), color='#808080')
        colorbar = False
    else:
        raise AttributeError('Failed to plot %s'%type(obj))
    
    plt.axis(show_axis)
        
    if colorbar:
        plt.colorbar(pp, fraction=.1, pad=0.2)
    else:
        plt.gca().set_aspect('equal')
        
    if mytitle is not None:
        plt.title(mytitle, fontsize=20)
        
    return pp
        
def multi1_plot(objs, titles, same_colorbar=True, show_axis='off', logscale=False):
           
    vmin = None
    vmax = None 
    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for f in objs:
            if isinstance(f, dl.Function):
                fmin = f.vector().min()
                fmax = f.vector().max()
                if fmin < vmin:
                    vmin = fmin
                if fmax > vmax:
                    vmax = fmax
                            
    nobj = len(objs)
    if nobj == 1:
        plt.figure(figsize=(7.5,5))
        subplot_loc = 110
    elif nobj == 2:
        plt.figure(figsize=(15,5))
        subplot_loc = 120
    elif nobj == 3:
        plt.figure(figsize=(18,4))
        subplot_loc = 130
    else:
        raise AttributeError("Too many figures")
             
    for i in range(nobj):
        plot(objs[i], colorbar=True,
             subplot_loc=(subplot_loc+i+1), mytitle=titles[i],
             show_axis='off', vmin=vmin, vmax=vmax, logscale=logscale)


def plot_pts(points, values, colorbar=True, subplot_loc=None, mytitle=None, show_axis='on', vmin=None, vmax=None, xlim=(0,1), ylim=(0,1)):
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    
    pp = plt.scatter(points[:,0], points[:,1], c=values.array(), marker=",", s=20, vmin=vmin, vmax=vmax)
        
    plt.axis(show_axis)
        
    if colorbar:
        plt.colorbar(pp, fraction=.1, pad=0.2)
    else:
        plt.gca().set_aspect('equal')
        
    if mytitle is not None:
        plt.title(mytitle, fontsize=20)
        
    if xlim is not None:
        plt.xlim(xlim)
        
    if ylim is not None:
        plt.ylim(ylim)
        
    return pp


def show_solution(Vh, ic, state, same_colorbar=True, colorbar=True, mytitle=None, show_axis='off', logscale=False, times=[0, .4, 1., 2., 3., 4.]):
    state.store(ic, 0)
    assert len(times) % 3 == 0
    nrows = len(times) / 3
    subplot_loc = nrows*100 + 30
    plt.figure(figsize=(18,4*nrows))
    
    if mytitle is None:
        title_stamp = "Time {0}s"
    else:
        title_stamp = mytitle + " at time {0}s" 
    
    vmin = None
    vmax = None
        
    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smax = s.max()
            smin = s.min()
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin
                
    counter=1
    myu = dl.Function(Vh)
    for i in times:
        try:
            state.retrieve(myu.vector(),i)
        except:
            print "Invalid time: ", i
            
        plot(myu, subplot_loc=(subplot_loc+counter), mytitle=title_stamp.format(i), colorbar=colorbar,
             logscale=logscale, show_axis=show_axis, vmin=vmin, vmax=vmax)
        counter = counter+1

    
        
def animate(Vh, state, same_colorbar=True, colorbar=True,
            subplot_loc=None, mytitle=None, show_axis='off', logscale=False):
    
    fig = plt.figure()
    
    vmin = None
    vmax = None
        
    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smax = s.max()
            smin = s.min()
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin
                
    def my_animate(i):
        time_stamp = "Time: {0:f} s"  
        obj = dl.Function(Vh, state.data[i])
        t = mytitle + time_stamp.format(state.times[i])
        plt.clf()
        return  plot(obj, colorbar=True, subplot_loc=None, mytitle=t, show_axis='off', vmin=vmin, vmax=vmax, logscale=False)
    
    return animation.FuncAnimation(fig, my_animate, np.arange(0, state.nsteps), blit=True)
    
def coarsen_v(fun, nx = 16, ny = 16):
    #mesh = dl.UnitSquareMesh(nx,ny)
    mesh = dl.Mesh("ad_20.xml")
    V_H = dl.VectorFunctionSpace(mesh, "CG", 1)
    dl.parameters['allow_extrapolation'] = True
    fun_H =  dl.interpolate(fun, V_H)
    dl.parameters['allow_extrapolation'] = False
    return fun_H

def plot_eigenvalues(d, mytitle = None, subplot_loc=None):
    k = d.shape[0]
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
    plt.yscale('log')
    if mytitle is not None:
        plt.title(mytitle)
    
def plot_eigenvectors(Vh, U, mytitle, which = [0,1,2,5,10,15]):
    assert len(which) % 3 == 0
    nrows = len(which) / 3
    subplot_loc = nrows*100 + 30
    plt.figure(figsize=(18,4*nrows))
    
    title_stamp = mytitle + " {0}" 
    u = dl.Function(Vh)
    counter=1
    for i in which:
        assert i < U.shape[1]
        Ui = U[:,i]
        if Ui[0] >= 0:
            s = 1./np.linalg.norm(Ui, np.inf)
        else:
            s = -1./np.linalg.norm(Ui, np.inf)
        u.vector().set_local(s*Ui)
        plot(u, subplot_loc=(subplot_loc+counter), mytitle=title_stamp.format(i), vmin=-1, vmax=1)
        counter = counter+1
    
    
    