#================================PlotFuncs.py==================================#
# Created by Ciaran O'Hare 2019

# Description:
# This file has many functions which are used throughout the project, but are
# all focused around the bullshit that goes into making the plots

#==============================================================================#

from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.stats import zscore,chi2,multivariate_normal
from scipy.special import erfinv
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

#==============================================================================#
from astropy import units as u
from astropy.coordinates import SkyCoord, get_constellation

# This commented out bit is what you need to run to get the coordinates of cygnus
# cygnus_stars = array(['β','η','γ','α','γ','δ','ι','κ','ι','δ','γ','ε','ζ'])
# #cygnus_stars = ['Deneb','gamma cyg']
# nst = size(cygnus_stars)
# cyg = zeros(shape=(nst,2))
# for i in range(0,nst):
#     c = SkyCoord.from_name(cygnus_stars[i]+' Cyg').galactic
#     cyg[i,:] = array([c.l.degree,c.b.degree])

# (l,b) of the stars in Cygnus
cyg = array([[ 62.10963941,   4.57150891],
       [ 71.01544763,   3.3646167 ],
       [ 78.14859103,   1.86708845],
       [ 84.28473664,   1.99754612],
       [ 78.14859103,   1.86708845],
       [ 78.70955616,  10.24302209],
       [ 83.61190613,  15.44876931],
       [ 84.40177176,  17.85322722],
       [ 83.61190613,  15.44876931],
       [ 78.70955616,  10.24302209],
       [ 78.14859103,   1.86708845],
       [ 75.95136158,  -5.71541249],
       [ 76.75381354, -12.45226928]])

#==============================================================================#

from mpl_toolkits.axes_grid1 import make_axes_locatable

def cbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    return fig.colorbar(mappable, cax=cax)

#==============================================================================#
def MySquarePlot(xlab='',ylab='',\
                 lw=2.5,lfs=45,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)

    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def MyDoublePlot(xlab1='',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2


def MyTriplePlot(xlab1='',ylab1='',xlab2='',ylab2='',xlab3='',ylab3='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=7,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    fig, axarr = plt.subplots(1, 3,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax3.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax3.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    ax3.set_xlabel(xlab3,fontsize=lfs)
    ax3.set_ylabel(ylab3,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
        ax3.grid()
    return fig,ax1,ax2,ax3
#==============================================================================#


#==============================================================================#
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r
#==============================================================================#


#==============================================================================#
def col_alpha(col,alpha=0.1):
    rgb = colors.colorConverter.to_rgb(col)
    bg_rgb = [1,1,1]
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]
#==============================================================================#



#==============================================================================#
def MollweideMap1(ax,TH,PH,fv0,cmin,cmax,nlevels,cmap,tfs,\
        PlotCygnus=False,gridlinecolor='k',GalacticPlane=False):
    plt.rcParams['axes.linewidth'] = 3
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=15)


    ax.contourf(rad2deg(PH), rad2deg(TH),fv0,nlevels,transform=ccrs.PlateCarree(),\
                cmap=cmap,vmin=cmin,vmax=cmax,linestyles='none',antialiased=True)

    gl = ax.gridlines(color=gridlinecolor,linewidth=1.5, linestyle='--',alpha=0.5)
    gl.ylocator = mticker.FixedLocator([-90,-60, -30, 0, 30, 60,90])
    ax.outline_patch.set_linewidth(3)


    tx = array([r'$-60^\circ$',r'$-30^\circ$',r'$0^\circ$',r'$+30^\circ$',r'$+60^\circ$'])
    xtx = array([0.17,0.05,-0.01,0.05,0.18])
    ytx = array([0.08,0.26,0.49,0.72,0.9])

    for i in range(0,size(xtx)):
        plt.text(xtx[i],ytx[i],tx[i],transform=ax.transAxes,horizontalalignment='right',verticalalignment='center',fontsize=tfs)


    if PlotCygnus==True:
        ax.plot(-cyg[0:4,0],cyg[0:4,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[4:,0],cyg[4:,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[:,0],cyg[:,1],'.',color='k',ms=5,transform=ccrs.PlateCarree())

    if GalacticPlane==True:
        ax.plot([-181,181],[0,0],'-',color=gridlinecolor,lw=1.5,transform=ccrs.PlateCarree())
        ax.text(125,4,'Galactic',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
        ax.text(135,-10,'plane',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
    return

#==============================================================================#
def MollweideMap(ax,TH,PH,fv0,cmin,cmax,nlevels,cmap,tfs,PlotCygnus=False,\
gridlinecolor='k',GalacticPlane=False):
    plt.rcParams['axes.linewidth'] = 3
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=15)


    ax.contourf(rad2deg(PH), rad2deg(TH), \
    fv0,nlevels,transform=ccrs.PlateCarree(),cmap=cmap,vmin=cmin,vmax=cmax)
    gl = ax.gridlines(color=gridlinecolor,linewidth=1.5, linestyle='--',alpha=0.5)
    gl.ylocator = mticker.FixedLocator([-90,-60, -30, 0, 30, 60,90])
    ax.outline_patch.set_linewidth(3)


    tx = array([r'$-60^\circ$',r'$-30^\circ$',\
    r'$0^\circ$',r'$+30^\circ$',r'$+60^\circ$'])
    xtx = array([0.17,0.05,-0.01,0.05,0.18])
    ytx = array([0.08,0.26,0.49,0.72,0.9])

    for i in range(0,size(xtx)):
        plt.text(xtx[i],ytx[i],tx[i],transform=ax.transAxes,\
        horizontalalignment='right',verticalalignment='center',fontsize=tfs)


    if PlotCygnus==True:
        ax.plot(-cyg[0:4,0],cyg[0:4,1],'-',color='crimson',\
        transform=ccrs.PlateCarree())
        ax.plot(-cyg[4:,0],cyg[4:,1],'-',color='crimson',\
        transform=ccrs.PlateCarree())
        ax.plot(-cyg[:,0],cyg[:,1],'.',color='k',ms=5,\
        transform=ccrs.PlateCarree())

    if GalacticPlane==True:
        ax.plot([-181,181],[0,0],'-',color=gridlinecolor,\
        lw=1.5,transform=ccrs.PlateCarree())
        ax.text(125,4,'Galactic',color=gridlinecolor,\
        transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
        ax.text(135,-10,'plane',color=gridlinecolor,\
        transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
    return
#==============================================================================#
