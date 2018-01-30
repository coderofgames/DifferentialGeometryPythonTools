### Differential Geometry Display Utils

from __future__ import division
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import pylab
from matplotlib.text import Annotation
from sympy.vector import *
from sympy import * 



#=================================================================
# 1) Fancy Arrow for plotting vectors
# http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
# posted this fancy arrow object
#=================================================================
class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)
		
#=================================================================     
# 2) Annotate 3D functions 
# also from stackoverflow
# annotate object
#=================================================================
class Annotation3D(Annotation):
	'''Annotate the point xyz with text s'''

	def __init__(self, s, xyz, *args, **kwargs):
		Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
		self._verts3d = xyz        

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.xy=(xs,ys)
		Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
	'''add anotation text s to to Axes3d ax'''

	tag = Annotation3D(s, *args, **kwargs)
	ax.add_artist(tag)
	
	
#=================================================================   
# 3) Function to plot the cartesian basis ijk
# ax is the matplotlib plot.axis()
#=================================================================
def PlotBasisCartesian(ax):
	a = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=5, lw=2, arrowstyle="-|>", color="k")
	ax.add_artist(a)
	a = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=5, lw=2, arrowstyle="-|>", color="k")
	ax.add_artist(a)
	a = Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=5, lw=2, arrowstyle="-|>", color="k")
	ax.add_artist(a) 
	annotate3D(ax, r'$ \hat{i}$', xyz=(1,0,0), fontsize=30, xytext=(-3,4),
			   textcoords='offset points', ha='right',va='bottom') 
	annotate3D(ax, r'$ \hat{j}$', xyz=(0,1,0), fontsize=30, xytext=(-3,4),
			   textcoords='offset points', ha='right',va='bottom') 
	annotate3D(ax, r'$ \hat{k}$', xyz=(0,0,1), fontsize=30, xytext=(-3,4),
				   textcoords='offset points', ha='right',va='bottom') 
	
	
#=================================================================
# 4) Draw an arrow in 3d space from origin o to point v1, with name and color    
# ax is the matplotlib plot.axis()
# o is the initial point
# v1 is the vector
# e is the CoordSys3D basis 
# name is a string e.g. "$\hat{v}$"
# col is a the color input to Arrow3D
#=================================================================
def plot_arrow(ax,o, v1, e, name,col):
		LX=[float(N(o.dot(e.i))), float(N(v1.dot(e.i) + o.dot(e.i)))]
		LY=[float(N(o.dot(e.j))), float(N(v1.dot(e.j) + o.dot(e.j)))]
		LZ=[float(N(o.dot(e.k))), float(N(v1.dot(e.k) + o.dot(e.k)))]
		a = Arrow3D(LX, LY, LZ, mutation_scale=5, lw=2, arrowstyle="-|>", color=col)
		ax.add_artist(a)
		xyz_ = (LX[1], LY[1], LZ[1])
		annotate3D(ax, name, xyz=xyz_, fontsize=17, xytext=(-3,4),
			   textcoords='offset points', ha='right',va='bottom')

#=================================================================
# 5) plot 3 basis vectors at a point o 
# they are automatically labelled e_1,e_2,e_3
# ax is the matplotlib plot.axis()        
# o is the initial point
# v1,v2,v3 are the vectors, 
# e is the CoordSys3D basis 
# col is a the color input to Arrow3D
#=================================================================
def plot_basis(ax,o,v1,v2,v3,e,col):
	plot_arrow(ax,o,v1,e,r'$ \hat{e_1}$',col)
	plot_arrow(ax,o,v2,e,r'$ \hat{e_2}$',col)
	plot_arrow(ax,o,v3,e,r'$ \hat{e_3}$',col)
	
#=================================================================
### Tangent vectors to a space curve
# returns the symbolic derivative with respect to arc length s
# r is the vector expression e.g. r = r_1*e.i + r_2 * s * e.j + r_3 * s**2 *e.k
# s is the symbol for arc length
#=================================================================
def unit_tangent_natural(r,s):
	return diff(r,s)

	
# returns the normalized tangent as a symbolic derivative expression 
# with respect to t parameter
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3( t) *e.k
# t is the symbol for time. 	
def unit_tangent_time(r,t):
	drdt = unit_tangent_natural(r,t)
	return drdt / drdt.magnitude()

#=================================================================
### Tangent Lines
#=================================================================

# returns an expression for an equation of a tangent line
# r is the vector expression e.g. r = r_1(s)*e.i + r_2 (s) * e.j + r_3 (s) *e.k
# s is the natural parameter (arc length)
# c is a symbolic constant multiplier	
def tangent_line_natural(r,s,c):
	tangent = unit_tangent_natural(r,s)
	return r + c * tangent

# returns an expression for an equation of a tangent line
# r is the vector expression e.g. r = r_1(s)*e.i + r_2 (s) * e.j + r_3 (s) *e.k
# s is the natural parameter (arc length)
# s0 is a number [0:s] for the point on s
# c is a symbolic constant multiplier		
# c0 is a number [0:cmax] for the multiplier
def tangent_line_natural_at_point(r,s,s0,c,c0):
	return tangent_line_natural(r,s,c).subs({s:s0,c:c0}) 

# returns an expression for an equation of a tangent line
# r is the vector expression e.g. r = r_1(t)*e.i + r_2 (t) * e.j + r_3 (t) *e.k
# t is the symbol for time
# c is a symbolic constant multiplier		
def tangent_line_time(r,t,c):
	tangent = unit_tangent_time(r,t)
	return r + c * tangent

# r is the vector expression e.g. r = r_1(t)*e.i + r_2 (t) * e.j + r_3 (t) *e.k
# t is the symbol for time
# t0 is a number [0:t] for the point on t
# c is a symbolic constant multiplier		
# c0 is a number [0:cmax] for the multiplier	
def tangent_line_time_at_point(r,t,t0,c, c0):
	return  tangent_line_time(r,t,c).subs({t:t0,c:c0}) 

#=================================================================
### Function to plot a space curve
#=================================================================
# ax is the plot axis, 
# e is the basis
# r is the curve, 
# t is the parameter in r, 
# tt is the numpy parameter np.arange[a,b]
def space_curve(ax, e,  r, t, tt):
	fx = lambdify( t, r.dot(e.i), "numpy" )
	fy = lambdify( t, r.dot(e.j), "numpy" )
	fz = lambdify( t, r.dot(e.k), "numpy" )

	# plot the lambda funcs
	ax.plot(fx(tt),fy(tt),fz(tt))
	
#=================================================================
### Curvature and Normal Vector
#=================================================================   

# returns the symbolic expression for the curvature vector wrt s
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for arc length parameter
def curv_vec_natural(r, s):
	tangent = unit_tangent_natural(r,s)
	return diff(tangent, s)

# returns the symbolic expression for the curvature vector wrt s
# tangent is the tangent vector with s as a parameter
# s is the symbol for arc length parameter	
def curv_vec_from_tangent(tangent, s):
	return diff(tangent,s)

# returns the symbolic expression for magnitude of curvature, kappa
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for arc length parameter		
def curvature_natural(r,s):
	return curv_vec_natural(r,s).magnitude()

# returns the symbolic expression for magnitude of curvature, kappa
# k is the curvature vector
def curv_mag(k):
	return k.magnitude()

# returns the symbolic expression for radius of curvature
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for arc length parameter	
def rad_of_curv_natural(r,s):
	return 1/curvature_natural(r,s)

# returns the symbolic expression for radius of curvature
# k is the curvature vector	
def rad_of_curv_from_curv_vec(k):
	return 1/curv_mag(k)

# returns the symbolic expression for the curvature vector wrt t
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3 (t) *e.k
# t is the symbol for the time parameter		
def curv_vec_time(r,t):
	drdt = diff(r,t)
	drdt_mag = drdt.magnitude()
	tangent = drdt / drdt_mag
	dTdt = diff(tangent,t)
	return dTdt / drdt_mag

# returns the symbolic expression for the normal vector wrt t
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3 (t) *e.k
# t is the symbol for the time parameter		
def normal_vec_time(r,t):
	k = curve_vec_time(r,t)
	return k/k.magnitude()

# returns the symbolic expression for the principal normal vector wrt s
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter		
def principle_normal_natural(r,s):
	k = curv_vec_natural(r,s)
	return k/k.magnitude()

# returns the symbolic expression for the principal normal vector
# k is the curvature vector		
def principle_normal_from_curv_vec(k):
	return k / k.magnitude()

# returns the symbolic expression for the principal normal vector
# tangent is the tangent vector
# s is the symbol for the natural parameter			
def principle_normal_from_tangent(tangent,s):
	k = diff(tangent,s)
	return k / k.magnitude()

#=================================================================
### Curvature of a space curve
#=================================================================  
# returns the symbolic expression for the magnitude or the curvature wrt t
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3 (t) *e.k
# t is the symbol for the time parameter	
def curvature_time(r,t):
	rprime1 = diff(r,t)
	rprime2 = diff(rprime1,t)
	return (rprime1.cross(rprime2)).magnitude() / (rprime1.magnitude()**3)

#=================================================================
### Osculating plane
#=================================================================  

# returns the symbolic expression for the normal plane wrt s at s0
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter	
# s0 is the number [0:s] for a point 
def normal_plane_natural(r, s, s0):
	tangent = unit_tangent_natural(r,s)
	return (r-r.subs({s:s0})).dot(tangent.subs({s:s0}))

# returns the symbolic expression for the normal plane wrt t at t0
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3 (t) *e.k
# t is the symbol for the time parameter	
# t0 is the number [0:t] for a point 	
def normal_plane_time(r,t,t0):
	tangent = unit_tangent_time(r,t)
	return (r-r.subs({t:t0})).dot(tangent.subs({t:t0}))

# returns the symbolic expression for the osculating plane wrt s 
# y is the vector to a point where the plane occurs (symbolic)
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter		
def osculating_plane_natural(y,r,s):
	tangent = diff(r,s)
	normal = diff(tangent,s)
	normal = normal / normal.magnitude()
	bi = tangent.cross(normal)
	return (y-r).dot(bi)

# Plot's a plane using matplotlib
# ax is the plot axis 
# e is the CoordSys3D 
# r1 is the vector to the point
# tn1 is the tangent to the curve traced by r1 at the point
# nm1 is the normal to the curve traced by r1 at the point
# X and Y are a matplotlib meshgrid
def plot_plane(ax,e,r1,tn1,nm1,X,Y):
	ax.plot_surface( float(r1.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
					float(r1.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
					float(r1.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), color = 'b',alpha = 0.1)

# Given a point and normal, this code finds two vectors perpendicular to the normal and 
# constructs the plane
# ax is the matplotlib plot.axis()
# e is the CoordSys3D
# P is the point as a symbolic vector with numerical components
# N is the normal as a symbolic vector with numerical components 
def plot_plane_from_point_normal(ax, e, P, N):
	a_1, a_2, a_3 = symbols("a_1, a_2, a_3")
	b_1, b_2, b_3 = symbols("b_1, b_2, b_3")
	n = a_1*e.i + a_2 * e.j + a_3 *e.k
	u = b_1*e.i + b_2 * e.j + b_3 *e.k
	A1 = 1
	A2 = 3
	A3 = -1
	if N.dot(e.i) != 0:
		new_b1_ = solve(n.dot(u),b_1)
		u_temp = u.subs({b_1:new_b1_[0]})
		
		B3 = solve(new_b1_[0].subs({a_1:A1,a_2:A2,a_3:A3}),b_3) 
		
		v = u_temp.cross(n)
		
		u_temp = u_temp.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		u_temp = u_temp.subs({b_3:B3[0]})
		u_temp = u_temp.subs({b_2:1})
		
		u_temp = u_temp.normalize()
		
		v_temp = v.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		v_temp = v_temp.subs({b_3:B3[0]})
		v_temp = v_temp.subs({b_2:1})
		
		v_temp = v_temp.normalize()
		
		X = np.linspace(-0.5,0.5,10)
		Y = np.linspace(-0.5,0.5,10)
		X,Y = np.meshgrid(X,Y)
		tn1 = u_temp
		nm1 = v_temp
		#print tn1
		#print nm1
		ax.plot_surface( float(P.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
		 float(P.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
		 float(P.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), alpha = 0.1)
	elif N.dot(e.j) != 0:
		new_b2_ = solve(n.dot(u),b_2)
		u_temp = u.subs({b_2:new_b2_[0]})
		
		B3 = solve(new_b1_[0].subs({a_1:A1,a_2:A2,a_3:A3}),b_3) 
		
		v = u_temp.cross(n)
		
		u_temp = u_temp.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		u_temp = u_temp.subs({b_3:B3[0]})
		u_temp = u_temp.subs({b_1:1})
		
		u_temp = u_temp.normalize()
		
		v_temp = v.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		v_temp = v_temp.subs({b_3:B3[0]})
		v_temp = v_temp.subs({b_1:1})
		
		v_temp = v_temp.normalize()
		
		X = np.linspace(-0.5,0.5,10)
		Y = np.linspace(-0.5,0.5,10)
		X,Y = np.meshgrid(X,Y)
		tn1 = u_temp
		nm1 = v_temp
		#print tn1
		#print nm1
		ax.plot_surface( float(P.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
		 float(P.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
		 float(P.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), alpha = 0.1)        
	elif N.dot(e.j) != 0:
		new_b3_ = solve(n.dot(u),b_3)
		u_temp = u.subs({b_3:new_b3_[0]})
		
		B2 = solve(new_b1_[0].subs({a_1:A1,a_2:A2,a_3:A3}),b_2) 
		
		v = u_temp.cross(n)
		
		u_temp = u_temp.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		u_temp = u_temp.subs({b_2:B2[0]})
		u_temp = u_temp.subs({b_1:1})
		
		u_temp = u_temp.normalize()
		
		v_temp = v.subs({a_1:N.dot(e.i),a_2:N.dot(e.j),a_3:N.dot(e.k) })
		v_temp = v_temp.subs({b_2:B2[0]})
		v_temp = v_temp.subs({b_1:1})
		
		v_temp = v_temp.normalize()
		
		X = np.linspace(-0.5,0.5,10)
		Y = np.linspace(-0.5,0.5,10)
		X,Y = np.meshgrid(X,Y)
		tn1 = u_temp
		nm1 = v_temp
		#print tn1
		#print nm1
		ax.plot_surface( float(P.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
		 float(P.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
		 float(P.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), alpha = 0.1)   
	
	
	
#=================================================================
### Binormal Vector
#=================================================================

# returns the symbolic expression for the binormal vector wrt s 
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter	
def binormal_natural(r,s):
	tangent = diff(r,s)
	norm = diff(tangent,s)
	norm = norm/norm.magnitude()
	return tangent.cross(normal)

# returns the symbolic expression for the binormal vector wrt s 
# tangent is the tangent vector as a function of natural parameter s
# s is the symbol for the natural parameter		
def binormal_natural_from_tangent(tangent,s):
	norm = diff(tangent,s)
	norm = norm/norm.magnitude()
	return tangent.cross(normal)

# returns the symbolic expression for the binormal from unit tangent and unit normal	
def binormal_from_tangent_normal(tanget,normal):
	return tangent.cross(normal)

# returns the symbolic expression for the binormal line with s as parameter
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the natural parameter (arc length)
# c is a symbolic constant multiplier	
def binormal_line(r,s,c):
	bi = binormal_natural(r,s)
	return r + c*bi

# returns the symbolic expression for the binormal line with s as parameter at point s0
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the natural parameter (arc length)
# s0 is a number on the interval [0:s]
# c is a symbolic constant multiplier		
# s0 is a number defining the length of the line
def binormal_line_at_point(r,s,s0,c,c0):
	return binormal_line(r,s,c).subs({s:s0,c:c0})

#=================================================================
### Torsion
#=================================================================

# returns the symbolic expression for the torsion wrt s 
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter	
def torsion_natural(r,s):
	tangent = unit_tangent_natural(r,s)
	normal = diff(tangent,s)
	normal = normal/normal.magnitude()
	binormal = tangent.cross(normal)
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

# returns the symbolic expression for the torsion wrt t
# r is the vector expression e.g. r = r_1(t)*e.i + r_2(t) * e.j + r_3 (t) *e.k
# t is the symbol for the time parameter	
def torsion_time(r,t):
	rprime1 = diff(r,t)
	rprime2 = diff(rprime1,t)
	rprime3 = diff(rprime2,t)
	num = rprime1.dot( rprime2.cross(rprime3))
	den = rprime1.cross(rprime2).magnitude() 
	return num / (den**2)

# returns the symbolic expression for the torsion wrt s
# tangent is vector function of s
# normal is a vector function of s
# t is the symbol for the time parameter		
def torsion_from_tangent_normal(tangent,normal,s):
	binormal = tangent.cross(normal)
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

# returns the symbolic expression for the torsion wrt s
# tangent is vector function of s
# binormal is a vector function of s
# t is the symbol for the time parameter	
def torsion_from_normal_binormal(normal,binormal,s):
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

# returns the symbolic expression for the derivative of binormal wrt s 
# r is the vector expression e.g. r = r_1(s)*e.i + r_2(s) * e.j + r_3 (s) *e.k
# s is the symbol for the natural parameter		
def dbds_natural(r,s):
	tangent = unit_tangent_natural(r,s)
	normal = diff(tangent,s)
	normal = normal/normal.magnitude()
	binormal = tangent.cross(normal)
	return diff(binormal, s)

# returns the symbolic expression for the torsion  
# dbds is the derivative of binormal wrt s
# normal is the normal vector as a function of s	
def torsion_from_dbds_normal(dbds,normal):
	return -dbds.dot(normal)


#=================================================================
### plot frenet frame  
#=================================================================

# plots 3 arrows
# ax is the matplotlib plot.axis()
# o is the point where the frame will appear
# T is a unit Tangent vector
# N is a unit Normal vector
# e is the CoordSys3D
# col is the color of the vector
def plot_frenet_frame(ax,o,T,N,e,col):
	plot_arrow(ax,o,T,e,r'$ \hat{T}$',col)
	plot_arrow(ax,o,N,e,r'$ \hat{N}$',col)
	B = T.cross(N)
	B = B/ B.magnitude()
	plot_arrow(ax,o,B,e,r'$ \hat{B}$',col)

	
# ax is the matplotlib plot.axis()
# r is the point where the frame will appear r=r(t)
# tangent is a unit Tangent vector as a function of time t
# normal is a unit normal vector as a function of time t
# binormal is a unit binormal vector as a function of time t
# e is the CoordSys3D
# time_point is a point on the interval t=[t0:tmax] 
# X,Y are a numpy meshgrid	
def plot_frenet_frame_2(ax, r, tangent, normal, binormal,e,time_point, X, Y):
	plot_arrow(ax,r.subs({t:time_point}), tangent.subs({t:time_point}), e, '','r')
	plot_arrow(ax,r.subs({t:time_point}), normal.subs({t:time_point}), e, '','g')
	plot_arrow(ax,r.subs({t:time_point}), binormal.subs({t:time_point}), e, '','b')
	r1 = r.subs({t:time_point})
	tn1 = tangent.subs({t:time_point})
	nm1 = normal.subs({t:time_point})
	bn1 = binormal.subs({t:time_point})
	ax.plot_surface( float(r1.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
		 float(r1.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
		 float(r1.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), color = 'b',alpha = 0.1)
	ax.plot_surface( float(r1.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(bn1.dot(e.i)),
		 float(r1.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(bn1.dot(e.j)),
		 float(r1.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(bn1.dot(e.k)),  color = 'g',alpha = 0.1)
	ax.plot_surface( float(r1.dot(e.i)) + X * float(bn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
		 float(r1.dot(e.j)) + X * float(bn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
		 float(r1.dot(e.k))+ X * float(bn1.dot(e.k)) + Y * float(nm1.dot(e.k)),  color = 'r',alpha = 0.1)
	
# returns a vector function for a mongepatch
# x*e.i + y*e.j + f(x,y)*e.k
# x and y are both symbols for coordinates 
# fxy is a function of x and y, such that z=f(x,y) 	
# e is the CoordSys3D
def monge_patch_z(x,y,fxy, e):
	return x*e.i + y*e.j + fxy * e.k
	
# plots a surface given 
# ax is the Matplotlib plot.axis()
# f is a function of x and y defining a surface such that z=f(x,y)
# x,y are symbols in f
# X,Y are a numpy meshgrid
# rs = rstride (row stride) for the matplotlib function ax.plot_surface
# cs = cstride (column stride) for the matplotlib function ax.plot_surface
def space_surface(ax, f, x, y, X,Y,rs,cs):
	fxy = lambdify( (x,y), f, "numpy" )
	# plot the lambda funcs
	ax.plot_surface(X,Y, fxy(X,Y),rstride=rs, cstride=cs,color='g',alpha=1,linewidth=0.0,antialiased=False)
	
# This returns a peaks function
# X,Y are numbers (or symbols)
def computePeaks( X, Y ):
	return 3*((1-X)**2)*exp(-(X**2) - (Y+1)**2) - 10*(X/5 - X**3 - Y**5)*exp(-X**2-Y**2) - (1/3)*exp(-(X+1)**2 - Y**2);

# returns a surface normal 
# r is a vector function defining a surface (e.g. monge patch)
# u is a paramter (r(u,v) can be a re-parameterization of r)
# v is a parameter for r
def surface_normal(r,u,v):
	ru_cross_rv = diff(r,u).cross(diff(r,v))
	return ru_cross_rv

# returns a unit surface normal 
# r is a vector function defining a surface (e.g. monge patch)
# u is a paramter (r(u,v) can be a re-parameterization of r)
# v is a parameter for r	
def surface_unit_normal(r,u,v):
	ru_cross_rv = diff(r,u).cross(diff(r,v))
	return ru_cross_rv/ru_cross_rv.magnitude()
	
# returns E, a first fundamental coefficient
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)	
def compute_E(x,u):
	xu = diff(x,u)
	return xu.dot(xu)

# returns F, a first fundamental coefficient
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_F(x,u,v):
	xu = diff(x,u)
	xv = diff(x,v)
	return xu.dot(xv)

# returns G, a first fundamental coefficient
# x is a vector function defining a surface (e.g. monge patch)
# v is a paramter (x(u,v) can be a re-parameterization of x)		
def compute_G(x,v):
	xv = diff(x,v)
	return xv.dot(xv)

# returns a list containing the first fundamental coefficients
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def first_fundamental_form(x,u,v):
	EFG = zeros(3,1)
	EFG[0] = compute_E(x,u)
	EFG[1] = compute_F(x,u,v)
	EFG[2] = compute_G(x,v)
	return EFG	
	
# returns the integral of surface area 
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
# u_min,u_max are the limits in the u parameter
# v_min,v_max are the limits in the v parameter
def surface_area_patch( x, u, v, u_min, u_max, v_min, v_max ):
	EFG = first_fundamental_form(x, u, v)
	xu_cross_xv = simplify(EFG[0]*EFG[2]- EFG[1]**2) #
	int_1 = simplify( integrate(sqrt(xu_cross_xv), (u, u_min, u_max)).doit())
	return N(integrate( int_1,(v,v_min,v_max)).doit())	

# return L the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_L_1(x,u,v):
	n = surface_unit_normal(x,u,v)
	nu = diff(n,u)
	xu = diff(x,u)
	return -xu.dot(nu)
	
# return M the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_M_1(x,u,v):
	n = surface_unit_normal(x,u,v)
	nu = diff(n,u)
	nv = diff(n,v)
	xu = diff(x,u)
	xv = diff(x,v)
	return (S(1)/2)*( -xu.dot(nv)-xv.dot(nu))

# return N the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_N_1(x,u,v):
	n = surface_unit_normal(x,u,v)
	nv = diff(n,v)
	xv = diff(x,v)
	return -xv.dot(nv)

# return L the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_L_2(x,u,v):
	n = surface_unit_normal(x,u,v)
	xuu = diff(x,u,2)
	return xuu.dot(n)
	
# return M the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x	
def compute_M_2(x,u,v):
	n = surface_unit_normal(x,u,v)
	xuv = diff(diff(x,v),u)
	return xuv.dot(n)

# return N the second fundamental coefficient	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def compute_N_2(x,u,v):
	n = surface_unit_normal(x,u,v)
	xvv = diff(diff(x,v),v)
	return xvv.dot(n)

# returns a list containing the coefficients of the second fundamental form	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def second_fundamental_form(x,u,v):
	LMN = zeros(3,1)
	LMN[0] = compute_L_1(x,u,v)
	LMN[1] = compute_M_1(x,u,v)
	LMN[2] = compute_N_1(x,u,v)
	return LMN

# returns a list containing the coefficients of the second fundamental form	
# x is a vector function defining a surface (e.g. monge patch)
# u is a paramter (x(u,v) can be a re-parameterization of x)
# v is a parameter for x		
def second_fundamental_form_version2(x,u,v):
	LMN = zeros(3,1)
	LMN[0] = compute_L_2(x,u,v)
	LMN[1] = compute_M_2(x,u,v)
	LMN[2] = compute_N_2(x,u,v)
	return LMN	
	
# computes the quadratic formula
# returns the roots of an equation a*x**2 + b*x + c = 0
def quadratic_formula(a,b,c):
	sols = zeros(2,1)
	sols[0]=simplify((-b**2 - sqrt(b**2-4*a*c))/(2*a))
	sols[1]=simplify((-b**2 + sqrt(b**2-4*a*c))/(2*a))
	return sols
	
# returns a vector containing a rectangular vector as a function of spherical coords
# e is the CoordSys3D
# Y is a matrix [r,phi,theta]	
def rect_vector_of_sphere_coords(e, Y):
	return Y[0] * cos (Y[2]) * sin(Y[1]) * e.i+ Y[0] * sin (Y[2]) * sin(Y[1]) * e.j +  Y[0] * cos(Y[1]) * e.k

# returns a vector containing a rectangular vector as a function of cylindrical coords
# e is the CoordSys3D
# Y is a matrix [r,phi,z]		
def rect_vector_of_cylinder_coords(e, Y):
	return Y[0] * cos (Y[1]) * e.i+ Y[0] * sin (Y[1]) * e.j +  Y[2] * e.k

# returns a vector containing a rectangular vector as a function of parabolic cylindrical coords
# e is the CoordSys3D
# Y is a matrix [y_1, y_2, y_3]		
def rect_vector_of_parab_cylinder_coords(e, Y):
	return (S(1)/2) (Y[0]**2 - Y[1]**2 ) * e.i+ Y[0] * Y[1] * e.j +  Y[2] * e.k

# returns a vector containing a rectangular vector as a function of parabololoidal coords
# e is the CoordSys3D
# Y is a matrix [y_1, y_2, y_3]		
def rect_vector_of_paraboloidal_coords(e, Y):
	return Y[0]*Y[1]*cos(Y[2]) * e.i+ Y[0]*Y[1]*sin(Y[2])  * e.j +  (S(1)/2) (Y[0]**2 - Y[1]**2 )* e.k

# returns a vector containing a rectangular vector as a function of elliptic cylindrical coords
# e is the CoordSys3D
# Y is a matrix [y_1, y_2, y_3]			
def rect_vector_of_elliptic_cylindrical_coords(e, Y):
	return cosh(Y[0]) *cos(Y[1])* e.i+  sinh(Y[0]) *sin(Y[1])* e.j +  Y[2]* e.k

# returns a vector containing a rectangular vector as a function of elliptic cylindrical coords
# e is the CoordSys3D
# Y is a matrix [y_1, y_2, y_3]			
def rect_vector_of_elliptic_cylindrical_coords(e, Y):
	return cosh(Y[0]) *cos(Y[1])* e.i+  sinh(Y[0]) *sin(Y[1])* e.j +  Y[2]* e.k


# converts a vector v in basis e into a matrix
# e is the CoordSys3D
# v is the vector
def vector_to_matrix_form(e,v):
	return Matrix([v.dot(e.i), v.dot(e.j),v.dot(e.k)])
	
# r is a vector function of coords i.e. rect_vector_of_sphere_coords()
# u1,u2,u3 are the coords 
def basis_vectors(r,u1,u2,u3):
	drdu1 = diff(r,u1)
	drdu2 = diff(r,u2)
	drdu3 = diff(r,u3)
	drdu1 = simplify( drdu1/drdu1.magnitude() )
	drdu2 = simplify( drdu2/drdu2.magnitude() )
	drdu3 = simplify( drdu3/drdu3.magnitude() )
	return [drdu1, drdu2, drdu3]
	
## e is the CoordSys3D
# v1, v2, v3 are the symbolic basis vectors
# e.g v1 = 1*e.i + 0*e.j+ 0*e.k   
def matrix_from_vector2(e,v1,v2,v3):
	return Matrix([[v1.dot(e.i),v1.dot(e.j),v1.dot(e.k)],
				   [v2.dot(e.i),v2.dot(e.j),v2.dot(e.k)],
				   [v3.dot(e.i),v3.dot(e.j),v3.dot(e.k)]])

# e is the coordSys3d
# v is the vector
# i is reserved
def vector_from_matrix(e,v,i):
	return v[0]*e.i + v[1]*e.j + v[2] * e.k	

# V and C are coordinate systems 	
# V = [x_1,x_2,x_3] where x_1=x_1(y_1,y_2,y_3) e.g y_1*cos(y_2) etc
# C = [y_1,y_2,y_3]
# returns Jacobian matrix of d x_i / d y_j
def Jacobian_WRT_coords(V,C):
	return V.jacobian(C)	

# A is a Jacobian matrix	
def metric_from_jacobian(A):
	return A.T * A
	
# computes the metric directly from the coords
def metric_from_coords(V,C):
	return metric_from_jacobian(Jacobian_WRT_coords(V,C))
	


# G is the metric
# p is the superscript
# a,b are the subscript indices
# X is the set of coordinates of the metric of dimension d
# d is the dimension of the space
# G_inv is the optional inverse of the metric
def christoffel_symbol_2_2(G, p, a, b, X, d, G_inv):
	output = 0
	if G_inv == 0:
		G_inv = G.inverse_ADJ()
	for c in range(0,d):
		output += G_inv[p,c]*(S(1)/2)*(diff(G[b,c],X[a]) +diff(G[c,a],X[b])-diff(G[a,b],X[c]))
	return output
	
# this was added because the original christoffel symbol code
# above does not extend to higher dimensions, neither does
# the inner function christoffel_symbol_2 (see above)
# and so
# metric is the metric d*d
# d is the dimension of the space
# basis is the basis from which the metric was derived [x_1, ... x_d]
# metric_inv is the optional inverse of the metric
def compute_christoffel_symbols_2(metric, d, basis, metric_inv):
	gamma2 = []
	for i in range(0,d):
		gamma2.append(zeros(d,d))
		
	for a in range(0,d):
		for b in range(0,d):
			for c in range(0,d):
				# puts this in matrix form with fancy text and indexing
				gamma2[a][b,c] = simplify(christoffel_symbol_2_2(metric, a, b, c, basis, d, metric_inv))
	
	return gamma2
	
# this was added because the original christoffel symbol code
# above does not extend to higher dimensions, neither does
# the inner function christoffel_symbol_2 (see above)
# and so
# metric is the metric d*d
# d is the dimension of the space
# basis is the basis from which the metric was derived [x_1, ... x_d]
# metric_inv is the optional inverse of the metric
def compute_christoffel_symbols_3(metric, d, basis, metric_inv):
	gamma2 = []
	for i in range(0,d):
		gamma2.append(zeros(d,d))
		
	for a in range(0,d):
		for b in range(0,d):
			for c in range(0,d):
				# puts this in matrix form with fancy text and indexing
				gamma2[a][b,c] = simplify(christoffel_symbol_2_2(metric, a, b, c, basis, d, metric_inv))
	
	
	return gamma2
	
from sympy import Array	

# T : covariant vector in matrix form
# Y : coordinates the vector is a function of
# gamma2: Christoffel symbols of the metric
# d : dimension of the space
def covariant_derivative_covariant_vector(T, Y, gamma2, d):
    d_r_T_i = zeros(d,d)
    for r in range(0,d): # free index r
        for i in range(0,d): # free index j
            d_r_T_i [i, r] = diff( T[i], Y[r] )
            for p in range(0,d): # dummy index p
        
                # using our vector matrix X=[y_1,y_2, y_3]
                 d_r_T_i [i, r] -= gamma2 [p][i, r]*T[p] ## perhaps one of the other gamma2 systems

    return d_r_T_i

# T : contravariant vector in matrix form
# Y : coordinates the vector is a function of
# gamma2: Christoffel symbols of the metric
# d : dimension of the space	
def covariant_derivative_contravariant_vector(T, Y, gamma2, d):
    d_r_T_i = zeros(d,d)
    for r in range(0,d): # free index r
        for i in range(0,d): # free index j
            d_r_T_i [i, r] = diff( T[i], Y[r] )
            for q in range(0,d): # dummy index q
        
                # using our vector matrix X=[y_1,y_2, y_3]
                d_r_T_i [i, r] += gamma2 [i][q, r]*T[q] ## perhaps one of the other gamma2 systems
            
    return d_r_T_i
	
# T : tensor to differentiate (vector as matrix)
# Y : coordinates
# DY: first time derivative of coordinates
# gamma_2: christoffel symbols of the metric
# d : dimension of the space
def absolute_derivative(T, Y, DY, gamma_2, d ):
	DT_i = zeros(d,1)
	for i in range(0,d):
		DT_i[i] =  diff(T[i],t)
		for q in range (0,d):
			for r in range(0,d):
				DT_i[i] += gamma_2[i][q,r]*T[q]*DY[r]
	return DT_i		

## 
## 
def absolute_acceleration(DY, D2Y, gamma_2, d ):
	acc = zeros(d,1)
	for i in range(0,d):
		acc[i] =  D2Y[i]
		for j in range (0,d):
			for s in range(0,d):
				acc[i] += gamma_2[i][j,s]*DY[j]*DY[s]
	return acc	
	
# untested, returns the physical acceleration corrected by obtaining
# the scale factors from the metric
# A is the absolute acceleration
# g is the metric	
def physical_acceleration_from_absolute_acceleration(abs_accel,g,d):
	abs_accel_corrected=zeros(d,1)

	for i in range(0,d):
		abs_accel_corrected[i] = simplify(sqrt(g[i,i])*abs_accel[i]) 
	return abs_accel_corrected
	

	

## returns the Riemann tensor with 1 contravariant index (the first) and 
## 3 covariant indices, accessed with R[a,b,c,d]
## gamma_2 is the set of christoffel symbols for the metric_
## X is the coord system X= [X_1,... X_d]  
## d is the number of dimensions of the metric e.g 4 
def Riemann_Tensor_2( gamma_2, X, d):
	# construct an array followed by a shape tuple
	# the array is of size d^4 and shape d,d,d,d
	Riem = MutableDenseNDimArray(zeros(d*d*d*d),(d,d,d,d))
	for a in range(0,d):
		for nu in range(0,d):
			for mu in range(0,d):
				for i in range(0,d):
					R =  diff(gamma_2[a][mu,i],X[nu]) 
					R -= diff(gamma_2[a][nu,i],X[mu])
					for beta in range(0,d):
						R += gamma_2[a][nu,beta]*gamma_2[beta][mu,i] 
						R -= gamma_2[a][mu,beta]*gamma_2[beta][nu,i]
					Riem[a,nu,mu,i] = simplify(trigsimp(R))
	return Riem
	
## new function, untested
# R_abcd : Reimann tensor ({a} contravariant, {bcd} covariant)
# d dimension of the space
# returns Ricci Tensor contracted on a and c
# note: destroys Riemann tensor in attempt to return Weyl tensor.
def compute_ricci_tensor(R_abcd, d):
    R_uv = MutableDenseNDimArray(zeros(d*d),(d,d))
    for u in range(0,d):
        for v in range(0,d):
            for a in range(0,d):
                for b in range(0,d):
                    if a == b:
                        R_uv[u,v] += R_abcd[a,u,b,v]
                        R_abcd[a,u,b,v] = 0 ## This presumabbly returns a weyl tensor?
    return R_uv
	
## R_uv : Ricci Tensor
# d dimension ofthe space	
def compute_ricci_scalar(R_uv,d):
	R=0
	for u in range(0,d):
		for v in range(0,d):
			if u==v:
				R += R_uv[u,v]
	return R

