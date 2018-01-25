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



def basis_vectors(r,u1,u2,u3):
	drdu1 = diff(r,u1)
	drdu2 = diff(r,u2)
	drdu3 = diff(r,u3)
	drdu1 = simplify( drdu1/drdu1.magnitude() )
	drdu2 = simplify( drdu2/drdu2.magnitude() )
	drdu3 = simplify( drdu3/drdu3.magnitude() )
	return [drdu1, drdu2, drdu3]
	
def matrix_from_vector2(e,v1,v2,v3):
	return Matrix([[v1.dot(e.i),v1.dot(e.j),v1.dot(e.k)],
				   [v2.dot(e.i),v2.dot(e.j),v2.dot(e.k)],
				   [v3.dot(e.i),v3.dot(e.j),v3.dot(e.k)]])

def vector_from_matrix(e,v,i):
	return v[0]*e.i + v[1]*e.j + v[2] * e.k

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
# requires plot_arrow
#=================================================================
def plot_basis(ax,o,v1,v2,v3,e,col):
	plot_arrow(ax,o,v1,e,r'$ \hat{e_1}$',col)
	plot_arrow(ax,o,v2,e,r'$ \hat{e_2}$',col)
	plot_arrow(ax,o,v3,e,r'$ \hat{e_3}$',col)
	
#=================================================================
### Tangent vectors to a space curve
#=================================================================
def unit_tangent_natural(r,s):
	return diff(r,s)

def unit_tangent_time(r,t):
	drdt = unit_tangent_natural(r,t)
	return drdt / drdt.magnitude()

#=================================================================
### Tangent Lines
#=================================================================
def tangent_line_natural(r,s,c):
	tangent = unit_tangent_natural(r,s)
	return r + c * tangent

def tangent_line_natural_at_point(r,s,s0,c,c0):
	return tangent_line_natural(r,s,c).subs({s:s0,c:c0}) 

def tangent_line_time(r,t,c):
	tangent = unit_tangent_time(r,t)
	return r + c * tangent

def tangent_line_time_at_point(r,t,t0,c, c0):
	return  tangent_line_time(r,t,c).subs({t:t0,c:c0}) 

#=================================================================
### Function to plot a space curve
#=================================================================
# ax is the plot axis, 
# r is the curve, 
# t is the parameter in r, 
# tt is the numpy parameter range
def space_curve(ax, e,  r, t, tt):
	fx = lambdify( t, r.dot(e.i), "numpy" )
	fy = lambdify( t, r.dot(e.j), "numpy" )
	fz = lambdify( t, r.dot(e.k), "numpy" )

	# plot the lambda funcs
	ax.plot(fx(tt),fy(tt),fz(tt))
	
#=================================================================
### Curvature and Normal Vector
#=================================================================   
def curv_vec_natural(r, s):
	tangent = unit_tangent_natural(r,s)
	return diff(tangent, s)

def curv_vec_from_tangent(tangent, s):
	return diff(tangent,s)

def curvature_natural(r,s):
	return curv_vec_natural(r,s).magnitude()

def curv_mag(k):
	return k.magnitude()

def rad_of_curv_natural(r,s):
	return 1/curvature_natural(r,s)

def rad_of_curv_from_curv_vec(k):
	return 1/curv_mag(k)

def curv_vec_time(r,t):
	drdt = diff(r,t)
	drdt_mag = drdt.magnitude()
	tangent = drdt / drdt_mag
	dTdt = diff(tangent,t)
	return dTdt / drdt_mag

def normal_vec_time(r,t):
	k = curve_vec_time(r,t)
	return k/k.magnitude()

def principle_normal_natural(r,s):
	k = curv_vec_natural(r,s)
	return k/k.magnitude()

def principle_normal_from_curv_vec(k):
	return k / k.magnitude()

def principle_normal_from_tangent(tangent,s):
	k = diff(tangent,s)
	return k / k.magnitude()

#=================================================================
### Curvature of a space curve
#=================================================================  
def curvature_time(r,t):
	rprime1 = diff(r,t)
	rprime2 = diff(rprime1,t)
	return (rprime1.cross(rprime2)).magnitude() / (rprime1.magnitude()**3)

#=================================================================
### Osculating plane
#=================================================================  
def normal_plane_natural(r, s, s0):
	tangent = unit_tangent_natural(r,s)
	return (r-r.subs({s:s0})).dot(tangent.subs({s:s0}))

def normal_plane_time(r,t,t0):
	tangent = unit_tangent_time(r,t)
	return (r-r.subs({t:t0})).dot(tangent.subs({t:t0}))

def osculating_plane_natural(y,r,s):
	tangent = diff(r,s)
	normal = diff(tangent,s)
	normal = normal / normal.magnitude()
	bi = tangent.cross(normal)
	return (y-r).dot(bi)

def plot_plane(ax,e,r1,tn1,nm1,X,Y):
	ax.plot_surface( float(r1.dot(e.i)) + X * float(tn1.dot(e.i)) + Y * float(nm1.dot(e.i)),
					float(r1.dot(e.j)) + X * float(tn1.dot(e.j)) + Y * float(nm1.dot(e.j)),
					float(r1.dot(e.k))+ X * float(tn1.dot(e.k)) + Y * float(nm1.dot(e.k)), color = 'b',alpha = 0.1)

# Given a point and normal, this code finds two vectors perpendicular to the normal and 
# constructs the plane

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
def binormal_natural(r,s):
	tangent = diff(r,s)
	norm = diff(tangent,s)
	norm = norm/norm.magnitude()
	return tangent.cross(normal)

def binormal_natural_from_tangent(tanget,s):
	norm = diff(tangent,s)
	norm = norm/norm.magnitude()
	return tangent.cross(normal)

def binormal_from_tangent_normal(tanget,normal):
	return tangent.cross(normal)

def binormal_line(r,s,c):
	bi = binormal_natural(r,s)
	return r + c*bi

def binormal_line_at_point(r,s,s0,c,c0):
	return binormal_line(r,s,c).subs({s:s0,c:c0})

#=================================================================
### Torsion
#=================================================================
def torsion_natural(r,s):
	tangent = unit_tangent_natural(r,s)
	normal = diff(tangent,s)
	normal = normal/normal.magnitude()
	binormal = tangent.cross(normal)
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

def torsion_time(r,t):
	rprime1 = diff(r,t)
	rprime2 = diff(rprime1,t)
	rprime3 = diff(rprime2,t)
	num = rprime1.dot( rprime2.cross(rprime3))
	den = rprime1.cross(rprime2).magnitude() 
	return num / (den**2)

	
def torsion_from_tangent_normal(tangent,normal,s):
	binormal = tangent.cross(normal)
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

def torsion_from_normal_binormal(normal,binormal,s):
	dbds = diff(binormal, s)
	return -dbds.dot(normal)

def dbds_natural(r,s):
	tangent = unit_tangent_natural(r,s)
	normal = diff(tangent,s)
	normal = normal/normal.magnitude()
	binormal = tangent.cross(normal)
	return diff(binormal, s)

def torsion_from_dbds_normal(dbds,normal):
	return -dbds.dot(normal)


#=================================================================
### plot frenet frame  
#=================================================================
def plot_frenet_frame(ax,o,T,N,e,col):
	plot_arrow(ax,o,T,e,r'$ \hat{T}$',col)
	plot_arrow(ax,o,N,e,r'$ \hat{N}$',col)
	B = T.cross(N)
	B = B/ B.magnitude()
	plot_arrow(ax,o,B,e,r'$ \hat{B}$',col)
	
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
	
def monge_patch_z(x,y,fxy, e):
	return x*e.i + y*e.j + fxy * e.k
	
def space_surface(ax, f, x, y, X,Y,rs,cs):
	fxy = lambdify( (x,y), f, "numpy" )
	# plot the lambda funcs
	ax.plot_surface(X,Y, fxy(X,Y),rstride=rs, cstride=cs,color='g',alpha=1,linewidth=0.0,antialiased=False)
	
def computePeaks( X, Y ):
	return 3*((1-X)**2)*exp(-(X**2) - (Y+1)**2) - 10*(X/5 - X**3 - Y**5)*exp(-X**2-Y**2) - (1/3)*exp(-(X+1)**2 - Y**2);

def surface_normal(s,u,v):
	su_cross_sv = diff(s,u).cross(diff(s,v))
	return su_cross_sv

def surface_unit_normal(s,u,v):
	su_cross_sv = diff(s,u).cross(diff(s,v))
	return su_cross_sv/su_cross_sv.magnitude()
	
def compute_E(x,u):
	xu = diff(x,u)
	return xu.dot(xu)

def compute_F(x,u,v):
	xu = diff(x,u)
	xv = diff(x,v)
	return xu.dot(xv)

def compute_G(x,v):
	xv = diff(x,v)
	return xv.dot(xv)

def first_fundamental_form(x,u,v):
	EFG = zeros(3,1)
	EFG[0] = compute_E(x,u)
	EFG[1] = compute_F(x,u,v)
	EFG[2] = compute_G(x,v)
	return EFG	
	
def surface_area_patch( x, u, v, u_min, u_max, v_min, v_max ):
	EFG = first_fundamental_form(x, u, v)
	xu_cross_xv = simplify(EFG[0]*EFG[2]- EFG[1]**2) #
	int_1 = simplify( integrate(sqrt(xu_cross_xv), (u, u_min, u_max)).doit())
	return N(integrate( int_1,(v,v_min,v_max)).doit())	

def compute_L_1(s,u,v):
	n = surface_unit_normal(s,u,v)
	nu = diff(n,u)
	xu = diff(s,u)
	return -xu.dot(nu)
	
def compute_M_1(s,u,v):
	n = surface_unit_normal(s,u,v)
	nu = diff(n,u)
	nv = diff(n,v)
	xu = diff(s,u)
	xv = diff(s,v)
	return (S(1)/2)*( -xu.dot(nv)-xv.dot(nu))

def compute_N_1(s,u,v):
	n = surface_unit_normal(s,u,v)
	nv = diff(n,v)
	xv = diff(s,v)
	return -xv.dot(nv)

def compute_L_2(s,u,v):
	n = surface_unit_normal(s,u,v)
	xuu = diff(s,u,2)
	return xuu.dot(n)
	
def compute_M_2(s,u,v):
	n = surface_unit_normal(s,u,v)
	xuv = diff(diff(s,v),u)
	return xuv.dot(n)

def compute_N_2(s,u,v):
	n = surface_unit_normal(s,u,v)
	xvv = diff(diff(s,v),v)
	return xvv.dot(n)

def second_fundamental_form(x,u,v):
	LMN = zeros(3,1)
	LMN[0] = compute_L_1(x,u,v)
	LMN[1] = compute_M_1(x,u,v)
	LMN[2] = compute_N_1(x,u,v)
	return LMN

def second_fundamental_form_version2(x,u,v):
	LMN = zeros(3,1)
	LMN[0] = compute_L_2(x,u,v)
	LMN[1] = compute_M_2(x,u,v)
	LMN[2] = compute_N_2(x,u,v)
	return LMN	
	
def quadratic_formula(a,b,c):
	sols = zeros(2,1)
	sols[0]=simplify((-b**2 - sqrt(b**2-4*a*c))/(2*a))
	sols[1]=simplify((-b**2 + sqrt(b**2-4*a*c))/(2*a))
	return sols
	
def rect_vector_of_sphere_coords(e, Y):
	return Y[0] * cos (Y[2]) * sin(Y[1]) * e.i+ Y[0] * sin (Y[2]) * sin(Y[1]) * e.j +  Y[0] * cos(Y[1]) * e.k

def rect_vector_of_cylinder_coords(e, Y):
	return Y[0] * cos (Y[1]) * e.i+ Y[0] * sin (Y[1]) * e.j +  Y[2] * e.k

def rect_vector_of_parab_cylinder_coords(e, Y):
	return (S(1)/2) (Y[0]**2 - Y[1]**2 ) * e.i+ Y[0] * Y[1] * e.j +  Y[2] * e.k

def rect_vector_of_paraboloidal_coords(e, Y):
	return Y[0]*Y[1]*cos(Y[2]) * e.i+ Y[0]*Y[1]*sin(Y[2])  * e.j +  (S(1)/2) (Y[0]**2 - Y[1]**2 )* e.k

def rect_vector_of_elliptic_cylindrical_coords(e, Y):
	return cosh(Y[0]) *cos(Y[1])* e.i+  sinh(Y[0]) *sin(Y[1])* e.j +  Y[2]* e.k

def rect_vector_of_elliptic_cylindrical_coords(e, Y):
	return cosh(Y[0]) *cos(Y[1])* e.i+  sinh(Y[0]) *sin(Y[1])* e.j +  Y[2]* e.k



def vector_to_matrix_form(e,v):
	return Matrix([v.dot(e.i), v.dot(e.j),v.dot(e.k)])


def Jacobian_WRT_coords(V,C):
	return V.jacobian(C)	

	
def christoffel_symbol_1(G, a, b, c, X):
	return (S(1)/2)*(diff(G[b,c],X[a]) +diff(G[c,a],X[b])-diff(G[a,b],X[c]))
	
def christoffel_symbol_2(G, p, a, b, X):
	output = 0
	G_inv = G.inverse_ADJ()
	for c in range(0,3):
		output += G_inv[p,c]*(S(1)/2)*(diff(G[b,c],X[a]) +diff(G[c,a],X[b])-diff(G[a,b],X[c]))
	return output
	
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
# metric is the metric
# d is the dimension of the space
# basis is the basis from which the metric was derived
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
	
def absolute_acceleration(DY, D2Y, gamma_2, d ):
	acc = zeros(d,1)
	for i in range(0,d):
		acc[i] =  D2Y[i]
		for j in range (0,d):
			for s in range(0,d):
				acc[i] += gamma_2[i][j,s]*DY[j]*DY[s]
	return acc	
	
from sympy import Array
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