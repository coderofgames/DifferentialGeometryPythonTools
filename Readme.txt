This is a set of differential geometry tools using sympy, numpy and matplotlib.

The python file is a module, not a package, so there is no need for setup.py etc.

The utils include space curve and surface plotting, computing and plotting tangent, 
normal vectors, planes. Fundamental forms, metrics, christoffel symbols, Reiman Tensors, etc.

Example notebook is WIP
   
Installation
------------
1. Download the zip file and copy sdg.py into a directory that you can include from
   in Anaconda2 on Windows a good location is C:\Users\your_user_name\Anaconda2\Lib\site-packages


#To import the module
import sdg

#to see help on a function
# unfortunately due to other dependencies help is not very useful yet.
print help(sdg.space_curve)

# to use a function
sdg.space_curve(...) 

see 

https://github.com/coderofgames/Python-Math/blob/master/Vectors/Lagrangian/Optics.ipynb

at the end of the file for an example usage. Documentation will be complete soon

