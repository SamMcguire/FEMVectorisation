from Vectorise import ImageVec 
import matplotlib.pyplot as plt
import numpy as np

from ngsolve import Mesh
from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()


kr = 20
vec = ImageVec('Images/Dino.png')
segments = vec.GetVectorisation(kWidth = 30, blurRadius = 60, numSegments = 20, 
	display = True, saveFig = "")

vec.GenerateGeometry(geo)
geo.SetDomainMaxH(4, 0.01)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)