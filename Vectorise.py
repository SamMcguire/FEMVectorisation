import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class ImageVec:
	WHITE_PIX = [1,1,1]
	##############################################
	##############Public functions################
	##############################################
	def __init__(self, fileName):
		self.img = mpimg.imread(fileName)
		#Removing alpha channel
		if(self.img.shape[2]==4):
			self.img = self.img[:,:, :3]

		self.pixelList = None
		self.kList = None
		self.breakPoints = None
		self.segments = None
		self.boundaryClasses = None

	#Function to get line vectorisation of image
	#Output: Numpy array of endpoints of line segments [[x_1,y_1],[x_2,y_2],...] where locations are pixel locations
	#INPUTS: 
	#	kWidth: How wide curves it recognises
	#	blurRadius: How susceptible to noise it is
	#	numSegments: number of line segments wanted
	# 	display: Wether to display vectorisation
	#	saveFig: file to save fig to(leave empty to not save)
	def GetVectorisation(self, kWidth = 20, blurRadius = 30, numSegments = 10,
	 display = True, saveFig = ""):
		self.pixelList = ImageVec.GetPixelList(self.img)
		boundaryPnts = ImageVec.GetBoundaryPoints(self.img, self.pixelList)
		self.boundaryClasses = ImageVec.GetBoundaryClasses(self.img, self.pixelList)

		kList = ImageVec.GetDotCurvatureList(kWidth, self.pixelList)	
		self.kList = ImageVec.GaussianFilt(blurRadius, kList)

		self.breakPoints = ImageVec.GetBreakPoints(numSegments, 1000, self.kList, boundaryPnts)
		self.breakPoints = np.unique(self.breakPoints) #removing duplicates 

		#Saving figures
		if(saveFig or display):
			ImageVis.QuadPlot(self.img, self.pixelList, self.kList, self.boundaryClasses, self.breakPoints)
		if(saveFig):
			ImageVis.SaveFig(saveFig)
		if(display):
			plt.show()

		self.segments = [(self.pixelList[brk][0], self.pixelList[brk][1]) for brk in self.breakPoints ]
		return self.segments


	#Generates geometry for NGSolve FEM
	#Output: sets geometry of geo
	#Inputs:
	#	geo : spline geometry object
	#	scale : scale of mesh
	def GenerateGeometry(self, geo, scale = 1):
		#Scaling points to be in range [0,scale]
		pnts = scale*(self.segments-np.min(self.segments))/(np.max(self.segments)-np.min(self.segments))
		#setting bottom left as (0,0)
		pnts = np.array([[pnt[0], scale-pnt[1]] for pnt in pnts ])
		#sometimes order is wrong, so fix it if so
		flipped = False
		if(pnts[0][0]>pnts[-1][0]):
			flipped = True
			pnts = np.flipud(pnts)

		newPts = [geo.AppendPoint(*pnt) for pnt in pnts]
		#Creating curves
		curves = []
		n = len(self.segments)
		for i in range(n):
			if(flipped):
				pixLoc = self.segments[n-i-1]
			else:
				pixLoc = self.segments[i]
			colStr = ImageVec.GetColorString(self.img[pixLoc[1], pixLoc[0]])
			name = str(self.boundaryClasses[colStr])
			curves.append([["line", newPts[i], newPts[(i+1)%n]], name])

		[geo.Append(c,bc=bc) for c,bc in curves];




	##############################################
	#############Private functions################
	##############################################
	#Function to get list of pixels in order
	#Output: list of pixels in form [[x1,y1],[x2,y2]...]
	#Input: img array
	def GetPixelList(_img):
		#padding with zeros to handle edge cases
		img = np.pad(_img, pad_width = ((1,1),(1,1),(0,0)), constant_values = 1)
		#Getting list of non-white pixels to pick 1 out to start walk
		indices = np.where(img != ImageVec.WHITE_PIX)
		x = indices[1][0]
		y = indices[0][0]
		#Getting the number of pixels in the domain border!
		numPixels = int(indices[0].shape[0]/3)
		#[[x1,y1],[x2,y2]...]
		pixelLst = np.zeros((numPixels, 2), int)
		for i in range(numPixels):
			pixelLst[i] = [x-1,y-1]
			#Setting to white so doesnt look at this pixel again
			img[y,x] = ImageVec.WHITE_PIX
			x,y = ImageVec.GetNextPixel(x,y,img)
			#If it can not find the next pixel print an error
			if(x == -1 and i < numPixels-1):
				print("Error in reading image!")
				ImageVis.PlotError(img, pixelLst)
				return -1
		return pixelLst

	#Function to get the raw curvature list
	#Output: list [k1,k2,k3,...], where ki is float from [0,2] representing curvature of pixel i
	def GetDotCurvatureList(r, pixelList):
		numPixels = pixelList.shape[0]
		kList = np.zeros(numPixels, float)
		for i in range(numPixels):
			#getting unit vectors in either direction
			l1 = ImageVec.GetUnitVec(pixelList[i], pixelList[(i+r)%numPixels])#pixel on left
			l2 = ImageVec.GetUnitVec(pixelList[i], pixelList[(i-r)%numPixels])#pixel on right
			#performing dot product (must clamp as in some cases is a small negative number)
			kList[i] = max(l1[0]*l2[0]+l1[1]*l2[1]+1, 0)
		return kList

	#Gaussian filter function:
	#Output: filtered curvature list
	#Inputs: 
	#	r: radius of blur
	#	kList: list to blur
	def GaussianFilt(r, kList):
		return gaussian_filter1d(kList, r, mode='wrap') 

	#Calculates repulsive force of pixel with curvature k at distance dist
	def GetForce(k, dist):
		return 1/((dist+1)**3*(k+0.0000001))

	#Function to get endpoints of line segments
	#Output: list of breakPoints [i1,i2,...], where ij is an index of which point in the kList is a breakpoint
	def GetBreakPoints(numSegments, numTrials, kList, boundaryPoints):
		#getting peaks
		peaks, _ = find_peaks(kList, prominence = 0.005) 
		#setting stationary points to peaks and boundary points
		still = np.concatenate((peaks, boundaryPoints))
		stepSize = (int)(len(kList)/numSegments)
		#uniformly distributing break points
		breakIndxs = np.arange(0,len(kList)-stepSize+stepSize*(len(still)+1),stepSize)
		brkCopy = breakIndxs.copy()

		#setting locations for stationary points
		for i in range(1,len(still)+1):
			brkCopy[-i] = still[i-1]
		#removing duplicates
		brkCopy = np.unique(brkCopy)

		for i in range(numTrials):
			breakIndxs = np.sort(brkCopy.copy())
			for j in range(1,len(breakIndxs)-1):
				#dont move stationary points
				if(brkCopy[j] in still):
					continue

				f = ImageVec.GetSumForce(j, breakIndxs, kList)
				newIndx = (breakIndxs[j]-round(f))%len(kList)
				#set new location if not going to land on another point
				if(newIndx not in breakIndxs):
					brkCopy[j] = newIndx

		return brkCopy 

	#Returns next colored pixel by anticlockwise rotatoin starting from top
	def GetNextPixel(x,y, img):
		if(img[y-1,x,0]<1):
			return (x,y-1)
		if(img[y,x-1,0]<1):
			return (x-1,y)
		if(img[y+1,x,0]<1):
			return (x,y+1)
		if(img[y,x+1,0]<1):
			return (x+1,y)

		if(img[y-1,x-1,0]<1):
			return (x-1,y-1)
		if(img[y+1,x-1,0]<1):
			return (x-1,y+1)
		if(img[y+1,x+1,0]<1):
			return (x+1,y+1)
		if(img[y-1,x+1,0]<1):
			return (x+1,y-1)
		return (-1,-1)

	def GetUnitVec(pixA, pixB):
		lineVec = [pixB[0]-pixA[0],pixB[1]-pixA[1]]
		vecMag =  math.sqrt(lineVec[0]**2+lineVec[1]**2)
		return [lineVec[0]/vecMag ,lineVec[1]/vecMag]

	#Gets points on the boundary between two colors
	def GetBoundaryPoints(img, pixelList):
		boundaryPnts = []
		n = len(pixelList)
		for i in range(len(pixelList)):
			thisPix = img[pixelList[i][1],pixelList[i][0]]
			nextPix = img[pixelList[(i+1)%n][1],pixelList[(i+1)%n][0]]
			if(not np.array_equal(thisPix, nextPix)):
				boundaryPnts.append(i)
		return boundaryPnts

	#Gets dictionary of boundary classes
	#Key: Color as string
	#Value: id
	#id is int between 0 and n(num of boundary classes)
	def GetBoundaryClasses(img, pixelList):
		boundaryClasses = {}
		j = 0
		for i in range(len(pixelList)):
			thisPix = img[pixelList[i][1],pixelList[i][0]]
			#getting color as a string
			colStr = ImageVec.GetColorString(thisPix)
			#if not already a boundary class, make a new class
			if(colStr not in boundaryClasses):
				boundaryClasses[colStr] = j
				j = j +1
		return boundaryClasses

	#Function to convert from color list [r,g,b] to string 'r,b,g'
	def GetColorString(color):
		colStr = np.array2string(color, precision=4, separator=',',
                      suppress_small=True)
		return colStr[1:-1]

	#Gets sum of forces of a particular point
	def GetSumForce(indx, breakIndxs, kList):
		leftDist = breakIndxs[indx]-breakIndxs[indx-1]
		leftK = kList[breakIndxs[indx-1]]
		leftF = ImageVec.GetForce(leftK, leftDist)

		rightDist = breakIndxs[indx+1]-breakIndxs[indx]
		rightK = kList[breakIndxs[indx+1]]
		rightF = ImageVec.GetForce(rightK, rightDist)	

		f = 3 if rightF > leftF else -3
		return f




class ImageVis:
	def QuadPlot(img, pixelList, kList, boundaryClasses, breakPoints = []):
		x = np.arange(0,10)
		y = np.arange(0,100, 10)
		fig, axs = plt.subplots(2, 2)
		ImageVis.PlotCurvedImg(img, pixelList, kList, fig, axs[0,0])
		ImageVis.PlotRawVectorization(img, pixelList, kList, axs[1,0], breakPoints)
		axs[1, 0].sharex(axs[0, 0])
		axs[1, 0].sharey(axs[0, 0])

		ImageVis.PlotOverlayVectorization(img, pixelList, kList, axs[0,1], breakPoints)
		axs[0,1].sharex(axs[0, 0])
		axs[0,1].sharey(axs[0, 0])

		ImageVis.PlotCurvature(kList, axs[1,1], breakPoints)
		# axs[1, 1].plot(x + 2, y + 2)
		# axs[1, 1].set_title("Curvature")
		#fig.tight_layout()
		ImageVis.MakeLegend(fig, boundaryClasses)

		

	def MakeLegend(fig, boundaryClasses):
		handles = []
		for key in boundaryClasses:
			col = [float(i) for i in key.split(",")]
			#print(boundaryClasses[key])
			patch = mpatches.Patch(color=col, label=boundaryClasses[key])
			handles.append(patch)

		#red_patch = mpatches.Patch(color='red', label='The red data')
		fig.legend(handles=handles)

	def PlotCurvature(kList, ax, breakIndxs = []):
		x = np.arange(0,kList.shape[0])
		ax.plot(x, kList)
		y = []
		x2 = []
		for brk in breakIndxs:
			y.append(kList[brk])
			x2.append(brk)
		ax.scatter(x2, y, color = 'orange')
		ax.set_title("Curvature")

	def PlotOverlayVectorization(img, pixelList, kList, ax, breakPoints):
		x = []
		y = []
		temp = np.unique(breakPoints)
		for i in temp:
			x.append(pixelList[i][0])
			y.append(pixelList[i][1])
		x.append(pixelList[0][0])
		y.append(pixelList[0][1])
		if(len(x)>1):
			ax.scatter(x,y, color = img[y,x], s = 5)
			ax.plot(x,y, alpha = 0.5)
		ax.imshow(img)
		ax.set_title("Overlay Vectorisation")

	def PlotRawVectorization(img, pixelList, kList, ax, breakPoints):
		x = []
		y = []
		temp = np.unique(breakPoints)
		for i in temp:
			x.append(pixelList[i][0])
			y.append(pixelList[i][1])
		x.append(pixelList[0][0])
		y.append(pixelList[0][1])
		if(len(x)>1):
			ax.plot(x,y, linewidth = '0.5')
		temp = img.copy()
		temp[:,:] = 1
		ax.imshow(temp)

		#ax.invert_yaxis()
		ax.set_title("Raw Vectorization")

	def PlotCurvedImg(img, pixelList, kList, fig, ax):
		k_norm = (kList-np.min(kList))/(np.max(kList)-np.min(kList))

		img = img.copy()
		i = 0
		for pixel in pixelList:
			val = 1
			valR = 0.2 + k_norm[i] * 7
			valB = 0.2 + k_norm[i] * 3
			valG = 0.2 + k_norm[i] * 0.5
			if(len(img[pixel[1],pixel[0]]) == 4):
				img[pixel[1],pixel[0]] = [valR,valB,valG,1]
			else:
				img[pixel[1],pixel[0]] = [valR,valB,valG]
			i+=1
		ax.imshow(img)

		ax.set_title("Curved Image")

	def PlotError(img, pixelList):
		img = img.copy()
		height = img.shape[0]
		width = img.shape[1]
		for x in range(width):
			for y in range(height):
				if(not np.array_equal(img[y,x], [1,1,1])):
					img[y,x] = [0,0,0]

		i = 0
		latest = 0
		for pixel in pixelList:
			img[pixel[1],pixel[0]] = [0.8,0.2,0.3]
			if(pixel[0] !=0 and pixel[1] != 0):
				latest = i
			i += 1	

		plt.imshow(img)
		y = [pixelList[latest][1]]
		x = [pixelList[latest][0]]
		plt.scatter(x,y, color = "blue", s = 50)

		plt.show()

	def SaveFig(fileName):
		plt.savefig(fileName, dpi=1000)