import matplotlib
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Circle, Rectangle, Arrow, PathPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from typing import Iterable
import numpy as np

from .discretization import Discretization
from .graph import Graph
from .edge import Edge
from .node import Node


class CLonGraphPlot:
  """This class provides a data structure for plots of solutions.
One object will hold one graph and provide the facilities to plot it"""


  def __init__(self, discreteEqn: Discretization, pltRefinement: int):
    self._gridSpecConventionalPlots = None
    self._Writer                    = animation.writers['ffmpeg']
    self._writer                    = self._Writer(fps=25, metadata=dict(artist='Markus Musch'), bitrate=1800)
# Data structures for the 3D Plot
    self._plot3D   = None 
    self._ax3DPlot = None
# Data structures for the 3D animation
    self._animatedSolution = None 
    self._axAnimation      = None
    self._edgeFrame        = []
    self._nodeFrame        = []
    self._exactFrameEdge   = []
    self._exactFrameNode   = []
    self._plotAnimation3d  = None
#
    self._story   = {}
    self._maxX    = 1.
    self._minX    = 0.
    self._maxY    = 1.
    self._minY    = 0.
    self._maxZ    = 0.
    self._minZ    = 0.
    self._xMargin = 0.1
    self._yMargin = 0.1
    self._zMargin = 0.01
#
    self._pltRefinement = pltRefinement
    self._discreteEqn   = discreteEqn
    self._graph         = self._discreteEqn.getGraph()
    self._story         = self._discreteEqn.getStory()


  def updateStory(self):
    self._story = self._discreteEqn.getStory()


  def updateMinMaxZ(self):
    self._minZ = self._discreteEqn.getMinSolutionVal()
    self._maxZ = self._discreteEqn.getMaxSolutionVal()


  def checkPlotBoundsX(self, newXBound):
    if self._maxX < newXBound:
      self._maxX = newXBound
    if self._minX > newXBound:
      self._minX = newXBound


  def checkPlotBoundsY(self, newYBound):
    if self._maxY < newYBound:
      self._maxY = newYBound
    if self._minY > newYBound:
      self._minY = newYBound


  def checkPlotBoundsZ(self, newZBound: float):
    if self._maxZ < newZBound:
      self._maxZ = newZBound
    if self._minZ > newZBound:
      self._minZ = newZBound


  def initAnimation2d(self) -> Iterable:
    maxInEdges  = max([len(self._graph.getConnectInEdges(x.getNodeID())) for x in \
      self._graph.nodeIterator()])
    maxOutEdges = max([len(self._graph.getConnectOutEdges(x.getNodeID())) for x in \
      self._graph.nodeIterator()])
    self._gridSpecConventionalPlot = GridSpec(max([maxInEdges, maxOutEdges]), \
      2*self._graph.getNumJointNodes(), figure=self._conventionalPlot)
    lines = []
    nodeCounter = 0
# Iterate over all nodes in the graph
    for node in self._graph.nodeIterator():
      if node.isJointNode():
        tmpNodeID          = node.getNodeID()
        connectInEdges     = self._graph.getConnectInEdges(tmpNodeID)
        connectOutEdges    = self._graph.getConnectOutEdges(tmpNodeID)
        numConnectInEdges  = len(connectInEdges)
        numConnectOutEdges = len(connectOutEdges)
# Iterate over all ingoing edges of the specific node
        for i, edgeID in zip(range(numConnectInEdges), connectInEdges):
          axTmp = self._conventionalPlot.add_subplot(self._gridSpecConventionalPlot[i, 2*nodeCounter])
          axTmp.set_xlim(0, 1)
          axTmp.set_ylim(self._minZ - self._zMargin, self._maxZ + self._zMargin)
# Format the plot
          axTmp.get_yaxis().set_visible(False)
          if not i is (numConnectInEdges-1):
            axTmp.get_xaxis().set_visible(False)
          if i == 0:
            axTmp.set_title('Node ' + str(tmpNodeID) + ': ingoing')
# Create a line object and add it to the dictionary
          lineTmp, = axTmp.plot([], [], 'o', markersize=2)
          lineTmp.set_data([], [])
          lines.append(lineTmp)
          self._conventionalPlotLineObj.update({(2*tmpNodeID, edgeID) : lineTmp})
# Iterate over all outgoing edges of the specific node
        for i, edgeID in zip(range(numConnectOutEdges), connectOutEdges):
          axTmp = self._conventionalPlot.add_subplot(self._gridSpecConventionalPlot[i, 2*nodeCounter+1])
          axTmp.set_xlim(0, 1)
          axTmp.set_ylim(self._minZ - self._zMargin, self._maxZ + self._zMargin)
# Set the position of the outgoing edges closer to the ingoing edges to create visual coherence
# Consider using the AxesGrid toolkit instead
          posOut = axTmp.get_position()
          pointsOut = posOut.get_points()
          pointsOut[0][0] -= 0.02
          posOut.set_points(pointsOut)
          axTmp.set_position(posOut)
# Format the plot
          if not i is (numConnectOutEdges-1):
            axTmp.get_xaxis().set_visible(False)
          axTmp.get_yaxis().set_visible(False)
          if i == 0:
            axTmp.set_title('Node ' + str(tmpNodeID) + ': outgoing')
# Create a line object and add it to the dictionary
          lineTmp, = axTmp.plot([], [], 'o', markersize=2)
          lines.append(lineTmp)
          self._conventionalPlotLineObj.update({(2*tmpNodeID+1, edgeID) : lineTmp})
        nodeCounter += 1
    return lines


  def updateAnimation2d(self, timePoint: int) -> Iterable:
    lines = []
# Iterate over all nodes in the graph
    for node in self._graph.nodeIterator():
# The 2d animation only has to be updated for joint nodes
      if node.isJointNode():
        tmpNodeID          = node.getNodeID()
        connectInEdges     = self._graph.getConnectInEdges(tmpNodeID)
        connectOutEdges    = self._graph.getConnectOutEdges(tmpNodeID)
        numConnectInEdges  = len(connectInEdges)
        numConnectOutEdges = len(connectOutEdges)
# Iterate over all ingoing edges of the specific node
        for i, edgeID in zip(range(numConnectInEdges), connectInEdges):
          lineTmp = self._conventionalPlotLineObj[(2*tmpNodeID, edgeID)]
          tmpData = self._story[('Edge', timePoint, edgeID)][2, :]
          lineTmp.set_data(np.linspace(0, 1, len(tmpData)), tmpData) 
          lines.append(lineTmp)
# Iterate over all outgoing edges of the specific node
        for i, edgeID in zip(range(numConnectOutEdges), connectOutEdges):
          lineTmp = self._conventionalPlotLineObj[(2*tmpNodeID+1, edgeID)]
          tmpData = self._story[('Edge', timePoint, edgeID)][2, :]
          lineTmp.set_data(np.linspace(0, 1, len(tmpData)), tmpData)
          lines.append(lineTmp)
    return lines 


  def animateGraph2d(self):
    self._plotAnimation2d = animation.FuncAnimation(fig=self._conventionalPlot, \
      func=self.updateAnimation2d, frames=self._discreteEqn.getNumTimeSteps(), \
      init_func=self.initAnimation2d, interval=40, blit=True, repeat=False) 


  def showAnimation2d(self):
    plt.figure(self._conventionalPlot.number)
    self._conventionalPlot.subplots_adjust(hspace=0)
    manager = plt.get_current_fig_manager()


  def saveAnimation2d(self, fileName: str):
    if(self._plotAnimation2d is None):
      print("No 2D plot animation is available")
    else:
      self._plotAnimation2d.save('./Plots/' + fileName + '.mp4', writer=self._writer)


  def plotGraphProjection2d(self):
    for node in self._graph.nodeIterator():
      self.plotNode(node, self._ax2dProjection, flag3d=False)
    for edge in self._graph.edgeIterator():
      self.plotEdge(edge, self._ax2dProjection, flag3d=False)


  def initAnimationProjection2d(self) -> Iterable:
    for edge in self._graph.edgeIterator():
      self._edgeFrame2dProjection.append(self._ax2dProjection.plot([], [])[0]) 
    for node in self._graph.nodeIterator(): 
      self._nodeFrame2dProjection.append(self._ax2dProjection.plot([], [])[0]) 
    self._ax2dProjection.set_xlim(self._minX - self._xMargin, self._maxX + self._xMargin)
    self._ax2dProjection.set_ylim(self._minY - self._yMargin, self._maxY + self._yMargin)
    return []
      

  def updateAnimationProjection2d(self, timeStep: int) -> Iterable:
    for dataSet, edge in zip(self._edgeFrame2dProjection, self._graph.edgeIterator()):
      edgeID = edge.getEdgeID()
      data = self._story[('Edge', timeStep, edgeID)]
      x = data[0, :]
      y = data[1, :]
      vals = data[2, :]
      col = vals 
      if (edge.getInNode().getNodeID() is edge.getOutNode().getNodeID()):
        piFrac = (2.*np.pi)/len(x)
        centerX = x[0]
        centerY = y[0]
        for i in range(len(x)):
          x[i] = centerX + np.cos((i+(1/2))*piFrac-0.5*np.pi)
          y[i] = centerY + np.sin((i+(1/2))*piFrac-0.5*np.pi) + 1.
        for i in range(len(x)):
          self.plotDot(self._ax2dProjection, x[i], y[i], (1-col[i], col[i], 0))
      else:
        for i in range(len(x)):
          self.plotSquare(self._ax2dProjection, x[i], y[i], (1-col[i], col[i], 0))
      tmpData = np.empty((2,2))
      dataSet.set_data(tmpData)
    for dataSet, node in zip(self._nodeFrame2dProjection, self._graph.nodeIterator()):
      nodeID = node.getNodeID()
      tmpData = self._story[('Node', timeStep, nodeID)][2, 1]
      col = tmpData/(self._maxZ - self._minZ)
      self.plotNode(node, self._ax2dProjection, faceColor=(0, 0, col), flag3d=False)
      tmpData = np.empty((2,2))
      dataSet.set_data(tmpData)
    return []


  def animateProjection2d(self):
    numFrames = self._discreteEqn.getNumTimeSteps()
    self._plotAnimation2dProjection = animation.FuncAnimation(fig=self._projection2d, \
      func=self.updateAnimationProjection2d, frames=numFrames, init_func=self.initAnimationProjection2d, \
      interval=40, blit=True, repeat=False)


  def showAnimationProjection2d(self):
    plt.figure(self._projection2d.number)
    manager = plt.get_current_fig_manager()


  def saveAnimationProjection2d(self, fileName: str):
    self._plotAnimation2dProjection.save('./Plots/' + fileName + '.mp4', writer=self._writer)


  def plotSquare(self, plotAx: Axes, xCoord: float, yCoord: float, \
    faceColor: tuple=(0,0,0), sqHeight: float=0.025):
    sq = Rectangle((xCoord-0.0375, yCoord), sqHeight, 0.025)
    sq.set_facecolor(faceColor)
    plotAx.add_patch(sq)


  def plotDot(self, plotAx: Axes, xCoord: float, yCoord: float, \
    faceColor: tuple=(0,0,0)):
    point = Circle((xCoord, yCoord), 0.025)
    point.set_facecolor(faceColor)
    plotAx.add_patch(point)


  def plotArrow(self, xCoord: float, yCoord: float, dx: float, dy: float, \
    plotAx: Axes, faceColor: tuple=(0, 0, 0)):
    arrow = Arrow(xCoord, yCoord, -dx, -dy, width=0.05, edgecolor='k', facecolor=faceColor)
    plotAx.add_patch(arrow)


  def plotEdge(self, pltEdge: Edge, plotAx: Axes, faceColor: tuple=(0, 0, 0), flag3d: bool=True):
    xCoordIn  = pltEdge.getInNode().getCoordinate()[0]
    xCoordOut = pltEdge.getOutNode().getCoordinate()[0]
    yCoordIn  = pltEdge.getInNode().getCoordinate()[1]
    yCoordOut = pltEdge.getOutNode().getCoordinate()[1]
    xSlope    = xCoordOut-xCoordIn
    ySlope    = yCoordOut-yCoordIn
    self.checkPlotBoundsX(xCoordIn)
    self.checkPlotBoundsY(yCoordIn)
    self.checkPlotBoundsX(xCoordOut)
    self.checkPlotBoundsY(yCoordOut)
    if (pltEdge.getInNode().getNodeID() is pltEdge.getOutNode().getNodeID()):
      piFrac = (2.*np.pi)/self._pltRefinement
      xOld   = xCoordIn
      yOld   = yCoordIn
      for i in range(0, self._pltRefinement):
        xValue = xCoordIn + np.cos((i+(1/2))*piFrac-0.5*np.pi)
        yValue = yCoordIn + np.sin((i+(1/2))*piFrac-0.5*np.pi) + 1.
        self.checkPlotBoundsX(xValue)
        self.checkPlotBoundsY(yValue)
        if i%2 == 0:
          dx     = xOld - xValue
          dy     = yOld - yValue
          arrow  = Arrow(xOld, yOld, -dx, -dy, width=0.05, facecolor=faceColor)
          plotAx.add_patch(arrow)
          if flag3d:
            art3d.pathpatch_2d_to_3d(arrow, z=0, zdir='z')
        xOld   = xValue
        yOld   = yValue
      dx    = xOld - xCoordOut
      dy    = yOld - yCoordOut
      arrow = Arrow(xOld, yOld, dx, dy, width=0.02, facecolor=faceColor)
      plotAx.add_patch(arrow)
      if flag3d:
        art3d.pathpatch_2d_to_3d(arrow, z=0, zdir='z')
    else:
      lineFrac = 1./self._pltRefinement
      xOld     = xCoordIn
      yOld     = yCoordIn
      for i in range(0, self._pltRefinement, 4):
        xValue = xSlope*(i+1)*lineFrac + xCoordIn
        yValue = ySlope*(i+1)*lineFrac + yCoordIn
        if i%8 == 0:
          dx     = xOld - xValue
          dy     = yOld - yValue
          arrow  = Arrow(xOld, yOld, -dx, -dy, width=0.05, facecolor=faceColor)
          plotAx.add_patch(arrow)
          if flag3d:
            art3d.pathpatch_2d_to_3d(arrow, z=0, zdir='z')
        xOld   = xValue
        yOld   = yValue


  def plotNode(self, pltNode: Node, plotAx: Axes, faceColor: tuple=(0,0,0), flag3d: bool=True):
    xCoord = pltNode.getCoordinate()[0]
    yCoord = pltNode.getCoordinate()[1]
    self.checkPlotBoundsX(xCoord)
    self.checkPlotBoundsY(yCoord)
    point = Circle((xCoord, yCoord), 0.075)
    point.set_facecolor(faceColor)
    plotAx.add_patch(point)
    if flag3d:
      art3d.pathpatch_2d_to_3d(point, z=0, zdir='z')


  def plotGraph3d(self, plotAx: Axes):
    for node in self._graph.nodeIterator():
      self.plotNode(node, plotAx, flag3d=True)
    for edge in self._graph.edgeIterator():
      self.plotEdge(edge, plotAx, flag3d=True)


  def plotInitialData3d(self):
    for edge in self._graph.edgeIterator():
      xCoordIn  = edge.getInNode().getCoordinate()[0]
      xCoordOut = edge.getOutNode().getCoordinate()[0]
      yCoordIn  = edge.getInNode().getCoordinate()[1]
      yCoordOut = edge.getOutNode().getCoordinate()[1]
      inNodeID  = edge.getInNode().getNodeID()
      outNodeID = edge.getOutNode().getNodeID()
      dx        = xCoordOut - xCoordIn
      dy        = yCoordOut - yCoordIn
      if (inNodeID is outNodeID):
        piFrac = (2.*np.pi)/self._pltRefinement
        x = []
        y = []
        for i in range(self._pltRefinement):
          x.append(xCoordIn + np.cos((i+(1/2))*piFrac-0.5*np.pi))
          y.append(yCoordIn + np.sin((i+(1/2))*piFrac-0.5*np.pi+1))
      else:
        x = np.linspace(xCoordIn+(cellSize/2), xCoordOut-(cellSize/2), self._pltRefinement)
        if dy == 0:
          y = np.zeros(len(x))
        else:
          y = (dx/dy) * x
      initialData = []
      for j in np.linspace(0., 1., self._pltRefinement):
        initialDatum = edge.getEquation().evaluateInitialData(j)
        initialData.append(initialDatum)
        self.checkPlotBoundsZ(initialDatum)
      dataTmp = np.empty((3, len(x)))
      dataTmp[0, :] = x[:]
      dataTmp[1, :] = y[:]
      dataTmp[2, :] = initialData[:]
      self._story.update({('Edge', 0, edge.getEdgeID()) : dataTmp})
      self._edgeFrame.append(self._axAnimation.plot([], [], 'o', markersize=5)[0])
    for node in self._graph.nodeIterator():
      xCoord = node.getCoordinate()[0]
      yCoord = node.getCoordinate()[1]
      if node.isBoundaryNode():
        nodeValue = node.evaluateBoundaryData(0.)
      else:
        nodeValue = node.evaluateInitialData()
      dataTmp = np.empty((3, 2))
      dataTmp[0, :] = [xCoord, xCoord]
      dataTmp[1, :] = [yCoord, yCoord]
      dataTmp[2, :] = [0., nodeValue]
      self._story.update({('Node', 0, node.getNodeID()) : dataTmp})
      self._nodeFrame.append(self._axAnimation.plot([], [])[0])


  def initAnimation3d(self) -> Iterable:
    self.plotGraph3d(self._axAnimation)
    for edge in self._graph.edgeIterator():
      self._edgeFrame.append(self._axAnimation.plot([], [], 'ob', markersize=1.5)[0])
      self._exactFrameEdge.append(self._axAnimation.plot([], [], 'or', markersize=1.5)[0])
    for node in self._graph.nodeIterator():
      self._nodeFrame.append(self._axAnimation.plot([], [], 'b')[0])
      self._exactFrameNode.append(self._axAnimation.plot([], [], 'r')[0])
    self._axAnimation.set_xlim3d([self._minX - self._xMargin, self._maxX + self._xMargin])
    self._axAnimation.set_ylim3d([self._minY - self._yMargin, self._maxY + self._yMargin])
    self._axAnimation.set_zlim3d([self._minZ - self._zMargin, self._maxZ + self._zMargin])
    return []


  def updateAnimation3d(self, timePoint: int) -> Iterable:
    # Iterate over all nodes and load the data for the recent timestep into the frame
    for dataSet, node in zip(self._nodeFrame, self._graph.nodeIterator()):
      nodeID = node.getNodeID()
      dataSet.set_data(self._story[('Node', timePoint, nodeID)][0:2, :])
      dataSet.set_3d_properties(self._story[('Node', timePoint, nodeID)][2, :])
    for dataSet, node in zip(self._exactFrameNode, self._graph.nodeIterator()):
      nodeID = node.getNodeID()
      dataSet.set_data(self._story[('ExactNode', timePoint, nodeID)][0:2, :])
      dataSet.set_3d_properties(self._story[('ExactNode', timePoint, nodeID)][2, :])
# Iterate over all edges and load the data for the recent timestep into the frame
    for dataSet, edge in zip(self._edgeFrame, self._graph.edgeIterator()):
      edgeID = edge.getEdgeID()
      dataSet.set_data(self._story[('Edge', timePoint, edgeID)][0:2, :])
      dataSet.set_3d_properties(self._story[('Edge', timePoint, edgeID)][2, :])
    for dataSet, edge in zip(self._exactFrameEdge, self._graph.edgeIterator()):
      edgeID = edge.getEdgeID()
      dataSet.set_data(self._story[('Exact', timePoint, edgeID)][0:2, :])
      dataSet.set_3d_properties(self._story[('Exact', timePoint, edgeID)][2, :])
    return self._edgeFrame + self._nodeFrame + self._exactFrameEdge + self._exactFrameNode


  def animateGraph3d(self):
    numFrames = self._discreteEqn.getNumTimeSteps()
    self._plotAnimation3d = animation.FuncAnimation(fig=self._animatedSolution, \
      func=self.updateAnimation3d, frames=numFrames, init_func=self.initAnimation3d, \
      interval=40, blit=True, repeat=False)


  def showAnimation3d(self):
    plt.figure(self._animatedSolution.number)
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry = (960, 0, 960, 540)


  def saveAnimation3d(self, fileName: str):
    self._plotAnimation3d.save('./Plots/' + fileName + '.mp4', writer=self._writer)

  def showAnimations(self, name: str):
    self._animatedSolution = plt.figure()
    self._axAnimation      = Axes3D(self._animatedSolution)
    self._animatedSolution.suptitle(name)
    yellow_patch = Patch(color='b', label='approximation edge')
    blue_patch   = Patch(color='r', label='exact solution edge')
    green_patch  = Patch(color='b', label='approximation node')
    red_patch    = Patch(color='r', label='exact solution node')
    self._axAnimation.legend(handles=[yellow_patch, blue_patch, green_patch, red_patch])
    self.updateStory()
    self.updateMinMaxZ()
    self.animateGraph3d()
    self.showAnimation3d()

  def plot3D(self, timePoint: int, name: str):
    tmpPlot     = plt.figure()
    tmpAx3DPlot = Axes3D(tmpPlot)
    tmpPlot.suptitle(name)
    self.updateStory()
    self.plotGraph3d(tmpAx3DPlot)
    for edge in self._graph.edgeIterator():
      edgeID = edge.getEdgeID()
      dataTmp = self._story[('Edge', timePoint, edgeID)]
      tmpAx3DPlot.plot(dataTmp[0,:], dataTmp[1,:], dataTmp[2,:], 'ob', markersize=2.5)
      dataTmp = self._story[('Exact', timePoint, edgeID)]
      tmpAx3DPlot.plot(dataTmp[0,:], dataTmp[1,:], dataTmp[2,:], 'or', markersize=2.5)
    for node in self._graph.nodeIterator():
      nodeID = node.getNodeID()
      dataTmp = self._story[('Node', timePoint, nodeID)]
      tmpAx3DPlot.plot(dataTmp[0,:], dataTmp[1,:], dataTmp[2,:], 'b')
      dataTmp = self._story[('ExactNode', timePoint, nodeID)]
      tmpAx3DPlot.plot(dataTmp[0,:], dataTmp[1,:], dataTmp[2,:], 'r')
    blue_patch = Patch(color='b', label='approximation')
    red_patch = Patch(color='r', label='exact solution')
    tmpAx3DPlot.legend(handles=[blue_patch, red_patch])
    if self._plot3D is None:
      self._plot3D = [tmpPlot]
    else:
      self._plot3D.append(tmpPlot)
    if self._ax3DPlot is None:
      self._ax3DPlot = [tmpAx3DPlot]
    else:
      self._ax3DPlot.append(tmpAx3DPlot)
    plt.savefig('./Plots/' + name + '.png', dpi=400) # , quality=95)
    plt.savefig('./Plots/' + name + '.pdf', dpi=400) # , quality=95)


  def showPlot(self):
    plt.show()
