import numpy as np
import scipy.integrate as integrate
from graph import Graph
from edge import Edge
from node import Node
from flux import Flux

class Discretization:

  def __init__(self, graph: Graph, finalTime: float, numCells: int, cflConstant: float, plotting=False):
    self._graph         = graph
    self._finalTime     = finalTime
    self._cflConstant   = cflConstant
    self._numCells      = numCells
    self._plotting      = plotting
    self._maxSolutionVal    = 0.
    self._minSolutionVal    = 0.
    self._deltaT            = 0.
    self._deltaX            = 1./self._numCells
    self._maxDerivativeFlux = None
    self._timePoint         = 0.
    self._timeStep          = 0
    self._errorL1           = 0.
    self._story             = {}
    for edge in self._graph.edgeIterator():
      edge.setGrid(numCells)


  def updateMinMaxSolutionVal(self, newZBound: float):
    if self._maxSolutionVal < newZBound:
      self._maxSolutionVal = newZBound
    if self._minSolutionVal > newZBound:
      self._minSolutionVal = newZBound


  def handlePlotDataEdge(self, values: list, exactValues: list, edge: Edge):
    xCoordIn  = edge.getInNode().getCoordinate()[0]
    xCoordOut = edge.getOutNode().getCoordinate()[0]
    yCoordIn  = edge.getInNode().getCoordinate()[1]
    yCoordOut = edge.getOutNode().getCoordinate()[1]
    edgeID    = edge.getEdgeID()
    inNodeID  = edge.getInNode().getNodeID()
    outNodeID = edge.getOutNode().getNodeID()
    cellSize  = self._deltaX
    dx        = xCoordOut - xCoordIn
    dy        = yCoordOut - yCoordIn
    if (inNodeID is outNodeID):
      piFrac = (2.*np.pi)/len(values)
      x = []
      y = []
      for i in range(len(values)):
        x.append(xCoordIn + np.cos((i+(1/2))*piFrac-0.5*np.pi))
        y.append(yCoordIn + np.sin((i+(1/2))*piFrac-0.5*np.pi) + 1.)
    else:
      x = np.linspace(xCoordIn+(cellSize/2), xCoordOut-(cellSize/2), len(values))
      if dy == 0:
        y = np.zeros(len(x))
      else:
        y = (dx/dy) * x
    dataTmp = np.empty((3, len(x)))
    dataTmp[0, :] = x[:]
    dataTmp[1, :] = y[:]
    dataTmp[2, :] = values[:]
    self._story.update({('Edge', self._timeStep, edgeID) : dataTmp})
    dataTmp = np.empty((3, len(x)))
    dataTmp[0, :] = x[:]
    dataTmp[1, :] = y[:]
    exactValues   = [x+0.01 for x in exactValues]
    dataTmp[2, :] = exactValues[:]
    self._story.update({('Exact', self._timeStep, edgeID): dataTmp})


  def handlePlotDataNode(self, values: list, exactValues: list, pltNode: Node):
    xCoord        = pltNode.getCoordinate()[0]
    yCoord        = pltNode.getCoordinate()[1]
    nodeID        = pltNode.getNodeID()
    dataTmp       = np.empty((3, 2))
    dataTmp[0, :] = [xCoord, xCoord]
    dataTmp[1, :] = [yCoord, yCoord]
    dataTmp[2, :] = values[:]
    self._story.update({('Node', self._timeStep, nodeID) : dataTmp})
    dataTmp       = np.empty((3, 2))
    dataTmp[0, :] = [xCoord+0.01, xCoord+0.01]
    dataTmp[1, :] = [yCoord+0.01, yCoord+0.01]
    dataTmp[2, :] = exactValues[:]
    self._story.update({('ExactNode', self._timeStep, nodeID) : dataTmp})


  def updateMaxDerivativeFlux(self, x: int, edge: Edge):
    tmpDerivativeFlux = edge.getEquation().evaluateDerivativeFlux(x)
    if self._maxDerivativeFlux is None:
      self._maxDerivativeFlux = tmpDerivativeFlux
    elif self._maxDerivativeFlux < tmpDerivativeFlux:
      self._maxDerivativeFlux = tmpDerivativeFlux


  def discretizeInitialData(self):
    for edge in self._graph.edgeIterator():
      grid           = edge.getGrid()
      eqn            = edge.getEquation()
      edgeID         = edge.getEdgeID()
      approxSolution = []
      exactSolution  = []
      grid.resetRecentCell()
      while not grid.recentIsLastCell():
        recentCell = grid.getRecentCell()
        leftEnd    = recentCell.getLeftEnd()
        rightEnd   = recentCell.getRightEnd()
        tmp        = integrate.quad(eqn.getInitialData(), leftEnd, rightEnd)
        avgValue   = (1./(rightEnd - leftEnd)) * tmp[0]
        self.updateMaxDerivativeFlux(avgValue, edge)
        recentCell.setAverage(avgValue)
        if self._plotting:
          self.updateMinMaxSolutionVal(avgValue)
          approxSolution.append(avgValue)
          exactValue = eqn.evaluateExactSolution(recentCell.getCellCenter(), self._timePoint)
          exactSolution.append(exactValue)
        grid.nextCell()
      recentCell = grid.getRecentCell()
      leftEnd    = recentCell.getLeftEnd()
      rightEnd   = recentCell.getRightEnd()
      tmp        = integrate.quad(eqn.getInitialData(), leftEnd, rightEnd)
      avgValue   = (1./(rightEnd - leftEnd))*tmp[0]
      self.updateMaxDerivativeFlux(avgValue, edge)
      recentCell.setAverage(avgValue)
      if self._plotting:
        self.updateMinMaxSolutionVal(avgValue)
        approxSolution.append(avgValue)
        exactValue = eqn.evaluateExactSolution(recentCell.getCellCenter(), 0.)
        exactSolution.append(exactValue)
        self.handlePlotDataEdge(approxSolution, exactSolution, edge)
    for node in self._graph.nodeIterator():
      nodeValue      = 0.
      if node.isBoundaryNode():
        nodeValue = node.evaluateBoundaryData(0.)
      else:
        nodeValue = node.evaluateInitialData()
        nodeID = node.getNodeID()
      node.setNodeValue(nodeValue)
      if self._plotting:
        self.handlePlotDataNode([0., nodeValue], [0., nodeValue], node)


  def discretizeEdges(self):
    if self._deltaT is None:
      raise TypeError("No delta t has been computed yet")
    for edge in self._graph.edgeIterator():
      grid           = edge.getGrid()
      approxSolution = []
      exactSolution  = []
      eqn            = edge.getEquation()
      numericalFlux  = edge.getNumericalFlux()
      grid.resetRecentCell()
# In case the grid has only two cells
      if not grid.recentIsLastCell():
        grid.nextCell()
      while not grid.recentIsLastCell():
        tmpCenterCell = grid.getRecentCell()
        avgValue      = 0.
        inFlux        = numericalFlux.evaluateFlux(tmpCenterCell.getPreviousCell(), tmpCenterCell, self._timePoint, self._deltaX, self._deltaT)
        outFlux       = numericalFlux.evaluateFlux(tmpCenterCell, tmpCenterCell.getNextCell(), self._timePoint, self._deltaX, self._deltaT)
        avgValue      = tmpCenterCell.getCellAverage() - (self._deltaT/self._deltaX) * (outFlux - inFlux)
        self.updateMaxDerivativeFlux(avgValue, edge)
        tmpCenterCell.setNextAverage(avgValue)
        exactValue = eqn.evaluateExactSolution(tmpCenterCell.getCellCenter(), self._timePoint)
        self._errorL1 += self.computeL1Error(exactValue, self._deltaX, avgValue)
        if self._plotting:
          self.updateMinMaxSolutionVal(avgValue)
          approxSolution.append(avgValue)
          exactSolution.append(exactValue)
        grid.nextCell()
      if self._plotting:
        self.handlePlotDataEdge(approxSolution, exactSolution, edge)


  def discretizeNodes(self):
    for node in self._graph.nodeIterator():
# If the node is a boundary node we just set its value to the value of the boundary data function
      if node.isBoundaryNode():
        exactValue = node.evaluateBoundaryData(self._timePoint)
        nodeValue  = exactValue
# If the node is a joint and not an exterior boundary the procedure is more complicated
      else:
        nodeValue   = node.getNodeValue()
        tmp         = 0.
        numInEdges  = len(self._graph.getConnectInEdges(node.getNodeID()))
        numOutEdges = len(self._graph.getConnectOutEdges(node.getNodeID()))
        for edgeID in self._graph.getConnectInEdges(node.getNodeID()):
          edge          = self._graph.getEdge(edgeID)
          grid          = edge.getGrid()
          numericalFlux = edge.getNumericalFlux()
          inFlux        = numericalFlux.evaluateFlux(grid.getLastCell().getPreviousCell(), grid.getLastCell(), self._timePoint, self._deltaX, self._deltaT)
          tmp          -= inFlux
        for edgeID in self._graph.getConnectOutEdges(node.getNodeID()):
          edge          = self._graph.getEdge(edgeID)
          grid          = edge.getGrid()
          numericalFlux = edge.getNumericalFlux()
          outFlux       = numericalFlux.evaluateFlux(grid.getFirstCell(), grid.getFirstCell().getNextCell(), self._timePoint, self._deltaX, self._deltaT)
          tmp          += outFlux
        dXZero          = (numInEdges+numOutEdges)*self._deltaX/2.
        tmp            *= self._deltaT/dXZero
        nodeValue      -= tmp
        exactValue      = node.evaluateExactSolution(self._timePoint)
        self._errorL1 += self.computeL1Error(exactValue, dXZero, nodeValue)
# Set the new value on the node
      node.setNextNodeValue(nodeValue)
      if self._plotting:
        self.updateMinMaxSolutionVal(nodeValue)
        self.handlePlotDataNode([0, nodeValue], [0, exactValue], node)


  def updateEdgeBoundaryValues(self):
    for node in self._graph.nodeIterator():
      nodeID = node.getNodeID()
      if node.isBoundaryNode():
        for edgeID in self._graph.getConnectInEdges(nodeID):
          edge          = self._graph.getEdge(edgeID) 
          eqn           = edge.getEquation()
          grid          = edge.getGrid()
          numericalFlux = edge.getNumericalFlux()
          tmpCell       = grid.getLastCell()
          tmpNodeCell   = node.getNodeCell()
          avgValue      = 0.
          inFlux        = numericalFlux.evaluateFlux(tmpCell.getPreviousCell(), tmpCell, self._timePoint, self._deltaX, self._deltaT)
          outFlux       = numericalFlux.evaluateFlux(tmpCell, tmpNodeCell, self._timePoint, self._deltaX, self._deltaT)
          avgValue      = tmpCell.getCellAverage() - (self._deltaT/self._deltaX) * (outFlux - inFlux)
          self.updateMaxDerivativeFlux(avgValue, edge)
          tmpCell.setNextAverage(avgValue)
          exactValue    = eqn.evaluateExactSolution(tmpCell.getCellCenter(), self._timePoint)
          self._errorL1 += self.computeL1Error(exactValue, self._deltaX, avgValue)
        for edgeID in self._graph.getConnectOutEdges(nodeID):
          edge          = self._graph.getEdge(edgeID) 
          eqn           = edge.getEquation()
          grid          = edge.getGrid()
          numericalFlux = edge.getNumericalFlux()
          tmpCell       = grid.getFirstCell()
          tmpNodeCell   = node.getNodeCell()
          avgValue      = 0.
          inFlux        = numericalFlux.evaluateFlux(tmpNodeCell, tmpCell, self._timePoint, self._deltaX, self._deltaT)
          outFlux       = numericalFlux.evaluateFlux(tmpCell, tmpCell.getNextCell(), self._timePoint, self._deltaX, self._deltaT)
          avgValue      = tmpCell.getCellAverage() - (self._deltaT/self._deltaX) * (outFlux - inFlux)
          self.updateMaxDerivativeFlux(avgValue, edge)
          tmpCell.setNextAverage(avgValue)
          exactValue = eqn.evaluateExactSolution(tmpCell.getCellCenter(), self._timePoint)
          self._errorL1 += self.computeL1Error(exactValue, self._deltaX, avgValue)
      else:
        nodeValue = node.getNextNodeValue()
        for edgeID in self._graph.getConnectInEdges(nodeID):
          edge = self._graph.getEdge(edgeID)
          edge.getGrid().getLastCell().setNextAverage(nodeValue)
          self.updateMaxDerivativeFlux(nodeValue, edge)
        for edgeID in self._graph.getConnectOutEdges(nodeID):
          edge = self._graph.getEdge(edgeID)
          edge.getGrid().getFirstCell().setNextAverage(nodeValue)
          self.updateMaxDerivativeFlux(nodeValue, edge)


  def evolveTimeStep(self) -> bool:
    if (self._finalTime is self._timePoint + self._deltaT):
      pass
    elif (self._finalTime < self._timePoint + self._deltaT):
      self._deltaT  = self._finalTime-self._timePoint
      self.step()
      self.swapAverages()
      return False
    else:
      if self._timePoint == 0.0:
        self.discretizeInitialData()
        self.computeNewDeltaT()
        self.incrementTimeStep()
      self.step()
      self.swapAverages()
      return True


  def step(self):
    self._errorL1 = 0.
    self.discretizeEdges()
    self.discretizeNodes()
    self.updateEdgeBoundaryValues()
    self.incrementTimePoint()
    self.incrementTimeStep()
    self.computeNewDeltaT()
    self._story.update({('errorL1', self._timeStep): self._errorL1})


  def swapAverages(self):
    for edge in self._graph.edgeIterator():
      grid = edge.getGrid()
      grid.resetRecentCell()
      while not grid.recentIsLastCell():
        tmpCenterCell = grid.getRecentCell()
        tmpCenterCell.swapAverageNextAverage()
        grid.nextCell()
      tmpCenterCell = grid.getRecentCell()
      tmpCenterCell.swapAverageNextAverage()
    for node in self._graph.nodeIterator():
      node.swapValueNextValue()


  def incrementTimeStep(self):
    self._timeStep += 1


  def incrementTimePoint(self):
    self._timePoint += self._deltaT


  def computeNewDeltaT(self):
    self._deltaT = self._cflConstant * self._deltaX * (1./self._maxDerivativeFlux)


  def computeL1Error(self, exactVal: float, cellSize: float, avg: float) -> float:
# Use the midpoint rule
    errorL1 = cellSize*0.5*np.abs(exactVal - avg)
    return errorL1


  def getTimePoint(self) -> float:
    return self._timePoint


  def getGraph(self) -> Graph:
    return self._graph


  def getNumTimeSteps(self) -> float:
    return self._timeStep


  def getStory(self) -> dict:
    return self._story


  def getMinSolutionVal(self) -> float:
    return self._minSolutionVal


  def getMaxSolutionVal(self) -> float:
    return self._maxSolutionVal


  def getL1Error(self) -> float:
    return self._story[('errorL1', self._timeStep)]
