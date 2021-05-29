from .grid import Grid
from .node import Node
from .equation import Equation
from .flux import Flux

class Edge:
  """This class defines a data structure for edges in the graph"""


  counter = 0


  def __init__(self, equation: Equation, numericalFlux: Flux, inNode: Node, outNode: Node):
    self._edgeID = Edge.counter
    Edge.counter += 1
    self._equation      = equation
    self._inNode        = inNode
    self._outNode       = outNode
    self._numericalFlux = numericalFlux
    self._grid = Grid(2) 


  def setGrid(self, numCells: int):
    self._grid = Grid(numCells)


  def getEquation(self) -> Equation:
    return self._equation


  def getEdgeID(self) -> int:
    return self._edgeID


  def getInNode(self) -> Node:
    return self._inNode


  def getOutNode(self) -> Node:
    return self._outNode


  def getGrid(self) -> Grid:
    return self._grid


  def getNumericalFlux(self) -> Flux:
    return self._numericalFlux
