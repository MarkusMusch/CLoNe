from .flux import Flux
from .cell import Cell

class Node:
  """This class defines a data structure for nodes in the graph"""


  counter = 0


  def __init__(self, coordinate: tuple, initialData: callable=None, boundaryData: callable=None, exactSolution: callable=None):
    self._nodeID = Node.counter
    Node.counter += 1
    self._coordinate    = coordinate
    self._boundaryData  = boundaryData
    self._initialData   = initialData
    self._exactSolution = exactSolution
    self._nextNodeValue = 0.
    self._cell          = Cell(1., 0., 1.)
    if not (boundaryData is None):
      self._nodeValue = boundaryData(0)
    else:
      self._nodeValue = initialData()
    self._cell.setAverage(self._nodeValue)


  def isBoundaryNode(self) -> bool:
    if (self._boundaryData is None):
      return False
    else:
      return True


  def isJointNode(self) -> bool:
    if (self._initialData is None):
      return False
    else:
      return True


  def evaluateBoundaryData(self, timePoint) -> float:
    if (self._boundaryData is None):
      print("No boundary data has been declared")
    else:
      return self._boundaryData(timePoint)


  def evaluateInitialData(self) -> float:
    if (self._initialData is None):
      print("No initial data has been declared")
    else:
      return self._initialData()


  def evaluateExactSolution(self, t: float):
    if (self._exactSolution is None):
      print("No exact solution has been declared")
    else:
      return self._exactSolution(t)


  def swapValueNextValue(self):
    self._nodeValue     = self._nextNodeValue
    self._nextNodeValue = 0.


  def setNodeValue(self, newValue: float):
# We check the input parameters for type consistency
    if not isinstance(newValue, float):
      raise TypeError("The value assigned to a node has to be a floating point number")
    else:
      self._nodeValue = newValue


  def setNextNodeValue(self, newValue: float):
      if not isinstance(newValue, float):
        raise TypeError("The value assigned to a node has to be a floating point number")
      else:
        self._nextNodeValue = newValue


  def getBoundaryData(self) -> callable:
    if (self._boundaryData is None):
      print("No boundary data has been declared")
    else:
      return self._boundaryData


  def getInitialData(self) -> callable:
    if (self._boundaryData is None):
      print("No boundary data has been declared")
    else:
      return self._initialData


  def getexactSolution(self) -> callable:
    if (self._exactSolution is None):
      print("No boundary data has been declared")
    else:
      return self._exactSolution


  def getNodeValue(self) -> float:
    return self._nodeValue


  def getNextNodeValue(self) -> float:
    return self._nextNodeValue


  def getCoordinate(self) -> tuple:
    return self._coordinate


  def getNodeID(self) -> int:
    return self._nodeID


  def getNodeCell(self) -> Cell:
    self._cell.setAverage(self._nodeValue)
    return self._cell
