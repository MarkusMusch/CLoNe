class Cell:
  """This class defines a cell that makes up a grid"""


  counter = 0


  def __init__(self, size: float, leftEnd: float, rightEnd: float):
    self._cellID       = Cell.counter
    Cell.counter      += 1
    self._cellAverage  = 0.
    self._nextAverage  = 0.
    self._previousCell = None
    self._nextCell     = None
#
    self._size     = size
    self._leftEnd  = leftEnd
    self._rightEnd = rightEnd
    self._center   = self._leftEnd + (self._rightEnd - self._leftEnd)/2.


  def setAverage(self, newAverage: float):
    self._cellAverage = newAverage


  def setNextAverage(self, newAverage: float):
    self._nextAverage = newAverage


  def swapAverageNextAverage(self):
    self._cellAverage = self._nextAverage
    self._nextAverage = 0.


  def setPreviousCell(self, prevCell):
    self._previousCell = prevCell


  def setNextCell(self, nextCell):
    self._nextCell = nextCell


  def getPreviousCell(self):
    return self._previousCell


  def getNextCell(self):
    return self._nextCell


  def getCellSize(self) -> float:
    return self._size


  def getLeftEnd(self) -> float:
    return self._leftEnd


  def getRightEnd(self) -> float:
    return self._rightEnd


  def getCellAverage(self) -> float:
    return self._cellAverage


  def getCellCenter(self) -> float:
    return self._center
