import math
from node import Node
from cell import Cell

class Grid:
  """This class defines a data structure for a grid"""


  def __init__(self, numCells: int):
    self._numCells = numCells
    self._deltaX     = 1./self._numCells
    self._firstCell  = Cell(self._deltaX, 0., self._deltaX)
    self._recentCell = self._firstCell
    for i in range(1, self._numCells):
      leftEnd  = i*self._deltaX
      rightEnd = (i+1)*self._deltaX
      tmpCell  = Cell(self._deltaX, leftEnd, rightEnd)
      self._recentCell.setNextCell(tmpCell) 
      tmpCell.setPreviousCell(self._recentCell)
      self._recentCell = tmpCell
    self._lastCell = self._recentCell


  def resetRecentCell(self):
    self._recentCell = self._firstCell


  def nextCell(self):
    if self.recentIsLastCell():
      print("The recent cell is the last cell.")
    else:
      self._recentCell = self._recentCell.getNextCell()


  def previousCell(self):
    if self.recentIsFirstCell():
      print("The recent cell is the first cell")
    else:
      self._recentCell = self._recentCell.getPreviousCell()


  def recentIsLastCell(self) -> bool:
    if (self._recentCell.getNextCell() is None):
      return True
    else:
      return False


  def recentIsFirstCell(self) -> bool:
    if self._recentCell.getPreviousCell() is None:
      return True
    else:
      return False


  def getRecentCell(self) -> Cell:
    return self._recentCell


  def getNumCells(self):
    return self._numCells 


  def getFirstCell(self) -> Cell:
    return self._firstCell


  def getLastCell(self) -> Cell:
    return self._lastCell

