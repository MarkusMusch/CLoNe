from .grid import Grid
from .flux import Flux

class Equation:
  """This class holds all the information that defines the equation to be solved"""


  def __init__(self, name: str, flux: Flux, derivativeFlux: Flux, initialData: callable, exactSolution: callable):
    self._name           = name
    self._flux           = flux
    self._initialData    = initialData
    self._exactSolution  = exactSolution
    self._derivativeFlux = derivativeFlux


  def evaluateInitialData(self, x: float) -> float:
    return self._initialData(x)


  def evaluateExactSolution(self, x: float, t: float) -> float:
    return self._exactSolution(x, t)

  def evaluateDerivativeFlux(self, x: float) -> float:
    return self._derivativeFlux.evaluateFlux(x)


  def getInitialData(self) -> callable:
    return self._initialData


  def getExactSolution(self) -> callable:
    return self._exactSolution

  def getFlux(self) -> callable:
    return self._flux
