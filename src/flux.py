from .cell import Cell

class Flux:
  """This class defines a data structure for a flux function"""


  def __init__(self, name: str, fluxFunction: callable, capacity: float=1.):
    self._name = name
    self._fluxFunction = fluxFunction
    self._capacity     = capacity


  def evaluateFlux(self, x: float, y: float=1., t: float=1.) -> float:
    return self._fluxFunction(x, y, self._capacity, t)

class NumericalFlux(Flux):
  """This class defines a data structure for numerical flux functions"""


  def __init__(self, name: str, fluxFunction: callable, analyticalFlux: callable, analyticalDerivative: callable, capacity: float=1.):
    super().__init__(name, fluxFunction, capacity)
    self._analyticalFlux       = analyticalFlux
    self._analyticalDerivative = analyticalDerivative

  
  def evaluateFlux(self, x, y, t, dx, dt) -> float:
    return self._fluxFunction(x, y, self._analyticalFlux, self._analyticalDerivative, self._capacity, t, dx, dt) 
