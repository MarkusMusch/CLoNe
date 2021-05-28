from cell import Cell
import scipy.integrate as integrate
import numpy as np

# Define the fluxes on the edges and nodes here
def linearAdvection(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  return x

def monotoneQuadratic(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x > 0:
    return x*x/capacity
  else:
    return 0.

def positiveBurgers(x: float, y: float=1, capacity: float=1., t: float=1.) -> float:
  if x > 0:
    return x*x/(2.*capacity)
  else:
    return 0.

def trafficFlow(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  elif (0. <= x) and (x <= capacity):
    return x*(1.-x/capacity)
  else:
    return 0.

def holdenRisebroFlow(x: float, y: float = 1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  elif (0. <= x) and (x <= capacity):
    return 4.*x*(1.-(x/capacity))
  else:
    return 0.

def timeDependentHoldenRisebro(x: float, y: float=1., capacity: float=1., t:float=1.) -> float:
  if x < 0.:
    return 0.
  elif (0. <= x) and (x <= capacity):
    return 4.*x*(1.-(x/capacity))
  else:
    return 0.

# Define the numerical flux here
def upwindFlux(cIn: Cell, cOut: Cell, flux: callable, derivativeFlux: callable, capacity: float=1., t: float=1., dx: float=1., dt:float=1.) -> float:
  return flux(cIn.getCellAverage(), 1., capacity, t)

def downwindFlux(cIn: Cell, cOut: Cell, flux: callable, derivativeFlux: callable, capacity: float=1., t: float=1., dx: float=1., dt:float=1.) -> float:
  return flux(cOut.getCellAverage(), 1., capacity, t)

def concaveGodunov(cIn: Cell, cOut: Cell, flux: callable, derivativeFlux: callable, capacity: float=1., t: float=1., dx: float=1., dt: float=1.) -> float:
  fluxValue = min([flux(min([cIn.getCellAverage(), capacity/2.]), 1., capacity, t), flux(max([cOut.getCellAverage(), capacity/2.]), 1., capacity, t)])
  return fluxValue

def laxFriedrichs(cIn: Cell, cOut: Cell, flux: callable, derivativeFlux: callable, capacity: float=1., t: float=1., dx: float=1., dt: float=1.) -> float:
  inVal     = cIn.getCellAverage()
  outVal    = cOut.getCellAverage()
  fluxValue = 0.5*(dx/dt)*(inVal - outVal) + 0.5*(flux(inVal, 1., capacity, t) + flux(outVal, 1., capacity, t))
  return fluxValue

def enquistOsher(cIn: Cell, cOut: Cell, flux: callable, derivativeFlux: callable, capacity: float=1., t: float=1., dx: float=1., dt: float=1.) -> float:
  inVal            = cIn.getCellAverage()
  outVal           = cOut.getCellAverage()
  fluxValIn        = flux(inVal, 1., capacity, t)
  fluxValOut       = flux(outVal, 1., capacity, t)
  intDerivative    = integrate.quad(lambda x: np.abs(derivativeFlux(x, 1., capacity, t)), inVal, outVal)[0]
  fluxValue = 0.5*(fluxValIn + fluxValOut) - 0.5*intDerivative
  return fluxValue
  
# Define the derivative of the fluxes on the edges here
def derivativeLinearAdvection(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  return 1.

def derivativeMonotoneQuadratic(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  else:
    return 2.*x/capacity

def derivativePositiveBurgers(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  else:
    return x/capacity

def derivativeTrafficFlow(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  elif (0. <= x) and (x <= capacity):
    return (1.-2.*x/capacity)
  else:
    return 0.

def derivativeHoldenRisebro(x: float, y: float=1., capacity: float=1., t: float=1.) -> float:
  if x < 0.:
    return 0.
  elif (0. <= x) and (x <= capacity):
    return 4*(1. - 2.*(x/capacity))
  else:
    return 0.
