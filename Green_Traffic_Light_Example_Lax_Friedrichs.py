import numpy as np

from src import fluxcollection
from src.clongraphplot import CLonGraphPlot
from src.discretization import Discretization
from src.graph import Graph
from src.edge import Edge
from src.node import Node
from src.cell import Cell
from src.equation import Equation
from src.flux import Flux, NumericalFlux

"""In this module one defines the equation and initial data and assembles a graph"""

# Define the initial data for your equation here
def initialDataInEdgeOne(x: float) -> float:
  if x < 0.5:
    return 0.6
  else:
    return 1.0

def initialDataInEdgeTwo(x: float) -> float:
  if x < 0.5:
    return 0.6
  else:
    return 1.0

def initialDataOutEdgeOne(x: float) -> float:
  return 0.0

def initialDataOutEdgeTwo(x: float) -> float:
  return 0.0

def initialDataOutEdgeThree(x: float) -> float:
  return 0.0

def initialDataOutEdgeFour(x: float) -> float:
  return 0.0

def boundaryDataInNodeOne(t: float) -> float:
  return 0.6

def boundaryDataInNodeTwo(t: float) -> float:
  return 0.6

def boundaryDataOutNodeOne(t: float) -> float:
  return 0.0

def boundaryDataOutNodeTwo(t: float) -> float:
  return 0.0

def boundaryDataOutNodeThree(t: float) -> float:
  return 0.0

def boundaryDataOutNodeFour(t: float) -> float:
  return 0.0

def initialDataCentralNode() -> float:
  return (3.-3.*np.sqrt(1./6.))/10.
#  return ((2.-np.sqrt(2.))/4.) 

def exactSolutionInEdgeOne(x: float, t: float) -> float:
  if x < 0.5-2.4*t:
    return 0.6
  elif 1.-4.*t <= x <= 1.:
    return 0.5*(1.+((1.-x)/(4.*t)))
  else:
    return 1.

def exactSolutionInEdgeTwo(x: float, t: float) -> float:
  if x < 0.5-2.4*t:
    return 0.6
  elif 1.-4.*t <= x <= 1.:
    return 0.5*(1.+((1.-x)/(4.*t)))
  else:
    return 1.

def exactSolutionOutEdgeOne(x: float, t: float) -> float:
  if x < 4*np.sqrt(1./6.)*t: 
    return (3.-3.*np.sqrt(1./6.))/10.
  elif x < 4.*t:
    return 0.3*(1.-(x/(4.*t))) 
  else:
    return 0.

def exactSolutionOutEdgeTwo(x: float, t: float) -> float:
  if x < 4*np.sqrt(1./6.)*t: 
    return (3.-3.*np.sqrt(1./6.))/10.
  elif x < 4.*t:
    return 0.3*(1.-(x/(4.*t))) 
  else:
    return 0.

def exactSolutionOutEdgeThree(x: float, t: float) -> float:
  if x < 4*np.sqrt(1./6.)*t: 
    return (3.-3.*np.sqrt(1./6.))/10.
  elif x < 4.*t:
    return 0.3*(1.-(x/(4.*t))) 
  else:
    return 0.

def exactSolutionOutEdgeFour(x: float, t: float) -> float:
  if x < 4*np.sqrt(1./6.)*t: 
    return (3.-3.*np.sqrt(1./6.))/10.
  elif x < 4.*t:
    return 0.3*(1.-(x/(4.*t))) 
  else:
    return 0.

def exactSolutionJointNode(t: float) -> float:
  return (3.-3.*np.sqrt(1./6.))/10.

# Create your edge fluxes here
trafficEdgeFluxInOne    = Flux(name='Traffic Flow Edge, One', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=1.)
trafficEdgeFluxInTwo    = Flux(name='Traffic Flow Edge, Two', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=1.)
trafficEdgeFluxOutOne   = Flux(name='Traffic Flow Edge, Three', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=0.6)
trafficEdgeFluxOutTwo   = Flux(name='Traffic Flow Edge, Four', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=0.6)
trafficEdgeFluxOutThree = Flux(name='Traffic Flow Edge, Five', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=0.6)
trafficEdgeFluxOutFour  = Flux(name='Traffic Flow Edge, Six', fluxFunction=fluxcollection.holdenRisebroFlow, capacity = 0.6)
# And numerical fluxes here
trafficNumericalInEdgeOne    = NumericalFlux(name='Traffic Flow Numerical, One', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficNumericalInEdgeTwo    = NumericalFlux(name='Traffic Flow Numerical, Two', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficNumericalOutEdgeOne   = NumericalFlux(name='Traffic Flow Numerical, Three', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficNumericalOutEdgeTwo   = NumericalFlux(name='Traffic Flow Numerical, Four', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficNumericalOutEdgeThree = NumericalFlux(name='Traffic Flow Numerical, Five', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficNumericalOutEdgeFour  = NumericalFlux(name='Traffic Flow Numerical, Six', fluxFunction=fluxcollection.laxFriedrichs, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
# And derivatives here
trafficDerivativeInEdgeOne    = Flux(name='Traffic Flow Derivative, One', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficDerivativeInEdgeTwo    = Flux(name='Traffic Flow Derivative, Two', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficDerivativeOutEdgeOne   = Flux(name='Traffic Flow Derivative, Three', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficDerivativeOutEdgeTwo   = Flux(name='Traffic Flow Derivative, Four', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficDerivativeOutEdgeThree = Flux(name='Traffic Flow Derivative, Five', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=0.6)
trafficDerivativeOutEdgeFour  = Flux(name='Traffic Flow Derivative, Six', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=0.6)

# Create your equation here
trafficEqnOne   = Equation(name='Nonlinear Conservation Law: Traffic Flow, One', flux=trafficEdgeFluxInOne, derivativeFlux=trafficDerivativeInEdgeOne, initialData=initialDataInEdgeOne, exactSolution=exactSolutionInEdgeOne)
trafficEqnTwo   = Equation(name='Nonlinear Conservation Law: Traffic Flow, Two', flux=trafficEdgeFluxInTwo, derivativeFlux=trafficDerivativeInEdgeTwo, initialData=initialDataInEdgeTwo, exactSolution=exactSolutionInEdgeTwo)
trafficEqnThree = Equation(name='Nonlinear Conservation Law: Traffic Flow, Three', flux=trafficEdgeFluxOutOne, derivativeFlux=trafficDerivativeOutEdgeOne, initialData=initialDataOutEdgeOne, exactSolution=exactSolutionOutEdgeOne)
trafficEqnFour  = Equation(name='Nonlinear Conservation Law: Traffic Flow, Four', flux=trafficEdgeFluxOutTwo, derivativeFlux=trafficDerivativeOutEdgeTwo, initialData=initialDataOutEdgeTwo, exactSolution=exactSolutionOutEdgeTwo)
trafficEqnFive  = Equation(name='Nonlinear Conservation Law: Traffic Flow, Five', flux=trafficEdgeFluxOutThree, derivativeFlux=trafficDerivativeOutEdgeThree, initialData=initialDataOutEdgeThree, exactSolution=exactSolutionOutEdgeThree)
trafficEqnSix   = Equation(name='Nonlinear Conservation Law: Traffic Flow, Six', flux=trafficEdgeFluxOutFour, derivativeFlux=trafficDerivativeOutEdgeFour, initialData=initialDataOutEdgeFour, exactSolution=exactSolutionOutEdgeFour)

# Create the nodes for your graph here
centralNode  = Node(coordinate=(0,0), initialData=initialDataCentralNode, boundaryData=None, exactSolution=exactSolutionJointNode)
inNodeOne    = Node(coordinate=(-1,-1), initialData=None, boundaryData=boundaryDataInNodeOne, exactSolution=None)
inNodeTwo    = Node(coordinate=(1,1), initialData=None, boundaryData=boundaryDataInNodeTwo, exactSolution=None)
outNodeOne   = Node(coordinate=(-1,-1), initialData=None, boundaryData=boundaryDataOutNodeOne, exactSolution=None)
outNodeTwo   = Node(coordinate=(1,1), initialData=None, boundaryData=boundaryDataOutNodeTwo, exactSolution=None)
outNodeThree = Node(coordinate=(1,-1), initialData=None, boundaryData=boundaryDataOutNodeThree, exactSolution=None)
outNodeFour  = Node(coordinate=(-1,1), boundaryData=boundaryDataOutNodeFour, exactSolution=None)

# Create your graph here with its' initial node
streets = Graph(initialNode=centralNode)

# Create the edges for your graph here
inEdgeOne      = Edge(equation=trafficEqnOne, numericalFlux=trafficNumericalInEdgeOne, inNode=inNodeOne, outNode=centralNode)
inEdgeTwo      = Edge(equation=trafficEqnTwo, numericalFlux=trafficNumericalInEdgeTwo, inNode=inNodeTwo, outNode=centralNode)
outEdgeOne     = Edge(equation=trafficEqnThree, numericalFlux=trafficNumericalOutEdgeOne, inNode=centralNode, outNode=outNodeOne)
outEdgeTwo     = Edge(equation=trafficEqnFour, numericalFlux=trafficNumericalOutEdgeTwo, inNode=centralNode, outNode=outNodeTwo)
outEdgeThree   = Edge(equation=trafficEqnFive, numericalFlux=trafficNumericalOutEdgeThree, inNode=centralNode, outNode=outNodeThree)
outEdgeFour    = Edge(equation=trafficEqnSix, numericalFlux=trafficNumericalOutEdgeFour, inNode=centralNode, outNode=outNodeFour)

# Assemble your graph adding all the edges
streets.addEdge(newEdge=inEdgeOne)
streets.addEdge(newEdge=inEdgeTwo)
streets.addEdge(newEdge=outEdgeOne)
streets.addEdge(newEdge=outEdgeTwo)
streets.addEdge(newEdge=outEdgeThree)
streets.addEdge(newEdge=outEdgeFour)

# Compute the solution to your problem here
errorL1       = 0.
errorL1New    = 0.
eoc           = 0.
cfl           = 0.5
T             = 0.1
numGridLevels = 7
eocList       = ['  - ']
errorL1List   = []
for i in range(3,2+numGridLevels):
  cells = 2**i
  dXNew = 1./cells
  # Create your discretization
  discreteEquation = Discretization(graph=streets, finalTime=T, numCells=cells, cflConstant=cfl, plotting=True)
  
  while discreteEquation.evolveTimeStep():
    pass
  
  errorL1New = np.round(discreteEquation.getL1Error(), decimals=5) 
  errorL1List.append(errorL1New)
  if errorL1 == 0.:
    pass
  else:
    eoc = np.round(np.log(errorL1New/errorL1)/np.log(dXNew/dX), decimals = 2)
    eocList.append(eoc)
  errorL1 = errorL1New
  dX      = dXNew
print('-------------')
print('|  T  | cfl |')
print('-------------')
print('|', T, '|', cfl, '|')
print('------------- \n')

print('------------------------------------')
print('| Grid Level | L^1 error |   eoc   |')
print('------------------------------------')
for i in range(len(eocList)):
  print('|    ', i+3,'     |', errorL1List[i], ' | ', eocList[i], '  |')
print('------------------------------------')

# Plot the final state here
streetsPlot = CLonGraphPlot(discreteEqn=discreteEquation, pltRefinement=100) 
# streetsPlot.showAnimations('Trafficlight switch to green')
# streetsPlot.saveAnimation3d('Traffic_Flow_Monotonous_Elementary_Waves_Example')
# streetsPlot.plot3D(0, 't=0.0')
# streetsPlot.plot3D(discreteEquation.getNumTimeSteps()-1, 't=0.1')
# streetsPlot.showPlot()
#streetsPlot.saveAnimationProjection2d('complexNetworkProjectionAnimation50CellsDots')
