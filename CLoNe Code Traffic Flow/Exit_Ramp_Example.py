import numpy as np
import fluxcollection
from clongraphplot import CLonGraphPlot
from discretization import Discretization
from graph import Graph
from edge import Edge
from node import Node
from cell import Cell
from equation import Equation
from flux import Flux, NumericalFlux

"""In this module one defines the equation and initial data and assembles a graph"""

# Define the initial data for your equation here
def initialDataInEdgeOne(x: float) -> float:
  if x < 0.98:
    return 1.0
  else:
    return 0.2

def initialDataOutEdgeOne(x: float) -> float:
  return 0.2

def initialDataOutEdgeTwo(x: float) -> float:
    return 0.1

def boundaryDataInNodeOne(t: float) -> float:
  return 1.0

def boundaryDataOutNodeOne(t: float) -> float:
  return 0.2

def boundaryDataOutNodeTwo(t: float) -> float:
  return 0.1

def initialDataCentralNode() -> float:
  return 0.2 

def exactSolutionInEdgeOne(x: float, t: float) -> float:
  return 1.

def exactSolutionOutEdgeOne(x: float, t: float) -> float:
  return 0.2

def exactSolutionOutEdgeTwo(x: float, t: float) -> float:
  return 0.1

def exactSolutionJointNode(t: float) -> float:
  return 0.2

# Create your edge fluxes here
trafficEdgeFluxInOne    = Flux(name='Traffic Flow Edge, One', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=1.)
trafficEdgeFluxOutOne   = Flux(name='Traffic Flow Edge, Three', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=1.)
trafficEdgeFluxOutTwo    = Flux(name='Traffic Flow Edge, Two', fluxFunction=fluxcollection.holdenRisebroFlow, capacity=0.2)
# And numerical fluxes here
trafficNumericalInEdgeOne    = NumericalFlux(name='Traffic Flow Numerical, One', fluxFunction=fluxcollection.enquistOsher, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficNumericalOutEdgeOne   = NumericalFlux(name='Traffic Flow Numerical, Three', fluxFunction=fluxcollection.enquistOsher, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficNumericalOutEdgeTwo    = NumericalFlux(name='Traffic Flow Numerical, Two', fluxFunction=fluxcollection.enquistOsher, analyticalFlux=fluxcollection.holdenRisebroFlow, analyticalDerivative=fluxcollection.derivativeHoldenRisebro, capacity=0.2)
# And derivatives here
trafficDerivativeInEdgeOne    = Flux(name='Traffic Flow Derivative, One', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficDerivativeOutEdgeOne   = Flux(name='Traffic Flow Derivative, Three', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=1.)
trafficDerivativeOutEdgeTwo    = Flux(name='Traffic Flow Derivative, Two', fluxFunction=fluxcollection.derivativeHoldenRisebro, capacity=0.2)

# Create your equation here
trafficEqnOne   = Equation(name='Nonlinear Conservation Law: Traffic Flow, One', flux=trafficEdgeFluxInOne, derivativeFlux=trafficDerivativeInEdgeOne, initialData=initialDataInEdgeOne, exactSolution=exactSolutionInEdgeOne)
trafficEqnTwo = Equation(name='Nonlinear Conservation Law: Traffic Flow, Two', flux=trafficEdgeFluxOutOne, derivativeFlux=trafficDerivativeOutEdgeOne, initialData=initialDataOutEdgeOne, exactSolution=exactSolutionOutEdgeOne)
trafficEqnThree   = Equation(name='Nonlinear Conservation Law: Traffic Flow, Three', flux=trafficEdgeFluxOutTwo, derivativeFlux=trafficDerivativeOutEdgeTwo, initialData=initialDataOutEdgeTwo, exactSolution=exactSolutionOutEdgeTwo)

# Create the nodes for your graph here
centralNode  = Node(coordinate=(0,0), initialData=initialDataCentralNode, boundaryData=None, exactSolution=exactSolutionJointNode)
inNodeOne    = Node(coordinate=(1,0), initialData=None, boundaryData=boundaryDataInNodeOne, exactSolution=None)
outNodeOne   = Node(coordinate=(-1,0), initialData=None, boundaryData=boundaryDataOutNodeOne, exactSolution=None)
outNodeTwo   = Node(coordinate=(-1,-1), initialData=None, boundaryData=boundaryDataOutNodeTwo, exactSolution=None)

# Create your graph here with its' initial node
streets = Graph(initialNode=centralNode)

# Create the edges for your graph here
inEdgeOne      = Edge(equation=trafficEqnOne, numericalFlux=trafficNumericalInEdgeOne, inNode=inNodeOne, outNode=centralNode)
outEdgeOne     = Edge(equation=trafficEqnTwo, numericalFlux=trafficNumericalOutEdgeOne, inNode=centralNode, outNode=outNodeOne)
outEdgeTwo      = Edge(equation=trafficEqnThree, numericalFlux=trafficNumericalOutEdgeTwo, inNode=centralNode, outNode=outNodeTwo)

# Assemble your graph adding all the edges
streets.addEdge(newEdge=inEdgeOne)
streets.addEdge(newEdge=outEdgeOne)
streets.addEdge(newEdge=outEdgeTwo)

# Compute the solution to your problem here
errorL1       = 0.
errorL1New    = 0.
eoc           = 0.
cfl           = 0.5
T             = 0.2
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
  if errorL1 is 0.:
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
streetsPlot.showAnimations('Trafficlight switch to green')
streetsPlot.saveAnimation3d('Traffic_Flow_Monotonous_Elementary_Waves_Example')
streetsPlot.plot3D(0, 't=0.0 ')
streetsPlot.plot3D(discreteEquation.getNumTimeSteps()-1, 't=0.1')
streetsPlot.showPlot()
#streetsPlot.saveAnimationProjection2d('complexNetworkProjectionAnimation50CellsDots')
