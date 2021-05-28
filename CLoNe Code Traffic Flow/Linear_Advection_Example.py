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
  if x < 0.8:
    return 2.0
  else:
    return 1.0 

def initialDataInEdgeTwo(x: float) -> float:
  return 1.0

def initialDataOutEdgeOne(x: float) -> float:
  return 2./3. 

def initialDataOutEdgeTwo(x: float) -> float:
  return 2./3.

def initialDataOutEdgeThree(x: float) -> float:
  return 2./3.

def boundaryDataInNodeOne(t: float) -> float:
  return 2.0

def boundaryDataInNodeTwo(t: float) -> float:
  return 1.0

def boundaryDataOutNodeOne(t: float) -> float:
  return 2./3.

def boundaryDataOutNodeTwo(t: float) -> float:
  return 2./3.

def boundaryDataOutNodeThree(t: float) -> float:
  return 2./3.

def initialDataCentralNode() -> float:
  return 2./3. 

def exactSolutionInEdgeOne(x: float, t: float) -> float:
  if x < t+0.8:
    return 2.
  else:
    return 1.

def exactSolutionInEdgeTwo(x: float, t: float) -> float:
  return 1.

def exactSolutionOutEdgeOne(x: float, t: float) -> float:
  if x < 1.*t-0.2:
    return 1.
  else:
    return 2./3. 

def exactSolutionOutEdgeTwo(x: float, t: float) -> float:
  if x < 1.*t-0.2:
    return 1.
  else:
    return 2./3. 


def exactSolutionOutEdgeThree(x: float, t: float) -> float:
  if x < 1.*t-0.2:
    return 1.
  else:
    return 2./3. 

def exactSolutionJointNode(t: float) -> float:
  if t < 0.2:
    return 2./3.
  else:
    return 1.

# Create your edge fluxes here
linearEdgeFlux = Flux(name='Linear Advection Edge', fluxFunction=fluxcollection.linearAdvection, capacity=1.)
# And numerical fluxes here
linearNumerical = NumericalFlux(name='Linear Advection Numerical', fluxFunction=fluxcollection.upwindFlux, analyticalFlux=fluxcollection.linearAdvection, capacity=1.)
# And derivatives here
linearDerivative = Flux(name='Linear Advection Derivative', fluxFunction=fluxcollection.derivativeLinearAdvection, capacity=1.)

# Create your equation here
linearEqnOne   = Equation(name='Linear Conservation Law: Linear Advection, One', flux=linearEdgeFlux, derivativeFlux=linearDerivative, initialData=initialDataInEdgeOne, exactSolution=exactSolutionInEdgeOne)
linearEqnTwo   = Equation(name='Linear Conservation Law: Linear Advection, Two', flux=linearEdgeFlux, derivativeFlux=linearDerivative, initialData=initialDataInEdgeTwo, exactSolution=exactSolutionInEdgeTwo)
linearEqnThree = Equation(name='Linear Conservation Law: Linear Advection, Three', flux=linearEdgeFlux, derivativeFlux=linearDerivative, initialData=initialDataOutEdgeOne, exactSolution=exactSolutionOutEdgeOne)
linearEqnFour  = Equation(name='Linear Conservation Law: Linear Advection, Four', flux=linearEdgeFlux, derivativeFlux=linearDerivative, initialData=initialDataOutEdgeTwo, exactSolution=exactSolutionOutEdgeTwo)
linearEqnFive  = Equation(name='Linear Conservation Law: Linear Advection, Five', flux=linearEdgeFlux, derivativeFlux=linearDerivative, initialData=initialDataOutEdgeThree, exactSolution=exactSolutionOutEdgeThree)

# Create the nodes for your graph here
centralNode  = Node(coordinate=(0,0), initialData=initialDataCentralNode, boundaryData=None, exactSolution=exactSolutionJointNode)
inNodeOne    = Node(coordinate=(-1,0), initialData=None, boundaryData=boundaryDataInNodeOne, exactSolution=None)
inNodeTwo    = Node(coordinate=(-1,-1), initialData=None, boundaryData=boundaryDataInNodeTwo, exactSolution=None)
outNodeOne   = Node(coordinate=(1,1), initialData=None, boundaryData=boundaryDataOutNodeOne, exactSolution=None)
outNodeTwo   = Node(coordinate=(1,-1), initialData=None, boundaryData=boundaryDataOutNodeTwo, exactSolution=None)
outNodeThree = Node(coordinate=(1,0), initialData=None, boundaryData=boundaryDataOutNodeThree, exactSolution=None)

# Create your graph here with its' initial node
streets = Graph(initialNode=centralNode)

# Create the edges for your graph here
inEdgeOne      = Edge(equation=linearEqnOne, numericalFlux=linearNumerical, inNode=inNodeOne, outNode=centralNode)
inEdgeTwo      = Edge(equation=linearEqnTwo, numericalFlux=linearNumerical, inNode=inNodeTwo, outNode=centralNode)
outEdgeOne     = Edge(equation=linearEqnThree, numericalFlux=linearNumerical, inNode=centralNode, outNode=outNodeOne)
outEdgeTwo     = Edge(equation=linearEqnFour, numericalFlux=linearNumerical, inNode=centralNode, outNode=outNodeTwo)
outEdgeThree   = Edge(equation=linearEqnFive, numericalFlux=linearNumerical, inNode=centralNode, outNode=outNodeThree)

# Assemble your graph adding all the edges
streets.addEdge(newEdge=inEdgeOne)
streets.addEdge(newEdge=inEdgeTwo)
streets.addEdge(newEdge=outEdgeOne)
streets.addEdge(newEdge=outEdgeTwo)
streets.addEdge(newEdge=outEdgeThree)

# Compute the solution to your problem here
errorL1       = 0.
errorL1New    = 0.
eoc           = 0.
cfl           = 0.5
T             = 0.5
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
streetsPlot = CLonGraphPlot(discreteEqn=discreteEquation, pltRefinement=1000) 
streetsPlot.showAnimations('Linear Advection')
#streetsPlot.saveAnimation3d('Linear_Advection_Example')
streetsPlot.plot3D(0, 't=0.0')
streetsPlot.plot3D(discreteEquation.getNumTimeSteps()-1, 't=0.5')
streetsPlot.showPlot()
#streetsPlot.saveAnimationProjection2d('complexNetworkProjectionAnimation50CellsDots')
#streetsPlot.saveAnimation2d('complexNetworkConventionalPlotAnimation2')
