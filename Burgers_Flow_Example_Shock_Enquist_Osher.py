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
  if x < 0.8:
    return 2.0
  else:
    return 1.0

def initialDataInEdgeTwo(x: float) -> float:
  return 1.0

def initialDataOutEdgeOne(x: float) -> float:
  return np.sqrt(2./3.)

def initialDataOutEdgeTwo(x: float) -> float:
  return np.sqrt(2./3.)

def initialDataOutEdgeThree(x: float) -> float:
  return np.sqrt(2./3.)

def boundaryDataInNodeOne(t: float) -> float:
  return 2.0

def boundaryDataInNodeTwo(t: float) -> float:
  return 1.0

def boundaryDataOutNodeOne(t: float) -> float:
  return np.sqrt(2./3.) 

def boundaryDataOutNodeTwo(t: float) -> float:
  return np.sqrt(2./3.)

def boundaryDataOutNodeThree(t: float) -> float:
  return np.sqrt(2./3.)

def initialDataCentralNode() -> float:
  return np.sqrt(2./3.) 

def exactSolutionInEdgeOne(x: float, t: float) -> float:
  if x < (3./2.)*t+0.8:
    return 2.
  else:
    return 1.

def exactSolutionInEdgeTwo(x: float, t: float) -> float:
  return 1.

def exactSolutionOutEdgeOne(x: float, t: float) -> float:
  if t < (2./15.):
    return np.sqrt(2./3.)
  else:
    if x < (1./(2.*(np.sqrt(5./3.)-np.sqrt(2./3.))))*(t-(2./15.)):
      return np.sqrt(5./3.)
    else:
      return np.sqrt(2./3.)

def exactSolutionOutEdgeTwo(x: float, t: float) -> float:
  if t < (2./15.):
    return np.sqrt(2./3.)
  else:
    if x < (1./(2.*(np.sqrt(5./3.)-np.sqrt(2./3.))))*(t-(2./15.)):
      return np.sqrt(5./3.)
    else:
      return np.sqrt(2./3.)

def exactSolutionOutEdgeThree(x: float, t: float) -> float:
  if t < (2./15.):
    return np.sqrt(2./3.)
  else:
    if x < (1./(2.*(np.sqrt(5./3.)-np.sqrt(2./3.))))*(t-(2./15.)):
      return np.sqrt(5./3.)
    else:
      return np.sqrt(2./3.)

def exactSolutionJointNode(t: float) -> float:
  if t < (2./15.): 
    return np.sqrt(2./3.)
  else:
    return np.sqrt(5./3.)
  
# Create your edge fluxes here
burgersEdgeFlux = Flux(name='Burgers Flow Edge', fluxFunction=fluxcollection.positiveBurgers, capacity=1.)
# And numerical fluxes here
burgersNumerical = NumericalFlux(name='Burgers Flow Numerical', fluxFunction=fluxcollection.enquistOsher, analyticalFlux=fluxcollection.positiveBurgers, analyticalDerivative=fluxcollection.derivativePositiveBurgers, capacity=1.)
# And derivatives here
burgersDerivative = Flux(name='Burgers Flow Derivative', fluxFunction=fluxcollection.derivativePositiveBurgers, capacity=1.)

# Create your equation here
burgersEqnOne   = Equation(name='Nonlinear Conservation Law: Burgers Flow, One', flux=burgersEdgeFlux, derivativeFlux=burgersDerivative, initialData=initialDataInEdgeOne, exactSolution=exactSolutionInEdgeOne)
burgersEqnTwo   = Equation(name='Nonlinear Conservation Law: Burgers Flow, Two', flux=burgersEdgeFlux, derivativeFlux=burgersDerivative, initialData=initialDataInEdgeTwo, exactSolution=exactSolutionInEdgeTwo)
burgersEqnThree = Equation(name='Nonlinear Conservation Law: Burgers Flow, Three', flux=burgersEdgeFlux, derivativeFlux=burgersDerivative, initialData=initialDataOutEdgeOne, exactSolution=exactSolutionOutEdgeOne)
burgersEqnFour  = Equation(name='Nonlinear Conservation Law: Burgers Flow, Four', flux=burgersEdgeFlux, derivativeFlux=burgersDerivative, initialData=initialDataOutEdgeTwo, exactSolution=exactSolutionOutEdgeTwo)
burgersEqnFive  = Equation(name='Nonlinear Conservation Law: Burgers Flow, Five', flux=burgersEdgeFlux, derivativeFlux=burgersDerivative, initialData=initialDataOutEdgeThree, exactSolution=exactSolutionOutEdgeThree)

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
inEdgeOne      = Edge(equation=burgersEqnOne, numericalFlux = burgersNumerical, inNode=inNodeOne, outNode=centralNode)
inEdgeTwo      = Edge(equation=burgersEqnTwo, numericalFlux = burgersNumerical, inNode=inNodeTwo, outNode=centralNode)
outEdgeOne     = Edge(equation=burgersEqnThree, numericalFlux = burgersNumerical, inNode=centralNode, outNode=outNodeOne)
outEdgeTwo     = Edge(equation=burgersEqnFour, numericalFlux = burgersNumerical, inNode=centralNode, outNode=outNodeTwo)
outEdgeThree   = Edge(equation=burgersEqnFive, numericalFlux = burgersNumerical, inNode=centralNode, outNode=outNodeThree)

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
streetsPlot.showAnimations('Burgers Flow')
#streetsPlot.saveAnimation3d('Burgers_Flow_Travelling_Shock_Example')
streetsPlot.plot3D(0, 't=0.0')
streetsPlot.plot3D(discreteEquation.getNumTimeSteps()-1, 't=0.5')
streetsPlot.showPlot()
#streetsPlot.saveAnimationProjection2d('complexNetworkProjectionAnimation50CellsDots')
#streetsPlot.saveAnimation2d('complexNetworkConventionalPlotAnimation2')
