from .edge import Edge
from .node import Node

class Graph:
  """This class provides a data-structure for a graph"""

####################

  class EdgeIterator:
    """This class provides an iterator over all edges of a graph"""


    def __init__(self, edges: Edge):
      self._iterator = iter(edges.values())


    def __iter__(self):
      return self


    def __next__(self):
      return self._iterator.__next__()

####################

  class NodeIterator:
    """This class provides an iterator over all nodes of a graph"""


    def __init__(self, nodes: Node):
      self._iterator = iter(nodes.values())


    def __iter__(self):
      return self


    def __next__(self) -> Node:
      return self._iterator.__next__()

####################

  def __init__(self, initialNode: Node):
    self._initialNode = initialNode
    self._nodes       = {self._initialNode.getNodeID() : self._initialNode}
    self._edges               = {}
    self._connectInEdgesNode  = {}
    self._connectOutEdgesNode = {}
    self._numEdges            = 0
    self._numJointNodes       = 0
    self._numBoundaryNodes    = 0
    if self._initialNode.isBoundaryNode():
      self._numBoundaryNodes += 1
    else:
      self._numJointNodes += 1


  def edgeIterator(self) -> EdgeIterator:
    return self.EdgeIterator(self._edges)


  def nodeIterator(self) -> NodeIterator:
    return self.NodeIterator(self._nodes)


  def addEdge(self, newEdge: Edge):
    inNode = newEdge.getInNode()
    outNode = newEdge.getOutNode()
    inNodeID = inNode.getNodeID()
    outNodeID = outNode.getNodeID()
    if (newEdge.getEdgeID() in self._edges):
      print("This edge already exists. \r" \
            "The edge will not be added.")
    elif not ((inNodeID in self._nodes) or (outNodeID in self._nodes)):
      print("Neither the in nor the out node is part of the graph yet. \r" \
            "Adding the edge would lead to the graph beeing not connected anymore. \r" \
            "The edge will not be added.")
    else:
      self._edges.update({newEdge.getEdgeID() : newEdge})
      if not (inNodeID in self._nodes):
        self.addNode(inNode)
      if not (outNodeID in self._nodes):
        self.addNode(outNode)
      self.connectInEdge(newEdge.getEdgeID(), outNodeID)
      self.connectOutEdge(newEdge.getEdgeID(), inNodeID)
      self._numEdges += 1


  def addNode(self, newNode: Node):
    if (newNode.getNodeID() in self._nodes):
      print("This node already exists.")
    else:
      if newNode.isBoundaryNode():
        self._numBoundaryNodes += 1
      else:
        self._numJointNodes += 1 
      self._nodes.update({newNode.getNodeID() : newNode})


  def connectInEdge(self, newEdgeID: int, connectNodeID: int):
    if not (connectNodeID in self._nodes):
      print("The node %d to which this edge should be connected ingoing is not in the graph" % (connectNodeID))
    elif not (connectNodeID in self._connectInEdgesNode):
      self._connectInEdgesNode.update({connectNodeID: [newEdgeID]})
    else:
      if newEdgeID in self._connectInEdgesNode[connectNodeID]:
        print("This edge is already registered as an ingoing edge to this node \r" \
              "The Edge will not be registered again")
      else:
        self._connectInEdgesNode[connectNodeID].append(newEdgeID)


  def connectOutEdge(self, newEdgeID: int, connectNodeID: int):
    if not (connectNodeID in self._nodes):
      print("The node %d% to which this edge should be connected outgoing is not in the graph" % (connectNodeID))
    elif not (connectNodeID in self._connectOutEdgesNode):
      self._connectOutEdgesNode.update({connectNodeID: [newEdgeID]})
    else:
      if newEdgeID in self._connectOutEdgesNode[connectNodeID]:
        print("This edge is already registed as an outgoing edge to this node \r" \
              "The Edge will not be registered again")
      else:
        self._connectOutEdgesNode[connectNodeID].append(newEdgeID)


  def getConnectInEdges(self, nodeID: int) -> list:
    if not (nodeID in self._connectInEdgesNode):
      return []
    else:
      return self._connectInEdgesNode[nodeID]


  def getConnectOutEdges(self, nodeID: int) -> list:
    if not (nodeID in self._connectOutEdgesNode):
      return []
    else:
      return self._connectOutEdgesNode[nodeID]


  def getEdge(self, edgeID: int) -> Edge:
    if (edgeID in self._edges):
      return self._edges[edgeID]
    else:
      print("The requested edge was not found in the graph")


  def getNode(self, nodeID: int) -> Node:
    if (nodeID in self._nodes):
      return self._nodes[nodeID]
    else:
      print("The requested node was not found in the graph")


  def getNumJointNodes(self) -> int:
    return self._numJointNodes


  def getNumBoundaryNodes(self) -> int:
    return self._numBoundaryNodes


  def getNumEdges(self) -> int:
    return self._numEdges
