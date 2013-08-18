import random
import math

def activationFunction(x):
   return 1.0 / (1.0 + math.exp(-x))

class Node:
   def __init__(self):
      self.lastOutput = None
      self.lastInput = None
      self.error = None
      self.outgoingEdges = []
      self.incomingEdges = []
      self.addBias()

   def addBias(self):
      self.incomingEdges.append(Edge(BiasNode(), self))

   def evaluate(self, inputVector):
      if self.lastOutput is not None:
         return self.lastOutput

      self.lastInput = []
      weightedSum = 0

      for e in self.incomingEdges:
         theInput = e.source.evaluate(inputVector)
         self.lastInput.append(theInput)
         weightedSum += e.weight * theInput

      self.lastOutput = activationFunction(weightedSum)
      self.evaluateCache = self.lastOutput
      return self.lastOutput

   def getError(self, label):
      ''' Get the error for a given node in the network. If the node is an
         output node, label will be used to compute the error. For an input node, we
         simply ignore the error. '''

      if self.error is not None:
         return self.error

      assert self.lastOutput is not None

      if self.outgoingEdges == []: # this is an output node
         self.error = label - self.lastOutput
      else:
         self.error = sum([edge.weight * edge.target.getError(label) for edge in self.outgoingEdges])

      return self.error

   def updateWeights(self, learningRate):
      ''' Update the weights of a node, and all of its successor nodes.
         Assume self is not an InputNode. If the error, lastOutput, and
         lastInput are None, then this node has already been updated. '''

      if (self.error is not None and self.lastOutput is not None
            and self.lastInput is not None):

         for i, edge in enumerate(self.incomingEdges):
            edge.weight += (learningRate * self.lastOutput * (1 - self.lastOutput) *
                           self.error * self.lastInput[i])

         for edge in self.outgoingEdges:
            edge.target.updateWeights(learningRate)

         self.error = None
         self.lastInput = None
         self.lastOutput = None

   def clearEvaluateCache(self):
      if self.lastOutput is not None:
         self.lastOutput = None
         for edge in self.incomingEdges:
            edge.source.clearEvaluateCache()


class InputNode(Node):
   ''' Input nodes simply evaluate to the value of the input for that index.
    As such, each input node must specify an index. We allow multiple copies
    of an input node with the same index (why not?). '''

   def __init__(self, index):
      Node.__init__(self)
      self.index = index;

   def evaluate(self, inputVector):
      self.lastOutput = inputVector[self.index]
      return self.lastOutput

   def updateWeights(self, learningRate):
      for edge in self.outgoingEdges:
         edge.target.updateWeights(learningRate)

   def getError(self, label):
      for edge in self.outgoingEdges:
         edge.target.getError(label)

   def addBias(self):
      pass

   def clearEvaluateCache(self):
      self.lastOutput = None


class BiasNode(InputNode):
   def __init__(self):
      Node.__init__(self)

   def evaluate(self, inputVector):
      return 1.0


class Edge:
   def __init__(self, source, target):
      self.weight = random.uniform(0,1)
      self.source = source
      self.target = target

      # attach the edges to its nodes
      source.outgoingEdges.append(self)
      target.incomingEdges.append(self)


class Network:
   def __init__(self):
      self.inputNodes = []
      self.outputNode = None

   def evaluate(self, inputVector):
      assert max([v.index for v in self.inputNodes]) < len(inputVector)
      self.outputNode.clearEvaluateCache()

      output = self.outputNode.evaluate(inputVector)
      return output

   def propagateError(self, label):
      for node in self.inputNodes:
         node.getError(label)

   def updateWeights(self, learningRate):
      for node in self.inputNodes:
         node.updateWeights(learningRate)

   def train(self, labeledExamples, learningRate=0.9, maxIterations=10000):
      while maxIterations > 0:
         for example, label in labeledExamples:
            output = self.evaluate(example)
            self.propagateError(label)
            self.updateWeights(learningRate)

            maxIterations -= 1

