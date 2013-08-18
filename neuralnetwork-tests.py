from unittest import *
from neuralnetwork import *

def binaryNumbersTest():
   network = Network()
   inputNodes = [InputNode(i) for i in range(3)]
   hiddenNodes = [Node() for i in range(3)]
   outputNode = Node()

   # weights are all randomized
   for inputNode in inputNodes:
      for node in hiddenNodes:
         Edge(inputNode, node)

   for node in hiddenNodes:
      Edge(node, outputNode)

   network.outputNode = outputNode
   network.inputNodes.extend(inputNodes)

   labeledExamples = [((0,0,0), 1),
                      ((0,0,1), 0),
                      ((0,1,0), 1),
                      ((0,1,1), 0),
                      ((1,0,0), 1),
                      ((1,0,1), 0),
                      ((1,1,0), 1),
                      ((1,1,1), 0)]
   network.train(labeledExamples, maxIterations=5000)

   # test for consistency
   for number, isEven in labeledExamples:
      print "Error for %r is %0.4f. Output was:%0.4f" % (number, isEven - network.evaluate(number), network.evaluate(number))


def makeNetwork(numInputs, numHiddenLayers, numInEachLayer):
   network = Network()
   inputNodes = [InputNode(i) for i in range(numInputs)]
   outputNode = Node()
   network.outputNode = outputNode
   network.inputNodes.extend(inputNodes)

   layers = [[Node() for _ in range(numInEachLayer)] for _ in range(numHiddenLayers)]

   # weights are all randomized
   for inputNode in inputNodes:
      for node in layers[0]:
         Edge(inputNode, node)

   for layer1, layer2 in [(layers[i], layers[i+1]) for i in range(numHiddenLayers-1)]:
      for node1 in layer1:
         for node2 in layer2:
            Edge(node1, node2)

   for node in layers[-1]:
      Edge(node, outputNode)

   return network


def sineTest(numLayers, numNodes):
   import math
   import random

   f = lambda x: 0.5 * (1.0 + math.sin(x))
   domain = lambda: [random.random()*math.pi*4 for _ in range(100)]

   network = makeNetwork(1, numLayers, numNodes)
   labeledExamples = [((x,), f(x)) for x in  domain()]
   network.train(labeledExamples, learningRate=0.25, maxIterations=100000)

   errors = [abs(f(x) - network.evaluate((x,))) for x in domain()]
   print "Avg error: %.4f" % (sum(errors) * 1.0 / len(errors))

   with open('sine.txt', 'a') as theFile:
      vals = tuple((x,network.evaluate((x,))) for x in domain())
      line = "{%s},\n" % (",".join(["{%s}" % ",".join([str(n) for n in x]) for x in vals]),)
      theFile.write(line)


def digitsTest():
   import random
   network = makeNetwork(256, 2, 15)

   digits = []

   with open('digits.dat', 'r') as dataFile:
      for line in dataFile:
         (exampleStr, classStr) = line.split(',')
         digits.append(([int(x) for x in exampleStr.split()], float(classStr) / 9))

   random.shuffle(digits)
   trainingData, testData = digits[:-500], digits[-500:]

   network.train(trainingData, learningRate=0.5, maxIterations=100000)
   errors = [abs(testPt[-1] - round(network.evaluate(testPt[0]))) for testPt in testData]
   print "Average error: %.4f" % (sum(errors)*1.0 / len(errors))


if __name__ == "__main__":
   #binaryNumbersTest()

   print "Sine"
   with open('sine.txt','w') as theFile:
      theFile.write("{")

   sineTest(1, 20)

   with open('sine.txt','a') as theFile:
      theFile.write("}\n")

   print "Digits"
   digitsTest()
