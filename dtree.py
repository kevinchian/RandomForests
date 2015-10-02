import numpy
import random
import math
from scipy import io
import csv

data = io.loadmat("./spam-dataset/spam_data.mat")
training_X = numpy.array(data["training_data"])
training_Y = numpy.array(data["training_labels"][0])
test_X = numpy.array(data["test_data"])
word_file = open('./feature_keys.txt', 'r')
words = []
for line in word_file:
	words.append(line)

def build_dtree(data, labels, depth):
	#Stopping Condition
	spam = numpy.sum(labels)
	if (entropy(spam, len(labels)) < 0.3):
		ham = len(labels) - spam
		if ham > spam:
			return LeafNode(0)
		if ham <= spam:
			return LeafNode(1)
	#Determine rule
	best_feature = None
	split = None
	info_val = -float('inf')
	random_features = random.sample(range(0,len(data[0])),int(math.sqrt(len(data[0]))))
	for x in random_features:
		values = data[:,x]
		zipped = zip(values, labels)
		zipped = sorted(zipped, key=lambda x: x[0])
		features, sorted_labels = [list(t) for t in zip(*zipped)]
		split_dictionary = {}
		spam_count = 0

		for y in xrange(0, len(features)):
			if sorted_labels[y] == 1:
				spam_count += 1

			split_dictionary[features[y]]={'left_spam': spam_count, 'left_total': y+1} 


		for value in split_dictionary:
			if (split_dictionary[value]['left_total'] == 0) or (split_dictionary[value]['left_total'] == len(labels)):
				continue
			temp = goodness(len(labels), spam_count, split_dictionary[value]['left_total'], split_dictionary[value]['left_spam'])
			if temp > info_val:
				best_feature = x
				split = value
				info_val = temp

	left_data = []
	left_label = []
	right_data = []
	right_label = []

	if best_feature is None:
		spam = numpy.sum(labels)
		ham = len(labels) - spam
		if ham > spam:
			return LeafNode(0)
		if ham <= spam:
			return LeafNode(1)

	for x in xrange(0, len(data)):
		if data[x][best_feature] <= split:
			left_data.append(data[x])
			left_label.append(labels[x])
		else:
			right_data.append(data[x])
			right_label.append(labels[x])
	left_data = numpy.array(left_data)
	left_label = numpy.array(left_label)
	right_data = numpy.array(right_data)
	right_label = numpy.array(right_label)
	node = Node((best_feature, split), depth)
	node.left = build_dtree(left_data, left_label, depth+1)
	node.right = build_dtree(right_data, right_label,depth+1)
	return node

def entropy(spam_count, total):
	spam = float(spam_count)/total
	ham = 1.0 - spam
	if (spam == 0) or (ham == 0):
		return 0
	return -spam*numpy.log2(spam) - ham*numpy.log2(ham)

def goodness(total, total_spam, left_total, left_spam):
	first = entropy(total_spam, total)
	second = float(left_total)/total*entropy(left_spam, left_total)
	third = float(total - left_total)/total*entropy(total_spam - left_spam, total - left_total)
	return first - second - third

class Node:
	def __init__(self, rule, depth):
		self.rule = rule
		self.left = None
		self.right = None
		self.depth = depth

class LeafNode:
	def __init__(self, value):
		self.value = value

class Decision_Tree:
	def __init__(self, data, labels):
		self.root = None
		self.data = data
		self.labels = labels

	def predict(self, observation):
		node = self.root
		while isinstance(node, Node):
			#Uncomment for a path trace for predictions
			#print "Feature: %s, Split: %f Observation: %f" % (words[node.rule[0]], node.rule[1], observation[node.rule[0]])
			if observation[node.rule[0]] <= node.rule[1]:
				node = node.left
			else:
				node = node.right
		return node.value

	def train(self):
		self.root = build_dtree(self.data, self.labels, 0)


class Random_Forest:
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.dtrees = []

	def populate(self, count):
		for x in xrange(0,count):
			#bagging to get set
			selection = [random.randint(0,len(self.data)-1) for x in xrange(0,len(self.data))]
			bag_data = numpy.array([self.data[x] for x in selection])
			bag_label = numpy.array([self.labels[x] for x in selection])
			dtree = Decision_Tree(bag_data, bag_label)
			dtree.train()
			self.dtrees.append(dtree)		

	def predict(self, observation):
		spam_count = 0
		ham_count = 0
		for x in self.dtrees:
			if x.predict(observation) == 0:
				ham_count+=1
			if x.predict(observation) == 1:
				spam_count+=1
		if ham_count > spam_count:
			return 0
		else:
			return 1

index = [x for x in xrange(0,5172)]
random.shuffle(index)
training_set = numpy.array([training_X[x] for x in index[0:4000]])
training_set_labels = numpy.array([training_Y[x] for x in index[0:4000]])
validation_set = numpy.array([training_X[x] for x in index[4000:5172]])
validation_val = numpy.array([training_Y[x] for x in index[4000:5172]])

forest = Random_Forest(training_X, training_Y)
forest.populate(50)
count = 1
with open('spam.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Id', 'Category'])
	for x in xrange(0,len(test_X)):
		writer.writerow([count, forest.predict(test_X[x])])
		count += 1

#Single Decision Tree
dtree = Decision_Tree(training_set, training_set_labels)
dtree.train()
success = 0
dtree.predict(validation_set[40]) #This is to get the path of one point

for x in xrange(0,1172):
	if dtree.predict(validation_set[x]) == validation_val[x]:
		success += 1
print "Success rate dtree: %f" % (success/1172.0)

#Random Forest w/ 25

forest = Random_Forest(training_set, training_set_labels)
forest.populate(25)

#To find common splits at top of trees
for dtree in forest.dtrees:
	node = dtree.root
	print "Feature: %s, Split %f" % (words[node.rule[0]], node.rule[1])

hit = 0
for x in xrange(0,1172):
	if forest.predict(validation_set[x]) == validation_val[x]:
		hit += 1
print "Success rate forest: %f" % (hit/1172.0)
"""
#Random Forest w/ 10

forest = Random_Forest(training_set, training_set_labels)
forest.populate(10)
hit = 0
for x in xrange(0,1172):
	if forest.predict(validation_set[x]) == validation_val[x]:
		hit += 1
print "Success rate forest: %f" % (success/1172.0)

#Random Forest w/ 40

forest = Random_Forest(training_set, training_set_labels)
forest.populate(40)
hit = 0
for x in xrange(0,1172):
	if forest.predict(validation_set[x]) == validation_val[x]:
		hit += 1
print "Success rate forest: %f" % (success/1172.0)

"""
