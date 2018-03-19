
import random
import math

import emtreesc
  
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = random.randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Tree representation as 2d array
# feature, value, left_child, right_child
# Leaf node: -1, class, -1, -1
# 	fields = { 'feature': 0, 'value': 1, 'left': 2, 'right': 3 }
def flatten_tree(tree):
	flat = []
	next_node_idx = 0

	def is_leaf(node):
		return not isinstance(node, dict)
	def get_node_idx(node):
		nonlocal next_node_idx
		idx = next_node_idx
		next_node_idx += 1
		return idx

	def flatten_node(node):
		if is_leaf(node):
			flat.append([-1, node, -1, -1])
			return get_node_idx(node)

		l = flatten_node(node['left'])
		r = flatten_node(node['right'])
		flat.append([node['index'], node['value'], l, r])
		return get_node_idx(node)

	r_idx = flatten_node(tree)
	assert r_idx == next_node_idx -1
	assert r_idx == len(flat) - 1

	return flat


def flatten_forest(trees):
	tree_roots = []
	tree_offset = 0
	forest_nodes = []

	for tree in trees: 
		flat = flatten_tree(tree)

		# Offset the nodes in tree, so they can be stored in one array 
		root = len(flat) - 1 + tree_offset
		for node in flat:
			if node[2] > 0:
				node[2] += tree_offset
			if node[3] > 0:
				node[3] += tree_offset
		tree_offset += len(flat)
		tree_roots.append(root)
		forest_nodes += flat

	return forest_nodes, tree_roots

def remap_node_references(nodes, remap):
	for n in nodes:
		n[2] = remap.get(n[2], n[2])
		n[3] = remap.get(n[3], n[3])

def remove_orphans(nodes, roots):
	referenced = []
	for n in nodes:
		if n[0] >= 0:
			referenced.append(n[2])
			referenced.append(n[3])
	referenced = set(referenced).union(roots)
	all_nodes = set(range(len(nodes))) 
	orphaned = all_nodes.difference(referenced)


	offsets = []	
	offset = 0
	for idx, node in enumerate(nodes):
		offsets.append(offset)
		if idx in orphaned:
			offset -= 1

	compacted = []
	for idx, node in enumerate(nodes):
		if idx in orphaned:
			continue

		if node[0] >= 0:
			node[2] += offsets[node[2]]
			node[3] += offsets[node[3]] 
		compacted.append(node)

	compacted_roots = [ r + offsets[r] for r in roots ]

	return compacted, compacted_roots


def remove_duplicate_leaves(forest):
	nodes, roots = forest

	unique_leaves = []
	unique_idx = []
	remap_leaves = {}
	for i, node in enumerate(nodes):
		if node[0] >= 0:
			# not a leaf
			continue
		found = unique_leaves.index(node) if node in unique_leaves else None
		if found is None:
			unique_leaves.append(node)
			unique_idx.append(i)
		else:
			remap_leaves[i] = unique_idx[found]	

	leaves = list(filter(lambda n: n[0] < 0, nodes))
	wasted = (len(leaves) - len(unique_leaves)) / len(nodes)
	
	remap_node_references(nodes, remap_leaves)
	compacted, compacted_roots = remove_orphans(nodes, roots)

	return compacted, compacted_roots

def traverse_dfs(nodes, idx, visitor):
	visitor(idx)
	if nodes[idx][0] < 0:
		return None
	traverse_dfs(nodes, nodes[idx][2], visitor)
	traverse_dfs(nodes, nodes[idx][3], visitor)

def dot_node(name, **opts):
	return '{name} [label={label}];'.format(name=name, label=opts['label'])
def dot_edge(src, tgt, **opts):
	return '{src} -> {tgt} [taillabel={label}, labelfontsize={f}];'.format(src=src,tgt=tgt,label=opts['label'], f=opts['labelfontsize'])
def dot_cluster(name, nodes, indent='  '):
	name = 'cluster_' + name
	n = ('\n'+indent).join(nodes)
	return 'subgraph {name} {{\n  {nodes}\n}}'.format(name=name, nodes=n)

def forest_to_dot(forest, name='emtrees', indent="  "):
	nodes, roots = forest

	leaf_nodes = list(filter(lambda i: nodes[i][0] < 0, range(len(nodes))))
	trees = [ [] for r in roots ]
	for tree_idx, root in enumerate(roots):
		collect = []
		traverse_dfs(nodes, root, lambda i: collect.append(i))
		trees[tree_idx] = set(collect).difference(leaf_nodes)

	edges = []
	leaves = []
	clusters = []

	# group trees using cluster
	for tree_idx, trees in enumerate(trees):
		decisions = []
		for idx in trees:
			node = nodes[idx]
			n = dot_node(idx, label='"{}: feature[{}] < {}"'.format(idx, node[0], node[1]))
			left = dot_edge(idx, node[2], label='"  1"', labelfontsize=8)
			right = dot_edge(idx, node[3], label='"  0"', labelfontsize=8)
			decisions += [ n ]
			edges += [ left, right]

		clusters.append(dot_cluster('_tree_{}'.format(tree_idx), decisions, indent=2*indent))

	# leaves shared between trees
	for idx in leaf_nodes:
		node = nodes[idx]
		leaves += [ dot_node(idx, label='"{}"'.format(node[1])) ]

	dot_items = clusters + edges + leaves

	graph_options = {
		#'rankdir': 'LR',
		#'ranksep': 0.07,
	}

	variables = {
		'name': name,
		'options': ('\n'+indent).join('{}={};'.format(k,v) for k,v in graph_options.items()),
		'items': ('\n'+indent).join(dot_items),
	}
	dot = """digraph {name} {{
	  // Graph options
	  {options}

	  // Nodes/edges
	  {items}
	}}""".format(**variables)

	return dot


def generate_c_nodes(flat, name):
	def node(n):
		return "{{ {}, {}, {}, {} }}".format(*n)

	nodes_structs = ',\n  '.join(node(n) for n in flat)
	nodes_name = name
	nodes_length = len(flat)
	nodes = "EmtreesNode {nodes_name}[{nodes_length}] = {{\n  {nodes_structs} \n}};".format(**locals());

	return nodes
	

def generate_c_forest(forest, name='myclassifier'):
	nodes, roots = forest

	nodes_name = name+'_nodes'
	nodes_length = len(nodes)
	nodes_c = generate_c_nodes(nodes, nodes_name)

	tree_roots_length = len(roots)
	tree_roots_name = name+'_tree_roots';
	tree_roots_values = ', '.join(str(t) for t in roots)
	tree_roots = 'int32_t {tree_roots_name}[{tree_roots_length}] = {{ {tree_roots_values} }};'.format(**locals())

	forest = """Emtrees {name} = {{
		{nodes_length},
		{nodes_name},	  
		{tree_roots_length},
		{tree_roots_name},
	}};""".format(**locals())
	
	return '\n\n'.join([nodes_c, tree_roots, forest]) 


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = random.randrange(len(dataset))
		sample.append(dataset[index])
	return sample
 
import copy
 
# TODO: implement max_nodes limit
class RandomForest:
	def __init__(self, n_features=None, max_depth=10, min_size=1, sample_size=1.0, n_trees=10):
		self.n_trees = n_trees
		self.sample_size = sample_size
		self.min_size = min_size
		self.max_depth = max_depth
		self.n_features = n_features

		self.forest = None
		self.classifier = None

	def get_params(self, deep=False):
		param_names = ['n_trees', 'sample_size', 'min_size', 'max_depth', 'n_features']
		params = {}
		for name in param_names:
			params[name] = getattr(self, name)
		return params

	def fit(self, X, Y):
		if self.n_features is None:
			self.n_features = math.sqrt(len(X[0]))

		# Internally targets are expected to be part of same list/array as features
		data = []
		for x, y in zip(X, Y):
			row = list(x) + [ y ]
			data.append(row)

		trees = list()
		for i in range(self.n_trees):
			sample = subsample(data, self.sample_size)
			tree = build_tree(sample, self.max_depth, self.min_size, self.n_features)
			trees.append(tree)

		self.forest = flatten_forest(trees)
		self.forest = remove_duplicate_leaves(self.forest)		
		nodes, roots = self.forest
		node_data = []
		for node in nodes:
			node_data += node # copy.copy(node)
		self.classifier = emtreesc.Classifier(node_data, roots)

	def predict(self, X):
		# TODO: only pass features
		predictions = [ self.classifier.predict(row) for row in X ]
		return predictions

	def output_c(self, name):
		return generate_c_forest(self.forest, name)

	def to_dot(self, **kwargs):
		return forest_to_dot(self.forest, **kwargs)

def main():
	# Example usage on Sonar Dataset
	import csv

	# Split a dataset into k folds
	def cross_validation_split(dataset, n_folds):
		dataset_split = list()
		dataset_copy = list(dataset)
		fold_size = int(len(dataset) / n_folds)
		for i in range(n_folds):
			fold = list()
			while len(fold) < fold_size:
				index = random.randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
			dataset_split.append(fold)
		return dataset_split
	 
	# Calculate accuracy percentage
	def accuracy_metric(actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0
	 
	# Evaluate an algorithm using a cross validation split
	def evaluate_algorithm(dataset, algorithm, n_folds):
		folds = cross_validation_split(dataset, n_folds)
		scores = list()
		for fold in folds:
			train_set = list(folds)
			train_set.remove(fold)
			train_set = sum(train_set, [])
			train_X = [row[:-1] for row in train_set]
			train_Y = [row[-1] for row in train_set] 
			test_X = [row[:-1] for row in fold]
			test_Y = [row[-1] for row in fold]

			algorithm.fit(train_X, train_Y)
			predicted = algorithm.predict(test_X)
			accuracy = accuracy_metric(test_Y, predicted)

			scores.append(accuracy)
		return scores

	# Load a CSV file
	def load_csv(filename):
		dataset = list()
		with open(filename, 'r') as file:
			csv_reader = csv.reader(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset
	 
	# Convert string column to float
	def str_column_to_float(dataset, column):
		for row in dataset:
			row[column] = float(row[column].strip())
	 
	# Convert string column to integer
	def str_column_to_int(dataset, column):
		class_values = [row[column] for row in dataset]
		unique = set(class_values)
		lookup = dict()
		for i, value in enumerate(unique):
			lookup[value] = i
		for row in dataset:
			row[column] = lookup[row[column]]
		return lookup

	# Test the random forest algorithm
	random.seed(3)
	# load and prepare data
	filename = 'sonar.all-data.csv'
	dataset = load_csv(filename)
	# convert string attributes to integers
	for i in range(0, len(dataset[0])-1):
		str_column_to_float(dataset, i)
	# convert class column to integers
	str_column_to_int(dataset, len(dataset[0])-1)

	# convert floats to integer
	for row in dataset:
		for idx, data in enumerate(row[:-1]):
			fixed = int(data * 2**16)
			row[idx] = fixed

	# evaluate algorithm
	n_folds = 5
	for n_trees in [1, 5, 10]:
		estimator = RandomForest(n_trees=n_trees, max_depth=10, min_size=10, sample_size=1.0)
		scores = evaluate_algorithm(dataset, estimator, n_folds)

		with open('mytree.h', 'w') as f:
			f.write(generate_c_forest(estimator.forest))

		print('Trees: %d' % n_trees)
		print("Node storage: {} bytes".format(len(estimator.forest[0]) * 9))
		print('Scores: %s' % scores)
		print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

if __name__ == '__main__':
	main()
