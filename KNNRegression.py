import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

session = tf.Session()
# calculate euclidean distance square
def D_eucl_sqr(X, Z):
	if X.shape[1] != Z.shape[1]:
		print("D_eucl_sqr: X and Z must have the same number of column")
		return tf.constant([])
	return tf.reduce_sum(tf.square(X[:, None] - Z[None, :]), 2)

# find k nearest neighbors, return responsility matrix
def K_NN(X, Z, k):
	if X.shape[0] < k:
		print("k exceeds the maximum allowed number by matrix!")
		return None;
	distance = D_eucl_sqr(Z, X)
	reciprocal = tf.reciprocal(distance)
	values, indices = tf.nn.top_k(reciprocal, k)
	kthValue = tf.reduce_min(values)
	# calculate condition mask
	cond = tf.greater_equal(tf.transpose(reciprocal), kthValue)
	# assign responsibility with 1 or 0
	R = tf.where(cond, tf.ones(cond.shape, tf.float64), tf.zeros(cond.shape, tf.float64))
	# normalize responsibilty to sum up to 1
	R = tf.divide(R, tf.reduce_sum(R, 0))
	return R

# find k nearest neighbors, use the majority vote X-training, Z-testing
def K_NN_Majority_Vote(X, Z, k):
	if X.shape[0] < k:
		print("k exceeds the maximum allowed number by matrix!")
		return None;
	distance = D_eucl_sqr(Z, X)
	reciprocal = tf.reciprocal(distance)
	values, indices = tf.nn.top_k(reciprocal, k)
	return indices

def report(training_data, training_target, testing_data, testing_target, k, name):
	R = K_NN(training_data, testing_data, k)
	predictions = tf.matmul(tf.transpose(R), training_target)
	error = tf.divide(tf.reduce_sum(tf.square(predictions - testing_target)), 2*testing_target.shape[0])
	print(name, " error: ", session.run(error))
	return predictions

def get_subplot_index(k):
	map = {}
	map[1] = 1
	map[3] = 2
	map[5] = 3
	map[50] = 4
	return map[k]
 
def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
	data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	target[rnd_idx[trBatch + validBatch + 1:-1], task]
	return trainData, validData, testData, trainTarget, validTarget, testTarget

def phote_predicting_class_label(training_data, training_target, testing_data, testing_target, k, name):
	indices = K_NN_Majority_Vote(training_data, testing_data, k)
	KVoteMatrix = tf.cast(tf.gather(training_target, indices), tf.int64)
	print(KVoteMatrix.shape)
	Prediction = tf.map_fn(lambda x: tf.gather(tf.unique_with_counts(x)[0],tf.argmax(tf.unique_with_counts(x)[2])),KVoteMatrix)
	error = tf.divide(tf.reduce_sum(tf.square(Prediction - testing_target)), 2*testing_target.shape[0])
	print(k, " error: ", session.run(error))
	return Prediction


