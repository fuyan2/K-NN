import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import KNNRegression



session = tf.Session()
trainData, validData, testData, trainTarget, validTarget, testTarget = KNNRegression.data_segmentation("data.npy", "target.npy", 0)

for k in {1, 5, 10, 25, 50, 100, 200}:
	predict = KNNRegression.phote_predicting_class_label(trainData, trainTarget, testData, testTarget, k, "Try")

print("predict is : ",session.run(predict))
