import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import KNNRegression

# generate test data
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


test_fig = plt.figure()
valid_fig = plt.figure()

session = tf.Session()
for k in {1, 3, 5, 50}:
	print("==========================")
	print("=        k = ", k)
	print("==========================")
	# plot training result
	KNNRegression.report(trainData, trainTarget, trainData, trainTarget, k, "training")
	# plot test result
	test_result = KNNRegression.report(trainData, trainTarget, testData, testTarget, k, "test")
	test_plot = test_fig.add_subplot(2, 2, KNNRegression.get_subplot_index(k))
	test_plot.plot(testData, testTarget, 'g^', testData, session.run(test_result), 'r*', trainData, trainTarget, 'b+')
	test_plot.set_title('Test k = ' + str(k))
	# plot validation result
	validate_result = KNNRegression.report(trainData, trainTarget, validData, validTarget, k, "validation")
	valid_plot = valid_fig.add_subplot(2, 2, KNNRegression.get_subplot_index(k))
	valid_plot.plot(validData, validTarget, 'g^', validData, session.run(validate_result), 'r*', trainData, trainTarget, 'b+')
	valid_plot.set_title('Validate k = ' + str(k))
	valid_fig.add_subplot(valid_plot)

plt.show()