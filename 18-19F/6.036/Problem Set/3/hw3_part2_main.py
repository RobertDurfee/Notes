import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# # Returns a list of dictionaries.  Keys are the column names, including mpg.
# auto_data_all = hw3.load_auto_data('auto-mpg.tsv')
#
# # The choice of feature processing for each feature, mpg is always raw and
# # does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# # 'name' is not numeric and would need a different encoding.
# features = [('cylinders', hw3.raw),
#             ('displacement', hw3.raw),
#             ('horsepower', hw3.raw),
#             ('weight', hw3.raw),
#             ('acceleration', hw3.raw),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.raw)]
#
# # Construct the standard data and label arrays
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)
#
# if False:                               # set to True to see histograms
#     import matplotlib.pyplot as plt
#     for feat in range(auto_data.shape[0]):
#         print('Feature', feat, features[feat][0])
#         # Plot histograms in one window, different colors
#         plt.hist(auto_data[feat,auto_labels[0,:] > 0])
#         plt.hist(auto_data[feat,auto_labels[0,:] < 0])
#         plt.show()
#         # Plot histograms in two windows, different colors
#         fig,(a1,a2) = plt.subplots(nrows=2)
#         a1.hist(auto_data[feat,auto_labels[0,:] > 0])
#         a2.hist(auto_data[feat,auto_labels[0,:] < 0])
#         plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

review_data = None

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# Perceptron
#-------------------------------------------------------------------------------

def y(x, th, th0):
    return np.dot(np.transpose(th), x) + th0


def positive(x, th, th0):
    return np.sign(y(x, th, th0))


def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0


def averaged_perceptron(data, labels, params = {}, hook = None):
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    theta_sum = theta.copy()
    theta_0_sum = theta_0.copy()
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
            theta_sum = theta_sum + theta
            theta_0_sum = theta_0_sum + theta_0
    theta_avg = theta_sum / (T*n)
    theta_0_avg = theta_0_sum / (T*n)
    if hook: hook((theta_avg, theta_0_avg))
    return theta_avg, theta_0_avg


# print('Results: ' + str(hw3.xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': 50})))
# print('Results: ' + str(hw3.xval_learning_alg(averaged_perceptron, auto_data, auto_labels, 10, {'T': 50})))
#
# features = [('cylinders', hw3.one_hot),
#             ('displacement', hw3.standard),
#             ('horsepower', hw3.standard),
#             ('weight', hw3.standard),
#             ('acceleration', hw3.standard),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.one_hot)]
#
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)
#
# print('Results: ' + str(hw3.xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': 50})))
# print('Results: ' + str(hw3.xval_learning_alg(averaged_perceptron, auto_data, auto_labels, 10, {'T': 50})))
#
# features = [## ('cylinders', hw3.one_hot),
#             ## ('displacement', hw3.standard),
#             ## ('horsepower', hw3.standard),
#             ('weight', hw3.standard),
#             ## ('acceleration', hw3.standard),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.one_hot)]
#
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)
#
# print('Results: ' + str(hw3.xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': 50})))
# print('Results: ' + str(hw3.xval_learning_alg(averaged_perceptron, auto_data, auto_labels, 10, {'T': 50})))
#
# features = [('cylinders', hw3.one_hot),
#             ('displacement', hw3.standard),
#             ('horsepower', hw3.standard),
#             ('weight', hw3.standard),
#             ('acceleration', hw3.standard),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.one_hot)]
#
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)
#
# print('best: ' + str(perceptron(auto_data, auto_labels, {'T': 1})))

print('Perceptron, T=1: ' + str(hw3.xval_learning_alg(perceptron, review_bow_data, review_labels, 10, { 'T': 1 })))
print('Averaged Perceptron, T=1: ' + str(hw3.xval_learning_alg(averaged_perceptron, review_bow_data, review_labels, 10, { 'T': 1 })))

print('Perceptron, T=10: ' + str(hw3.xval_learning_alg(perceptron, review_bow_data, review_labels, 10, { 'T': 10 })))
print('Averaged Perceptron, T=10: ' + str(hw3.xval_learning_alg(averaged_perceptron, review_bow_data, review_labels, 10, { 'T': 10 })))

print('Perceptron, T=20: ' + str(hw3.xval_learning_alg(perceptron, review_bow_data, review_labels, 10, { 'T': 20 })))
print('Averaged Perceptron, T=20: ' + str(hw3.xval_learning_alg(averaged_perceptron, review_bow_data, review_labels, 10, { 'T': 20 })))



