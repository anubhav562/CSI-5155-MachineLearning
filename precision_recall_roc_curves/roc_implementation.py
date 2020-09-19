from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


"""
make_classification is a function provided with sklearn.datasets which takes a set of parameters 
and provides us with feature vectors and class labels to play around with.
"""

feature_vectors, class_labels = make_classification(n_samples=500, n_features=10, n_classes=2)

training_feature_vectors, test_feature_vectors, training_class_labels,  test_class_labels = \
    train_test_split(feature_vectors, class_labels, test_size=0.2)

logistic_regression_classifier = LogisticRegression()

logistic_regression_classifier.fit(training_feature_vectors, training_class_labels)

test_class_predictions = logistic_regression_classifier.predict(test_feature_vectors)

test_class_prediction_probabilities = logistic_regression_classifier.predict_proba(test_feature_vectors)[:, 1]

"""
Now we will be implementing a function to calculate the false_positive_rates and true_positive_rates and plot the curve
for us
"""


def self_implemented_roc_curve(y_test_labels, y_test_prediction_probabilities):
    """
    :param y_test_labels: the class labels of the test dataset
    :param y_test_prediction_probabilities: the predicted probabilities calculated using the classifier
    :return: false_positive_rate (list)
           : true_positive_rate (list)

    Can be only used for binary classification, one positive class (0) and one negative class (1)

    This function is the simulation of the roc_curve from sklearn.metrics.

    The algorithm is stated below:
    1. Take the input: y_test_labels and y_predicted_probabilities

    2. Sort the probability in descending order and sort the y_test_labels such that
       order pair of y_test_labels and y_probabilities is not changed.

    3. Now instance by instance we need to calculate the cumulative count/sum of the positive and negative classes both.
       If the 1st instance belongs to class 0, then cumulative count/sum of that class for that instance would be 1,
       if the 2nd instance also belongs to class 0, then the value changes to 2,
       but if the 3rd instance belongs to class 1, the value remains 2 -> the cumulative class for class 1 becomes 1.

    4. Now we can calculate the
       True Positive Rate for each instance=Cumulative sum of +ve class at instance i / Total positives in the test set
       False Positive Rate (each instance) = Cumulative sum of -ve class at instance i / Total negatives in the test set

    Note: If you are having difficulty in understanding the above explanation,
          you can refer to the following link :
          https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used
    """
    # sorting of y_test_labels
    probability_sorted_indices = list(y_test_prediction_probabilities.argsort())
    y_test_labels = list(y_test_labels[probability_sorted_indices])

    cumulative_sum_positive_class = [y_test_labels[:i].count(1) for i in range(1, len(y_test_labels)+1)]
    cumulative_sum_negative_class = [y_test_labels[:i].count(0) for i in range(1, len(y_test_labels)+1)]

    true_positive_rate_list = [a/sum(y_test_labels) for a in cumulative_sum_positive_class]
    false_positive_rate_list = [a / (len(y_test_labels) - sum(y_test_labels)) for a in cumulative_sum_negative_class]

    return false_positive_rate_list, true_positive_rate_list


false_positive_rate, true_positive_rate =\
    self_implemented_roc_curve(test_class_labels, test_class_prediction_probabilities)


pyplot.plot(true_positive_rate, false_positive_rate, marker='.')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.show()
