# Imbalanced Domain Learning

**Standard Machine Learning vs Imbalanced Machine Learning.**

First of all we are going to have a look on how the imbalanced domain learning
is different from regular/standard machine learning modelling which we do.

In Standard machine learning modelling, we assume/consider the following:
- All/Both (multiclass or binary class) classes are balanced, i.e they are 
  uniformly present. (if we have n instances of class A, then we have ~n instances
  of class B, .... and so on).
  
- The user/expert (the one building the model or solving the problem) is interested
  equally in all the classes. Eg. , if we have a problem statement, in which
  we have a dataset having images of dog, cat, lion, camel, and the user is 
  equally interested in knowing whether the image is of a dog, cat, lion or a camel.
  
 
 While in case of Imbalanced domain learning this is not the case, the following
 are the assumptions:
 
 - In Imbalanced domain learning the distribution of the classes is imbalanced, 
 this simply means that the cardinality of each class is different. 
 (In most of the cases, the extent of imbalance is quite heavy, i.e. one class
 is present in huge numbers while the other has only a few instances.)
 
 - The user has a bias/preference towards a subset of classes, these are the
 classes which are rare (are the minority class(es))
 
 
 
**An ML problem can be said to be belonging to the imbalanced domain learning only if both
of the above-mentioned points hold true for a problem statement.**


### Imbalanced domain learning - why is it challenging?

1. The subset of classes that we want to learn are rare, so how do we learn? How do we fit a
model?
2. Most of the standard machine learning algorithms are based on averaging the performance
metrics over various classes. But, in this case, our area of interest in a subset of target
variables which have rarely occurred. So how do we evaluate that how are we doing?


### Evaluation metrics:

The evaluation metrics in standard machine learning are generally accuracy or error rate.
It should be noted that these metrics are in terms of both the classes (minority as well as
majority). So if we have a model which is doing good only on the majority class will have a 
good accuracy.

For imbalanced domain learning we have the following metrics:

- Precision (TP/TP+FP)
- Recall (TP/TP+FN) (also called sensitivity)
- Geometric Mean (sqrt(sensitivity*specificity))
- F beta scores -> these can be F1, F2, F0.5 scores depending upon the beta we provide
- ROC curve
- Precision Recall Curve.
