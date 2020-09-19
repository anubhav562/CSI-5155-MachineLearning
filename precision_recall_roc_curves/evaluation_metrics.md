## Evaluation Metrics for Binary classification

Hello everyone, thanks for dropping in if you are following along with me.

In this introduction and the code which would be following, we will be hearing the following terms a lot:
- Contingency table / Confusion matrix
- True Positives Rate, True Negative Rate, False Positive Rate, False Negative Rate
- Sensitivity, Specificity, Inverse Specificity
- Precision, Recall
- Coverage plots, ROC curve, Precision Recall curve.


Let's first start with what binary classification is?

It simply means, we want to predict the class of a test instance out of the two possible classes.
(generally we refer to these classes by calling them as positive class and negative class)

## Definitions / Formulas

- True Positives (TP) ~~~~ The number of instances which actually belonged to +ve class and the classifier also 
predicted them to be of +ve class.

- True Negatives (TN) ~~~~ The number of instances which actually belonged to -ve class and the classifier also 
predicted them to be of -ve class.

- False Positives (FP) ~~~~ the number of instances which actually belonged to -ve class but were wrongly classified as 
+ve by the classifier.

- False Negatives (FN) ~~~~ the number of instances which actually belonged to +ve class but were wrongly classified as 
-ve by the classifier.

#### True Positive Rate
True Positive Rate = No. of True Positives / Total number of instances actually belonging to +ve class

In other words this can be also written as:

TPR = TP / TP + FN

**True positive rate is also termed as sensitivity.**


#### True Negative Rate
True Negative Rate = No. of True Negatives / Total number of instances actually belonging to -ve class

In other words this can be also written as:

TNR = TN / TN + FP

**True negative rate is also termed as specificity.**


#### False Positive Rate
False Positive Rate = No. of False Positives / Total number of instances actually belonging to -ve class

In other words this can be also written as:

FPR = FP / FP + TN

**True positive rate is also termed as inverse sensitivity.**

FPR = 1 - TNR

FPR = 1 - (TN / (TN + FP))

FPR = (TN + FP - TN)/(TN + FP) = FP/(FP + TN)

**False positive rate is sometimes also referred to as false alarm rate**


#### False Negative Rate
False Negative Rate = No. of False Negatives / Total number of instances actually belonging to +ve class

In other words this can be also written as:

FNR = FN / FN + TP


### Precision

Precision = TP / TP + FP

Precision gives us an overall idea about how the model is doing in terms of predicting the positive class 
(or the class of interest)

### Recall 

Recall = TP / TP + FN

Recall == Sensitivity == TPR
