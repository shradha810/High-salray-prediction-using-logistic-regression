# High-salray-prediction-using-logistic-regression
To build prediction system using ML to predict if a candidate will get high salary or not.
Data Preparation: -
Deleted the columns of ID, Gender, DOB, 10th board, 12th board, college ID, college city ID, college 
city tier and college state; as these attributes have little/no effect on deciding whether a person will 
get high income/low income salary.
Converted college tier, degree and specialisation to categorical codes.
Shuffled the dataset.
Experiments: -
Performed Logistic regression on the training set and prediction on test set. 
Calculated and displayed accuracy, class wise accuracy and the confusion matrix.
Tried on several values of test set like 0.1, 0.2, 0.3, 0.4
