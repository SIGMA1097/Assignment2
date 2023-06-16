# Assignment2
Git, GitHub, ML
Explaination of the code

The provided code carries out the subsequent actions:

Imported are the necessary libraries, including pandas, numpy, and matplotlib.


Using pandas, the data is loaded into a DataFrame called x1 from a CSV file called "data.csv."

The 'Unnamed: 32' column has been removed from the DataFrame x1.

The describe() function is used to display some of the DataFrame x1's most important statistics.

The 'diagnostic' column of x1's unique classifications are identified.

The DataFrame x1 is divided into the features (x) and the target variable (y).

The train_test_split function from scikit-learn is used to divide the data into training and testing datasets. at guarantee a balanced class distribution, the test dataset size is set at 20% of the total data.

The StandardScaler from scikit-learn is used to scale the data using standardisation.

Scikit-learn's SVC and GaussianNB classes are used, respectively, to create the Support Vector Machine (SVM) and Naive Bayes classification models.

For each model, predictions are made using the test data after the model has been trained using the training data.

Scikit-learn's confusion_matrix and classification_report functions are used to output the classification report (precision, recall, F1-score, and support) for each model.

Performance metrics for both the SVM and Naive Bayes models are shown in the results.

The programme evaluates how well SVM and Naive Bayes performed on the provided dataset using these two classification models.
