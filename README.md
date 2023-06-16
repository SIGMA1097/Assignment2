# Assignment2
Git, GitHub, ML
Explaination of the code

•	The provided code carries out the subsequent actions:
•	Imported are the necessary libraries, including pandas, numpy, and matplotlib.
•	Using pandas, the data is loaded into a DataFrame called x1 from a CSV file called "data.csv."
•	The 'Unnamed: 32' column has been removed from the DataFrame x1.
•	The describe() function is used to display some of the DataFrame x1's most important statistics.
•	The 'diagnostic' column of x1's unique classifications are identified.
•	The DataFrame x1 is divided into the features (x) and the target variable (y).
•	The train_test_split function from scikit-learn is used to divide the data into training and testing datasets. at guarantee a balanced class distribution, the test dataset size is set at 20% of the total data.
•	The StandardScaler from scikit-learn is used to scale the data using standardisation.
