# Machine_Learning

## IT354 Machine Learning

### Week 1

1. Explore google colab (Online tool to run ML codes) and make a experimental report on features of Google colab with how to access them.
2. Study of computing facilty for machine learning available offline.

   - NVIDIA GPU computer @711 lab
   - Super Micro @718
   - Param Shavak @ wincell
   - DGX server @Center of Excellence
 
### Week 2
Numpy
- Creating blank array, with predefined data, with pattern specific data
- Slicing and Updating elements,
- Shape manipulations
- Looping over arrays.
- Reading files in numpy
- Use numpy vs list for matrix multiplication of 1000 X 1000 array and evaluate computing performance.

For Help:
https://www.dataquest.io/m/289-introduction-to-numpy/
https://cloudxlab.com/blog/numpy-pandas-introduction/

Pandas
- Creating data frame
- Reading files
- Slicing manipulations
- Exporting data to files
- Columns and row manipulations with loops
- Use pandas for masking data and reading if in Boolean format.

For Help:
https://www.hackerearth.com/practice/machine-learning/data-manipulation-visualisation-r-python/tutorial-data-manipulation-numpy-pandas-python/tutorial/

Matplotlib
- Importing matplotlib
- Simple line chart
- Correlation chart
- Histogram
- Plotting of Multivariate data
- Plot Pi Chart

For Help:
https://towardsdatascience.com/data-visualization-using-matplotlib-16f1aae5ce70

### Week 3
Linear Regression
The goal of this assignment is to study and implement simple linear regression using three different approaches:
- Ordinary Least Squares (OLS) method
- SKLearn library
- Gradient Descent
Select a real dataset from UCI machine  Learning Repository with one dependent variable and one independent variable to compare the results of each approach and respond to the following questions.
Discuss the full story of the dataset and discuss why regression is applicable on the dataset.
- Write a code to show 
- How many total observations in data?
- Data Distribution of independent and independent variables?
- Relationship between dependent and independent variables(Correlation analysis).
- Write a code to implement linear regression using the Ordinary Least Squares method on selected dataset. 
- Use sklearn API to create linear regression using selected dataset. Print intercept and slope of a model. 
- Write a code to implement linear regression using Gradient Descent from scratch on selected dataset. 
- Quantify the goodness of your model using a table to display the result of predictions using SSE, RMSE and R2Score and discuss interpretation of errors and  steps taken for improvement of errors.
- Prepare presentation for this work in group of 5
Refrences:
- Sklearn API: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
- Kaggle Notebook: https://www.kaggle.com/code/nargisbegum82/step-by-step-ml-linear-regression
- Complete Tutorial: https://realpython.com/linear-regression-in-python/
- API reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- Dataset Reference: https://archive.ics.uci.edu/datasets



### Week 4
Two Class Classification (Logistic Regression)
Select Dataset of your choice and respond to following questions.
- Why you want to apply classification on selected dataset? Discuss full story behind dataset.
- How many total observations in data?
- How many independent variables?
- Which is dependent variable?
- Which are most useful variable in classification? Prove using correlation.
- Implement logistic function.
- Implement Log-loss function.
- Implement Logistic regression from scratch.
- Implement Logistic regression using sklearn API.
- Quantify goodness of your model and discuss steps taken for improvement (Accuracy, Confusion matrices, F-measure).
- Discuss comparison of different methods.
- Prepare presentation for this work in group of 5

For Help:
1. https://medium.com/@anishsingh20/logistic-regression-in-python-423c8d32838b
2. https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
3. https://towardsdatascience.com/logistic-regression-python-7c451928efee
4. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
5. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### Week 5
Select Dataset of your choice with 3 or more classes and respond to following questions.
- Discuss full story of the dataset including 
   - Origin and revisions of dataset
   - How it was collected
   - Which are sources of data download 
   - Associated research papers etc..
- Explain size of dataset on disc and in memory including shape
- Which are dependent and independent variables and their distributions?
- Analyze dataset to check correlations between independent and dependent variable. Also explain how and why relevant variables are useful in estimating independent variable.
- Explain pseudo code/steps involved in using KNN.
- Implement KNN from pseudo code/steps.
- Implement KNN using sklearn api.
- Quantify goodness of your model and discuss steps taken for improvement.
- Can we use KNN for regression also? Why / Why not?
- Discuss drawbacks of algorithms such as KNN.
- Prepare presentation for this work in group of 5


For Help:
https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/

### Week 6
Comparative analysis of models using quantitative measures.
(F-measures, confusion Matrix, RMSE etc.).
https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/

### Week 7
Find a dataset with number of samples smaller than number of features. Apply principle component analysis to select K best features. Use Support Vector Machines/Naïve Bayes to train predictive model. Compare model accuracy and time required for training with full dataset and with selected K features. (use Sci-kit-learn library)
https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
https://www.dataquest.io/blog/sci-kit-learn-tutorial/

### Week 8
Perceptron algorithm for logic gates.
https://www.mldawn.com/train-a-perceptron-to-learn-the-and-gate-from-scratch-in-python/
Implement Convolutional neural network for hand written digits classification. Tune it and compare it with practical 8. Apply Convolutional neural network on image 

### Week 9
classification data of your choice and write all steps for hyper parameter optimization. (use Keras library)
https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

### Week 10
Use K-Means Clustering algorithm for clustering customer groups for optimizing product delivery.
https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
https://www.datacamp.com/community/tutorials/k-means-clustering-python

### Week 11
Make a presentation on any one application currently you see in the market. Discuss technical, pros and cons, before after, and ongoing development in the same applications.





