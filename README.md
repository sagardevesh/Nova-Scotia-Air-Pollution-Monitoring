# Nova-Scotia-Air-Pollution-Monitoring

This project aims to build a decision tree model that classifies the PM2.5 content into 'Low' and 'High' bins based on input variables. 

Here is the link for [Nova Scotia Provincial Ambient Fine Particulate Matter Hourly Data](https://data.novascotia.ca/Environment-and-Energy/Nova-Scotia-Provincial-Ambient-Fine-Particulate-Ma/5hnc-kmpv/data) and [Traffic Volumes - Provincial Highway System](https://data.novascotia.ca/Roads-Driving-and-Transport/Traffic-Volumes-Provincial-Highway-System/8524-ec3n/data).

## TASK 1: Description of the motivation and the dataset, explaining each column and the labels

The motivation of the two datasets we used is to monitor and protect the outdoor air quality by keeping a track of the ‘ambient fine particulate matter (PM2.5)’ pollutant and the traffic volume in Nova Scotia. In the first dataset, the hourly Nova Scotia provincial ambient fine particulate matter is being monitored. The PM2.5 pollutant data is collected with the help of 2 instruments- BAM 1020 and API T640. On the other hand, traffic volume dataset provides the traffic information for different provincial highways.

There are 6 columns in the hourly ambient fine particulate matter (PM2.5) dataset, namely ‘Date & Time’, ‘Pollutant’, ‘Unit’, ‘Station’, ‘Instrument’, and ‘Average’. The ‘Date & Time’ column gives us the date and the hourly time at which the pollutant PM2.5 content was recorded. Therefore, each day has 24 records pertaining to PM2.5 being recorded 24 times. The datatype of the ‘Date & Time’ column is floating timestamp. The second column ‘Pollutant’ gives us name of the pollutant being monitored. This column has only one label which is PM2.5. The third column ‘Unit’ depicts the unit of the pollutant which is micrograms per cubic metre. The fourth column ‘Station’ provides information regarding the air quality station which is measuring the air quality. There are 2 stations measuring the air quality – Halifax station measured the air quality until 2017, however since 2018, Halifax Johnston has been measuring the air quality of the Nova Scotia province. The fifth column shows the instruments used for monitoring the ambient fine particulate matter – BAM 1020 and API T640. The last column ‘Average’ gives us the average of the PM2.5 content per hour.

In the traffic volume dataset, there are 16 columns in total. Each column provides information about different aspects related to traffic. For example, the ‘HIGHWAY’ feature gives information about the traffic volume of a particular highway, ‘DESCRIPTION’ gives information about the exact location that is being monitored. The column ‘ADT (Average Daily Traffic)’ provides information on the average number of vehicles passing that count location in a 24-hour period, whereas ‘AADT (Annual Average Daily Traffic)’ measures the average number of vehicles passing the count location in a 24-hour period, averaged on the basis of one year. The ‘GROUP’ column groups the count locations according to their seasonal variation patterns. The ‘COUNTY’ feature shows the county or the town in which the traffic is being monitored. ‘PTRUCK’ shows the percentage of trucks in that location for that given time. The column ‘Priority Points’ shows the signal analysis points based on 2014 TAC (Transportation Association of Canada). [10]

## TASK 2: Pre-processing, Normalization and Discretization

In the second task, I have pre-processed both the datasets. For the PM2.5 dataset, first I converted the values of ‘Date & Time’ to datetime format. This helped me to filter out all the data only for the year 2019. I built a continuous feature report which gave me an idea about the percentage of missing values for each feature along with other parameters. I dropped all the rows with any missing values to get rid of Nan values in my dataset

For the traffic volume dataset, I built a continuous feature report again, which showed that 'PRIORITY_POINTS', '85PCT', and 'PTRUCKS' had a lot of missing values. Similarly, the categorical feature ‘DIRECTION’ had a lot of missing values too. Therefore, I went ahead and dropped all of the above 4 features from the dataset. Next, I filtered to dataset to get data only for ‘HFX’ county, which reduced the number of records to 2044. Just like in PM2.5 dataset, I converted the ‘Date’ column to datetime format and sorted the dataset by date and set it as the index. Following the pre-processing of the 2 datasets, I proceeded to combine them both based on the date index. I used ‘Inner Join’ for this task.

Once I obtained the combined dataset, I performed normalization on the ‘Average’ column using the below formula:

![image](https://github.com/sagardevesh/Nova-Scotia-Air-Pollution-Monitoring/assets/25725480/a1feb12d-1c7e-4459-84c1-60db9b55e725)

Once I had the normalized values of the PM2.5 content data (Average), I discretized and created 2 bins – ‘Low’ and ‘High’, keeping 0.5 as the threshold. I created a new column ‘Bin’ and stored the discretized results in it. Following the above, I dropped the columns 'Average', 'SECTION DESCRIPTION', 'DESCRIPTION' and 'COUNTY' as they were no longer needed. 'SECTION DESCRIPTION' and 'DESCRIPTION' had string values, ‘COUNTY’ had only 1 label i.e., ‘HFX’, and ‘Average’ depicted the same information that the new column ‘Bin’ depicted.

The next task was to perform one hot encoding for the features ‘GROUP’ and ‘TYPE’ since they were classification features and contained strings, following which I separated the input variables and the target variable (‘Bin’) into 2 dataframes. The summary visualisation of the target variable ‘Bin’ showed the counts of the labels ‘Low’ and ‘High’ in the pre-processed dataset.

![image](https://github.com/sagardevesh/Nova-Scotia-Air-Pollution-Monitoring/assets/25725480/572f3617-44c7-4538-8757-5698a42ec60c)
Visualisation of the target variable

## Task 3: Feature Selection

For feature selection, I used 2 methods for the sake of comparison - Select K Best algorithm and Pearson correlation coefficient. Performing the Select K-Best algorithm gave 'SECTION_ID', 'HIGHWAY', 'GROUP', 'ADT', 'AADT' and 'SECTION LENGTH' as the top 6 features. On the other hand, the Pearson Correlation Coefficient method gave 'SECTION_ID', 'SECTION', 'SECTION LENGTH', 'GROUP', ‘TYPE’ and ‘ADT’ as the top 6 features. I went ahead with the features obtained from the Pearson Correlation Coefficient method for the forthcoming tasks.

Pearson Correlation Coefficient feature selection method helped in selecting only those features that are potentially more useful to the decision tree model for predicting the target variable ‘Bin’. The features that were eliminated would not provide substantial value to the model and would have only made the model predictions slower, hence they were eliminated.

In our case, the task is for the model to classify PM2.5 level into a ‘Low’ or a ‘High’ class based on the input variables provided. A decision tree is a more reasonable model in our case because it can handle both numerical and categorical features, as is the case with our resulting dataset. Also, a decision tree is a better model for exploratory analysis, as to understand the data relationship in a tree hierarchy structure. Plotting the tree gives us a very clear picture of the relationships between the variables.

### i. The most influential factor for PM2.5 level:

As we saw earlier, I performed 2 feature selection techniques – Select K Best and Pearson Correlation Coefficient method. The Select K Best algorithm gave us the order of the most influential features in predicting the PM2.5 level. Below is the result that we obtained:

{'SECTION ID': 5.448275069450755, 'HIGHWAY': 5.447461127850749, 'GROUP': 4.268685064623846, 'ADT': 2.0667938027990727, 'AADT': 1.9819256511998695, 'SECTION LENGTH': 1.3382980976909435, 'TYPE': 0.9902543144002058, 'SECTION': 0.3092450124885276}

As per the above, the feature ‘SECTION ID’ is the most influential feature according to Select K Best algorithm. This in other words implies that ‘SECTION ID’ will provide the maximum information gain.

### ii. Using Information Gain to select the best attribute to split on

For this task, let us consider the decision tree that we obtained for the 50-50 train test split. The tree can be seen in the next section. The model I trained has chosen ‘ADT’ as the root node. Here, I ‘ll justify why ‘ADT’ has been chosen as the best attribute to split on.

As per the decision tree, ‘ADT’ is split into 2 classes (<=13870 and >13870). Both the classes have equal number of instances i.e., 76, since the total number of instances we have is 152. Now, we will calculate the entropy for this root node. Entropy is given by:
• Entropy = -P(0) * log(P(0)) - P(1) * log(P(1))
Where P(0) and P(1) are the probability of the 2 classes.

For ‘ADT’, the probability of 2 classes is 76/152=0.5 each. Hence, the entropy will be as below:
Entropy = -0.5*log(0.5) -0.5*log(0.5) = 1

Now, information gain is the difference in the entropy of the parent node and the child nodes. Since the entropy of parent node in our case is 1, which signifies maximum impurity, the child nodes will definitely have an entropy value of 1 or less than 1 (Since 1 is the maximum value that entropy can take). So, there can’t a scenario where entropy of child node is more than that of the parent node in our case, and hence we can be sure that there will be a positive information gain. This implies that ‘ADT’ is the best attribute to make a split on since it has maximum entropy.

### iii. (a) Fitting a decision tree with the default parameters with 50-50 train test split:

For the first instance, I split the dataset into train and test data with 50-50 weightage. When I trained the model and fit it, I got an accuracy of 85.52%. Below is the plot of the decision tree obtained. It chose ‘ADT’ as the root node for splitting further. Looking closely, we realise that as we go down from the root node to the subsequent child nodes, the Gini impurity keeps on decreasing, and it finally becomes 0 when a node is completely pure, i.e., if we reach a leaf node.

![image](https://github.com/sagardevesh/Nova-Scotia-Air-Pollution-Monitoring/assets/25725480/610aa38c-83d7-4eb4-84d1-5137a97dc741)
Decision tree obtained with 50-50 train-test split

**(b)** In the next instance, I took a train-test split of 70-30, for 10-fold cross validation data. The classification report shows the statistics (including accuracy) of the model for each fold. In this instance, the model has chosen ‘SECTION_LENGTH’ as the root node to do further splits. Below is the plot of the tree that I obtained on the 10-fold cross validation data.

![image](https://github.com/sagardevesh/Nova-Scotia-Air-Pollution-Monitoring/assets/25725480/c7ccc1df-e41b-484f-9254-0c9057cb90f3)
Decision tree obtained for cross validation data

For the cross-validation data, I generated 10 classification reports and 10 confusion matrices for the 10 folds. Among the 10 folds, the highest accuracy for a model was 100%, whereas the lowest accuracy was 70%. The overall cross-predicted accuracy obtained was 82.23%.

**(c)** The model that I obtained is optimal, however there can still be some improvements. There are small leaves at the beginning of the tree, where the number of observations is 5 in one of the leaves after the first split. To address this, we can try to increase the number of observations in the root node. An overly small number of observations in the root node may result in the tree going too deep which would imply that the model is overfitting. Taking an extensively large number of observations in the root node may result in tree ending within 2 or 3 splits. In that case we would get a poor predictive performance. Overall, we need to find the right balance for the number of observations in the root node to get an optimum size of the tree, resulting in having not a lot of small sized leaves in the subsequent branches. [8]

### (d) Evaluation metrics:
For the 2 decision tree models, I have used accuracy as the metric to compare the 2 models. Even for the 10 models for 10-fold cross-validation, I have used model accuracy as the metric to gauge the performances of different models. Accuracy basically depicts the predictive accuracy rate of a decision tree model. In other words, it shows the correct classification rate. Since, trees are used for classification tasks (in our case, we are classifying data into ‘Low’ and ‘High’ bins), accuracy is the most appropriate evaluation metric to use.

### iv. Hyperparameter tuning: 

For hyperparameter tuning, I used randomized Search CV to get the best parameters. As per the results that I obtained, the model gives the best accuracy of 82.91% for parameters 'max_depth'=None 'min_samples_leaf'=1, 'min_samples_split'=6. 

Next, I experimented further by tweaking the parameters to see how it impacts the accuracy of the model. Firstly, I changed the max_depth to 2, which gave an accuracy of 84.21%. Increasing the max_depth to 3 increased the accuracy to 93.42%. On the flip side, increasing the min_samples_leaf parameter to 2 or 3 made no difference to the performance of the model. Furthermore, decreasing the min_samples_split increased the model accuracy. Whereas, if we increase the min_samples_split, model accuracy goes down a bit.

## Summary:

The aim of the project was to build a decision a decision tree model that best classifies the PM2.5 content into ‘Low’ and ‘High’ bins based on input variables. The performance of the model depends on various factors. As we saw, those factors may include the relevancy of input features that we choose, the hyperparameters of the model such as max_depth and number of observations in a leaf, and choosing best root node. We learned that the best combination of the above factors will give is the best results in terms of model performance, and this what we need to strive for during the model building process.

## REFERENCES: 

[1] Normalization, N., 2022. Normalize columns of a dataframe. [online] Stack Overflow. Available at: <https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe> .

[2] Google Developers. 2022. Normalization | Machine Learning | Google Developers. [online] Available at: <https://developers.google.com/machine-learning/data-prep/transform/normalization>


[3] Pandas.pydata.org. 2022. pandas.DataFrame.join — pandas 1.5.0 documentation. [online] Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html> . 

[4] GitHub. 2022. Complete-Feature-Selection/2-Feature Selection- Correlation.ipynb at main · teja0508/Complete-Feature-Selection. [online] Available at: https://github.com/teja0508/Complete-Feature-Selection/blob/main/2-Feature%20Selection-%20Correlation.ipynb 

[5] GitHub. 2022. Complete-Feature-Selection/2-Feature Selection- Correlation.ipynb at main · teja0508/Complete-Feature-Selection. [online] Available at: <https://github.com/teja0508/Complete-Feature-Selection/blob/main/2-Feature%20Selection-%20Correlation.ipynb> 

[6] values), C. and Dey, S., 2022. Classification report with Nested Cross Validation in SKlearn (Average/Individual values). [online] Stack Overflow. Available at: <https://stackoverflow.com/questions/42562146/classification-report-with-nested-cross-validation-in-sklearn-average-individua> [Accessed 8 October 2022]. 

[7] GeeksforGeeks. 2022. Countplot using seaborn in Python - GeeksforGeeks. [online] Available at: <https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/> 

[8] Data Science, Analytics and Big Data discussions. 2022. What is good in a decision tree, a large or a small leaf size?. [online] Available at: <https://discuss.analyticsvidhya.com/t/what-is-good-in-a-decision-tree-a-large-or-a-small-leaf-size/2108/2> 

[9] Data, P., Coder, N. and Kelechi, C., 2022. Pandas Calculate Daily & Monthly Average from Hourly Data. [online] Stack Overflow. Available at: <https://stackoverflow.com/questions/62397287/pandas-calculate-daily-monthly-average-from-hourly-data> 

[10] Novascotia.ca. 2022. [online] Available at: <https://novascotia.ca/tran/publications/Primary_Roads_-_Traffic_Volume_Data.pdf>

## Steps to run on local machine/jupyter notebook:

To run this assignment on your local machine, you need the following software installed.

*********************************************
Anaconda Environment

Python
*********************************************

To install Python, download it from the following link and install it on your local machine:

https://www.python.org/downloads/

To install Anaconda, download it from the following link and install it on your local machine:

https://www.anaconda.com/products/distribution

After installing the required software, clone the repo, and run the following command for opening the code in jupyter notebook.

jupyter notebook Nova_Scotia_Particulate_Matter.ipynb
This command will open up a browser window with jupyter notebook loading the project code.

You need to upload the dataset from the git repo to your jupyter notebook environment.





