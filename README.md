# NYC_Taxi_TripDuration
Predicting NYC taxi trip duration using RF, XGboost and LightGBM (Kaggle competition) </br>
By Miriam Assraf, Elior Shriki, Aviv  Shabtay


## Topic:
New York City Taxi Trip Duration

## Competition link:
https://www.kaggle.com/c/nyc-taxi-trip-duration/overview

## The problem:
Improve the efficiency of electronic taxi dispatching systems it is important to be able to predict how long a driver will have his taxi occupied.  </br>
If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better able to identify which driver to assign to each pickup request. </br>

## Project objectives:
build a model that predicts the total ride duration of taxi trips in New York City.

## Dataset:
The competition dataset is based on the 2016 NYC Yellow Cab trip record data.  </br>
The data was originally published by the NYC Taxi and Limousine Commission (TLC).  </br>
The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes,  </br>
participants should predict the duration of each trip in the test set. </br>
 </br>
The dataset consists of taxi trips over 2016 and already split by kaggle to train with about 1.5 million records and test with about 600K records. </br>
Moreover, The dataset has 9 feature and 1 target as follow: </br>
- **id:** A unique ID given to each ride made by the passenger. </br>
- **vendor_id:** The ID of the vendor who owns the taxi. </br>
- **pickup_datetime:** The start date and the time of the ride. </br>
- **dropoff_datetime:** The end date and time of the ride. No test! </br>
- **passenger_count:** The number of passengers traveled in the ride. </br>
- **pickup_longitude and pickup_latitude:** This specifies the location of the passenger pickup.  </br>
 Latitude and longitude both help us measure the location of pickup. (These both have to be taken together in order to make sense). </br>
- **dropoff_longitude and dropoff_latitude:** This specifies the location of the passenger dropoff. Similar measures are used as the above one. </br>
- **store_and_fwd_flag:** This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle  </br>
 did not have a connection to the server - Y=store and forward; N=not a store and forward trip. </br>
- **trip_duration:** The total duration of the trip measures in seconds. Predict value </br>
Along our project we conduct a deep data analysis and feature enrichment such joining other datasets and apply algorithms like PCA or KMEANS  </br>
in order to get benefit insight on the data. </br>
 </br>
Also, We used plots such as scatter plots, histograms, box plots and interactive maps. </br>

## Technical issues:
During the work we were faced with many technical difficulties.  </br>
Due to the complex data engineering we performed, the runtime of the notebook was long so each time we wanted to test our model it took a long time until we  </br>
even got to run the model because the notebook was producing the data.  </br>
Therefore, we decided to build an absent data production notebook and a notebook for each of the 3 models we implemented that consume the datasets from Amazon S3 Storage. </br>

## Methodology:
#### Random Forest Regressor:
RF Regressor is an ensemble method for regression using multiple decision trees. </br>
The model prediction is made by the average of the predictions of all trees in the forest, which should reduce variance compared to a single decision tree. </br> 
This is because a single tree is sensitive to noise in the dataset, while the average is less sensitive. </br>
To ensure reduction of variance, the trees should be uncorrelated. RF uses bootstrap aggregation technique for building the trees - for each tree create a new dataset by choosing randomly n samples (as the size of the original dataset) with replacement.  </br>
This leads to less correlated trees by using different datasets for each tree.  </br>
Another method for reducing variance is Random Subspace Method- for each bootstrapped dataset consider only a subset of features,  </br>
this will assure more uncorrelated trees since there may be some features that are very strong predictors for the target variable,  </br>
and these features will be selected in many of the trees, causing them to become correlated. </br>
 </br>
Split method for Regression Trees is reduction in variance- for each split we calculate the average variance by three steps: </br>
1. We calculate the variance of each child node using the formula:  </br>
    where X is value of the feature, μis the mean and N is the number of samples in that node </br>
2. We calculate the variance of each split by the weighted average variance of child nodes </br>
3. We choose the split with minimum variance </br>
 </br>
For reducing more variance, we can use a subset of each tree instead of using a whole tree.  </br>
One way to do this is by building a whole tree and pruning it using Minimal Cost-Complexity method, which calculates the SSR of all the subtrees  </br>
of the tree and “punishes” the number of leaf in the tree using a hyper parameter α, called the complexity parameter.  </br>
The higher α is, the greater the punishment is and the less number of leaves are allowed. </br>
Because this way is time consuming, we chose to use the second way- early stopping.  </br>
In this way, we don’t get to build the whole tree and stop the splitting when reaching the stop criteria. </br>
For our model we chose a set of parameters: 
- **n_estimators :** 100 (number of GBDT in forest)
- **max_features :** sqrt (use a subset of features at size of sqrt num_features)
- **max_depth :** 4 (don’t allow grow a full tree)
- **min_sample_split :** 0.01 (don’t split if less than min_sample_split in leaf)
- **max_samples :** 0.6 (use max_samples for bootstrap)

#### XGboost and LighGBM:
They are both based on Gradient Boosting ensemble technique </br>
Gradient Boosting = Gradient Descent + Boosting </br>
GDBT method builds weak learners sequentially - first tree learns how to fit to the average target  </br>
(for all in regression), and the other trees learns how to fit the residuals of all previous ones. </br>
all those trees are trained by propagating the gradients of errors through. </br>
The main drawback is finding the best split points in each tree node, which is time and memory consuming. </br>

#### XGboost:
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. </br>
When it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. </br>
 </br>
XGBoost algorithm was developed as a research project at the University of Washington in 2016 and caught the Machine Learning world by fire. </br>
Since its introduction, this algorithm has not only been credited with winning numerous Kaggle competitions but also for being the driving force under the hood for several cutting-edge industry applications. </br>
As a result, there is a strong community of data scientists contributing to the XGBoost open source project. The algorithm differentiates itself in the following ways: </br>
- **A wide range of applications:** Can be used to solve regression, classification, ranking, and user-defined prediction problems.
- **Portability:** Runs smoothly on Windows, Linux, and OS X.
- **Languages:** Supports all major programming languages including C++, Python, R, Java, Scala, and Julia.
- **Cloud Integration:** Supports AWS, Azure, and Yarn clusters and works well with Flink, Spark, and other ecosystems.

XGboost splits up to the specified max_depth hyperparameter and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. </br>
It uses this approach since sometimes a split of no loss reduction may be followed by a split with loss reduction. XGBoost can also perform leaf-wise tree growth (as LightGBM). </br>
In XGBoost missing values will be allocated to the side that reduces the loss in each split. </br>
XGBoost have two similar methods: The first is “Gain” which is the improvement in accuracy (or total gain) brought by a feature to the branches it is on. </br>
The second method is “Frequency”/”Weight”. This method calculates the relative number of times a particular feature occurs in all splits of the model’s trees.  </br>
This method can be biased by categorical features with a large number of categories. </br>
 </br>
XGBOOST has built in CV and we used it to fit the model to the data with 500 number of boosting iteration and early stop (number of iteration without loss change) of 50. </br>
With 3-fold the model gain RMSLE score of 0.04660 on validation and 0.48575 in kaggle competition. </br>
 </br>
 
#### LightGBM:
Unlike other algorithms, like XGboost, LightGBM uses depth-wise tree growth - instead of splitting each leaf node and making a balanced tree,  </br>
the split is done on a single leaf node. That way less loss is reached faster but the chance of overfitting increases. </br>
The most consuming time task in GBDT is to find the best split- the algorithm loops over all features and for each feature loops over all of it’s values. </br>
LightGBM reduces the time complexity of the first loop by being a Histogram-based algorithm. Instead of sorting all values for each feature and checking the split score for each value,  </br>
it buckets continuous values into discrete bins and each bin is assigned a unique integer such that the ordinal relationship between the bins is preserved.  </br>
In this way, the algorithm checks the split score for each bin and reduces time complexity from O(#data) to O(#bins). </br>
It also reduced the time complexity of the second loop by using Exclusive Feature Building (EFB).  </br>
The main idea relies on the fact that High-dimensional data are usually very sparse and  many features are mutually exclusive, i.e., they never take nonzero values simultaneously.  </br>
EFB uses a greedy algorithm to bundle these mutually exclusive into a single feature and thus reduce the dimensionality, in a way that the original values of each feature can be extracted and exclusive values of features  </br>
in a bundle are put in different bins, which can be achieved by adding offsets to the original feature values.  </br>
The complexity of creating feature histograms is now proportional to the number of bundles instead of the number of features and reduces time complexity from O(#data x #features) to O(#data x #bundles). </br>
 </br>
To avoid overfitting we need to tune LightGBM hyperparameters: 
- **reg_alpha:** L1 regularization term on weights
- **reg_lambda:** L2 regularization term on weights
- **num_leaves:** large number of leaves increases accuracy to training set and increases the chance of over fitting, so we want to limit the number of leaves 
- **max_depth:** avoid growing deep trees
- **bagging_fraction:** use bagging on a percentage of the data and specify after how many trees re-subsampling by bagging_freq.  </br>
This improves generalization (by not using the same data for each tree) and reduces training time (by not using all the data). </br>
For tuning the hyperparameters we used RandomisezSearchCV over a set of parameters: </br>
params = { </br>
    "n_estimators" : [100, 200, 500, 1000], </br>
    "max_depth" : [6, 10, 14, 20, -1], </br>
    "bagging_fraction" : [0.4, 0.6, 1], </br>
    "bagging_freq" : [2,4,6,8], </br>
    "reg_alpha" : [0.0,0.6,0.8,1.0], </br>
    "reg_lambda" : [0.0,0.6,0.8,1.0], </br>
    "learning_rate" : [0.1, 0.01, 0.001] </br>
} </br>
 </br>
Fitting 10 models with 5-fold CV, totaling 50 models. </br>
We trained our model with the best estimator and it took 67.1948 seconds for 500 trees, which is much faster than XGboost with a very similar score for the test in kaggle. </br>

## Results:
#### RF:
- **Time to train:** 47.8614 minutes(only Fit)
- **Score in Kaggle(RMSLE):** 0.5823
- **RMSLE:** 0.4478
- **Num trees:** 100

#### XGboost:
- **Time to train:** 45 minutes (with CV)
- **Score in Kaggle(RMSLE):** 0.48172	
- **RMSLE:** 0.04660	
- **Num trees:** 500

#### LightGBM:
- **Time to train:** 47.8614 minutes(only Fit)
- **Score in Kaggle(RMSLE):** 0.49039	
- **RMSLE:** 0.04885
- **Num trees:** 500

