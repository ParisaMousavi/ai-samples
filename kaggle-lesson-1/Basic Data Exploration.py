# Databricks notebook source
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC The most important part of the Pandas library is the DataFrame. A DataFrame holds the type of data you might think of as a table. This is similar to a sheet in Excel, or a table in a SQL database.
# MAGIC
# MAGIC Pandas has powerful methods for most things you'll want to do with this type of data.
# MAGIC
# MAGIC The example (Melbourne) data is at the file path ../input/melbourne-housing-snapshot/melb_data.csv.
# MAGIC /Workspace/Repos/parisamoosavinezhad@hotmail.com/ai-samples/kaggle-lesson-1/input/metadata.json
# MAGIC
# MAGIC We load and explore the data with the following commands:

# COMMAND ----------

# save filepath to variable for easier access
melbourne_file_path = '/Workspace/Repos/parisamoosavinezhad@hotmail.com/ai-samples/kaggle-lesson-1/input/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # nterpreting Data Description
# MAGIC
# MAGIC The results show 8 numbers for each column in your original dataset. The first number, the count, shows how many rows have non-missing values.
# MAGIC
# MAGIC Missing values arise for many reasons. For example, the size of the 2nd bedroom wouldn't be collected when surveying a 1 bedroom house. We'll come back to the topic of missing data.
# MAGIC
# MAGIC The second value is the mean, which is the average. Under that, std is the standard deviation, which measures how numerically spread out the values are.
# MAGIC
# MAGIC To interpret the min, 25%, 50%, 75% and max values, imagine sorting each column from lowest to highest value. The first (smallest) value is the min. If you go a quarter way through the list, you'll find a number that is bigger than 25% of the values and smaller than 75% of the values. That is the 25% value (pronounced "25th percentile"). The 50th and 75th percentiles are defined analogously, and the max is the largest number.

# COMMAND ----------

# MAGIC %md
# MAGIC # Selecting Data for Modeling
# MAGIC
# MAGIC Your dataset had too many variables to wrap your head around, or even to print out nicely. How can you pare down this overwhelming amount of data to something you can understand?
# MAGIC
# MAGIC We'll start by picking a few variables using our intuition. Later courses will show you statistical techniques to automatically prioritize variables.
# MAGIC
# MAGIC To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).

# COMMAND ----------

melbourne_data.columns

# COMMAND ----------

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC There are many ways to select a subset of your data. The Pandas course covers these in more depth, but we will focus on two approaches for now.
# MAGIC
# MAGIC   1. Dot notation, which we use to select the "prediction target"
# MAGIC   2. Selecting with a column list, which we use to select the "features"
# MAGIC
# MAGIC # Selecting The Prediction Target
# MAGIC You can pull out a variable with **dot-notation**. This single column is stored in a **Series**, which is broadly like a DataFrame with only a single column of data.
# MAGIC
# MAGIC We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y. So the code we need to save the house prices in the Melbourne data is.

# COMMAND ----------

y = melbourne_data.Price

# COMMAND ----------

# MAGIC %md
# MAGIC # Choosing "Features"
# MAGIC
# MAGIC The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.
# MAGIC
# MAGIC For now, we'll build a model with only a few features. Later on you'll see how to iterate and compare models built with different features.
# MAGIC
# MAGIC We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).
# MAGIC
# MAGIC Here is an example:
# MAGIC

# COMMAND ----------

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# COMMAND ----------

# MAGIC %md
# MAGIC By convention, this data is called X.

# COMMAND ----------

X = melbourne_data[melbourne_features]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's quickly review the data we'll be using to predict house prices using the describe method and the head method, which shows the top few rows.
# MAGIC

# COMMAND ----------

X.describe()

# COMMAND ----------

X.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Visually checking your data with these commands is an important part of a data scientist's job. You'll frequently find surprises in the dataset that deserve further inspection.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Building Your Model¶
# MAGIC You will use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.
# MAGIC
# MAGIC The steps to building and using a model are:
# MAGIC
# MAGIC   * Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# MAGIC   * Fit: Capture patterns from provided data. This is the heart of modeling.
# MAGIC   * Predict: Just what it sounds like
# MAGIC   * Evaluate: Determine how accurate the model's predictions are.
# MAGIC
# MAGIC   Here is an example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.
# MAGIC
# MAGIC We now have a fitted model that we can use to make predictions.
# MAGIC
# MAGIC In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first few rows of the training data to see how the predict function works.
# MAGIC

# COMMAND ----------

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# COMMAND ----------

# MAGIC %md
# MAGIC # What is Model Validation¶
# MAGIC You'll want to evaluate almost every model you ever build. In most (though not all) applications, the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions be close to what actually happens.
# MAGIC
# MAGIC Many people make a huge mistake when measuring predictive accuracy. They make predictions with their training data and compare those predictions to the target values in the training data. You'll see the problem with this approach and how to solve it in a moment, but let's think about how we'd do this first.
# MAGIC
# MAGIC You'd first need to summarize the model quality into an understandable way. If you compare predicted and actual home values for 10,000 houses, you'll likely find mix of good and bad predictions. Looking through a list of 10,000 predicted and actual values would be pointless. We need to summarize this into a single metric.
# MAGIC
# MAGIC There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). Let's break down this metric starting with the last word, error.
# MAGIC
# MAGIC The prediction error for each house is:
# MAGIC
# MAGIC error=actual−predicted
# MAGIC So, if a house cost $150,000 and you predicted it would cost $100,000 the error is $50,000.
# MAGIC
# MAGIC With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. This is our measure of model quality. In plain English, it can be said as
# MAGIC
# MAGIC On average, our predictions are off by about X.
# MAGIC
# MAGIC To calculate MAE, we first need a model. That is built in a hidden cell below, which you can review by clicking the code button.
# MAGIC
# MAGIC Once we have a model, here is how we calculate the mean absolute error:
# MAGIC

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC # The Problem with "In-Sample" Scores
# MAGIC The measure we just computed can be called an "in-sample" score. We used a single "sample" of houses for both building the model and evaluating it. Here's why this is bad.
# MAGIC
# MAGIC Imagine that, in the large real estate market, door color is unrelated to home price.
# MAGIC
# MAGIC However, in the sample of data you used to build the model, all homes with green doors were very expensive. The model's job is to find patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with green doors.
# MAGIC
# MAGIC Since this pattern was derived from the training data, the model will appear accurate in the training data.
# MAGIC
# MAGIC But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.
# MAGIC
# MAGIC Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.
# MAGIC
# MAGIC # Coding It
# MAGIC The scikit-learn library has a function train_test_split to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate mean_absolute_error.
# MAGIC
# MAGIC Here is the code:

# COMMAND ----------

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

# Give it the argument random_state=1 so the check functions know what to expect when verifying your code.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC Wow!
# MAGIC Your mean absolute error for the in-sample data was about 500 dollars. Out-of-sample it is more than 250,000 dollars.
# MAGIC
# MAGIC This is the difference between a model that is almost exactly right, and one that is unusable for most practical purposes. As a point of reference, the average home value in the validation data is 1.1 million dollars. So the error in new data is about a quarter of the average home value.
# MAGIC
# MAGIC There are many ways to improve this model, such as experimenting to find better features or different model types.

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC # Code you have previously used to load data
# MAGIC import pandas as pd
# MAGIC from sklearn.tree import DecisionTreeRegressor
# MAGIC
# MAGIC # Path of the file to read
# MAGIC iowa_file_path = '../input/home-data-for-ml-course/train.csv'
# MAGIC
# MAGIC home_data = pd.read_csv(iowa_file_path)
# MAGIC y = home_data.SalePrice
# MAGIC feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# MAGIC X = home_data[feature_columns]
# MAGIC
# MAGIC # Specify Model
# MAGIC iowa_model = DecisionTreeRegressor()
# MAGIC # Fit Model
# MAGIC iowa_model.fit(X, y)
# MAGIC
# MAGIC print("First in-sample predictions:", iowa_model.predict(X.head()))
# MAGIC print("Actual target values for those homes:", y.head().tolist())
# MAGIC
# MAGIC # Set up code checking
# MAGIC from learntools.core import binder
# MAGIC binder.bind(globals())
# MAGIC from learntools.machine_learning.ex4 import *
# MAGIC print("Setup Complete")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate the Mean Absolute Error in Validation Data

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

# COMMAND ----------

# MAGIC %md
# MAGIC # Underfitting and Overfitting
# MAGIC
# MAGIC At the end of this step, you will understand the concepts of underfitting and overfitting, and you will be able to apply these ideas to make your models more accurate.
# MAGIC
# MAGIC # Experimenting With Different Models¶
# MAGIC Now that you have a reliable way to measure model accuracy, you can experiment with alternative models and see which gives the best predictions. But what alternatives do you have for models?
# MAGIC
# MAGIC You can see in scikit-learn's documentation that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the tree's depth. Recall from the first lesson in this course that a tree's depth is a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree
# MAGIC
# MAGIC [](./kaggle-lesson-1/img/house-ai-model.png)
# MAGIC
# MAGIC In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. As the tree gets deeper, the dataset gets sliced up into leaves with fewer houses. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have  210
# MAGIC   groups of houses by the time we get to the 10th level. That's 1024 leaves.
# MAGIC
# MAGIC When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
# MAGIC
# MAGIC This is a phenomenon called overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.
# MAGIC
# MAGIC At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.
# MAGIC
# MAGIC Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting. Visually, we want the low point of the (red) validation curve in the figure below.
# MAGIC
# MAGIC
