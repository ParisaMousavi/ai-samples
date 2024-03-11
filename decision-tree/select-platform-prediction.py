# Databricks notebook source
import pandas as pd

dt_file_path = '/Workspace/Repos/parisamoosavinezhad@hotmail.com/ai-samples/decision-tree/input/decision-tree-scenarios-samples.csv'

dt_data = pd.read_csv(dt_file_path) 

dt_data.describe()

# COMMAND ----------

dt_data = dt_data.dropna(axis=0)

y = dt_data.Platform

dt_data.columns

# COMMAND ----------

dt_features = ['FR_AS_SaaS', 'NFR_AS_SaaS', 'MS_365', 'Salesforce_Cloud', 'Other', 'Cloud_native', 'Hybrid', 'Azure_native_indepth', 'technology_specific', 'WebApp', 'ContainerApp', 'hybrid_multi_tenant', 'Has_Azure_Skill', 'Has_AWS_Skill', 'AWS_native_indepth', 'Aftersales_cloud', 'Need_Analytics']

X = dt_data[dt_features]

X.describe()

# COMMAND ----------

X.head()

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
dt_model = DecisionTreeRegressor(random_state=1)

# Fit model
dt_model.fit(X, y)

# COMMAND ----------

print("Making predictions for the following 5 first rows:")
print(X.head())
print("The predictions are")
print(dt_model.predict(X.head()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Validation (in-sample)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

predicted_platform = dt_model.predict(X)

mean_absolute_error(y, predicted_platform)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Validation

# COMMAND ----------


test_file_path = '/Workspace/Repos/parisamoosavinezhad@hotmail.com/ai-samples/decision-tree/input/test-sample.csv'

test_data = pd.read_csv(test_file_path) 

test_y = test_data.Platform

test_X = test_data[dt_features]

val_predictions = dt_model.predict(test_X)

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(test_y, val_predictions)

print(val_mae)

# COMMAND ----------

# MAGIC %md
# MAGIC # Underfitting and Overfitting

# COMMAND ----------

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [2, 4, 6, 8]:
    my_mae = get_mae(max_leaf_nodes, X, test_X, y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# COMMAND ----------

candidate_max_leaf_nodes = [2, 4, 6, 8, 12]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, X, test_X, y, test_y) for leaf_size in candidate_max_leaf_nodes}
print(scores)
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forests

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(X, y)
dt_preds = forest_model.predict(test_X)
print(mean_absolute_error(test_y, dt_preds))

# COMMAND ----------

output = pd.DataFrame({'Id': 1, 'Platform': dt_preds})
output.to_csv('submission.csv', index=False)
