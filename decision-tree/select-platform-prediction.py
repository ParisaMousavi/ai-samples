# Databricks notebook source
import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = '/Workspace/Repos/parisamoosavinezhad@hotmail.com/ai-samples/decision-tree/input/decision-tree-scenarios-samples.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()
