# Databricks notebook source
# MAGIC %pip install langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read the text file
df = spark.read.text("/Volumes/demo_one/default/csv_vol/dataset_website-content-crawler_2024-07-11_06-35-41-477.csv")

# Collect all the text into a single string
text_column = " ".join([row.value for row in df.collect()])

length_function = len

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=length_function
)
chunks = splitter.split_text(text_column)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType
import pandas as pd

@pandas_udf("array<string>")
def get_chunk(dummy):
    return pd.Series([chunks])

# Register the UDF
spark.udf.register("get_chunk_udf", get_chunk)

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into llm.rag.docs_text (text)
# MAGIC select explode(get_chunk_udf('dummy')) as text;
