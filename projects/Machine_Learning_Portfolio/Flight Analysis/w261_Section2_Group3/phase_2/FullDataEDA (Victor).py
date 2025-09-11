# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from functools import reduce

# COMMAND ----------

blob_container = "lmorin" # The name of your container created in https://portal.azure.com
storage_account = "teamlrfv" # The name of your Storage account created in https://portal.azure.com
secret_scope = "lmorin" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "lmorinkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

merged_df2 = spark.read.parquet(f'{blob_url}/landon&rathin/final_full_merge6')

# COMMAND ----------

merged_df2.count()

# COMMAND ----------

display(merged_df2)

# COMMAND ----------

merged_df2.select(count('FL_DATE')).show()
len(merged_df2.columns)

# COMMAND ----------

# df schema
merged_df2.printSchema()

# COMMAND ----------

# show stats for all col
for col in merged_df2.columns:
   print(merged_df2.select(col).describe().toPandas())

# COMMAND ----------



# COMMAND ----------

