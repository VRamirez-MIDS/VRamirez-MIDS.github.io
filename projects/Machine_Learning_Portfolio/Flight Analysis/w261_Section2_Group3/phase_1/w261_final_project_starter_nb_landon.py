# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

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

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

from pyspark.sql.functions import count
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_airlines.select(count('DISTANCE')).show()
len(df_airlines.columns)

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_airlines = df_airlines.sample(fraction=0.15)

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
import pyspark.sql.functions as F
amount_missing_df = df_airlines.select([(count(when(isnan(c) | col(c).isNull(), c))/count(F.lit(1))).alias(c) for c in df_airlines.columns])
print(amount_missing_df.toPandas().transpose().sort_values(0, ascending=False))

# COMMAND ----------

sums = amount_missing_df.select(*[F.sum(c).alias(c) for c in amount_missing_df.columns]).first()

cols_to_drop = amount_missing_df.select([c for c in sums.asDict() if sums[c] >= 0.8])

# COMMAND ----------

cols_to_drop.toPandas()

# COMMAND ----------

len(cols_to_drop.columns)

# COMMAND ----------

df_airlines2 = df_airlines.drop(*cols_to_drop.columns)


# COMMAND ----------

display(df_airlines2)

# COMMAND ----------

from pyspark.sql.functions import col,isnan,when,count
import pyspark.sql.functions as F
amount_missing_df = df_airlines2.select([(count(when(isnan(c) | col(c).isNull(), c))/count(F.lit(1))).alias(c) for c in df_airlines2.columns])


# COMMAND ----------

# MAGIC %md

# COMMAND ----------

null_val = amount_missing_df.toPandas().transpose().sort_values(0, ascending=False).rename(columns = {0:'%Null'})
null_val

# COMMAND ----------

null_val[null_val['%Null']==0]

# COMMAND ----------

df_airlines2.write.parquet(f'{blob_url}/landon/eda')

# COMMAND ----------

display(df_airlines2.select('ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN'))

# COMMAND ----------

for col in df_airlines2.columns:
   print(df_airlines2.select(col).describe().toPandas())

# COMMAND ----------

display(df_airlines.select('DEP_DELAY_GROUP'))

# COMMAND ----------

display(df_airlines.select('DEP_TIME'))

# COMMAND ----------

display(df_airlines.select('DEP_TIME', 'DEP_DELAY'))

# COMMAND ----------

#median departure delay by carrier
display(df_airlines.select('OP_CARRIER', 'DEP_DELAY'))

# COMMAND ----------

display(df_airlines.select('OP_CARRIER'))

# COMMAND ----------

display(df_airlines.sample(fraction=0.2).select('DISTANCE', 'DEP_DELAY_NEW'))

# COMMAND ----------

display(df_airlines.sample(fraction=0.2).select('WEATHER_DELAY'))

# COMMAND ----------

display(df_airlines.select('DISTANCE', 'DEP_DELAY_GROUP'))

# COMMAND ----------

'OP_CARRIER', 'ORIGIN_AIRPORT_ID', 'MONTH', 'DAY'df_airlines2.sample(fraction = 0.1).toPandas().boxplot(by="DEP_DELAY_GROUP", column = ['OP_CARRIER', 'ORIGIN_AIRPORT_ID', 'MONTH', 'DAY'], vert=False, grid=True, figsize=(30,10), layout = (2,2))

# COMMAND ----------

display(df_airlines2)

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import numpy as np

corr = df_airlines2.sample(fraction=0.1).toPandas().select_dtypes(include=[float, int])
corr_spark = spark.createDataFrame(corr)
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=list(corr.columns), 
                            outputCol=vector_col)
myGraph_vector = assembler.transform(corr_spark).select(vector_col)
matrix = Correlation.corr(myGraph_vector, vector_col)

# COMMAND ----------

corr

# COMMAND ----------

df = df_airlines2.sample(fraction = 0.2).describe().toPandas().transpose()

# COMMAND ----------

df.columns = df.iloc[0]

# COMMAND ----------

df = df.drop(df.index[0])

# COMMAND ----------

df.dropna(inplace=True)

# COMMAND ----------

df[(~df.index.str.contains('_ID'))&(~df.index.str.contains('_NUM'))]

# COMMAND ----------



# COMMAND ----------

df_airlines2 = spark.read.format('parquet').option("inferSchema", "true").load(f'{blob_url}landon/eda')


# COMMAND ----------

df_airlines2 = df_airlines2.withColumn('CRS_DEP_TIME_padded', format_string("%04d", col('CRS_DEP_TIME').cast('int')))
df_airlines2.show()

# COMMAND ----------

df_stations6_test = df_stations6_test.withColumn("CRS_DEP_TIME_formatted", regexp_replace(col("CRS_DEP_TIME_padded"),  "(\\d{2})(\\d{2})", "$1:$2"))
df_stations6_test = df_stations6_test.withColumn("dt", concat(col("FL_DATE"), lit(" "), col("CRS_DEP_TIME_formatted")))
df_stations6_test = df_stations6_test.withColumn("dt_timestamp", to_timestamp(col("dt")))
df_stations6_test = df_stations6_test.withColumn("dt_timestamp_new", (F.unix_timestamp("dt_timestamp") + 3600 * col("gmt_offset")).cast('timestamp'))

# COMMAND ----------

df_weather2 = df_weather.withColumn("NAME_country", split(col("NAME"),", ")[1])
df_weather_US = df_weather2.filter(col('NAME_country')  == "US")
df_weather_US.count()

# COMMAND ----------

