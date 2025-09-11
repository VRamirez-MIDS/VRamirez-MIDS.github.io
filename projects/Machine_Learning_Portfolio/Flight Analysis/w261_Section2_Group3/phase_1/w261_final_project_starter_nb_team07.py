# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

from pyspark.sql.functions import col
print("Welcome to the W261 final project HEY") #hey everyone!

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

#try to view how many nulls are in each column for feature selection 
from pyspark.sql.functions import col,isnan,when,count
df_Columns= df_airlines
df_airlines.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]
   ).toPandas()

# COMMAND ----------

#try to view how many nulls are in each column for feature selection
# duplicate of next cell
from pyspark.sql.functions import col,isnan,when,count
df_Columns= df_airlines
df_airlines.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]
   ).toPandas()

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') > "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
display(df_stations)

# COMMAND ----------

#create join of 

# COMMAND ----------

for col in df_airlines.columns:
   print(df_airlines.select(col).describe().toPandas())

# COMMAND ----------

display(df_airlines.select('DEP_DELAY'))

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

display(df_airlines.sample(fraction=0.2).select('DISTANCE'))

# COMMAND ----------

display(df_airlines.sample(fraction=0.2).select('WEATHER_DELAY'))

# COMMAND ----------

df_weather.sample(fraction=0.2).describe().toPandas()

# COMMAND ----------

df_stations.sample(fraction=0.2).describe().toPandas()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
windowneighbor = Window.partitionBy("neighbor_call").orderBy(col("distance_to_neighbor").asc())
df_stations.withColumn("row",row_number().over(windowneighbor)) \
  .filter(col("row") == 1).drop("row") \
  .show()


# COMMAND ----------

