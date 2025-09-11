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

# MAGIC %md # Victor Station Dataset Phase 3 EDA

# COMMAND ----------

df_weather1 = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') < "2015-04-01T00:00:00.000")
# display(df_weather1)

df_airlines1 = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
# display(df_airlines1)

# Load all the stations data dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
# display(df_stations)


# COMMAND ----------

df_stations.select(count('usaf')).show()
len(df_stations.columns)

df_weather1.select(count('STATION')).show()
len(df_weather1.columns)

df_airlines1.select(count('FL_DATE')).show()
len(df_airlines1.columns)

# COMMAND ----------

df_weather2 = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
# display(df_weather1)

df_airlines2 = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
# display(df_airlines1)

# COMMAND ----------

df_weather2.select(count('STATION')).show()
len(df_weather2.columns)

df_airlines2.select(count('FL_DATE')).show()
len(df_airlines2.columns)

# COMMAND ----------

# df type
type(df_stations)

# df schema
df_stations.printSchema()

# show 4 rows
df_stations.show(4)

# show top 5 rows
df_stations.head(5)

# show stats for all col
for col in df_stations.columns:
   print(df_stations.select(col).describe().toPandas())



# COMMAND ----------

# df schema
df_weather.printSchema()

# COMMAND ----------

display(df_stations.select('usaf', 'station_id', 'neighbor_id'))

# COMMAND ----------


#try to view how many nulls are in each column for feature selection
# duplicate of next cell
from pyspark.sql.functions import col,isnan,when,count
df_Columns= df_stations
df_stations.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_Columns.columns]
   ).toPandas()


# COMMAND ----------

display(df_stations.select('neighbor_state'))

# COMMAND ----------

display(df_stations.select('neighbor_call'))


# COMMAND ----------

# display(df_stations.select('neighbor_call').distinct().count())

# distinct_ids = [x.id for x in df_stations.select('station_id').distinct().collect()]

distinct_ids = df_stations.select("neighbor_call").distinct()

display(distinct_ids)


airport_ids = sqlContext.sql("""

SELECT neighbor_call AS ids,

FROM df_stations

ORDER BY day""").toPandas()



# COMMAND ----------


display(df_stations.select('neighbor_name'))


# COMMAND ----------

# get the lat and long of all the neighbor stations and map viz of stations 
display(df_stations.select('neighbor_lat', 'neighbor_lon'))


# COMMAND ----------

display(df_stations.select('distance_to_neighbor'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Victor Final Project EDA 
# MAGIC ## EDA Findings
# MAGIC 
# MAGIC Project stations dataset
# MAGIC 
# MAGIC Explored the stations datasets. Identified the following schema:
# MAGIC 
# MAGIC   - usaf: string (nullable = true)
# MAGIC   
# MAGIC   - wban: string (nullable = true)
# MAGIC   
# MAGIC   - station_id: string (nullable = true)
# MAGIC   
# MAGIC   - lat: double (nullable = true)
# MAGIC   
# MAGIC   - lon: double (nullable = true)
# MAGIC   
# MAGIC   - neighbor_id: string (nullable = true)
# MAGIC   
# MAGIC   - neighbor_name: string (nullable = true)
# MAGIC   
# MAGIC   - neighbor_state: string (nullable = true)
# MAGIC   
# MAGIC   - neighbor_call: string (nullable = true)
# MAGIC   
# MAGIC   - neighbor_lat: double (nullable = true)
# MAGIC   
# MAGIC   - neighbor_lon: double (nullable = true)
# MAGIC   
# MAGIC   - distance_to_neighbor: double (nullable = true)
# MAGIC   
# MAGIC **usaf** = A character string identifying the fixed weather station from the USAF Master Station Catalog. USAF is an acronym for United States Air Force.
# MAGIC 
# MAGIC **wban** = A character string for the fixed weather station NCDC WBAN identifier. NCDC is an acronym for National Climatic Data Center. WBAN is an acronym for Weather Bureau, Air Force and Navy.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## EDA Null Findings
# MAGIC 
# MAGIC # stations 
# MAGIC None of the feature data contains any missing or NULL values
# MAGIC | Feature | Missing / NULLS | 
# MAGIC | ------ | ------ |
# MAGIC | usaf | None |
# MAGIC | wban | None |
# MAGIC | station_id  | None |
# MAGIC | lat | None |
# MAGIC | long | None |
# MAGIC | neighbor_id  | None |
# MAGIC | neighbor_name  | None |
# MAGIC | neighbor_state  | None |
# MAGIC | neighbor_call  | None |
# MAGIC | neighbor_lat  | None |
# MAGIC | neighbor_lon  | None |
# MAGIC | distance_to_neighbor | None |
# MAGIC 
# MAGIC ### neighbor_state: Max: 167
# MAGIC ### neighbor_state: Min: 3
# MAGIC ### station: neighbor_call Max: 2
# MAGIC ###  station: neighbor_call Min: 1
# MAGIC 
# MAGIC # distance_to_neighbor
# MAGIC | missing | mean | std dev | min | mean | max |
# MAGIC | ------ | ------ | ------ | ------ | ------ | ------ | 
# MAGIC | 0% | 1,343.52 | 948.85 | 0 | 1078.12 | 6,435.97 | 