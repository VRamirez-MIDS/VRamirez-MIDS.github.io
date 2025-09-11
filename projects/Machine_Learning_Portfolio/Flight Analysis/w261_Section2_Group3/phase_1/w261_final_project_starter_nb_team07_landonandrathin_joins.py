# Databricks notebook source
# MAGIC %md
# MAGIC # Data Join Notebook

# COMMAND ----------

!pip install airporttime
## Package Import
from pyspark.sql.functions import col
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from functools import reduce


from airporttime import AirportTime






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

# Load Data
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
df_airport_codes = spark.read.table('airport_codes')


# COMMAND ----------

# df_stations.select('station_id').distinct().count()

# COMMAND ----------

# df_stations.select('neighbor_call').distinct().count()

# COMMAND ----------

amount_missing_df = df_airlines.select([(count(when(isnan(c) | col(c).isNull(), c))/count(F.lit(1))).alias(c) for c in df_airlines.columns])
sums = amount_missing_df.select(*[F.sum(c).alias(c) for c in amount_missing_df.columns]).first()

cols_to_drop = amount_missing_df.select([c for c in sums.asDict() if sums[c] >= 0.8])
df_airlines = df_airlines.drop(*cols_to_drop.columns)


# COMMAND ----------

amount_missing_df = df_weather.select([(count(when(isnan(c) | col(c).isNull(), c))/count(F.lit(1))).alias(c) for c in df_weather.columns])
sums = amount_missing_df.select(*[F.sum(c).alias(c) for c in amount_missing_df.columns]).first()

cols_to_drop = amount_missing_df.select([c for c in sums.asDict() if sums[c] >= 0.8])
df_weather = df_weather.drop(*cols_to_drop.columns)


# COMMAND ----------

df_airlines = df_airlines.distinct()
df_weather = df_weather.distinct()
df_stations = df_stations.distinct()
df_airport_codes = df_airport_codes.distinct()

# COMMAND ----------

df_airlines.write.parquet(f'{blob_url}/landon/df_airlines1')
df_weather.write.parquet(f'{blob_url}/landon/df_weather1')

# COMMAND ----------

df_airlines = spark.read.parquet(f'{blob_url}/landon/df_airlines1')
df_weather = spark.read.parquet(f'{blob_url}/landon/df_weather1')

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
windowDept = Window.partitionBy('neighbor_call').orderBy(col('distance_to_neighbor').asc())
df_closest_station = df_stations.withColumn("row",row_number().over(windowDept)).filter(col("row") == 1).drop("row") 

# COMMAND ----------

df_closest_station_with_iata = df_closest_station.join(df_airport_codes, df_closest_station.neighbor_call == df_airport_codes.ident, 'left').select('station_id', 'neighbor_call', 'distance_to_neighbor', 'iata_code', 'neighbor_lat', 'neighbor_lon')

# COMMAND ----------

df_airlines_with_stations = df_airlines.join(df_closest_station_with_iata, df_airlines.ORIGIN == df_closest_station_with_iata.iata_code, 'left')

# COMMAND ----------

## Date Parse

# COMMAND ----------

@F.udf('timestamp')
def airport_tz(ORIGIN,datetime):
    apt=AirportTime(iata_code=ORIGIN)
    if datetime:
      utc_time=apt.to_utc(datetime)
      return utc_time
    else:
      return None
df_airlines_date_adjust = df_airlines_with_stations.withColumn('CRS_DEP_TIME_PAD', format_string("%04d", col('CRS_DEP_TIME').cast('int')))\
                                                   .withColumn("CRS_DEP_TIME_PAD", regexp_replace(col("CRS_DEP_TIME_PAD"),  "(\\d{2})(\\d{2})", "$1:$2"))\
                                                   .withColumn('FL_DATE', regexp_replace(col('FL_DATE'), " (\\d{2}):(\\d{2}):(\\d{2})", ''))\
                                                   .withColumn("DateTimeLocal", concat(col("FL_DATE"), lit(" "), col("CRS_DEP_TIME_PAD")))\
                                                   .withColumn("DateTimeLocal", to_timestamp(col("DateTimeLocal")))\
                                                   .withColumn("DateTimeUTC", airport_tz(col('ORIGIN'), 'DateTimeLocal')).cache()

# COMMAND ----------

df_airlines_date_adjust.write.parquet(f'{blob_url}/landon/df_airlines_stations3')

# COMMAND ----------

df_airlines_date_adjust = spark.read.parquet(f'{blob_url}/landon/df_airlines_stations3')

# COMMAND ----------

unique_stations = list(df_airlines_date_adjust.toPandas()['station_id'].unique())
unique_days = list(df_airlines_date_adjust.toPandas()['FL_DATE'].unique())

# COMMAND ----------

df_weather = df_weather.withColumn("Day", split(col("DATE"),"T")[0])
df_weather = df_weather.filter(df_weather.STATION.isin(unique_stations)).filter(df_weather.Day.isin(unique_days))

# COMMAND ----------

## Weather Join

# COMMAND ----------

df_weather = df_weather.withColumn('TimeUpper', F.to_timestamp(df_weather.DATE + F.expr('INTERVAL 120 MINUTES')))\
                       .withColumn('TimeLower', F.to_timestamp(df_weather.DATE + F.expr('INTERVAL 135 MINUTES')))



df_weather.write.parquet(f'{blob_url}/landon/weather_datasetprejoin3')


# COMMAND ----------

df_weather = spark.read.parquet(f'{blob_url}/landon/weather_datasetprejoin3')

# COMMAND ----------

display(df_weather)

# COMMAND ----------

merged_dataset = df_airlines_date_adjust.join(df_weather, (df_airlines_date_adjust.station_id == df_weather.STATION) & (df_airlines_date_adjust.FL_DATE == df_weather.Day))



# COMMAND ----------

display(merged_dataset)

# COMMAND ----------

df_weather.registerTempTable("weather")
df_airlines_date_adjust.registerTempTable("airlines")
results = sqlContext.sql("SELECT * FROM weather INNER JOIN airlines ON weather.TimeLower < airlines.DateTimeUTC and  airlines.DateTimeUTC < weather.TimeUpper")
display(results)

# COMMAND ----------



# COMMAND ----------

merged_dataset.write.parquet(f'{blob_url}/landon&rathin/merged_dataset4_3m')

# COMMAND ----------

merged_df2 = spark.read.parquet(f'{blob_url}/landon&rathin/merged_dataset4_3m')

# COMMAND ----------

