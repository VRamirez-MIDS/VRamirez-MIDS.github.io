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
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/")
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
df_airport_codes = spark.read.table('airport_codes')


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

report = ['FM-16', 'FM-15']
df_weather = df_weather.filter(df_weather.REPORT_TYPE.isin(report))

# COMMAND ----------

df_weather = df_weather.withColumn('TimeLag', F.to_timestamp(df_weather.DATE + F.expr('INTERVAL 120 MINUTES'))).cache()


# COMMAND ----------

df_weather = df_weather.withColumn("TimeLagRounded", F.to_timestamp(df_weather['TimeLag'] + F.expr('INTERVAL 1 HOURS')).cast("timestamp"))\
                       .withColumn('TimeLagFormatted', regexp_replace(col("TimeLagRounded"),  "(\\d{2}):(\\d{2}):(\\d{2})", "$1:00:00").cast('timestamp')).drop('YEAR')

# COMMAND ----------

df_weather = df_weather.withColumn("LATITUDE", df_weather["LATITUDE"].cast('float'))\
                       .withColumn("LONGITUDE", df_weather["LONGITUDE"].cast('float'))\
                      .withColumn("ELEVATION", df_weather["ELEVATION"].cast('float'))\
                      .withColumn("HourlyAltimeterSetting", df_weather["HourlyAltimeterSetting"].cast('float'))\
                      .withColumn("HourlyDewPointTemperature", df_weather["HourlyDewPointTemperature"].cast('float'))\
                      .withColumn("HourlyDryBulbTemperature", df_weather["HourlyDryBulbTemperature"].cast('float'))\
                      .withColumn("HourlyPressureChange", df_weather["HourlyPressureChange"].cast('float'))\
                      .withColumn("HourlyRelativeHumidity", df_weather["HourlyRelativeHumidity"].cast('float'))\
                      .withColumn("HourlySeaLevelPressure", df_weather["HourlySeaLevelPressure"].cast('float'))\
                      .withColumn("HourlyVisibility", df_weather["HourlyVisibility"].cast('float'))\
                      .withColumn("HourlyWetBulbTemperature", df_weather["HourlyWetBulbTemperature"].cast('float'))\
                      .withColumn("HourlyWindDirection", df_weather["HourlyWindDirection"].cast('float'))\
                      .withColumn("HourlyWindSpeed", df_weather["HourlyWindSpeed"].cast('float'))


# COMMAND ----------

df_weather = df_weather.groupBy(['TimeLagFormatted', 'STATION']).agg(F.mean('LATITUDE'),
                                                     F.mean('LONGITUDE'),
                                                     F.mean('ELEVATION'),
                                                     F.mean('HourlyAltimeterSetting'),
                                                     F.mean('HourlyDewPointTemperature'),
                                                     F.mean('HourlyDryBulbTemperature'),
                                                     F.mean('HourlyPressureChange'),
                                                     F.mean('HourlyPressureTendency'),
                                                     F.mean('HourlyRelativeHumidity'),
                                                     F.mean('HourlySeaLevelPressure'),
                                                     F.mean('HourlyStationPressure'),
                                                     F.mean('HourlyVisibility'),
                                                     F.mean('HourlyWetBulbTemperature'),
                                                     F.mean('HourlyWindDirection'),
                                                     F.mean('HourlyWindSpeed'),
                                                     F.max('DATE'),
                                                     F.max('HourlySkyConditions'),
                                                     F.max('REM'),
                                                     F.max('SOURCE'),
                                                     F.max('NAME'))
                                                    

# COMMAND ----------

df_weather.write.parquet(f'{blob_url}/landon/weather_tojoinfull2')

# COMMAND ----------

df_weather = spark.read.parquet(f'{blob_url}/landon/weather_tojoinfull2')

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
                                                   .withColumn("DateTimeUTC", airport_tz(col('ORIGIN'), 'DateTimeLocal'))
df_airlines_date_adjust = df_airlines_date_adjust.withColumn("DepTimeRounded", F.to_timestamp(df_airlines_date_adjust['DateTimeUTC']).cast("timestamp"))\
                                                   .withColumn('DepTimeFormatted', regexp_replace(col("DepTimeRounded"),  "(\\d{2}):(\\d{2}):(\\d{2})", "$1:00:00").cast('timestamp')).drop('YEAR')

# COMMAND ----------

display(df_airlines_date_adjust)

# COMMAND ----------

df_airlines_date_adjust.write.parquet(f'{blob_url}/landon/df_airlines_stationsfull2')

# COMMAND ----------

df_airlines_date_adjust = spark.read.parquet(f'{blob_url}/landon/df_airlines_stationsfull2')

# COMMAND ----------

## Weather Join

# COMMAND ----------

# df_weather = df_weather.withColumn('TimeUpper', F.to_timestamp(df_weather.DATE + F.expr('INTERVAL 120 MINUTES')))\
#                        .withColumn('TimeLower', F.to_timestamp(df_weather.DATE + F.expr('INTERVAL 135 MINUTES')))



# df_weather.write.parquet(f'{blob_url}/landon/weather_datasetprejoin3')


# COMMAND ----------

df_airlines_date_adjust = df_airlines_date_adjust.withColumn('DepTimeFormattedv2', regexp_replace(col("DateTimeUTC"),  "(\\d{2}):(\\d{2}):(\\d{2})", "$1:00:00").cast('timestamp')).drop('YEAR')

# COMMAND ----------

merged_dataset = df_airlines_date_adjust.join(df_weather, (df_airlines_date_adjust.station_id == df_weather.STATION) & (df_airlines_date_adjust.DepTimeFormattedv2 == df_weather.TimeLagFormatted))



# COMMAND ----------

merged_dataset = merged_dataset.drop('DepTimeFormatted')

# COMMAND ----------

display(merged_dataset.select('DateTimeUTC', 'max(DATE)'))

# COMMAND ----------

merged_dataset.write.parquet(f'{blob_url}/landon&rathin/merged_dataset4_full3')

# COMMAND ----------

merged_df2 = spark.read.parquet(f'{blob_url}/landon&rathin/merged_dataset4_full3')

# COMMAND ----------

display(merged_df2.select('DateTimeUTC', 'max(DATE)'))

# COMMAND ----------

#get station ID for dest and join to weather on time. Destination weather process below.


# COMMAND ----------

df_airlines_dep_stations = df_airlines_date_adjust.select('DEST').join(df_closest_station_with_iata, df_airlines_date_adjust.DEST == df_closest_station_with_iata.iata_code, 'left')

# COMMAND ----------

df_airlines_dep_stations = df_airlines_dep_stations.select('station_id', 'DEST').withColumnRenamed('station_id', 'dest_station_id').withColumnRenamed('DEST', 'DEST2')

# COMMAND ----------

df_airlines_dep_stations.write.parquet(f'{blob_url}/landon&rathin/dest_stations2')

# COMMAND ----------

dest_df = spark.read.parquet(f'{blob_url}/landon&rathin/dest_stations2')

# COMMAND ----------

dest_df = dest_df.distinct()

# COMMAND ----------

merged_df3 = merged_df2.join(dest_df, dest_df.DEST2 == merged_df2.DEST, 'left')


# COMMAND ----------

merged_df3.write.parquet(f'{blob_url}/landon&rathin/df_merged14')
merged_df3 = spark.read.parquet(f'{blob_url}/landon&rathin/df_merged14')

# COMMAND ----------

merged_df2.distinct().count()

# COMMAND ----------

df_weather = df_weather.select([F.col(c).alias("DEST"+c) for c in df_weather.columns])

# COMMAND ----------

df_weather.write.parquet(f'{blob_url}/landon&rathin/dest_weather')

# COMMAND ----------

df_weather = spark.read.parquet(f'{blob_url}/landon&rathin/dest_weather')

# COMMAND ----------



# COMMAND ----------

merged_df3 = merged_df3.drop('ORIGIN_AIRPORT_SEQ_ID',
 'ORIGIN_CITY_MARKET_ID','ORIGIN_CITY_NAME',
 'ORIGIN_STATE_ABR',
 'ORIGIN_STATE_FIPS',
 'ORIGIN_STATE_NM',
 'ORIGIN_WAC','DEST_AIRPORT_SEQ_ID',
 'DEST_CITY_MARKET_ID', 'DEST_CITY_NAME',
 'DEST_STATE_ABR',
 'DEST_STATE_FIPS',
 'DEST_STATE_NM',
 'DEST_WAC','TAXI_OUT',
 'WHEELS_OFF',
 'WHEELS_ON',
 'TAXI_IN','DISTANCE_GROUP',
 'DIV_AIRPORT_LANDINGS','CRS_DEP_TIME_PAD','CRS_DEP_TIME', 'DEST').cache()

# COMMAND ----------

merged_df3.write.parquet(f'{blob_url}/landon&rathin/df_merged15')
merged_df3 = spark.read.parquet(f'{blob_url}/landon&rathin/df_merged15')

# COMMAND ----------

display(df_weather)

# COMMAND ----------

final_full_merge = merged_df3.join(df_weather, (merged_df3.dest_station_id == df_weather.DESTSTATION) & (merged_df3.DepTimeFormattedv2 == df_weather.DESTTimeLagFormatted), 'left')

# COMMAND ----------

display(final_full_merge)

# COMMAND ----------

final_full_merge.write.parquet(f'{blob_url}/landon&rathin/final_full_merge6')

# COMMAND ----------

display()