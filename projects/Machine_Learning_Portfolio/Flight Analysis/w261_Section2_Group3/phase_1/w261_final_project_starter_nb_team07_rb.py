# Databricks notebook source
# MAGIC %md
# MAGIC # Data Join Notebook

# COMMAND ----------

## Package Import
from pyspark.sql.functions import col

# COMMAND ----------

# Load Data
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
df_airlines = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data/")
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') < "2015-04-01T00:00:00.000")
df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
df_airport_codes = spark.read.table('airport_codes')

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

display(df_weather)

# COMMAND ----------

display(df_stations)

# COMMAND ----------

display(df_airport_codes)

# COMMAND ----------

df_stations.select('station_id').distinct().count()

# COMMAND ----------

df_stations.select('neighbor_call').distinct().count()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
windowDept = Window.partitionBy('neighbor_call').orderBy(col('distance_to_neighbor').asc())
df_closest_station = df_stations.withColumn("row",row_number().over(windowDept)).filter(col("row") == 1).drop("row") 

# COMMAND ----------

display(df_closest_station)

# COMMAND ----------

df_closest_station_with_iata = df_closest_station.join(df_airport_codes, df_closest_station.neighbor_call == df_airport_codes.ident, 'left').select('station_id', 'neighbor_call', 'distance_to_neighbor', 'iata_code')

# COMMAND ----------

display(df_closest_station_with_iata)

# COMMAND ----------

df_closest_station_with_iata.filter(df_closest_station_with_iata.iata_code.isNotNull()).count()

# COMMAND ----------

df_airlines_with_stations = df_airlines.join(df_closest_station_with_iata, df_airlines.ORIGIN == df_closest_station_with_iata.iata_code, 'left')

# COMMAND ----------

display(df_airlines_with_stations)

# COMMAND ----------

df_airlines_with_stations.count()

# COMMAND ----------

df_airlines_with_stations.filter(df_airlines_with_stations.station_id.isNull()).count()

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

df_closest_station.write.parquet(f'{blob_url}/landon&rathin/joinprocess')

# COMMAND ----------

