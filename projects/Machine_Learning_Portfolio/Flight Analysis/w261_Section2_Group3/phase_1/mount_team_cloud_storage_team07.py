# Databricks notebook source
# MAGIC %md
# MAGIC # How to Mount Your Team's Cloud Storage

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Databricks CLI
# MAGIC 
# MAGIC **Note:** All Databricks CLI commands shhould be run on your local computer, not in the cluster.
# MAGIC 
# MAGIC 1. Install the Databricks CLI by running this command:
# MAGIC `python3 -m pip install databricks-cli`
# MAGIC 2. Go to the top right corner of this UI and click on the box **databricks_poc_clus...**, click on **User Settings**, finally click on **Generate New Token**. You will only have one chance to copy the token to a safe place.
# MAGIC 3. Run the following command to configure the **CLI**:
# MAGIC `databricks configure --token`
# MAGIC 4. Provide this url when prompted with Databricks Host: `https://adb-731998097721284.4.azuredatabricks.net`
# MAGIC 5. Paste the Token when prompted.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Blob Storage
# MAGIC 
# MAGIC **Special Note:** Creating a Storage account, only needs to be performed by one member of the team. The token needs to be shared among the rest of the members via a Secrets ACL. Please be responsible.
# MAGIC 
# MAGIC ### Create Storage Account
# MAGIC 1. Navigate to https://portal.azure.com
# MAGIC 2. Login using Calnet credentials *myuser@berkeley.edu*
# MAGIC 3. Click on the top right corner on the User Icon.
# MAGIC 4. Click on Switch directory. Make sure you switch to **UC Berkeley berkeley.onmicrosoft.com**, this would be your personal space.
# MAGIC 5. Click on the Menu Icon on the top left corner, navigate to **Storage accounts**.
# MAGIC 6. Choose the option **Azure for Students** to take advantage of $100 in credits. Provide you *berkeley.edu* email and follow the prompts.
# MAGIC 7. Once the subscription is in place, navigate back to Storage accounts, refresh if needed. Hit the button **+ Create** in the top menu.
# MAGIC   - Choose **Azure for Students** as Subscription.
# MAGIC   - Create a new Resource group. Name is irrelevant here.
# MAGIC   - Choose a **Storage account name**, you will need this in the *Init Script* below. (e.g., jshanahan)
# MAGIC   - Go with the defaults for the rest of the form.
# MAGIC   - Hit the **Review + create** button.
# MAGIC 8. Once the **Storage account** is shown in your list:
# MAGIC   - Click on it. This will open a sub-window.
# MAGIC   - Under *Data Storage*, click on **container**.
# MAGIC   - Hit the **+ Container** in the top menu.
# MAGIC   - Choose a name for your container, you might need this if you choose a SAS token in the *Init Script* below.
# MAGIC   
# MAGIC **Note:** Create your Blob Storage in the US West 2 Region.
# MAGIC 
# MAGIC ### Obtain Credentials
# MAGIC 
# MAGIC First, you need to choose between using an Access Key or a SAS tokens. Bottom line, SAS tokens would be recommended since it's a token in which you have control on permissions and TTL (Time to Live). On the other hand, an Access Key, would grant full access to the Storage Account and will generate SAS tokens in the backend when these expire.
# MAGIC 
# MAGIC To obtain the **Access Key**:
# MAGIC 1. Navigate back to *Storage accounts**.
# MAGIC 2. Click on the recently created account name.
# MAGIC 3. In the sub-window, under *Security + networking*, click on **Access Keys**.
# MAGIC 4. Hit the **Show keys** button.
# MAGIC 5. Copy the **Key**, you don't need the Connection string. It's irrelevant if you choose *key1* or *key2*.
# MAGIC 
# MAGIC To obtain a **SAS Token**:
# MAGIC 1. Navigate to the containers list.
# MAGIC 2. At the far right, click on the `...` for the container you just created.
# MAGIC 3. Check the boxes of the permissions you want.
# MAGIC 4. Select an expiration you are comfortable with.
# MAGIC 5. Hit the **Generate SAS token and URL** button.
# MAGIC 6. Scroll down and copy only the **Blob SAS token**.
# MAGIC 
# MAGIC **Note:** SAS stands for *Shared access signature*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Credentials as Databricks Secret
# MAGIC 
# MAGIC **Special Note:** Only the member that created the Storage account should perform this step.
# MAGIC 
# MAGIC 1. Create a scope:
# MAGIC `databricks secrets create-scope --scope <choose-any-name>`
# MAGIC 2. Load the key/token:
# MAGIC `databricks secrets put --scope <name-from-above> --key <choose-any-name> --string-value '<paste-token-here>'`
# MAGIC 3. Add a principal to the Secret Scope ACL to share token with your teammates. **Careful:** make sure you type the right Team number.
# MAGIC `databricks secrets put-acl --scope <name-from-above> --principal team<your-team-number> --permission READ`
# MAGIC 
# MAGIC **Note:** This has been tested only on Mac/Linux. It might be different in Windows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Init Script 

# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "lmorin" # The name of your container created in https://portal.azure.com
storage_account = "teamlrfv" # The name of your Storage account created in https://portal.azure.com
secret_scope = "lmorin" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "lmorinkey" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# MAGIC %md
# MAGIC Run one of the next two cells.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Access Key

# COMMAND ----------

# spark.conf.set(
#   f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
#   dbutils.secrets.get(scope = secret_scope, key = secret_key)
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ### SAS Token

# COMMAND ----------

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test it!
# MAGIC A *Read Only* mount has been made available to all clusters in this Databricks Platform. It contains data you will use for **HW5** and **Final Project**. Feel free to explore the files by running the cell below.

# COMMAND ----------

display(dbutils.fs.ls(f"{mount_path}/datasets_final_project_2022"))

# COMMAND ----------

df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")# Load the Jan 1st, 2015 for Weather
df_weather = spark.read.parquet(f"{mount_path}/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-01-02T00:00:00.000").cache()
display(df_weather)

# COMMAND ----------

# This command will write to your Cloud Storage if right permissions are in place. 
# Navigate back to your Storage account in https://portal.azure.com, to inspect the files.
df_weather.write.parquet(f"{blob_url}/test")

# COMMAND ----------

# Load it the previous DF as a new DF
df_weather_new = spark.read.parquet(f"{blob_url}/weather_data_1d")
display(df_weather_new)

# COMMAND ----------

print(f"Your new df_weather has {df_weather_new.count():,} rows.")
print(f'Max date: {df_weather_new.select([max("DATE")]).collect()[0]["max(DATE)"].strftime("%Y-%m-%d %H:%M:%S")}')

# COMMAND ----------

display(dbutils.fs.ls(f"{mount_path}/HW5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using RDD API
# MAGIC 
# MAGIC When reading/writing using the RDD API, configuration cannot happen at runtime but at cluster creation.
# MAGIC Ping Dante Malagrino with the following information to be added in your Cluster as Spark Configuration:
# MAGIC - Storage Account name
# MAGIC - Container name
# MAGIC - Secret Scope name
# MAGIC - Secret Key name
# MAGIC 
# MAGIC **Important:** Do not share the actual SAS token.
# MAGIC 
# MAGIC After this is added as Spark Configuration, try the scripts provided below to test the Hadoop plug-in to connect to your Azure Blob Storage.
# MAGIC ```
# MAGIC spark.hadoop.fs.azure.sas.<container_name>.<storage_account>.blob.core.windows.net {{secrets/<scope>/<key>}}
# MAGIC ```

# COMMAND ----------

rdd = sc.textFile('/mnt/mids-w261/HW5/test_graph.txt')

parsed_rdd = rdd.map(lambda line: tuple(line.split('\t')))
parsed_rdd.take(3)

# COMMAND ----------

parsed_rdd.saveAsTextFile(f"{blob_url}/graph_test")

# COMMAND ----------

