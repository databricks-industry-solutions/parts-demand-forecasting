# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

import os
import re
import mlflow
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", "10")
db_prefix = "demand_planning"
dbname = db_prefix
catalog = 'ENTERYOURCATALOG'

# COMMAND ----------

#spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
#dbutils.fs.rm(cloud_storage_path, True)

# COMMAND ----------

# Get dbName and cloud_storage_path, reset and create database
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

#dbName = db_prefix+"_"+current_user_no_at
#cloud_storage_path = f"/Users/{current_user}/field_demos/{db_prefix}"
volume_name = f"field_demos_{db_prefix}"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP schema IF EXISTS {catalog}.{dbname} CASCADE")
  #dbutils.fs.rm(cloud_storage_path, True)

# COMMAND ----------

spark.sql(f"""create schema if not exists {catalog}.{dbname}""")
spark.sql(f"""USE {catalog}.{dbname}""")
spark.sql(f"""create volume if not exists {volume_name}""")

# COMMAND ----------

volume_path = f"dbfs:/Volumes/{catalog}/{dbname}/{volume_name}"

# COMMAND ----------

print(volume_path)
print(dbname)

# COMMAND ----------

reset_all = dbutils.widgets.get('reset_all_data')
reset_all_bool = (reset_all == 'true')

# COMMAND ----------

path = volume_path

dirname = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
filename = "01-data-generator"
if (os.path.basename(dirname) != '_resources'):
  dirname = os.path.join(dirname,'_resources')
generate_data_notebook_path = os.path.join(dirname,filename)

def generate_data():
  dbutils.notebook.run(generate_data_notebook_path, 600, {"reset_all_data": reset_all, "dbname": dbname, "catalog": catalog})

if reset_all_bool:
  generate_data()
else:
  try:
    dbutils.fs.ls(path)
  except: 
    generate_data()

# COMMAND ----------

mlflow.set_experiment('/Users/{}/parts_demand_forecasting'.format(current_user))