# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Technical Setup notebook. Hide this cell results
# MAGIC Initialize dataset to the current user and cleanup data when reset_all_data is set to true
# MAGIC 
# MAGIC Do not edit

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text("db_prefix", "retail", "Database prefix")
dbutils.widgets.text("min_dbr_version", "9.1", "Min required DBR version")

# COMMAND ----------

from delta.tables import *
import pandas as pd
import logging
from pyspark.sql.functions import to_date, col, regexp_extract, rand, to_timestamp, initcap, sha1
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType, input_file_name
import re


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------

try:
  min_required_version = dbutils.widgets.get("min_dbr_version")
except:
  min_required_version = "9.1"

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "


#python Imports for ML...
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from sklearn.model_selection import GroupKFold
from pyspark.sql.functions import pandas_udf, PandasUDFType
import os
import pandas as pd
from hyperopt import space_eval
import numpy as np
from time import sleep


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

#force the experiment to the field demos one. Required to launch as a batch
def init_experiment_for_batch(demo_name, experiment_name):
  notebook_path = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()
  if notebook_path.startswith('/Repos/field-demos/field-demo-read-only-do-not-edit'):
    xp = f'/Repos/field-demos/field-demo-read-only-do-not-edit/{demo_name}/_experiments/Field Demos - {experiment_name}'
    print(f"Using common experiment under {xp}")
    mlflow.set_experiment(xp)

# COMMAND ----------

def get_cloud_name():
  return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()

# COMMAND ----------

mount_name = "field-demos"

try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  workspace_id = dbutils.entry_point.getDbutils().notebook().getContext().workspaceId().get()
  url = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
  if workspace_id == '8194341531897276':
    print("CSE2 bucket isn't mounted, mount the demo data under %s" % mount_name)
    dbutils.fs.mount(f"s3a://databricks-field-demos/" , f"/mnt/{mount_name}")
  elif "azure" in url:
    print("ADLS2 isn't mounted, mount the demo data under %s" % mount_name)
    configs = {"fs.azure.account.auth.type": "OAuth",
              "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
              "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope = "common-sp", key = "common-sa-sp-client-id"),
              "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope = "common-sp", key = "common-sa-sp-client-secret"),
              "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/9f37a392-f0ae-4280-9796-f1864a10effc/oauth2/token"}

    dbutils.fs.mount(
      source = "abfss://field-demos@fielddemosdatasets.dfs.core.windows.net/field-demos",
      mount_point = "/mnt/"+mount_name,
      extra_configs = configs)
  else:
    aws_bucket_name = ""
    print("bucket isn't mounted, mount the demo bucket under %s" % mount_name)
    dbutils.fs.mount(f"s3a://databricks-datasets-private/field-demos" , f"/mnt/{mount_name}")

# COMMAND ----------

spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", "10")
#spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

db_prefix = dbutils.widgets.get("db_prefix")

dbName = db_prefix+"_"+current_user_no_at
cloud_storage_path = f"/Users/{current_user}/field_demos/{db_prefix}"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)

spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

print("using cloud_storage_path {}".format(cloud_storage_path))

# COMMAND ----------

print(dbName)

# COMMAND ----------



# COMMAND ----------

def display_slide(slide_id, slide_number):
  displayHTML(f'''
  <div style="width:1150px; margin:auto">
  <iframe
    src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}"
    frameborder="0"
    width="1150"
    height="683"
  ></iframe></div>
  ''')

# COMMAND ----------

mlflow.set_experiment(f"/Users/{current_user}/part_demand_forecasting") 

# COMMAND ----------


