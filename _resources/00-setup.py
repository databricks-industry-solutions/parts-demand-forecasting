# Databricks notebook source
import os

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

print("Starting ./_resources/00-setup")

# COMMAND ----------

#If True, all output files are in user specific databases, If False, a global database for the report is used
user_based_data = True

# COMMAND ----------

# MAGIC %run ../_resources_outside/00-global-setup $reset_all_data=$reset_all_data $db_prefix=demand_level_forecasting

# COMMAND ----------

if (not user_based_data):
  cloud_storage_path = '/FileStore/tables/demand_forecasting_solution_accelerator/'
  dbName = 'demand_db' 

# COMMAND ----------

reset_all = dbutils.widgets.get('reset_all_data')
reset_all_bool = (reset_all == 'true')

# COMMAND ----------

path = cloud_storage_path

dirname = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
filename = "01-data-generator"
if (os.path.basename(dirname) != '_resources'):
  dirname = os.path.join(dirname,'_resources')
generate_data_notebook_path = os.path.join(dirname,filename)

def generate_data():
  dbutils.notebook.run(generate_data_notebook_path, 600, {"reset_all_data": reset_all, "dbName": dbName, "cloud_storage_path": cloud_storage_path})

if reset_all_bool:
  generate_data()
else:
  try:
    dbutils.fs.ls(path)
  except: 
    generate_data()

# COMMAND ----------

print("Ending ./_resources/00-setup")
