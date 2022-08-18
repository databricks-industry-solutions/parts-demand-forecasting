# Databricks notebook source
dbutils.widgets.dropdown('reset_all_data', 'false', ['true', 'false'], 'Reset all data')
dbutils.widgets.text('dbName',  'demand_db' , 'Database Name')
dbutils.widgets.text('cloud_storage_path',  '/FileStore/tables/demand_forecasting_solution_accelerator/', 'Storage Path')

# COMMAND ----------

print("Starting ./_resources/01-data-generator")

# COMMAND ----------

cloud_storage_path = dbutils.widgets.get('cloud_storage_path')
dbName = dbutils.widgets.get('dbName')
reset_all_data = dbutils.widgets.get('reset_all_data') == 'true'

# COMMAND ----------

print(cloud_storage_path)
print(dbName)
print(reset_all_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Hierarchical Time Series Generator
# MAGIC This notebook-section simulates hierarchical time series data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate demand series data

# COMMAND ----------

#################################################
# Python Packages
#################################################
import pandas as pd
import numpy as np
import datetime

from dateutil.relativedelta import relativedelta
from dateutil import rrule

import os
import string
import random

from pyspark.sql.functions import pandas_udf, PandasUDFType, concat_ws
from pyspark.sql.types import StructType,StructField, StringType, DateType

# COMMAND ----------

#################################################
# Parameters
#################################################
n=10 # Number of SKU's per product
ts_length_in_years = 3 # Length of a time series in years
corona_breakpoint = datetime.date(year=2020, month=3, day=1) # date information: at which date do Corona effects come into play
percentage_decrease_corona_from = 20 # date information: define the decline after Corona comes into play
percentage_decrease_corona_to = 7 # date information: define the decline after Corona comes into play
trend_factor_before_corona = 100 # date information: trend before Corona

# COMMAND ----------

#################################################
# Create a Product Table
#################################################

data = [("Long Range Lidar",  "LRL"),
    ("Short Range Lidar", "SRL"),
    ("Camera", "CAM"),
    ("Long Range Radar", "LRR"),
    ("Short Range Radar", "SRR")
  ]

schema = StructType([ \
    StructField("Product",StringType(),True), \
    StructField("SKU_Prefix",StringType(),True)
  ])
 
product_identifier_lookup = spark.createDataFrame(data=data,schema=schema)

display(product_identifier_lookup)

# COMMAND ----------

#################################################
# Create a product hierarchy by simulating SKUs for each product
#################################################

# Define schema of output data-frame
product_hierarchy_schema = StructType([StructField("SKU_Postfix", StringType(), True)] + product_identifier_lookup.schema.fields)

# Help-function to generate a random string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# Create a Pandas UDF to simulate unique SKU's, i.e. n random strings without repetition
def id_sequence_generator(pdf):
  random.seed(123)
  res = set()
  while True:
    res.add(id_generator())
    if len(res) >= n:
      break
  
  pdf_out = pd.DataFrame()
  pdf_out["SKU_Postfix"] = list(res)
  pdf_out["Product"] = pdf["Product"].iloc[0]
  pdf_out["SKU_Prefix"] = pdf["SKU_Prefix"].iloc[0]
  
  return pdf_out

# Apply the Pandas UDF and clean up
product_hierarchy = ( \
  product_identifier_lookup \
  .groupby("SKU_Prefix", "Product") \
  .applyInPandas(id_sequence_generator, product_hierarchy_schema) \
  .withColumn("SKU", concat_ws('_',"SKU_Prefix","SKU_Postfix")) \
  .select("Product","SKU")
      )

# Check that the number of rows is what is expected
assert product_hierarchy.count() == (n * product_identifier_lookup.count()), "Number of rows in final table contradicts with input parameters"

display(product_hierarchy)

# COMMAND ----------

#################################################
# Create a Pandas DataFrame with common dates for ALL time series
#################################################
# End Date: Make it a Monday
end_date = datetime.datetime(2021, 7, 19)
end_date = end_date + datetime.timedelta(-end_date.weekday()) #Make sure to get the monday before

# Start date: Is a monday, since we will go back integer number of weeks
start_date = end_date + relativedelta(weeks= (-ts_length_in_years * 52))

# Make a sequence 
date_range = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))
date_range = [x.date() for x in date_range]

date_range = pd.DataFrame(date_range, columns =['Date'])


# Derive Corona Time Range, 0 for for pre-Corona and a count after corona
min_date = np.min(date_range.Date[ date_range.Date >=  corona_breakpoint ])
breakpoint = date_range.Date.searchsorted(min_date, side='left')
help_list = [0] * (breakpoint -1) + list(range(0, len(date_range) - breakpoint + 1))
assert len(help_list) == len(date_range), "Length of help_list is ambigious"
date_range = date_range.assign(Corona_Breakpoint_Helper = help_list)

# Derive Corona correction factor for demand
percentage_decrease = [  percentage_decrease_corona_from -  ( (percentage_decrease_corona_from - percentage_decrease_corona_to ) / max(help_list) ) * k  if k > 0 else 0 for k in  help_list]
factor = [ 1 if k == 0 else (100 - k) / 100 for k in percentage_decrease]
date_range = date_range.assign(Corona_Factor=factor)

# Derive X-mas Correction factor for demand
date_range = date_range.assign(Week = pd.DatetimeIndex(date_range['Date']).isocalendar().week.tolist())

conditions_xmas = [
      date_range.Week == 51,
      date_range.Week >= 52,
      date_range.Week == 1,
      date_range.Week == 2,
      date_range.Week == 3,
      date_range.Week == 4
    ]

choices_xmas = [
  0.85,
  0.8,
  1.1,
  1.15,
  1.1,
  1.05
]

date_range[ "Factor_XMas" ] = np.select(conditions_xmas, choices_xmas, default= 1.0)

display(date_range)

# COMMAND ----------

#################################################
# Enhance te product table with parameters for simulating time series
#################################################

# Get a list of all products from the hierarchy table and generate a list 
from  pyspark.sql.types import FloatType, ArrayType, IntegerType
from pyspark.sql.functions import monotonically_increasing_id


# Define schema for new columns
arma_schema = StructType(
  [
    StructField("Variance_RN", FloatType(), True),
    StructField("Offset_RN", FloatType(), True),
    StructField("AR_Pars_RN", ArrayType(FloatType()), True),
    StructField("MA_Pars_RN", ArrayType(FloatType()), True)
  ]
)

# Generate random numbers for the ARMA process
np.random.seed(123)
n_ = product_identifier_lookup.count()
variance_random_number = list(abs(np.random.normal(100, 50, n_)))
offset_random_number = list(np.maximum(abs(np.random.normal(10000, 5000, n_)), 4000))
ar_length_random_number = np.random.choice(list(range(1,4)), n_)
ar_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ar_length_random_number] 
ma_length_random_number = np.random.choice(list(range(1,4)), n_)
ma_parameters_random_number = [np.random.uniform(low=0.1, high=0.9, size=x) for x in ma_length_random_number] 


# Collect in a dataframe
pdf_helper = (pd.DataFrame(variance_random_number, columns =['Variance_RN']). 
              assign(Offset_RN = offset_random_number).
              assign(AR_Pars_RN = ar_parameters_random_number).
              assign(MA_Pars_RN = ma_parameters_random_number) 
             )

# Append column-wise
spark_df_helper = spark.createDataFrame(pdf_helper, schema=arma_schema)
spark_df_helper = spark_df_helper.withColumn("row_id", monotonically_increasing_id())
product_identifier_lookup = product_identifier_lookup.withColumn("row_id", monotonically_increasing_id())
product_identifier_lookup_extended = product_identifier_lookup.join(spark_df_helper, ("row_id")).drop("row_id")
product_identifier_lookup = product_identifier_lookup.drop("row_id")
product_hierarchy_extended = product_hierarchy.join(product_identifier_lookup_extended.drop("SKU_Prefix"), ["Product"], how = "inner")
assert product_identifier_lookup_extended.count() == product_identifier_lookup.count(), "Ambigious number of rows after join"

display(product_hierarchy_extended)

# COMMAND ----------

import statsmodels.api as sm
import matplotlib.pyplot as plt

from pyspark.sql.functions import col,when

from pyspark.sql.functions import row_number, sqrt, round

#################################################
# Generate an individual time series for each Product-SKU combination
#################################################

# function to generate an ARMA process
def generate_arma(arparams, maparams, var, offset, number_of_points, plot):
  np.random.seed(123)
  ar = np.r_[1, arparams] 
  ma = np.r_[1, maparams] 
  y = sm.tsa.arma_generate_sample(ar, ma, number_of_points, scale=var, burnin= 3000) + offset
  #y = np.round(y).astype(int)
  
  if plot:
    x = np.arange(1, len(y) +1)
    plt.plot(x, y, color ="red")
    plt.show()
    
  return(y)


#Schema for output dataframe
sku_ts_schema = StructType(  product_hierarchy.schema.fields + 
                    [
                      StructField("Date", DateType(), True),
                      StructField("Demand", FloatType(), True),
                      StructField("Corona_Factor", FloatType(), True),
                      StructField("Factor_XMas", FloatType(), True),
                      StructField("Corona_Breakpoint_Helper", FloatType(), True),
                      StructField("Row_Number", FloatType(), True)
                      
                      
                    ])

# Generate an ARMA
#pdf = product_hierarchy_extended.toPandas().head(1)

# Generate a time series with random parameters
# @pandas_udf(schema, PandasUDFType.GROUPED_MAP)

def time_series_generator_pandas_udf(pdf):
  out_df = date_range.assign(
   Demand = generate_arma(arparams = pdf.AR_Pars_RN.iloc[0], 
                          maparams= pdf.MA_Pars_RN.iloc[0], 
                          var = pdf.Variance_RN.iloc[0], 
                          offset = pdf.Offset_RN.iloc[0], 
                          number_of_points = date_range.shape[0], 
                          plot = False),
    Product = pdf.Product.iloc[0],
    SKU = pdf.SKU.iloc[0]
  )

  out_df = out_df[["Product", "SKU", "Date", "Demand", "Corona_Factor", "Factor_XMas", "Corona_Breakpoint_Helper"]]
  
  out_df["Row_Number"] = range(0,len(out_df))

  return(out_df)

# Apply the Pandas UDF and clean up
demand_df = ( 
  product_hierarchy_extended 
  .groupby("Product", "SKU") 
  .applyInPandas(time_series_generator_pandas_udf, sku_ts_schema) 
  .withColumn("Demand" , col("Demand") * col("Corona_Factor")) 
  .withColumn("Demand", when(col("Corona_Breakpoint_Helper") == 0,   
                             col("Demand") + trend_factor_before_corona * sqrt(col("Row_Number"))) 
                        .otherwise( col("Demand")))  
  .withColumn("Demand" , col("Demand") * col("Factor_XMas"))
  .withColumn("Demand" , round(col("Demand")))
  .select(col("Product"), col("SKU"), col("Date"), col("Demand") )
   )


display(demand_df)

# COMMAND ----------

# Plot individual series
res_table = demand_df.toPandas()
all_combis = res_table[[ "Product" , "SKU" ]].drop_duplicates()
random_series_to_plot = pd.merge(  res_table,   all_combis.iloc[[random.choice(list(range(len(all_combis))))]] ,  on =  [ "Product" , "SKU" ], how = "inner" )
selected_product = random_series_to_plot[ 'Product' ].iloc[0]
selected_sku = random_series_to_plot[ 'SKU' ].iloc[0]
random_series_to_plot = random_series_to_plot[["Date","Demand"]]

#Plot
plt.plot_date(random_series_to_plot.Date, random_series_to_plot.Demand, linestyle='solid')
plt.gcf().autofmt_xdate()
plt.title(f"Product: {selected_product}, SKU: {selected_sku}.")
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# COMMAND ----------

# Plot a sepecific time series
display(demand_df.join(demand_df.sample(False, 1 / demand_df.count(), seed=0).limit(1).select("SKU"), on=["SKU"], how="inner"))

# COMMAND ----------

demand_df_delta_path = os.path.join(cloud_storage_path, 'demand_df_delta')

# COMMAND ----------

# Write the data 
demand_df.write \
.mode("overwrite") \
.format("delta") \
.save(demand_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.part_level_demand")
spark.sql(f"CREATE TABLE {dbName}.part_level_demand USING DELTA LOCATION '{demand_df_delta_path}'")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dbName}.part_level_demand"))

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) as row_count FROM {dbName}.part_level_demand"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Simulate BoM Data
# MAGIC This notebook section simulates Bill-Of-Material data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate data

# COMMAND ----------

import string
import networkx as nx
import random
import numpy as np
import os

# COMMAND ----------

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_random_strings(n):
  random.seed(123)
  random_mat_numbers = set()
  while True:
    random_mat_numbers.add(id_generator(size=5))
    if len(random_mat_numbers) >= n:
      break
  return(random_mat_numbers)

# COMMAND ----------

def extend_one_step(node_from_):
  res_ = [  ]
  node_list_to_be_extended_ = [  ]
  # second level
  random_split_number = random.randint(2, 4)
  for i in range(random_split_number):
    node_to = random_mat_numbers.pop()
    node_list_to_be_extended_.append(node_to)
    res_.append((node_to, node_from_))
  return res_, node_list_to_be_extended_

# COMMAND ----------

def extend_one_level(node_list_to_be_extended, level, sku):
  
  
  print(f"""In  'extend_one_level': level={level} and sku = {sku}  """)
  
  if level == 1:
    head_node = random_mat_numbers.pop() 
    node_list_to_be_extended_one_level = [ ]
    node_list_to_be_extended_one_level.append(head_node)
    res_one_level = [ (head_node, sku) ]
  else:
    res_one_level = [ ]
    node_list_to_be_extended_one_level = [ ]
    
    if len(node_list_to_be_extended) > 2:
      node_list_to_be_extended_ = node_list_to_be_extended[ : 3 ]
    else:
      node_list_to_be_extended_ = node_list_to_be_extended

    for node in node_list_to_be_extended_:
      res_one_step = [ ]
      node_list_to_be_extended_one_step = [ ]
      
      res_one_step, node_list_to_be_extended_one_step = extend_one_step(node)    
      res_one_level.extend(res_one_step)
      node_list_to_be_extended_one_level.extend(node_list_to_be_extended_one_step)
  
  return res_one_level, node_list_to_be_extended_one_level

# COMMAND ----------

#Generate a set of material numbers
random_mat_numbers = generate_random_strings(1000000)

# COMMAND ----------

#Create a listof all SKU's
demand_df = spark.read.table(f"{dbName}.part_level_demand")
all_skus = demand_df.select('SKU').distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# Generaze edges
depth = 3
edge_list = [ ]

for sku in all_skus: 
  new_node_list = [ ]
  for level_ in range(1, (depth + 1)):
    new_edge_list, new_node_list = extend_one_level(new_node_list, level = level_, sku=sku)
    edge_list.extend(new_edge_list)

# COMMAND ----------

# Define the graph 
g=nx.DiGraph()
g.add_edges_from(edge_list)  

# COMMAND ----------

# Assign a quantity for the graph
edge_df = nx.to_pandas_edgelist(g)
edge_df = edge_df.assign(qty = np.where(edge_df.target.str.len() == 10, 1, np.random.randint(1,4, size=edge_df.shape[0])))

# COMMAND ----------

#Create the fnal mat number to sku mapper 
final_mat_number_to_sku_mapper = edge_df[edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')][["source","target"]]
final_mat_number_to_sku_mapper = final_mat_number_to_sku_mapper.rename(columns={"source": "final_mat_number", "target": "sku"} )

# COMMAND ----------

#Create the fnal mat number to sku mapper
final_mat_number_to_sku_mapper = edge_df[edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')][["source","target"]]
final_mat_number_to_sku_mapper = final_mat_number_to_sku_mapper.rename(columns={"source": "final_mat_number", "target": "sku"} )

# COMMAND ----------

# Create BoM
bom = edge_df[~edge_df.target.str.match('SRL|LRL|CAM|SRR|LRR_.*')]
bom = bom.rename(columns={"source": "material_in", "target": "material_out"} )

# COMMAND ----------

bom_df = spark.createDataFrame(bom) 
final_mat_number_to_sku_mapper_df = spark.createDataFrame(final_mat_number_to_sku_mapper)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register tables in database

# COMMAND ----------

bom_df_delta_path = os.path.join(cloud_storage_path, 'bom_df_delta')

# COMMAND ----------

# Write the data 
bom_df.write \
.mode("overwrite") \
.format("delta") \
.save(bom_df_delta_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.bom")
spark.sql(f"CREATE TABLE {dbName}.bom USING DELTA LOCATION '{bom_df_delta_path}'")

# COMMAND ----------

final_mat_number_to_sku_mapper_df_path = os.path.join(cloud_storage_path, 'sku_mapper_df_delta')

# COMMAND ----------

final_mat_number_to_sku_mapper_df.write \
.mode("overwrite") \
.format("delta") \
.save(final_mat_number_to_sku_mapper_df_path)

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {dbName}.sku_mapper")
spark.sql(f"CREATE TABLE {dbName}.sku_mapper USING DELTA LOCATION '{final_mat_number_to_sku_mapper_df_path}'")

# COMMAND ----------

display(spark.sql(f"select * from {dbName}.sku_mapper"))

# COMMAND ----------

display(spark.sql(f"select * from {dbName}.bom"))

# COMMAND ----------

print("Ending ./_resources/01-data-generator")

# COMMAND ----------


