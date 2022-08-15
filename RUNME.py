# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if another user alters the workflow and cluster via the UI, running this script again after the modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "MFG"
        },
        "tasks": [
            {
                "job_cluster_key": "PDF_cluster",
                "notebook_task": {
                    "notebook_path": f"01_Introduction_And_Setup",
                    "base_parameters": {
                         "reset_all_data": "true"
                     }
                },
                "task_key": "PDF_01"
            },
            {
                "job_cluster_key": "PDF_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Fine_Grained_Demand_Forecasting"
                },
                "task_key": "PDF_02",
                "depends_on": [
                    {
                        "task_key": "PDF_01"
                    }
                ]
            },
            {
                "job_cluster_key": "PDF_cluster",
                "notebook_task": {
                    "notebook_path": f"03_Derive_Raw_Material_Demand"
                },
                "task_key": "PDF_03",
                "depends_on": [
                    {
                        "task_key": "PDF_02"
                    }
                ]
            },
            {
                "job_cluster_key": "PDF_cluster",
                "notebook_task": {
                    "notebook_path": f"04_Manage_Material_Shortages"
                },
                "task_key": "PDF_04",
                "depends_on": [
                    {
                        "task_key": "PDF_03"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "PDF_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


