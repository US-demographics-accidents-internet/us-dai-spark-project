from pyspark.sql import SparkSession
from extract import load_dataframes, merge_equal_dataframes
from transformation.internet_demographic import InternetDemographicTransformation
from transform.accidents import transform_accidents_df
from transform.demographics import transform_demographics_df 
from transform.accidents_demographics import AccidentsDemographicsTransformation
from transform_pipeline import transform_stage

if __name__ == "__main__":
    # SparkSession initialization
    spark = SparkSession.builder.appName("ExtractStage").getOrCreate()

    # Path to the Data directory (inside the container)
    base_path = "/app/Data"

    # Loading all CSVs into a DataFrame
    print("=== STAGE 1: EXTRACT ===")
    dfs = load_dataframes(spark, base_path)

    print("\nRows number in each dataset:")
    for name, df in dfs.items():
        print(f"{name}: {df.count()}")

    # Transformational stage
    transform_stage(dfs, spark)

    print("\n=== Завершення роботи =====")
    spark.stop()
