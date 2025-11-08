from pyspark.sql import SparkSession
from extract import load_dataframes
from transform.accidents import transform_accidents_df 

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

    print("\n=== STAGE 2: TRANSFORM (Business-questions) ===")
    
    transform_accidents_df(dfs)
    # TODO: insert transform stage for other datasets

    print("\n=== Завершення роботи =====")
    spark.stop()
