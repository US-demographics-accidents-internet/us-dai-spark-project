from pyspark.sql import SparkSession
from extract import load_dataframes

if __name__ == "__main__":
    # SparkSession initialization
    spark = SparkSession.builder.appName("ExtractStage").getOrCreate()

    # Path to the Data directory (inside the container)
    base_path = "/app/Data"

    # Loading all CSVs into a DataFrame
    dfs = load_dataframes(spark, base_path)

    print("\n📊 Rows number in each dataset:")
    for name, df in dfs.items():
        print(f"{name}: {df.count()}")

    spark.stop()
