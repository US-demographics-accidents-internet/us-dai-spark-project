from pyspark.sql import SparkSession
from extract import load_dataframes

if __name__ == "__main__":
    # SparkSession initialization
    spark = SparkSession.builder.appName("ExtractStage").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Path to the Data directory (inside the container)
    base_path = "/app/Data"

    # Завантаження всіх датасетів
    dfs = load_dataframes(spark, base_path)

    print("\n📊 Короткий огляд датасетів")
    print("════════════════════════════════════════════════════")

    for name, df in dfs.items():
        print(f"\n📁 {name}")
        print("──────────────────────────────────────────────")

        row_count = df.count()
        col_count = len(df.columns)

        print(f"   • Рядків: {row_count}")
        print(f"   • Колонок: {col_count}")
        print("──────────────────────────────────────────────")

    spark.stop()
