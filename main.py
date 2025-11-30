from pyspark.sql import SparkSession
from extract import load_dataframes, merge_equal_dataframes
from transformation.internet_demographic import InternetDemographicTransformation
from transform.accidents import transform_accidents_df
from transform.demographics import transform_demographics_df 
from transform.accidents_demographics import AccidentsDemographicsTransformation
from transform_pipeline import transform_stage
from training.internet_demographic import InternetDemographicTraining
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
import mlflow


if __name__ == "__main__":
    # SparkSession initialization
    spark = SparkSession.builder.appName("ExtractStage").getOrCreate()

    mlflow.set_tracking_uri("http://mlflow:5000")

    print("Tracking URI:", mlflow.get_tracking_uri())
    # Path to the Data directory (inside the container)
    base_path = "/app/Data"

    # Loading all CSVs into a DataFrame
    print("=== STAGE 1: EXTRACT ===")
    dfs = load_dataframes(spark, base_path)

    print("\nRows number in each dataset:")
    for name, df in dfs.items():
        print(f"{name}: {df.count()}")

    # Transformational stage
    internet_demographic_results = transform_stage(dfs, spark)
    internet_demographic_training = InternetDemographicTraining(experiment_name="internet_demographic_regression_experiment")
    internet_demographic_results_10_df = internet_demographic_results.sample(withReplacement=False, fraction=0.1, seed=42)
    internet_demographic_training.invoke_training_pipeline(
        df=internet_demographic_results_10_df,
        target_col="income_deviation",
        algorithm=RandomForestRegressor(labelCol="income_deviation", max_bins=32),
        )

    print("\n=== Завершення роботи =====")
    spark.stop()