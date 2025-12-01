from pyspark.sql import SparkSession
from extract import load_dataframes, merge_equal_dataframes
from transformation.internet_demographic import InternetDemographicTransformation
from transform.accidents import transform_accidents_df
from transform.demographics import transform_demographics_df 
from transform.accidents_demographics import AccidentsDemographicsTransformation
from transform_pipeline import transform_stage
from training.internet_demographic import InternetDemographicTraining
from pyspark.ml.regression import LinearRegression, RandomForestRegressor


if __name__ == "__main__":
    # SparkSession initialization
    spark = SparkSession.builder.appName("ExtractStage").getOrCreate()

    # Path to the Data directory (inside the container)
    base_path = "/app/Data"

    # Loading all CSVs into a DataFrame
    print("=== STAGE 1: EXTRACT ===")
    dfs = load_dataframes(spark, base_path)

    # Transformational stage
    # internet_demographic_results = transform_stage(dfs)
    internet_demographic_training = InternetDemographicTraining(experiment_name="internet_demographic_regression_experiment")
    internet_df = dfs["internet"].sample(withReplacement=False, fraction=0.01, seed=42)
    print("rows", internet_df.count())
    internet_demographic_training.invoke_training_pipeline(
        df=internet_df,
        target_col="MaxAdDown",
        algorithm=RandomForestRegressor(labelCol="MaxAdDown")
    )

    print("\n=== Завершення роботи =====")
    spark.stop()