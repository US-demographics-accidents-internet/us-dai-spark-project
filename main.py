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

    # # Merge 2 DataFrames (pusa, pusb) into demographic_df
    # demographic_df = merge_equal_dataframes(dfs["pusa"], dfs["pusb"])
    # dfs["demographic_df"] = demographic_df

    
    # print("\n=== STAGE 2: TRANSFORM (Business-questions) ===")

    # internet_demographic_transformation = InternetDemographicTransformation() 

    # results = internet_demographic_transformation.invoke_pipeline(spark=spark,
    #                                                               demographics_df=dfs["demographic_df"], 
    #                                                               internet_df=dfs["internet"])

    # transform_accidents_df(dfs)

    # transform_demographics_df(dfs, spark)

    # accidents_demographics_transformation = AccidentsDemographicsTransformation() 

    # accidents_demographics_transformation.invoke_pipeline(spark=spark,
    #                                                               df_pusa=dfs["pusa"],
    #                                                               df_pusb=dfs["pusb"], 
    #                                                               df_accidents=dfs["accidents"])

    transform_stage(dfs, spark)

    print("\n=== Завершення роботи =====")
    spark.stop()
