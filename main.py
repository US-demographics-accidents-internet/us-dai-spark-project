from pyspark.sql import SparkSession
from extract import load_dataframes, merge_equal_dataframes
from transformation.internet_demographic import InternetDemographicTransformation


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

    # Merge 2 DataFrames (pusa, pusb) into demographic_df
    demographic_df = merge_equal_dataframes(dfs["pusa"], dfs["pusb"])
    dfs["demographic_df"] = demographic_df

    # Delete unused dataframes from dict
    pusa = dfs.pop("pusa")
    pusb = dfs.pop("pusb")

    internet_demographic_transformation = InternetDemographicTransformation() 

    results = internet_demographic_transformation.invoke_pipeline(spark=spark,
                                                                  demographics_df=dfs["demographic_df"], 
                                                                  internet_df=dfs["internet"])

    spark.stop()
