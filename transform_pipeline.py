from extract import merge_equal_dataframes
from transformation.internet_demographic import InternetDemographicTransformation
from transform.accidents import transform_accidents_df
from transform.demographics import transform_demographics_df 
from transform.accidents_demographics import AccidentsDemographicsTransformation
import os 

def transform_stage(dfs: dict):
    """
    Transformational stage:
    - General information
    - Statistics
    - Business questions
    """

    # ----------------------------------------------------------------
    # ------------------- General information ------------------------
    # ----------------------------------------------------------------

    df_pusa = dfs["pusa"]
    df_pusb = dfs["pusb"]
    df_demographics = dfs["demographics"]
    df_accidents = dfs["accidents"]
    df_internet = dfs["internet"]

    # print("\n=== General information ===")
    # target_keys = ["demographics", "accidents", "internet"]

    # for name, df in dfs.items():
    #     if name in target_keys:
    #         print(f"\n{name.upper()}:")
    #         print(f"  • Rows: {df.count()}")
    #         print(f"  • Columns: {len(df.columns)}")
    #         df.printSchema()


    # # ----------------------------------------------------------------
    # # ---------------- Numeric column statistics ---------------------
    # # ----------------------------------------------------------------

    # # Folder for results of numeric column statistics
    # os.makedirs("numeric_column_statistics", exist_ok=True)

    # print("\n=== 5 Numeric column statistics (Demographics) ===")
    # numeric_cols = [
    #     f.name for f in df_demographics.schema.fields
    #     if f.dataType.simpleString() in ["int", "double", "float"]
    # ]
    # df_d = df_demographics.select(numeric_cols).describe()
    # df_d.coalesce(1).write.mode("overwrite").option("header", True).csv("numeric_column_statistics/demographics_stats")
    # numeric_cols = numeric_cols[:5]
    # df_demographics.select(numeric_cols).describe().show(truncate=False)


    # print("\n=== 5 Numeric column statistics (Accidents) ===")
    # numeric_cols = [
    #     f.name for f in df_accidents.schema.fields
    #     if f.dataType.simpleString() in ["int", "double", "float"]
    # ]
    # df_a = df_accidents.select(numeric_cols).describe()
    # df_a.coalesce(1).write.mode("overwrite").option("header", True).csv("numeric_column_statistics/accidents_stats")
    # numeric_cols = numeric_cols[:5]
    # df_accidents.select(numeric_cols).describe().show(truncate=False)

    # print("\n=== 5 Numeric column statistics (Internet) ===")
    # numeric_cols = [
    #     f.name for f in df_internet.schema.fields
    #     if f.dataType.simpleString() in ["int", "double", "float"]
    # ]
    # df_i = df_internet.select(numeric_cols).describe()
    # df_i.coalesce(1).write.mode("overwrite").option("header", True).csv("numeric_column_statistics/internet_stats")
    # numeric_cols = numeric_cols[:5]
    # df_internet.select(numeric_cols).describe().show(truncate=False)


    # ----------------------------------------------------------------
    # --------------------- Business questions  ----------------------
    # ----------------------------------------------------------------

    # Merge 2 DataFrames (pusa, pusb) into demographic_df
    demographic_df = merge_equal_dataframes(dfs["pusa"], dfs["pusb"])
    dfs["demographic_df"] = demographic_df

    
    print("\n=== STAGE 2: TRANSFORM (Business-questions) ===")

    # internet_demographic_transformation = InternetDemographicTransformation() 

    # internet_demographic_results = internet_demographic_transformation.invoke_pipeline(spark=spark,
    #                                                               demographics_df=dfs["demographic_df"], 
    #                                                               internet_df=dfs["internet"])

    # transform_accidents_df(dfs)

    # transform_demographics_df(dfs, spark)

    # accidents_demographics_transformation = AccidentsDemographicsTransformation() 

    # accidents_demographics_transformation.invoke_pipeline(spark=spark,
    #                                                               df_pusa=dfs["pusa"],
    #                                                               df_pusb=dfs["pusb"], 
    #                                                               df_accidents=dfs["accidents"])

    return df_internet