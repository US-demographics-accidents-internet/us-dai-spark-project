from pyspark.sql import SparkSession
from extract import load_dataframes
from transformation.filter import FilterTransormation
from transformation.group_by import GroupByTransormation
from transformation.join import JoinTransormation
from transformation.window_functions import WindowFunctionTransormation

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

    filter_transformations = FilterTransormation() 
    join_transformations = JoinTransormation() 
    groupBy_transformations = GroupByTransormation() 
    window_function_transformations = WindowFunctionTransormation() 


    for name, df in dfs.items():
        filters = filter_transformations.invoke_pipeline(dataframe=df, dataframe_name=name)
        joins = join_transformations.invoke_pipeline(dataframe=df, dataframe_name=name)
        groupBys = groupBy_transformations.invoke_pipeline(dataframe=df, dataframe_name=name)
        window_functions = window_function_transformations.invoke_pipeline(dataframe=df, dataframe_name=name)




    spark.stop()
