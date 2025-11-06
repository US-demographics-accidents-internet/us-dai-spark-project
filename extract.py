from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType, BooleanType

def load_dataframes(spark: SparkSession, base_path: str):
    """
    Loads all CSV files from the base Data directory as a DataFrame.
    Uses explicitly specified schemas.
    """

    # ------ Internet data ------

    internet_schema = StructType([
        StructField("LogRecNo", IntegerType(), True),
        StructField("Provider_Id", IntegerType(), True),
        StructField("FRN", IntegerType(), True),
        StructField("ProviderName", StringType(), True),
        StructField("DBAName", StringType(), True),
        StructField("HoldingCompanyName", StringType(), True),
        StructField("HocoNum", IntegerType(), True),
        StructField("HocoFinal", StringType(), True),
        StructField("StateAbbr", StringType(), True),
        StructField("BlockCode", IntegerType(), True),
        StructField("TechCode", IntegerType(), True),
        StructField("Consumer", IntegerType(), True),
        StructField("MaxAdDown", FloatType(), True),
        StructField("MaxAdUp", FloatType(), True),
        StructField("Business", IntegerType(), True),
    ])

    # --- Read CSV ---
    df_internet = spark.read.csv(f"{base_path}/internet/fbd_us_with_satellite_dec2021_v1.csv", header=True, schema=internet_schema)

    # --- Light check ---
    print("Data read:")
    # First 10 columns
    print("Internet:")
    # First 10 columns
    df_internet.select(df_internet.columns[:10]).show(5)

    return {
        "internet": df_internet
    }