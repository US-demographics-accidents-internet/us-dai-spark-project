from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType, BooleanType

def load_dataframes(spark: SparkSession, base_path: str):
    """
    Loads all CSV files from the base Data directory as a DataFrame.
    Uses explicitly specified schemas.
    """

    # ---------------------
    # ------ Schemes ------
    # ---------------------

    # ------ Demographics data ------
    demographics_schema = StructType([
        StructField("RT", StringType(), True),
        StructField("SERIALNO", StringType(), True)
    ])

    # ------ Accidents data ------
    accidents_schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Source", StringType(), True)
    ])

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
    df_pusa = spark.read.csv(f"{base_path}/demographics/psam_pusa.csv", header=True, schema=demographics_schema)
    df_pusb = spark.read.csv(f"{base_path}/demographics/psam_pusb.csv", header=True, schema=demographics_schema)
    df_accidents = spark.read.csv(f"{base_path}/accidents/US_Accidents_March23.csv", header=True, schema=accidents_schema)
    df_internet = spark.read.csv(f"{base_path}/internet/fbd_us_with_satellite_dec2021_v1.csv", header=True, schema=internet_schema)

    # --- Light check ---
    print("Data read:")
    # First 10 columns
    print("Internet:")
    # First 10 columns
    df_internet.select(df_internet.columns[:10]).show(5)

    return {
        "pusa": df_pusa,
        "pusb": df_pusb,
        "accidents": df_accidents,
        "internet": df_internet
    }
