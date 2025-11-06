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
    fields_list = ['ID', 'Source', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng',
               'End_Lat', 'End_Lng', 'Distance(mi)', 'Description', 'Street', 'City', 'County', 'State',
               'Zipcode', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)',
               'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
               'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump',
               'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
               'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset',
               'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

    int_fields = ['Severity']
    float_fields = ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Temperature(F)',
                    'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                    'Precipitation(in)']
    string_fields = ['ID', 'Source', 'Start_Time', 'End_Time', 'Description', 'Street', 'City', 'County',
                    'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp',
                    'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight',
                    'Nautical_Twilight', 'Astronomical_Twilight']
    bool_fields = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout',
                'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

    accidents_schema = StructType([
        StructField(f, IntegerType() if f in int_fields else 
                    FloatType() if f in float_fields else 
                    BooleanType() if f in bool_fields else 
                    StringType(), True)
        for f in fields_list
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
