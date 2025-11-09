from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import broadcast

def create_state_lookup_df(spark):
    state_data = [
        ("01", "AL", "Alabama"), ("02", "AK", "Alaska"), ("04", "AZ", "Arizona"),
        ("05", "AR", "Arkansas"), ("06", "CA", "California"), ("08", "CO", "Colorado"),
        ("09", "CT", "Connecticut"), ("10", "DE", "Delaware"), ("11", "DC", "District of Columbia"),
        ("12", "FL", "Florida"), ("13", "GA", "Georgia"), ("15", "HI", "Hawaii"),
        ("16", "ID", "Idaho"), ("17", "IL", "Illinois"), ("18", "IN", "Indiana"),
        ("19", "IA", "Iowa"), ("20", "KS", "Kansas"), ("21", "KY", "Kentucky"),
        ("22", "LA", "Louisiana"), ("23", "ME", "Maine"), ("24", "MD", "Maryland"),
        ("25", "MA", "Massachusetts"), ("26", "MI", "Michigan"), ("27", "MN", "Minnesota"),
        ("28", "MS", "Mississippi"), ("29", "MO", "Missouri"), ("30", "MT", "Montana"),
        ("31", "NE", "Nebraska"), ("32", "NV", "Nevada"), ("33", "NH", "New Hampshire"),
        ("34", "NJ", "New Jersey"), ("35", "NM", "New Mexico"), ("36", "NY", "New York"),
        ("37", "NC", "North Carolina"), ("38", "ND", "North Dakota"), ("39", "OH", "Ohio"),
        ("40", "OK", "Oklahoma"), ("41", "OR", "Oregon"), ("42", "PA", "Pennsylvania"),
        ("44", "RI", "Rhode Island"), ("45", "SC", "South Carolina"), ("46", "SD", "South Dakota"),
        ("47", "TN", "Tennessee"), ("48", "TX", "Texas"), ("49", "UT", "Utah"),
        ("50", "VT", "Vermont"), ("51", "VA", "Virginia"), ("53", "WA", "Washington"),
        ("54", "WV", "West Virginia"), ("55", "WI", "Wisconsin"), ("56", "WY", "Wyoming")
    ]
    return spark.createDataFrame(state_data, ["STATE", "STUSAB", "STATE_NAME"])

def question_1(df: DataFrame):
    """
    1. Як сімейний стан (MAR) і кількість дітей (NOP)
       впливають на фінансову стабільність (POVPIP)?
    """
    print("1. Вплив шлюбу та наявності дітей на POVPIP")

    df_filtered = df.filter(F.col("POVPIP").isNotNull())
    df_cleaned = df_filtered.withColumn(
        "NOP",
        F.when(F.col("NOP").isNull(), F.lit(0)).otherwise(F.col("NOP"))
    )
    df_grouped = (
        df_cleaned.groupBy("MAR", "NOP")
        .agg(F.avg("POVPIP").alias("avg_poverty_ratio"))
    )

    df_labeled = df_grouped.withColumn(
        "Marital_Status",
        F.when(F.col("MAR") == 1, "Married")
         .when(F.col("MAR") == 2, "Widowed")
         .when(F.col("MAR") == 3, "Divorced")
         .when(F.col("MAR") == 4, "Separated")
         .when(F.col("MAR") == 5, "Never married")
         .otherwise("Unknown")
    )
    window_spec = Window.partitionBy("Marital_Status").orderBy(F.col("avg_poverty_ratio").desc())
    df_ranked = df_labeled.withColumn("rank", F.rank().over(window_spec))

    df_ranked.select(
        "Marital_Status",
        "NOP",
        F.round("avg_poverty_ratio", 2).alias("avg_poverty_ratio"),
        "rank"
    ).orderBy("Marital_Status", "rank").show(30, truncate=False)


def question_2(df: DataFrame):
    """
    2. Які групи (стать, раса, статус зайнятості) мають найнижчий рівень POVPIP?
    """
    print("2. Групи з найнижчим POVPIP")

    df_filtered = df.filter(F.col("POVPIP").isNotNull())

    df_labeled = (
        df_filtered
        .withColumn(
            "SEX_Label",
            F.when(F.col("SEX") == 1, "Male")
             .when(F.col("SEX") == 2, "Female")
             .otherwise("Unknown")
        )
        .withColumn(
            "RAC1P_Label",
            F.when(F.col("RAC1P") == 1, "White")
             .when(F.col("RAC1P") == 2, "Black or African American")
             .when(F.col("RAC1P") == 3, "American Indian or Alaska Native")
             .when(F.col("RAC1P") == 4, "Asian")
             .when(F.col("RAC1P") == 5, "Native Hawaiian or Pacific Islander")
             .when(F.col("RAC1P") == 6, "Other Race")
             .when(F.col("RAC1P") == 7, "Two or More Races")
             .otherwise("Unknown")
        )
        .withColumn(
            "ESR_Label",
            F.when(F.col("ESR") == 1, "Employed - at work")
             .when(F.col("ESR") == 2, "Employed - not at work")
             .when(F.col("ESR") == 3, "Unemployed")
             .when(F.col("ESR") == 4, "Armed Forces - at work")
             .when(F.col("ESR") == 5, "Armed Forces - not at work")
             .when(F.col("ESR") == 6, "Not in labor force")
             .otherwise("Unknown")
        )
    )
    df_grouped = (
        df_labeled.groupBy("SEX_Label", "RAC1P_Label", "ESR_Label")
        .agg(F.round(F.avg("POVPIP"), 2).alias("avg_poverty_ratio"))
        .orderBy(F.asc("avg_poverty_ratio"))
    )

    df_grouped.show(20, truncate=False)

def question_3(df: DataFrame):
    """
    3. Як відрізняється середній дохід (PINCP)
       між народженими у США (NATIVITY=1) та іммігрантами (NATIVITY=2)
       у різних галузях (INDP)?
    """
    print("3. Порівняння доходів між NATIVITY=1 (народженні у США) та NATIVITY=2 (іммігранти)")
    df_filtered = df.filter(F.col("PINCP").isNotNull())

    df_cleaned = df_filtered.withColumn(
        "INDP",
        F.when(F.col("INDP").isNull(), F.lit(0))
         .otherwise(F.col("INDP").cast("int"))
         .cast("int")
    )
    df_grouped = (
        df_cleaned.groupBy("INDP", "NATIVITY")
        .agg(F.round(F.avg("PINCP"), 2).alias("avg_income"))
    )

    window_spec = Window.partitionBy("INDP").orderBy(F.col("avg_income").desc())
    df_ranked = df_grouped.withColumn("rank", F.rank().over(window_spec))

    df_ranked.orderBy("INDP", "rank").show(30, truncate=False)


def question_4(df: DataFrame):
    """
    4. Як середня кількість відпрацьованих годин на тиждень (WKHP)
       впливає на рівень доходу (WAGP) у різних галузях (INDP)?
    """
    print("4. Продуктивність праці: дохід відносно відпрацьованих годин")

    df_cleaned = df.withColumn(
        "INDP",
        F.when(F.col("INDP").isNull(), F.lit(0))
         .otherwise(F.col("INDP").cast("int"))
         .cast("int")
    )
    df_filtered = df_cleaned.filter(
        F.col("WAGP").isNotNull() &
        F.col("WKHP").isNotNull() &
        (F.col("WKHP") > 0)
    )
    df_grouped = (
        df_filtered.groupBy("INDP")
        .agg(
            F.avg("WAGP").alias("avg_income"),
            F.avg("WKHP").alias("avg_hours")
        )
    )

    df_result = df_grouped.withColumn(
        "income_per_hour",
        F.round(F.col("avg_income") / F.col("avg_hours"), 2)
    )

    df_result.orderBy(F.col("income_per_hour").desc()).show(20)


def question_5(df_demo: DataFrame, df_accidents: DataFrame, spark):
    """
    5. Які вікові групи найчастіше потрапляють у аварії
       за несприятливої погоди (дощ, сніг, туман, гроза тощо)?
    """
    print("5. Вікові групи, що найчастіше потрапляють у аварії під час поганої погоди")

    state_lookup_df = create_state_lookup_df(spark)

    df_demo_small = df_demo.select("ST", "AGEP").filter(F.col("AGEP").isNotNull())

    df_demo_with_state = df_demo_small.join(
        state_lookup_df.select(F.col("STATE").alias("ST"), "STUSAB"),
        "ST", "left"
    ).withColumnRenamed("STUSAB", "StateAbbr")

    df_demo_age_groups = df_demo_with_state.withColumn(
        "Age_Group",
        F.when(F.col("AGEP") < 25, "18-24")
         .when(F.col("AGEP") < 40, "25-39")
         .when(F.col("AGEP") < 60, "40-59")
         .otherwise("60+")
    )

    df_population_by_age = (
        df_demo_age_groups.groupBy("StateAbbr", "Age_Group")
        .agg(F.count("*").alias("Population_Count"))
    )
    bad_weather_conditions = [
        "Rain", "Snow", "Drizzle", "Thunder", "T-Storm",
        "Fog", "Haze", "Overcast", "Windy", "Storm", "Ice"
    ]
    pattern = "(?i)" + "|".join(bad_weather_conditions)

    df_bad_weather = df_accidents.filter(F.col("Weather_Condition").rlike(pattern))

    df_weather_stats = (
        df_bad_weather.groupBy("State", "Weather_Condition")
        .agg(F.count("*").alias("Accident_Count"))
    )
    df_join = df_weather_stats.join(
        df_population_by_age,
        df_weather_stats.State == df_population_by_age.StateAbbr,
        "inner"
    )
    df_result = (
        df_join.withColumn(
            "Accidents_per_1000_people",
            F.round((F.col("Accident_Count") / F.col("Population_Count")) * 1000, 2)
        )
        .select("State", "Age_Group", "Weather_Condition", "Accidents_per_1000_people")
        .orderBy(F.desc("Accidents_per_1000_people"))
    )
    df_result.show(15, truncate=False)

def question_6(df_demo: DataFrame, df_accidents: DataFrame, spark):
    """
    6. Аналіз кількості аварій за статтю (SEX) у штатах із високою кількістю ДТП.
    """
    print("6. Аналіз кількості аварій за статтю у штатах із високою кількістю ДТП")

    state_lookup_df = create_state_lookup_df(spark)

    df_gender = (
        df_demo.select("ST", "SEX")
        .filter(F.col("SEX").isNotNull())
        .groupBy("ST", "SEX")
        .agg(F.count("*").alias("Population_Count"))
    )
    df_gender_with_abbr = df_gender.join(
        state_lookup_df.select(F.col("STATE").alias("ST"), "STUSAB"),
        "ST",
        "left"
    ).withColumnRenamed("STUSAB", "StateAbbr")

    df_gender_with_abbr = df_gender_with_abbr.withColumn(
        "Sex_Label",
        F.when(F.col("SEX") == 1, "Male")
         .when(F.col("SEX") == 2, "Female")
         .otherwise("Unknown")
    )
    df_accidents_by_state = (
        df_accidents.select("State")
        .filter(F.col("State").isNotNull())
        .groupBy("State")
        .agg(F.count("*").alias("Accident_Count"))
    )

    df_join = df_accidents_by_state.join(
        broadcast(df_gender_with_abbr),
        df_accidents_by_state.State == df_gender_with_abbr.StateAbbr,
        "inner"
    )
    df_with_ratio = df_join.withColumn(
        "Accidents_per_1000",
        F.round((F.col("Accident_Count") / F.col("Population_Count")) * 1000, 2)
    )

    window_spec = Window.partitionBy("Sex_Label").orderBy(F.desc("Accidents_per_1000"))
    df_ranked = df_with_ratio.withColumn("Rank", F.rank().over(window_spec))

    df_top10 = (
        df_ranked.filter(F.col("Rank") <= 10)
        .select("State", "Sex_Label", "Accidents_per_1000", "Rank")
        .orderBy("Sex_Label", "Rank")
    )
    df_top10.show(20, truncate=False)



def transform_demographics_df(dataframes: dict, spark):
    print("\n" + "="*20 + " Running Demographics Analysis " + "="*20)

    df_demo = dataframes.get("demographics")
    df_accidents = dataframes.get("accidents")

    if df_demo is None or df_accidents is None:
        print("DataFrames 'demographics' або 'accidents' не знайдено.")
        return

    df_demo = (
        df_demo.withColumn("PINCP", F.col("PINCP").cast("float"))
               .withColumn("POVPIP", F.col("POVPIP").cast("float"))
               .withColumn("WAGP", F.col("WAGP").cast("float"))
               .withColumn("WKHP", F.col("WKHP").cast("float"))
               .withColumn("AGEP", F.col("AGEP").cast("integer"))
    )

    question_1(df_demo)
    question_2(df_demo)
    question_3(df_demo)
    question_4(df_demo)
    question_5(df_demo, df_accidents, spark)
    question_6(df_demo, df_accidents, spark)

    print("\n" + "="*20 + " Demographics Analysis Complete " + "="*20)