from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- Питання 1 ---

def question_1(df_accidents: DataFrame):
    """
    1. Для кожного штату, які є топ-3 години дня, 
       коли трапляється найбільша кількість аварій?
    """
    print("--- 1. Топ-3 небезпечних годин по кожному штату ---")
    
    df_with_hour = df_accidents.withColumn("Hour", F.hour(F.col("Start_Time")))
    
    df_hourly_counts = df_with_hour.groupBy("State", "Hour") \
                                   .count() \
                                   .withColumnRenamed("count", "incident_count")
    
    window_spec = Window.partitionBy("State") \
                        .orderBy(F.col("incident_count").desc())
    
    df_ranked = df_hourly_counts.withColumn("rank", F.rank().over(window_spec))
    
    df_top3 = df_ranked.filter(F.col("rank") <= 3) \
                       .orderBy("State", "rank")
                       
    df_top3.show(30)

# --- Питання 2 ---

def question_2(df_accidents: DataFrame):
    """
    2. Який відсоток від загальної кількості аварій 
       у кожному штаті припадає на вихідні дні?
    """
    print("--- 2. Відсоток аварій на вихідних по кожному штату ---")
    
    df_with_weekend = df_accidents.withColumn("day_of_week", F.dayofweek(F.col("Start_Time"))) \
                                  .withColumn("is_weekend", 
                                              F.when(F.col("day_of_week").isin([1, 7]), 1)
                                               .otherwise(0))
    
    window_spec = Window.partitionBy("State")
    
    df_state_stats = df_with_weekend.withColumn("total_accidents", F.count("*").over(window_spec)) \
                                    .withColumn("weekend_accidents", F.sum("is_weekend").over(window_spec))
    
    df_percentage = df_state_stats.withColumn("weekend_percentage", 
                                              (F.col("weekend_accidents") / F.col("total_accidents")) * 100) \
                                  .select("State", "weekend_percentage") \
                                  .distinct() \
                                  .orderBy(F.col("weekend_percentage").desc())
                                  
    df_percentage.show()

# --- Питання 3 ---

def question_3(df_accidents: DataFrame):
    """
    3. Знайти всі пари аварій, які сталися в одному й тому ж поштовому індексі, 
       але на різних дорогах, з інтервалом менше ніж 24 години.
    """
    print("--- 3. Пари аварій (Zipcode, <24 год, різні вулиці) ---")
    
    df_prepared = df_accidents.select(
        F.col("ID"),
        F.col("Zipcode"),
        F.col("Street"),
        F.unix_timestamp(F.col("Start_Time")).alias("timestamp")
    ).filter(F.col("Zipcode").isNotNull() & F.col("Street").isNotNull()) # Очистка

    df1 = df_prepared.alias("df1")
    df2 = df_prepared.alias("df2")
    
    time_window_seconds = 24 * 60 * 60
    
    join_condition = (
        (F.col("df1.Zipcode") == F.col("df2.Zipcode")) &
        (F.col("df1.Street") != F.col("df2.Street")) &
        (F.col("df1.ID") < F.col("df2.ID")) &
        (F.abs(F.col("df1.timestamp") - F.col("df2.timestamp")) < time_window_seconds)
    )
    
    df_pairs = df1.join(df2, join_condition) \
                  .select(
                      F.col("df1.Zipcode"),
                      F.col("df1.ID").alias("Accident_A_ID"),
                      F.col("df1.Street").alias("Accident_A_Street"),
                      F.col("df2.ID").alias("Accident_B_ID"),
                      F.col("df2.Street").alias("Accident_B_Street")
                  )
                  
    df_pairs.show()

# --- Питання 4 ---

def question_4(df_accidents: DataFrame):
    """
    4. Для кожного штату, який відсоток аварій трапляється 
       за умов дуже низької видимості (< 1 милі)?
    """
    print("--- 4. Відсоток аварій при низькій видимості (<1 миля) ---")
    
    df_with_low_vis = df_accidents.filter(F.col("Visibility(mi)").isNotNull()) \
                                  .withColumn("low_visibility", 
                                              F.when(F.col("Visibility(mi)") < 1.0, 1)
                                               .otherwise(0))
    
    window_spec = Window.partitionBy("State")
    
    df_state_stats = df_with_low_vis.withColumn("total_accidents", F.count("*").over(window_spec)) \
                                    .withColumn("low_vis_accidents", F.sum("low_visibility").over(window_spec))
    
    df_percentage = df_state_stats.withColumn("low_visibility_percentage", 
                                              (F.col("low_vis_accidents") / F.col("total_accidents")) * 100) \
                                  .select("State", "low_visibility_percentage") \
                                  .distinct() \
                                  .orderBy(F.col("low_visibility_percentage").desc())
                                  
    df_percentage.show()

# --- Питання 5 ---

def question_5(df_accidents: DataFrame):
    """
    5. Знайдіть список унікальних міст (City, State), в яких 
       в один і той же календарний день трапилися аварії 
       з умовами "Дощ" і "Сніг".
    """
    print("--- 5. Міста, де був і дощ, і сніг в один день ---")
    
    df_with_conditions = df_accidents.filter(
        F.col("City").isNotNull() & F.col("Weather_Condition").isNotNull()
    ) \
    .withColumn("date", F.date_format(F.col("Start_Time"), "yyyy-MM-dd")) \
    .withColumn("has_rain", F.col("Weather_Condition").like("%Rain%")) \
    .withColumn("has_snow", F.col("Weather_Condition").like("%Snow%"))
    
    df_daily_weather = df_with_conditions.groupBy("City", "State", "date") \
                                         .agg(
                                             F.max(F.when(F.col("has_rain"), 1).otherwise(0)).alias("had_rain"),
                                             F.max(F.when(F.col("has_snow"), 1).otherwise(0)).alias("had_snow")
                                         )
    
    df_rain_and_snow_days = df_daily_weather.filter(
        (F.col("had_rain") == 1) & (F.col("had_snow") == 1)
    )
    
    df_unique_cities = df_rain_and_snow_days.select("City", "State").distinct()
    
    df_unique_cities.show()

# --- Питання 6 ---

def question_6(df_accidents: DataFrame):
    """
    6. Скільки аварій сталося за наявності знаку "Stop" 
       порівняно зі світлофором?
    """
    print("--- 6. Аварії на знаках Stop проти Світлофорів ---")
    
    df_counts = df_accidents.agg(
        F.count(F.when(F.col("Stop") == True, 1)).alias("stop_sign_accidents"),
        F.count(F.when(F.col("Traffic_Signal") == True, 1)).alias("traffic_signal_accidents")
    )
    
    df_counts.show()

# --- Головна функція ---

def transform_accidents_df(dataframes: dict):
    """
    Головна функція, яка запускає всі аналізи 
    для набору даних 'accidents'.
    """
    print("\n" + "="*20 + " Running Accidents Analysis " + "="*20)
    
    df_accidents = dataframes.get("accidents")
    
    if df_accidents is None:
        print("ПОМИЛКА: 'accidents' DataFrame не знайдено.")
        return
    
    df_accidents = df_accidents.withColumn(
        "Start_Time",
        F.to_timestamp(F.substring(F.col("Start_Time"), 0, 19), "yyyy-MM-dd HH:mm:ss")
    ).withColumn(
        "End_Time",
        F.to_timestamp(F.substring(F.col("End_Time"), 0, 19), "yyyy-MM-dd HH:mm:ss")
    )

    # Launch every question
    question_1(df_accidents)
    question_2(df_accidents)
    question_3(df_accidents)
    question_4(df_accidents)
    question_5(df_accidents)
    question_6(df_accidents)
    
    print("\n" + "="*20 + " Accidents Analysis Complete " + "="*20)