from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.functions import broadcast
import os


class AccidentsDemographicsTransformation:
    def invoke_pipeline(self, spark, df_pusa, df_pusb, df_accidents):
        # Demographic data is stored in two datasets, to use all available data we need to merge them.
        # Also we take only those columns that will be used for business questions.
        df_demo = df_pusa.select("SERIALNO", "AGEP", "SEX", "ST", "ADJINC", "CIT", "HINS1", "HINS2", "SCHL", "WKHP","PINCP") \
        .unionByName(df_pusb.select("SERIALNO", "AGEP", "SEX", "ST", "ADJINC", "CIT", "HINS1", "HINS2", "SCHL", "WKHP", "PINCP"))

        # For demographic data, states are denoted by a number rather than an abbreviation, so for 
        # convenient join with accidents and the internet, you need to map the resulting dataframe 
        # with a dictionary file.
        state_map = spark.read.option("header", True).option("sep", "|").csv("/app/Data/demographics/state_matcher.txt")
        state_map = state_map.select(F.col("STUSAB").alias("StateAbbr"), F.col("STATE").alias("ST"))

        # Join StateAbbr to df_demo 
        df_demo = df_demo.join(state_map, on="ST", how="inner")

        # Folder for results of business questions
        os.makedirs("business_questions_results", exist_ok=True)

        print("\n" + "="*20 + " Running Accidents/Demographics Analysis " + "="*20)

        df_visibility_impact = self.visibility_impact(df_accidents)
        df_visibility_impact.show(truncate=False)

        df_risk_index = self.risk_index(df_accidents)
        df_risk_index.show(10, truncate=False)

        education_vs_accidents_df = self.education_vs_accidents(df_demo,
                                                                df_accidents)
        education_vs_accidents_df.show(10)

        work_hours_vs_accidents_df = self.work_hours_vs_accidents(df_demo,
                                                                  df_accidents)
        work_hours_vs_accidents_df.show(10)

        top5_wether_conditions_df = self.top5_wether_conditions(df_accidents)
        top5_wether_conditions_df.show(10)

        under40_income_rank_state_df = self.under40_income_rank_state(df_demo)
        under40_income_rank_state_df.show(10, truncate=False)

        print("\n" + "="*20 + " Accidents/Demographics Analysis Complete " + "="*20)

    # ----------------------------------------------------------------
    # ------------------- Business questions -------------------------
    # ----------------------------------------------------------------

    # ---------- GrpupBy 1x, Filter 1x ----------
    def visibility_impact(self, df_accidents: DataFrame) -> DataFrame:
        """
        Computes accident severity statistics across visibility categories.
        Categorizes visibility ranges, then aggregates average severity and
        accident counts to quantify the impact of visibility conditions.
        """
        print("\n--- 1. The impact of visibility on accident severity ---")
        df_visibility_impact = (
            df_accidents
            .filter(F.col("Visibility(mi)").isNotNull())
            .withColumn("Visibility_Category", 
                        F.when(F.col("Visibility(mi)") < 1, "Low (<1 mi)")
                        .when(F.col("Visibility(mi)") < 5, "Moderate (1-5 mi)")
                        .otherwise("High (>5 mi)"))
            .groupBy("Visibility_Category")
            .agg(
                F.avg("Severity").alias("Avg_Severity"),
                F.count("*").alias("Accident_Count")
            )
            .orderBy(F.desc("Avg_Severity"))
        )

        df_visibility_impact.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("business_questions_results/am_group_visibility_impact")
        return df_visibility_impact


    # ---------- GrpupBy 1x ----------
    def risk_index(self, df_accidents: DataFrame) -> DataFrame:
        """
        Calculates a composite risk index for each state based on average accident
        severity and accident frequency. Produces a ranked list of states ordered
        by overall riskiness.
        Index = average severity + (number of accidents / 10,000)
        """
        print("\n--- 2. Ranking of states by “riskiness” (composite index) ---")
        df_risk_index = (
            df_accidents.groupBy("State")
            .agg(
                F.avg("Severity").alias("Avg_Severity"),
                F.count("*").alias("Accident_Count")
            )
            .withColumn("Risk_Index", F.col("Avg_Severity") + (F.col("Accident_Count") / 10000))
            .orderBy(F.desc("Risk_Index"))
        )

        df_risk_index.coalesce(1).write.mode("overwrite").option("header", True).csv("business_questions_results/am_group_risk_index")
        return df_risk_index


    # ---------- Join 1x, GroupBy 2x ----------
    def education_vs_accidents(self, df_demo: DataFrame, 
                               df_accidents: DataFrame) -> DataFrame:
        """
        Calculates the relationship between average education level and accident severity 
        by state,returning a DataFrame that links educational attainment to accident outcomes.
        """
        print("\n--- 3. Number of accidents depending on the level of education ---")
        df_demo_avg_edu = df_demo.select("StateAbbr", "SCHL") \
            .groupBy("StateAbbr") \
            .agg(F.avg("SCHL").alias("Avg_Education"))

        df_accidents_avg_sev = df_accidents.select("State", "Severity") \
            .groupBy("State") \
            .agg(F.avg("Severity").alias("Avg_Severity"))

        education_vs_accidents_df = df_accidents_avg_sev.join(
            broadcast(df_demo_avg_edu),
            df_accidents_avg_sev.State == df_demo_avg_edu.StateAbbr,
            how="inner"
        ).select("State", "Avg_Education", "Avg_Severity")

        education_vs_accidents_df.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("business_questions_results/am_join_edu_vs_accidents")
        return education_vs_accidents_df
    

    # # ---------- Join 1x, GroupBy 2x, Filter 1x ----------
    def work_hours_vs_accidents(self, df_demo: DataFrame, 
                                df_accidents: DataFrame) -> DataFrame:
        """
        Analyzes how average weekly working hours relate to accident counts by state,
        highlighting correlations for states where work hours exceed 35 per week.
        """
        print("\n--- 4. Number of accidents depending on the number of working hours (overtime) ---")
        df_work = df_demo.select("StateAbbr", "WKHP") \
            .groupBy("StateAbbr") \
            .agg(F.avg("WKHP").alias("Avg_Work_Hours"))

        df_acc = df_accidents.select("State") \
            .groupBy("State") \
            .agg(F.count("*").alias("Total_Accidents"))

        work_hours_vs_accidents_df = df_acc.join(
            broadcast(df_work),
            df_acc.State == df_work.StateAbbr,
            how="inner"
        ).filter(F.col("Avg_Work_Hours") > 35
        ).select("State", "Total_Accidents", "Avg_Work_Hours")

        work_hours_vs_accidents_df.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("business_questions_results/am_join_work_hours_vs_accidents")
        return work_hours_vs_accidents_df


    # ---------- Window 1x, Filter 1x ----------
    def top5_wether_conditions(self, df_accidents: DataFrame) -> DataFrame:
        """
        Identifies the top 5 weather conditions contributing to accidents in each state,
        ranking them by frequency of occurrence.
        """
        print("\n--- 5. The 5 most active weather conditions in the state by number of accidents ---")
        window_weather = Window.partitionBy("State").orderBy(F.desc("Weather_Count"))
        top5_wether_conditions_df = df_accidents.groupBy("State", "Weather_Condition").agg(F.count("*").alias("Weather_Count")) \
            .withColumn("Weather_Rank", F.rank().over(window_weather)) \
            .filter(F.col("Weather_Rank") <= 5)
        
        top5_wether_conditions_df.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("business_questions_results/am_window_top5_weather_conditions")
        return top5_wether_conditions_df


    # # ---------- Window 1x, Filter 1x ----------
    def under40_income_rank_state(self, df_demo: DataFrame) -> DataFrame:
        """
        Ranks individuals under 40 by adjusted income within each state,
        providing a state-level income hierarchy for younger populations.
        """
        print("\n--- 6. People under 40 ranked by adjusted income within each state ---")
        df_demo = df_demo.withColumn(
            "Adjusted_Income",
            (F.col("PINCP") * F.col("ADJINC")) / 1_000_000
        )

        window_age_income = Window.partitionBy("StateAbbr").orderBy(F.desc("Adjusted_Income"))

        under40_income_rank_state_df = (
            df_demo.filter(F.col("AGEP") < 40)
            .withColumn("Income_Rank_in_State", F.rank().over(window_age_income))
            .select("StateAbbr", "AGEP", "Adjusted_Income", "Income_Rank_in_State")
        )

        under40_income_rank_state_df.coalesce(1).write.mode("overwrite") \
        .option("header", True).csv("business_questions_results/am_window_under40_income_rank_state")
        return under40_income_rank_state_df