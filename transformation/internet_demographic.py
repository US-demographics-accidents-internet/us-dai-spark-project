import pyspark.sql.functions as F
from pyspark.sql.functions import col, avg, when
from pyspark.sql import DataFrame, SparkSession


class InternetDemographicTransformation:
    def invoke_pipeline(self, spark, demographics_df, internet_df):
        state_data = [
            ("01", "AL"), ("02", "AK"), ("04", "AZ"), ("05", "AR"), ("06", "CA"), 
            ("08", "CO"), ("09", "CT"), ("10", "DE"), ("11", "DC"), ("12", "FL"), 
            ("13", "GA"), ("15", "HI"), ("16", "ID"), ("17", "IL"), ("18", "IN"), 
            ("19", "IA"), ("20", "KS"), ("21", "KY"), ("22", "LA"), ("23", "ME"), 
            ("24", "MD"), ("25", "MA"), ("26", "MI"), ("27", "MN"), ("28", "MS"), 
            ("29", "MO"), ("30", "MT"), ("31", "NE"), ("32", "NV"), ("33", "NH"), 
            ("34", "NJ"), ("35", "NM"), ("36", "NY"), ("37", "NC"), ("38", "ND"), 
            ("39", "OH"), ("40", "OK"), ("41", "OR"), ("42", "PA"), ("44", "RI"), 
            ("45", "SC"), ("46", "SD"), ("47", "TN"), ("48", "TX"), ("49", "UT"), 
            ("50", "VT"), ("51", "VA"), ("53", "WA"), ("54", "WV"), ("55", "WI"), 
            ("56", "WY"), ("72", "PR")
        ]

        state_lookup_df = spark.createDataFrame(state_data, ["ST", "StateAbbr"])

        result_df = self.average_salary_by_speed(demographics_df=demographics_df,
                                                 internet_df=internet_df,
                                                 state_lookup_df=state_lookup_df)

        result_df.show()

    # What is the average salary in states with "fast" vs. "slow" internet?
    def average_salary_by_speed(self, demographics_df: DataFrame, 
                                  internet_df: DataFrame, 
                                  state_lookup_df: DataFrame, 
                                  speed_threshold: int = 100) -> DataFrame:
        """
        Calculates the average salary (WAGP) in states, 
        grouped by internet speed.
        
        :param demographics_df: DataFrame with demographic data (including WAGP, ST)
        :param internet_df: DataFrame with internet data (including MaxAdDown, StateAbbr)
        :param state_lookup_df: DataFrame to join ST and StateAbbr
        :param speed_threshold: Threshold (in Mbps) to define "fast" internet
        :return: DataFrame with columns [speed_tier, average_salary]
        """
        
        # 1. Calculate the average download speed for each state
        avg_speed_per_state = internet_df.groupBy("StateAbbr") \
            .agg(avg("MaxAdDown").alias("avg_download_speed"))

        # 2. Add the state code (ST) to join with demographics
        avg_speed_with_st = avg_speed_per_state.join(state_lookup_df, "StateAbbr")

        # 3. Create "fast" / "slow" categories
        speed_tiers = avg_speed_with_st.withColumn("speed_tier",
            when(col("avg_download_speed") > speed_threshold, "fast_internet")
            .otherwise("slow_internet")
        ).select("ST", "speed_tier")

        # 4. Join the speed categories to the demographic data
        demo_with_speed = demographics_df.join(speed_tiers, "ST")

        # 5. Calculate the average salary, filtering for those who have a salary
        q1_result = demo_with_speed.filter(col("WAGP") > 0) \
            .groupBy("speed_tier") \
            .agg(avg("WAGP").alias("average_salary"))
        
        return q1_result