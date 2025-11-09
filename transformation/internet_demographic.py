import pyspark.sql.functions as F
from pyspark.sql.functions import col, avg, when, desc, rank, col, countDistinct, expr
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window


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

        avarage_salary_df = self.average_salary_by_speed(demographics_df=demographics_df,
                                                 internet_df=internet_df,
                                                 state_lookup_df=state_lookup_df)
        avarage_salary_df.show()

        top_professions = self.rank_top_professions_by_education(demographics_df=demographics_df)
        top_professions.show()

        time_by_biz_internet = self.compare_commute_time_by_biz_internet(demographics_df=demographics_df,
                                                 internet_df=internet_df,
                                                 state_lookup_df=state_lookup_df)
        time_by_biz_internet.show()

        income_comparation = self.compare_income_to_peer_group(demographics_df=demographics_df)
        income_comparation.show()

        providers_correlation = self.correlate_providers_with_income(demographics_df=demographics_df,
                                                 internet_df=internet_df,
                                                 state_lookup_df=state_lookup_df)
        providers_correlation.show()

        income_inequality_with_digital_divide_df = self.correlate_income_inequality_with_digital_divide(demographics_df=demographics_df,
                                                 internet_df=internet_df,
                                                 state_lookup_df=state_lookup_df)
        income_inequality_with_digital_divide_df.show()

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
    
    def rank_top_professions_by_education(self, demographics_df: DataFrame, top_n: int = 3) -> DataFrame:
        """
        Ranks the top N professions (OCCP) by average income (PINCP) 
        within each education group (SCHL).
    
        :param demographics_df: DataFrame with demographic data (incl. PINCP, OCCP, SCHL)
        :param top_n: The number of top professions to return for each group (default: 3)
        :return: DataFrame with columns [SCHL, OCCP, avg_income, rank]
        """
    
        # 1. Filter for people with income and occupation,
        #    then calculate average income for each (Education, Occupation) pair.
        income_by_job_edu = demographics_df.filter(
            (col("PINCP") > 0) & col("OCCP").isNotNull()
            ) \
            .groupBy("SCHL", "OCCP") \
            .agg(avg("PINCP").alias("avg_income"))

        # 2. Create a "window" to partition data by education level (SCHL)
        window_spec = Window.partitionBy("SCHL").orderBy(desc("avg_income"))

        # 3. Assign a rank to each profession WITHIN its education group
        ranked_professions = income_by_job_edu.withColumn("rank", rank().over(window_spec))

        # 4. Select only the top N professions from each group
        q2_result = ranked_professions.filter(col("rank") <= top_n)

        return q2_result
    
    def compare_commute_time_by_biz_internet(self, 
                                           demographics_df: DataFrame, 
                                           internet_df: DataFrame, 
                                           state_lookup_df: DataFrame, 
                                           biz_threshold: float = 0.5) -> DataFrame:
        """
        Calculates average commute time (JWMNP) based on the state's 
        business internet availability.

        :param demographics_df: DataFrame with demographic data (JWMNP, WRK, ST)
        :param internet_df: DataFrame with internet data (Business, StateAbbr)
        :param state_lookup_df: DataFrame to join ST and StateAbbr
        :param biz_threshold: Threshold (0.0 to 1.0) to define 'high' business internet
        :return: DataFrame with [internet_tier, average_commute_time]
        """
        
        # 1. Calculate the average 'Business' metric per state
        #    (Assuming 1=yes, 0=no, avg gives a ratio)
        biz_internet_state = internet_df.groupBy("StateAbbr") \
            .agg(avg("Business").alias("business_internet_ratio"))

        # 2. Join with lookup to get ST code
        biz_internet_with_st = biz_internet_state.join(state_lookup_df, "StateAbbr")

        # 3. Classify states into tiers
        biz_internet_tiers = biz_internet_with_st.withColumn("internet_tier",
            when(col("business_internet_ratio") > biz_threshold, "high_biz_internet")
            .otherwise("low_biz_internet")
        ).select("ST", "internet_tier")

        # 4. Join tiers to demographic data
        demo_with_biz_net = demographics_df.join(biz_internet_tiers, "ST")

        # 5. Filter for people who work (e.g., WRK == 1) and commute (JWMNP > 0),
        #    then calculate average commute time.
        q3_result = demo_with_biz_net.filter(
                (col("WRK") == 1) & (col("JWMNP") > 0)
            ) \
            .groupBy("internet_tier") \
            .agg(avg("JWMNP").alias("average_commute_time"))

        return q3_result
    
    def compare_income_to_peer_group(self, demographics_df: DataFrame) -> DataFrame:
        """
        Calculates the deviation of each person's income (PINCP) from the 
        average income of their peer group (defined by State and Age Group).

        This uses a window function to meet the requirement.

        :param demographics_df: DataFrame with demographic data (PINCP, ST, AGEP)
        :return: DataFrame with [SERIALNO, ST, age_group, PINCP, avg_group_income, income_deviation]
        """
        
        # 1. Create age groups (a form of filtering/categorization)
        df_with_age_group = demographics_df.withColumn("age_group",
            when((col("AGEP") >= 18) & (col("AGEP") <= 25), "18-25")
            .when((col("AGEP") >= 26) & (col("AGEP") <= 35), "26-35")
            .when((col("AGEP") >= 36) & (col("AGEP") <= 45), "36-45")
            .when((col("AGEP") >= 46) & (col("AGEP") <= 55), "46-55")
            .when((col("AGEP") >= 56), "56+")
            .otherwise("Under 18")
        )

        # 2. Define the window: partition by State (ST) and the new age_group
        window_spec = Window.partitionBy("ST", "age_group")

        # 3. Filter for relevant people and calculate average income OVER the window
        #    This adds 'avg_group_income' to every row, matching its group
        df_with_group_avg = df_with_age_group.filter(
                (col("PINCP") > 0) & (col("age_group") != "Under 18")
            ) \
            .withColumn("avg_group_income", avg("PINCP").over(window_spec))

        # 4. Calculate the deviation for each person
        q4_result = df_with_group_avg.withColumn(
                "income_deviation", col("PINCP") - col("avg_group_income")
            ) \
            .select("SERIALNO", "ST", "age_group", "PINCP", "avg_group_income", "income_deviation") \
            .orderBy(desc("income_deviation")) # Show highest earners above average first
        
        return q4_result
    
    def correlate_providers_with_income(self, 
                                        demographics_df: DataFrame, 
                                        internet_df: DataFrame, 
                                        state_lookup_df: DataFrame) -> DataFrame:
        """
        Correlates the number of unique internet providers in a state 
        with the average income (PINCP) of that state.

        :param demographics_df: DataFrame with demographic data (PINCP, ST)
        :param internet_df: DataFrame with internet data (ProviderName, StateAbbr)
        :param state_lookup_df: DataFrame to join ST and StateAbbr
        :return: DataFrame with [ST, StateAbbr, avg_income, provider_count]
        """
        
        # 1. Calculate average income per state (ST) from demographics
        avg_income_state = demographics_df.filter(col("PINCP") > 0) \
            .groupBy("ST") \
            .agg(avg("PINCP").alias("avg_income"))

        # 2. Count unique providers per state (StateAbbr) from internet data
        provider_count_state = internet_df.groupBy("StateAbbr") \
            .agg(countDistinct("ProviderName").alias("provider_count"))

        # 3. Join provider count with the lookup table to get 'ST'
        providers_with_st = provider_count_state.join(state_lookup_df, "StateAbbr")

        # 4. Join the two aggregated datasets (income and provider count) on 'ST'
        q5_result = avg_income_state.join(providers_with_st, "ST") \
            .select("ST", "StateAbbr", "avg_income", "provider_count") \
            .orderBy(desc("avg_income"))

        return q5_result
    
    def correlate_income_inequality_with_digital_divide(self, 
                                                       demographics_df: DataFrame, 
                                                       internet_df: DataFrame, 
                                                       state_lookup_df: DataFrame) -> DataFrame:
        """
        Calculates and correlates income inequality (p90-p10 income) 
        with the digital divide (p90-p10 internet speed) for each state.
        
        This is a complex query involving multiple parallel aggregations and joins.

        :param demographics_df: DataFrame with (PINCP, ST)
        :param internet_df: DataFrame with (MaxAdDown, StateAbbr)
        :param state_lookup_df: DataFrame to join ST and StateAbbr
        :return: DataFrame with [StateAbbr, ST, income_gap, speed_gap]
        """
        
        # --- Step 1: Calculate Income Inequality (from demographics_df) ---
        # We calculate the 90th and 10th percentile income for each state
        income_percentiles = demographics_df.filter(col("PINCP") > 0) \
            .groupBy("ST") \
            .agg(
                expr("approx_percentile(PINCP, 0.90)").alias("p90_income"),
                expr("approx_percentile(PINCP, 0.10)").alias("p10_income")
            ) \
            .withColumn("income_gap", col("p90_income") - col("p10_income"))

        # --- Step 2: Calculate Digital Divide (from internet_df) ---
        # We calculate the 90th and 10th percentile speed for each state
        speed_percentiles = internet_df.filter(col("MaxAdDown") > 0) \
            .groupBy("StateAbbr") \
            .agg(
                expr("approx_percentile(MaxAdDown, 0.90)").alias("p90_speed"),
                expr("approx_percentile(MaxAdDown, 0.10)").alias("p10_speed")
            ) \
            .withColumn("speed_gap", col("p90_speed") - col("p10_speed"))

        # --- Step 3: Join the two results ---

        # Join speed data with lookup to get 'ST'
        speed_gap_with_st = speed_percentiles.join(state_lookup_df, "StateAbbr")

        # Join the income gap data with the speed gap data
        q6_result = income_percentiles.join(speed_gap_with_st, "ST") \
            .select("StateAbbr", "ST", "income_gap", "speed_gap") \
            .orderBy(desc("income_gap"))

        return q6_result