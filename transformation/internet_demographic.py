import pyspark.sql.functions as F
from pyspark.sql.functions import col, avg, when
from pyspark.sql import DataFrame
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

        print("\n Top-3 internet providers by state:")
        top3_df = self.top3_providers_by_state(internet_df=internet_df)
        top3_df.show(30, truncate=False)

        print("\n Gender pay gap by state and internet speed:")
        gender_gap_df = self.gender_pay_gap_by_state_speed(
            demographics_df=demographics_df,
            internet_df=internet_df,
            state_lookup_df=state_lookup_df
        )
        gender_gap_df.show(20, truncate=False)

        print("\n Digital Resilience Index (DRI):")
        dri_df = self.digital_resilience_index(
            demographics_df=demographics_df,
            internet_df=internet_df,
            state_lookup_df=state_lookup_df
        )
        dri_df.show(20, truncate=False)

        print("\n National champion providers:")
        champions_df = self.provider_national_champions(internet_df=internet_df)
        champions_df.show(10, truncate=False)

        print("\n Providers with large sub-threshold pools:")
        subx_df = self.subthreshold_pool_by_provider(internet_df=internet_df)
        subx_df.show(20, truncate=False)

        print("\n States with affordability gaps:")
        afford_df = self.affordability_gap_states(
            internet_df=internet_df,
            demographics_df=demographics_df,
            state_lookup_df=state_lookup_df
        )
        afford_df.show(15, truncate=False)
    
    def top3_providers_by_state(
        self,
        internet_df: DataFrame,
        min_consumer: int = 100,
        min_share: float = 0.0  
    ) -> DataFrame:
        """
        Top-3 провайдери у кожному штаті з найбільшою часткою абонентів.
        Якщо частки рівні — перевага за більшою середньою швидкістю (avg_down_mbps).

        Повертає:
        StateAbbr, ProviderName, avg_down_mbps, consumers, share, rank_in_state, cumulative_share
        """

        # 1) Агрегація на рівні (штат, провайдер)
        provider_stats = (
            internet_df
            .groupBy("StateAbbr", "ProviderName")
            .agg(
                F.avg("MaxAdDown").alias("avg_down_mbps"),
                F.sum("Consumer").alias("consumers")
            )
        )

        # 2) Загальна база штату + розрахунок частки провайдера у штаті
        state_totals = (
            provider_stats
            .groupBy("StateAbbr")
            .agg(F.sum("consumers").alias("state_consumers"))
        )

        with_share = (
            provider_stats
            .join(state_totals, on="StateAbbr", how="inner")
            .withColumn("share", F.col("consumers") / F.col("state_consumers"))
        )

        # 3) Відсікання дрібних провайдерів
        filtered = with_share.filter(F.col("consumers") >= F.lit(min_consumer))
        if min_share > 0.0:
            filtered = filtered.filter(F.col("share") >= F.lit(min_share))

        # 4) Ранг у межах штату: спершу частка ↓, потім швидкість ↓ (tie-breaker)
        w_rank = Window.partitionBy("StateAbbr").orderBy(
            F.col("share").desc(),
            F.col("avg_down_mbps").desc()
        )
        ranked = filtered.withColumn("rank_in_state", F.row_number().over(w_rank))

        # 5) Топ-3 на штат
        top3 = ranked.filter(F.col("rank_in_state") <= 3)

        # 6) Кумулятивна частка у сталому порядку топ-3
        w_cum = (
            Window.partitionBy("StateAbbr")
            .orderBy(F.col("rank_in_state"))
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        result = (
            top3
            .withColumn("cumulative_share", F.sum("share").over(w_cum))
            .select(
                "StateAbbr", "ProviderName", "avg_down_mbps",
                "consumers", "share", "rank_in_state", "cumulative_share"
            )
            .orderBy("StateAbbr", "rank_in_state")
        )

        return result

    def digital_resilience_index(
        self,
        demographics_df: DataFrame,
        internet_df: DataFrame,
        state_lookup_df: DataFrame
    ) -> DataFrame:
        """
        Обчислює Digital Resilience Index (DRI) для кожного штату.
        Враховує середню швидкість, медіанний дохід і рівень освіти.
        """

        # 1) Швидкість інтернету
        speed = (
            internet_df
            .groupBy("StateAbbr")
            .agg(F.avg("MaxAdDown").alias("avg_speed"))
        )

        # 2) Медіанний дохід 25–64
        income = (
            demographics_df
            .filter((F.col("AGEP") >= 25) & (F.col("AGEP") <= 64) & (F.col("WAGP") > 0))
            .join(state_lookup_df, "ST", "inner")
            .groupBy("StateAbbr")
            .agg(F.expr("percentile_approx(WAGP, 0.5)").alias("median_income"))
        )

        # 3) Освіта: частка без бакалавра
        edu = (
            demographics_df
            .filter((F.col("AGEP") >= 25) & (F.col("AGEP") <= 64))
            .join(state_lookup_df, "ST", "inner")
            .groupBy("StateAbbr")
            .agg(F.avg(F.when(F.col("SCHL") < 21, 1).otherwise(0)).alias("low_edu_share"))
        )

        # 4) Об’єднання
        combined = (
            speed.join(income, "StateAbbr", "inner")
                .join(edu, "StateAbbr", "inner")
        )

        w_stats = Window.partitionBy(F.lit(1))

        # 5) Нормалізація (мін-макс) із захистом від нульового діапазону
        normalized = (
            combined
            .withColumn(
                "speed_norm",
                F.when(
                    (F.max("avg_speed").over(w_stats) - F.min("avg_speed").over(w_stats)) == 0, F.lit(0.5)
                ).otherwise(
                    (F.col("avg_speed") - F.min("avg_speed").over(w_stats)) /
                    (F.max("avg_speed").over(w_stats) - F.min("avg_speed").over(w_stats))
                )
            )
            .withColumn(
                "income_norm",
                F.when(
                    (F.max("median_income").over(w_stats) - F.min("median_income").over(w_stats)) == 0, F.lit(0.5)
                ).otherwise(
                    (F.col("median_income") - F.min("median_income").over(w_stats)) /
                    (F.max("median_income").over(w_stats) - F.min("median_income").over(w_stats))
                )
            )
            .withColumn(
                "edu_norm",
                F.when(
                    (F.max("low_edu_share").over(w_stats) - F.min("low_edu_share").over(w_stats)) == 0, F.lit(0.5)
                ).otherwise(
                    (F.col("low_edu_share") - F.min("low_edu_share").over(w_stats)) /
                    (F.max("low_edu_share").over(w_stats) - F.min("low_edu_share").over(w_stats))
                )
            )
            .withColumn(
                "digital_resilience_index",
                0.4 * F.col("speed_norm") + 0.3 * F.col("income_norm") + 0.3 * (1 - F.col("edu_norm"))
            )
        )

        w_rank = Window.partitionBy(F.lit(1)).orderBy(F.col("digital_resilience_index").desc())
        ranked = normalized.withColumn("rank", F.dense_rank().over(w_rank))

        return ranked.select(
            "StateAbbr", "avg_speed", "median_income", "low_edu_share",
            "digital_resilience_index", "rank"
        )

    
    def gender_pay_gap_by_state_speed(
        self,
        demographics_df: DataFrame,
        internet_df: DataFrame,
        state_lookup_df: DataFrame,
        speed_threshold: int = 100,
        min_group_size: int = 500,
        restrict_age_25_64: bool = True
    ) -> DataFrame:
        """
        Гендерний розрив зарплат (Male - Female) по кожному штату з розбивкою на fast/slow.
        Повертає:
        StateAbbr, speed_tier, male_avg, female_avg, gender_gap, n_obs, rank_in_tier
        """

        # 1) Середня швидкість по штату -> категорія fast/slow + ST
        avg_speed_per_state = (
            internet_df
            .groupBy("StateAbbr")
            .agg(F.avg("MaxAdDown").alias("avg_download_speed"))
        )

        speed_tiers = (
            avg_speed_per_state
            .join(state_lookup_df, on="StateAbbr", how="inner")                    
            .withColumn(
                "speed_tier",
                when(col("avg_download_speed") > speed_threshold, "fast_internet")
                .otherwise("slow_internet")
            )
            .select("ST", "StateAbbr", "speed_tier")
        )

        # 2) Демографія: валідні зарплати (та опційно вік 25–64)
        demo = demographics_df.filter(col("WAGP") > 0)                               
        if restrict_age_25_64:
            demo = demo.filter((col("AGEP") >= 25) & (col("AGEP") <= 64))            

        demo = (
            demo
            .filter(col("SEX").isin([1, 2]))                                        
            .select("ST", "SEX", "WAGP")
        )

        # 3) Прив’язка демографії до tier швидкості через ST
        demo_with_speed = demo.join(speed_tiers, on="ST", how="inner")              

        # 4) Середні зарплати по (штат, tier, стать) + кількість спостережень
        per_group = (
            demo_with_speed
            .groupBy("StateAbbr", "speed_tier", "SEX")                               
            .agg(
                F.avg("WAGP").alias("avg_wagp"),
                F.count("*").alias("n_obs")
            )
        )

        # 5) Вікно по (штат, tier): витягуємо чоловічу та жіночу середні та сумарний розмір групи
        w = Window.partitionBy("StateAbbr", "speed_tier")
        with_both = (
            per_group
            .withColumn("male_avg",   F.max(when(col("SEX") == 1, col("avg_wagp"))).over(w))  
            .withColumn("female_avg", F.max(when(col("SEX") == 2, col("avg_wagp"))).over(w))  
            .withColumn("total_n",    F.sum("n_obs").over(w))                                   
        )

        # 6) Один рядок на (штат, tier) + фільтр мінімального обсягу
        collapsed = (
            with_both
            .select("StateAbbr", "speed_tier", "male_avg", "female_avg", "total_n")
            .dropDuplicates(["StateAbbr", "speed_tier"])
            .withColumn("gender_gap", col("male_avg") - col("female_avg"))
            .filter(col("total_n") >= F.lit(min_group_size))                          
        )

        # 7) Ранжування всередині tier за величиною гендерного розриву
        w_rank = Window.partitionBy("speed_tier").orderBy(col("gender_gap").desc())
        ranked = (
            collapsed
            .withColumn("rank_in_tier", F.dense_rank().over(w_rank))                  
            .orderBy("speed_tier", "rank_in_tier")
        )

        return ranked.select("StateAbbr", "speed_tier", "male_avg", "female_avg", "gender_gap", "total_n", "rank_in_tier")
    
    def provider_national_champions(
        self,
        internet_df: DataFrame,
        min_share: float = 0.3
    ) -> DataFrame:
        """
        Бізнес-питання: Які провайдери є «національними чемпіонами» — посідають 1-ше місце у найбільшій кількості штатів (та мають суттєву частку)?
        Вихід: ProviderName, states_as_no1, avg_top_share, states_list
        """
        # (1) Частки провайдерів у кожному штаті
        prov = (internet_df.groupBy("StateAbbr", "ProviderName")
                .agg(F.sum("Consumer").alias("consumers")))
        tot = prov.groupBy("StateAbbr").agg(F.sum("consumers").alias("state_cons"))
        shares = prov.join(tot, "StateAbbr").withColumn("share", F.col("consumers")/F.col("state_cons"))

        # (2) Знаходимо топ-1 провайдера у кожному штаті
        w = Window.partitionBy("StateAbbr").orderBy(F.col("share").desc())
        top1 = (shares
                .withColumn("rk", F.dense_rank().over(w))
                .filter((F.col("rk") == 1) & (F.col("share") >= F.lit(min_share)))  # тільки істотні лідери
                .select("StateAbbr", "ProviderName", "share"))

        # (3) Підсумок по провайдеру: у скількох штатах він #1 і середня частка серед тих перемог
        result = (top1.groupBy("ProviderName")
                .agg(
                    F.countDistinct("StateAbbr").alias("states_as_no1"),
                    F.avg("share").alias("avg_top_share"),
                    F.collect_list("StateAbbr").alias("states_list")
                )
                .orderBy(F.col("states_as_no1").desc(), F.col("avg_top_share").desc()))

        # кого вважати стратегічним директором каналів продажу
        return result

    def subthreshold_pool_by_provider(
        self,
        internet_df: DataFrame,
        speed_threshold: int = 100,
        min_share: float = 0.05
    ) -> DataFrame:
        """
        Бізнес-питання: У яких штатах і в яких провайдерів найбільший пул клієнтів зі швидкістю нижче порога (для таргету апґрейду)?
        Вихід: StateAbbr, ProviderName, subx_consumers, provider_cons, subx_share, state_share, rank_in_state
        """
        # (1) База провайдера у штаті: скільки всього споживачів та скільки з них < порога
        prov = (
            internet_df
            .groupBy("StateAbbr", "ProviderName")
            .agg(
                F.sum("Consumer").alias("provider_cons"),
                F.sum(F.when(F.col("MaxAdDown") < F.lit(speed_threshold), F.col("Consumer")).otherwise(0)).alias("subx_consumers")
            )
        ).filter(F.col("provider_cons") > 0)

        # (2) Частка «саб-порогових» у провайдера та частка провайдера в штаті
        state_tot = prov.groupBy("StateAbbr").agg(F.sum("provider_cons").alias("state_cons"))
        with_shares = (
            prov.join(state_tot, "StateAbbr", "inner")
                .withColumn("subx_share", F.col("subx_consumers")/F.col("provider_cons"))  
                .withColumn("state_share", F.col("provider_cons")/F.col("state_cons"))     
                .filter(F.col("state_share") >= F.lit(min_share))                          
        )

        # (3) Ранг усередині штату за абсолютним пулом для апґрейду
        w = Window.partitionBy("StateAbbr").orderBy(F.col("subx_consumers").desc())
        ranked = (
            with_shares
            .withColumn("rank_in_state", F.dense_rank().over(w))
            .orderBy("StateAbbr", "rank_in_state")
        )

        # → Де саме (штат×провайдер) концентруються клієнти для апсейлу/кампаній заміни тарифу/технології
        return ranked.select("StateAbbr", "ProviderName", "subx_consumers", "provider_cons", "subx_share", "state_share", "rank_in_state")

    def affordability_gap_states(
        self,
        internet_df: DataFrame,
        demographics_df: DataFrame,
        state_lookup_df: DataFrame,
        restrict_age_25_64: bool = True
    ) -> DataFrame:
        """
        У яких штатах середня швидкість інтернету вища за національну медіану,
        але медіанний дохід (25–64) нижчий за національну медіану?
        Вихід: StateAbbr, state_avg_mbps, median_wagp, adult_pop, affordability_gap_score, rank
        """

        # 1) Середня швидкість по штату
        speed = (
            internet_df
            .groupBy("StateAbbr")
            .agg(F.avg("MaxAdDown").alias("state_avg_mbps"))
        )

        # 2) Медіанний дохід 25–64 та чисельність дорослого населення
        demo = demographics_df.filter(F.col("WAGP").isNotNull() & (F.col("WAGP") > 0))
        if restrict_age_25_64:
            demo = demo.filter((F.col("AGEP") >= 25) & (F.col("AGEP") <= 64))

        income = (
            demo.select("ST", "WAGP")
                .join(state_lookup_df, "ST", "inner")
                .groupBy("StateAbbr")
                .agg(
                    F.expr("percentile_approx(WAGP, 0.5)").alias("median_wagp"),
                    F.count("*").alias("adult_pop")
                )
        )

        # 3) Об’єднання та нац. медіани (однією агрегацією)
        base = speed.join(income, "StateAbbr", "inner")
        meds = base.agg(
            F.expr("percentile_approx(state_avg_mbps, 0.5)").alias("sp_med"),
            F.expr("percentile_approx(median_wagp, 0.5)").alias("inc_med")
        ).first()
        sp_med, inc_med = float(meds["sp_med"]), float(meds["inc_med"])

        # 4) Відбір штатів з «ґепом доступності» + скор
        result = (
            base
            .withColumn("is_high_speed", F.col("state_avg_mbps") > F.lit(sp_med))
            .withColumn("is_low_income", F.col("median_wagp")   < F.lit(inc_med))
            .filter(F.col("is_high_speed") & F.col("is_low_income"))
            .withColumn(
                "affordability_gap_score",
                ((F.col("state_avg_mbps") - F.lit(sp_med)) / F.lit(sp_med)) * F.log1p(F.col("adult_pop"))
            )
        )

        # 5) Ранг: додаємо «штучну» партицію, щоб прибрати WARN
        w = Window.partitionBy(F.lit(1)).orderBy(F.col("affordability_gap_score").desc())
        ranked = (
            result
            .withColumn("rank", F.dense_rank().over(w))
            .orderBy("rank")
            .select("StateAbbr", "state_avg_mbps", "median_wagp", "adult_pop",
                    "affordability_gap_score", "rank")
        )

        return ranked