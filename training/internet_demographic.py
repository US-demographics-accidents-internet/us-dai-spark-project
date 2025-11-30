import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType, LongType, DoubleType, FloatType,
    ShortType, DecimalType, StringType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor


class InternetDemographicTraining:

    def __init__(self, experiment_name="internet_demographic_experiment"):

        self.experiment_name = experiment_name

        self.spark = SparkSession.builder \
            .appName("InternetDemographicTraining") \
            .master("local[*]") \
            .getOrCreate()
        
    def is_regression_model(self, algorithm):
        return isinstance(algorithm, (LinearRegression, RandomForestRegressor, GBTRegressor))

    def is_classification_model(self, algorithm):
        return isinstance(algorithm, (LogisticRegression, RandomForestClassifier, GBTClassifier))

    def detect_features(self, df, target_col):
        """
        Automatically detects numeric and categorical features based on Spark schema.
        """
        numeric_types = (
            IntegerType, LongType, DoubleType,
            FloatType, ShortType, DecimalType
        )

        numeric_features = []
        categorical_features = []

        for field in df.schema.fields:
            if field.name == target_col:
                continue  # skip label

            dtype = field.dataType

            if isinstance(dtype, numeric_types):
                numeric_features.append(field.name)

            elif isinstance(dtype, StringType):
                categorical_features.append(field.name)

        print("Auto-detected numeric features:", numeric_features)
        print("Auto-detected categorical features:", categorical_features)

        return numeric_features, categorical_features

    def build_preprocessing_pipeline(self, numeric_features, categorical_features):
        stages = []

        # Encode categorical
        for col in categorical_features:
            stages.append(
                StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_idx",
                    handleInvalid="keep"
                )
            )

        # Assemble features vector
        feature_cols = numeric_features + [f"{c}_idx" for c in categorical_features]

        stages.append(
            VectorAssembler(
                inputCols=feature_cols,
                outputCol="features"
            )
        )

        return stages

    def invoke_training_pipeline(
            self,
            df,
            target_col: str,
            algorithm,                 # e.g. LogisticRegression()
            params: dict = None,
            experiment_name: str = "default_experiment"
    ):
        """
        Executes ML training pipeline with auto-detected features.
        """

        mlflow.set_experiment(experiment_name)

        # Auto-detect features
        numeric_features, categorical_features = self.detect_features(df, target_col)

        # Build preprocessing
        preprocessing_stages = self.build_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features
        )

        # Set model params if any
        if params:
            algorithm = algorithm.setParams(**params)

        # Full pipeline: preprocessing + ML model
        pipeline = Pipeline(stages=preprocessing_stages + [algorithm])

        # Train-test split
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

        # MLflow logging run
        with mlflow.start_run():

            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)

            # -----------------------------
            # CLASSIFICATION vs REGRESSION
            # -----------------------------
            if self.is_classification_model(algorithm):
                evaluator = MulticlassClassificationEvaluator(
                    labelCol=target_col,
                    predictionCol="prediction",
                    metricName="accuracy"
                )
                metric = evaluator.evaluate(predictions)
                mlflow.log_metric("accuracy", metric)

            elif self.is_regression_model(algorithm):
                evaluator = RegressionEvaluator(
                    labelCol=target_col,
                    predictionCol="prediction",
                    metricName="rmse"
                )
                metric = evaluator.evaluate(predictions)
                mlflow.log_metric("rmse", metric)

            else:
                raise Exception("Unsupported model type.")

            print(f"Training finished. Metric: {metric}")

        return model, metric
