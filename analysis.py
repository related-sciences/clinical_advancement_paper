import functools
import logging
import math
import re
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar

import fire
import numpy as np
import pandas as pd
import psutil
from xarray import Dataset
from dotenv import load_dotenv
from IPython.display import display
from lightgbm import LGBMClassifier
from pandas import DataFrame as PandasDataFrame
from pandas import Series as PandasSeries
from pandas.io.formats.style import Styler
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark_gcs import get_gcs_enabled_config
from scipy.stats import wilcoxon
from scipy.stats.contingency import relative_risk
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
import xrml

logger = logging.getLogger(__name__)

###############################################################################
# Configuration
###############################################################################

SEED = 0
FEATURES_PRIMARY_KEY = ["target_id", "disease_id", "year"]
MIN_EVIDENCE_YEAR = 1990
MAX_EVIDENCE_YEAR = 2024
DEFAULT_MAX_TRAINING_TRANSITION_YEAR = 2015
DEFAULT_CLINICAL_ADVANCEMENT_WINDOW = 2
DATASETS = {
    "23.12": (
        default_tables := {
            "evidence": "evidence",
            "targets": "targets",
            "diseases": "diseases",
            "targetPrioritisation": "targetPrioritisation",
            "associationByOverallDirect": "associationByOverallDirect",
            "associationByOverallIndirect": "associationByOverallIndirect",
        }
    ),
    "23.09": default_tables,
    # Note: this typo in "Priorisation" is real for 23.06
    "23.06": {**default_tables, **{"targetPrioritisation": "targetsPriorisation"}},
}
PRIMARY_TABLES = {
    "relative_risk_p_values": {"caption": "P-values by model and limit"},
    "relative_risk_averages_by_ta": {
        "caption": "Average relative risk across therapeutic areas by model and limit"
    },
}


@dataclass
class AnalysisConfig:
    data_dir: str
    open_targets_version: str
    clinical_transition_phase: int
    clinical_advancement_window: int
    validate_modeling_features: bool
    min_training_transition_year: int
    max_training_transition_year: int
    max_training_advancement_year: int
    min_evaluation_transition_year: int
    max_evaluation_advancement_year: int
    max_evaluation_transition_year: int


def get_analysis_config(
    data_dir: str,
    open_targets_version: str,
    max_training_transition_year: int,
    clinical_advancement_window: int,
    clinical_transition_phase: int = 2,
    validate_modeling_features: bool = True,
) -> AnalysisConfig:
    assert isinstance(data_dir, str)
    assert isinstance(open_targets_version, str)
    assert isinstance(max_training_transition_year, int)
    assert isinstance(clinical_advancement_window, int)
    assert isinstance(clinical_transition_phase, int)
    assert isinstance(validate_modeling_features, bool)
    return AnalysisConfig(
        data_dir=data_dir,
        open_targets_version=open_targets_version,
        clinical_transition_phase=clinical_transition_phase,
        clinical_advancement_window=clinical_advancement_window,
        validate_modeling_features=validate_modeling_features,
        min_training_transition_year=MIN_EVIDENCE_YEAR,
        max_training_transition_year=max_training_transition_year,
        max_training_advancement_year=max_training_transition_year
        + clinical_advancement_window,
        min_evaluation_transition_year=max_training_transition_year + 1,
        max_evaluation_advancement_year=MAX_EVIDENCE_YEAR,
        max_evaluation_transition_year=MAX_EVIDENCE_YEAR - clinical_advancement_window,
    )


###############################################################################
# Data
###############################################################################


def load_publication_years(spark: SparkSession) -> SparkDataFrame:
    path = Path(tempfile.gettempdir()) / "pub_years.parquet"
    if not path.exists():
        # Load stable export from https://gist.github.com/eric-czech/44a9054e0d95b2fa9de3d37ba01a7527
        url = "https://github.com/eric-czech/public-data/blob/74586e5aa694bc4f00b715e201d709617b639917/pubmed/pub_years.parquet?raw=true"
        pd.read_parquet(url).to_parquet(path)
    return spark.read.parquet(str(path))


def load_evidence(spark: SparkSession, version: str) -> SparkDataFrame:
    return spark.read.parquet(get_gcs_path(version, "evidence"))


def load_association_scores(spark: SparkSession, version: str) -> SparkDataFrame:
    return (
        spark.read.parquet(get_gcs_path(version, "associationByOverallDirect"))
        .select("targetId", "diseaseId", F.col("score").alias("directScore"))
        .join(
            spark.read.parquet(
                get_gcs_path(version, "associationByOverallIndirect")
            ).select("targetId", "diseaseId", F.col("score").alias("indirectScore")),
            on=["targetId", "diseaseId"],
            how="outer",
        )
        .transform(to_snake_case)
    )


def load_targets(
    spark: SparkSession, version: str, include_prioritisation: bool = True
) -> SparkDataFrame:
    targets = spark.read.parquet(get_gcs_path(version, "targets"))
    if include_prioritisation:
        targets = targets.join(
            load_target_prioritisation(spark, version).withColumnRenamed(
                "targetId", "id"
            ),
            on="id",
            how="left",
        )
    return targets


def load_target_prioritisation(spark: SparkSession, version: str) -> SparkDataFrame:
    return spark.read.parquet(get_gcs_path(version, "targetPrioritisation"))


def load_diseases(spark: SparkSession, version: str) -> SparkDataFrame:
    return spark.read.parquet(get_gcs_path(version, "diseases"))


def get_disease_id_to_name(identifiers: SparkDataFrame) -> PandasSeries:
    return (
        identifiers.filter(F.col("disease_id").isNotNull())  # type: ignore[attr-defined]
        .select(
            "disease_id",
            F.coalesce("disease_name", F.col("disease_id")).alias("disease_name"),
        )
        .distinct()
        .toPandas()
        .set_index("disease_id")["disease_name"]
    )


def get_target_id_to_symbol(identifiers: SparkDataFrame) -> PandasSeries:
    return (
        identifiers.filter(F.col("target_id").isNotNull())  # type: ignore[attr-defined]
        .select(
            "target_id",
            F.coalesce("target_symbol", F.col("target_id")).alias("target_symbol"),
        )
        .distinct()
        .toPandas()
        .set_index("target_id")["target_symbol"]
    )


def get_identifiers(
    targets: SparkDataFrame, diseases: SparkDataFrame, evidence: SparkDataFrame
) -> SparkDataFrame:
    return (
        evidence.select(
            "targetId",
            "diseaseId",
        )
        .distinct()
        .withColumn("inEvidence", F.lit(True))
        .join(
            targets.select(
                F.col("id").alias("targetId"),
                F.col("approvedSymbol").alias("targetSymbol"),
            )
            .distinct()
            .withColumn("inTargets", F.lit(True)),
            on="targetId",
            how="outer",
        )
        .join(
            diseases.select(
                F.col("id").alias("diseaseId"), F.col("name").alias("diseaseName")
            )
            .distinct()
            .withColumn("inDiseases", F.lit(True)),
            on="diseaseId",
            how="outer",
        )
        .transform(to_snake_case)
    )


def get_target_features(targets: SparkDataFrame) -> SparkDataFrame:
    return (
        targets.select(
            F.col("id").alias("targetId"),
            "tissueSpecificity",
            "tissueDistribution",
            "geneticConstraint",
            *(
                [F.col("mouseKOScore").alias("mouseKoScore")]
                if "mouseKOScore" in targets.columns
                else []
            ),
        )
        .transform(to_snake_case)
        .transform(
            spark_lambda(
                lambda df: df.select(
                    "target_id",
                    *[
                        F.col(c).alias(f"target__{c}")
                        for c in df.columns
                        if c != "target_id"
                    ],
                )
            )
        )
    )


def get_target_tractability(targets: SparkDataFrame) -> SparkDataFrame:
    return targets.select("id", F.explode("tractability").alias("tractability")).select(
        F.col("id").alias("target_id"),
        F.col("tractability.id").alias("tractability_id"),
        F.col("tractability.value").alias("tractability_value"),
        F.col("tractability.modality").alias("tractability_modality"),
    )


def get_disease_therapeutic_areas(diseases: SparkDataFrame) -> SparkDataFrame:
    return (
        diseases.select(
            F.col("id").alias("diseaseId"),
            F.explode("therapeuticAreas").alias("therapeuticAreaId"),
        )
        .join(
            diseases.filter(F.col("ontology.isTherapeuticArea")).select(
                F.col("id").alias("therapeuticAreaId"),
                F.col("name").alias("therapeuticAreaName"),
            ),
            on="therapeuticAreaId",
            how="inner",
        )
        .transform(
            spark_lambda(
                lambda df: df.unionByName(
                    df.select(
                        "diseaseId",
                        F.lit("ALL_0").alias("therapeuticAreaId"),
                        F.lit("all").alias("therapeuticAreaName"),
                    ).distinct()
                )
            )
        )
        .select("diseaseId", "therapeuticAreaId", "therapeuticAreaName")
        .transform(to_snake_case)
    )


def get_clinical_evidence(evidence: SparkDataFrame) -> SparkDataFrame:
    return (
        evidence.filter(F.col("datasourceId") == "chembl")
        .filter(F.col("clinicalPhase").isNotNull())
        .select(
            "targetId",
            "diseaseId",
            "studyStartDate",
            "clinicalPhase",
        )
    )


def get_clinical_stages(clinical_evidence: SparkDataFrame) -> SparkDataFrame:
    return (
        clinical_evidence
        # Ignore phase 0 trials
        .filter(F.col("clinicalPhase") >= 1)
        .withColumn("year", F.year("studyStartDate"))
        # Implicitly ignore untemporalized approvals since they can't be used
        .filter(F.col("year").isNotNull())
        .transform(to_clamped_year)
        .groupby("targetId", "diseaseId", "year")
        .agg(
            F.max("clinicalPhase").cast("int").alias("clinicalPhaseMaxReached"),
            *[
                F.max(F.when(F.col("clinicalPhase") == phase, F.lit(1))).alias(
                    f"clinicalPhase{phase}Reached"
                )
                for phase in [1, 2, 3, 4]
            ],
        )
    )


def augment_evidence(evidence: SparkDataFrame) -> SparkDataFrame:
    return evidence.unionByName(
        # Add a genetic evidence group corresponding to EVA associations with linked
        # publications, which is effectively only OMIM submissions
        evidence.filter(F.col("datatypeId") == "genetic_association")
        .filter(F.col("datasourceId") == "eva")
        .filter(F.col("literature").isNotNull() & (F.size(F.col("literature")) > 0))
        .withColumn("datasourceId", F.lit("omim"))
    ).unionByName(
        # Add an aggregate genetic evidence data source as a combination of sources
        # that are curated, at least in part, to improve power for that kind of association;
        # notable omissions from this are 'gene_burden' and 'ot_genetics_portal'
        evidence.filter(F.col("datatypeId") == "genetic_association")
        .filter(
            F.col("datasourceId").isin(
                [
                    "genomics_england",
                    "orphanet",
                    "gene2phenotype",
                    "uniprot_literature",
                    "uniprot_variants",
                    "clingen",
                    "eva",
                ]
            )
        )
        .withColumn("datasourceId", F.lit("curated"))
    )


def temporalize_evidence(
    evidence: SparkDataFrame, pub_years: SparkDataFrame
) -> SparkDataFrame:
    primary_key = ["datatypeId", "datasourceId", "targetId", "diseaseId"]
    return (
        (
            evidence.filter(
                F.col("literature").isNotNull() & (F.size(F.col("literature")) > 0)
            )
            .select(*primary_key, "score", F.explode("literature").alias("pmid"))
            .withColumn("pmid", F.col("pmid").cast("long"))
            .join(
                pub_years.withColumnRenamed("pub_year", "year"), on="pmid", how="inner"
            )
            .drop("pmid")
        )
        .unionByName(
            evidence.filter(F.col("publicationYear").isNotNull()).select(
                *primary_key, "score", F.col("publicationYear").alias("year")
            )
        )
        .transform(to_clamped_year)
        .groupby(*primary_key, "year")
        .agg(F.max("score").alias("score"))
        .unionByName(
            # Add time-independent max scores as well
            evidence.groupby(*primary_key)
            .agg(F.max("score").alias("score"))
            .withColumn("year", F.lit(None).cast("int"))
        )
    )


def to_null_safe_year(df: SparkDataFrame) -> SparkDataFrame:
    return df.withColumn("year", F.coalesce(F.col("year"), F.lit(-1)))


def from_null_safe_year(df: SparkDataFrame) -> SparkDataFrame:
    return df.withColumn(
        "year",
        F.when(F.col("year") == -1, F.lit(None).cast("int")).otherwise(F.col("year")),
    )


def to_clamped_year(df: SparkDataFrame) -> SparkDataFrame:
    return df.withColumn(
        "year",
        F.when(F.col("year") < MIN_EVIDENCE_YEAR, F.lit(MIN_EVIDENCE_YEAR - 1))
        .when(F.col("year") > MAX_EVIDENCE_YEAR, F.lit(MAX_EVIDENCE_YEAR + 1))
        .otherwise(F.col("year")),
    )


def get_evidence_features(
    temporalized_evidence: SparkDataFrame, clinical_stages: SparkDataFrame
) -> tuple[SparkDataFrame, SparkDataFrame]:
    phase_column = "target_disease__clinical__phase_max__reached"
    features = (
        clinical_stages.transform(to_snake_case)
        .transform(
            spark_lambda(
                lambda df: df.select(
                    *[
                        F.col(c).alias(
                            "target_disease__"
                            + re.sub("_(phase(\\d|_max))_", r"__\1__", c)
                        )
                        if c.startswith("clinical")
                        else F.col(c)
                        for c in df.columns
                    ]
                )
            )
        )
        .transform(to_null_safe_year)
    ).join(
        temporalized_evidence.withColumn(
            "feature",
            F.concat_ws(
                "__",
                F.lit("target_disease"),
                F.col("datatypeId"),
                F.col("datasourceId"),
            ),
        )
        .groupby("targetId", "diseaseId", "year")
        .pivot("feature")
        .agg(F.max("score"))
        .transform(to_snake_case)
        .transform(to_null_safe_year),
        on=FEATURES_PRIMARY_KEY,
        how="outer",
    )

    temporal_features = (
        features.join(
            features.filter(F.col(phase_column).isNotNull())
            .select("target_id", "disease_id")
            .distinct(),
            on=["target_id", "disease_id"],
            how="semi",
        )
        .transform(
            spark_lambda(
                lambda df: df.join(
                    df.groupby(["target_id", "year"])
                    .agg(
                        F.max(phase_column).alias(
                            "target__clinical__phase_max__reached"
                        )
                    )
                    .crossJoin(df.select("disease_id").drop_duplicates()),
                    on=FEATURES_PRIMARY_KEY,
                    how="outer",
                ).join(
                    df.groupby(["disease_id", "year"])
                    .agg(
                        F.max(phase_column).alias(
                            "disease__clinical__phase_max__reached"
                        )
                    )
                    .crossJoin(df.select("target_id").drop_duplicates()),
                    on=FEATURES_PRIMARY_KEY,
                    how="outer",
                )
            )
        )
        .transform(from_null_safe_year)
        .transform(
            spark_lambda(
                lambda df: df.select(
                    *FEATURES_PRIMARY_KEY,
                    *[
                        F.max(F.col(c))
                        .over(
                            Window.partitionBy("target_id", "disease_id")
                            .orderBy(F.asc_nulls_last("year"))
                            .rowsBetween(Window.unboundedPreceding, 0)
                        )
                        .alias(c)
                        for c in df.columns
                        if c not in FEATURES_PRIMARY_KEY
                    ],
                )
            )
        )
    )

    static_features = (
        features.withColumn("year", F.lit(None).cast("int"))
        .transform(
            spark_lambda(
                lambda df: df.groupby(*FEATURES_PRIMARY_KEY).agg(
                    *[
                        F.max(F.col(c)).alias(c)
                        for c in df.columns
                        if c not in FEATURES_PRIMARY_KEY
                    ]
                )
            )
        )
        .join(
            temporal_features.groupby("target_id").agg(
                F.max("target__clinical__phase_max__reached").alias(
                    "target__clinical__phase_max__reached"
                )
            ),
            on="target_id",
            how="left",
        )
        .join(
            temporal_features.groupby("disease_id").agg(
                F.max("disease__clinical__phase_max__reached").alias(
                    "disease__clinical__phase_max__reached"
                )
            ),
            on="disease_id",
            how="left",
        )
        .select(*temporal_features.columns)
    )

    return temporal_features, static_features


def get_aggregated_features(
    features: SparkDataFrame, diseases: SparkDataFrame
) -> SparkDataFrame:
    indirect_feature_names = [
        c
        for c in features.columns
        if c.startswith("target_disease__") and "__clinical__" not in c
    ]
    direct_feature_names = [
        c
        for c in features.columns
        if c not in indirect_feature_names + FEATURES_PRIMARY_KEY
    ]
    logger.info(
        f"Aggregating indirect evidence for the following features: {indirect_feature_names}"
    )
    aggregated_features = (
        (base_features := features.transform(to_null_safe_year))
        .select(*FEATURES_PRIMARY_KEY, *direct_feature_names)
        .join(
            base_features.select(*FEATURES_PRIMARY_KEY, *indirect_feature_names)
            .withColumnRenamed("disease_id", "disease_id_src")
            .join(
                diseases.select("id", "ancestors")
                .withColumn(
                    "ancestors", F.array_union(F.array("id"), F.col("ancestors"))
                )
                .select(
                    F.col("id").alias("disease_id_dst"),
                    F.explode("ancestors").alias("disease_id_src"),
                ),
                on="disease_id_src",
                how="inner",
            )
            .withColumnRenamed("disease_id_dst", "disease_id")
            .drop("disease_id_src")
            .groupby("target_id", "disease_id", "year")
            .agg(*[F.max(c).alias(c) for c in indirect_feature_names]),
            on=FEATURES_PRIMARY_KEY,
            how="inner",
        )
        .transform(from_null_safe_year)
        .select(*features.columns)
    )
    return aggregated_features


def parquet_writer(
    spark: SparkSession, output_dir: Path
) -> Callable[[SparkDataFrame, str], tuple[SparkDataFrame, Path]]:
    def fn(df: SparkDataFrame, table: str) -> tuple[SparkDataFrame, Path]:
        path = output_dir / f"{table}.parquet"
        logger.info(f"Saving {table!r} to {path!r})")
        df.printSchema()
        df.write.parquet(str(path), mode="overwrite")
        df = spark.read.parquet(str(path))
        return df, path

    return fn


def aggregate_features(output_path: str, version: str) -> None:
    logger.info(f"Aggregating features for version {version!r} to {output_path!r}")
    spark = get_spark()
    output_dir = Path(output_path) / "features" / str(version)
    write_parquet = parquet_writer(spark, output_dir)

    features = spark.read.parquet(str(output_dir / "features.parquet"))
    diseases = load_diseases(spark, version=version)
    aggregated_features = get_aggregated_features(features=features, diseases=diseases)
    aggregated_features, _ = write_parquet(aggregated_features, "aggregated_features")
    assert aggregated_features.count() == features.count()
    logger.info("Feature aggregation complete")


def get_feature_info(
    feature_names: list[str], raise_on_uknown: bool = False
) -> PandasDataFrame:
    result = []
    for feature in feature_names:
        entity = feature.split("__")[0]
        if entity not in {"target", "disease", "target_disease"}:
            if raise_on_uknown:
                raise ValueError(f"Unknown entity: {entity}")
            else:
                continue
        kind = "temporal"
        if entity in {"target", "disease"} and "__clinical__" not in feature:
            kind = "static"
        elif entity == "target_disease" and feature.startswith(
            "target_disease__genetic_association__"
        ):
            kind = "static"
        result.append({"feature": feature, "entity": entity, "kind": kind})
    return pd.DataFrame(result)


def validate_features(features: SparkDataFrame, has_timeseries: bool = True) -> None:
    feature_info = get_feature_info(
        [c for c in features.columns if c not in FEATURES_PRIMARY_KEY]
    )
    feature_groups = (
        feature_info.groupby(["entity", "kind"])["feature"]
        .unique()
        .apply(list)
        .to_dict()
    )

    def _distinct_counts(
        df: SparkDataFrame, columns: list[str], by: list[str]
    ) -> PandasSeries:
        return (
            df.groupby(*by)  # type: ignore[attr-defined]
            .agg(
                *[
                    (
                        F.count_distinct(F.col(c))
                        + F.max(F.col(c).isNull()).cast("int")
                    ).alias(c)
                    for c in columns
                ]
            )
            .agg(*[F.max(c).alias(c) for c in columns])
            .toPandas()
            .iloc[0]
        )

    def _is_monotonic_increasing(
        df: SparkDataFrame, columns: list[str], by: list[str]
    ) -> PandasSeries:
        return (
            df.select(  # type: ignore[attr-defined]
                *[
                    F.when(
                        F.col(c)
                        < F.lag(F.col(c), offset=1).over(
                            Window.partitionBy(*by).orderBy(F.asc_nulls_last("year"))
                        ),
                        1,
                    )
                    .otherwise(0)
                    .alias(c)
                    for c in columns
                ]
            )
            .agg(*[(F.max(F.col(c)) <= 0).alias(c) for c in columns])
            .toPandas()
            .iloc[0]
        )

    def _validate_static_features(entity: str) -> None:
        names = feature_groups.get((entity, "static"), [])
        logger.info(f"Validating static features for entity {entity!r}: {names}")
        if len(names) == 0:
            return
        counts = _distinct_counts(features, names, by=[f"{entity}_id"])
        for feature in counts.index:
            if (count := counts.loc[feature]) > 1:
                raise ValueError(
                    f"Found {count} distinct values for static {entity} feature {feature!r}"
                )

    def _validate_temporal_features(entity: str) -> None:
        names = feature_groups.get((entity, "temporal"), [])
        logger.info(f"Validating temporal features for entity {entity!r}: {names}")
        if len(names) == 0:
            return
        # Ensure that there is more than one unique value for the feature
        # across years if it is supposed to differ in time
        if has_timeseries:
            counts = _distinct_counts(features, names, by=["year"])
            for feature in counts.index:
                if (count := counts.loc[feature]) <= 1:
                    raise ValueError(
                        f"Found only {count} distinct values across years for "
                        f"temporal {entity} feature {feature!r}"
                    )

        # Ensure that all temporal features only increase or stay the same in time
        by = [f"{entity}_id"]
        if entity == "target_disease":
            by = ["target_id", "disease_id"]
        if has_timeseries:
            is_monotonic = _is_monotonic_increasing(features, names, by=by)
            if (~is_monotonic).any():
                invalid = is_monotonic[~is_monotonic].index.tolist()
                raise ValueError(
                    f"Found non-monotonic temporal features for entity {entity!r}: {invalid}"
                )

        # Ensure that there is only one value for a given (target|disease) + year
        # combination for each feature; e.g. the same target + year combination should
        # always have the same value regardless of the associated disease
        if len(by) == 1:
            counts = _distinct_counts(features, names, by=by + ["year"])
            for feature in counts.index:
                if (count := counts.loc[feature]) > 1:
                    raise ValueError(
                        f"Found {count} distinct values across years for temporal {entity} feature {feature!r}"
                    )

    for entity in ["target", "disease"]:
        _validate_static_features(entity)
    for entity in ["target", "disease", "target_disease"]:
        _validate_temporal_features(entity)


def export_features(output_path: str, version: str | float) -> None:
    logger.info(f"Exporting features for version {version!r} to {output_path!r}")
    spark = get_spark()
    if isinstance(version, float):
        version = str(version)
    output_dir = Path(output_path) / "features" / version
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    write_parquet = parquet_writer(spark, output_dir)

    publication_years = load_publication_years(spark)
    targets = load_targets(spark, version=version)
    diseases = load_diseases(spark, version=version)
    evidence = load_evidence(spark, version=version)

    write_parquet(get_target_tractability(targets), "tractability")
    write_parquet(get_disease_therapeutic_areas(diseases), "therapeutic_areas")
    clinical_evidence = get_clinical_evidence(evidence)
    clinical_stages = get_clinical_stages(clinical_evidence)
    temporalized_evidence = temporalize_evidence(
        augment_evidence(evidence), publication_years
    )

    clinical_stages, _ = write_parquet(clinical_stages, "clinical_stages")
    temporalized_evidence, _ = write_parquet(
        temporalized_evidence, "temporalized_evidence"
    )
    identifiers = get_identifiers(targets, diseases, temporalized_evidence)
    identifiers, _ = write_parquet(identifiers, "identifiers")

    temporal_features, static_features = get_evidence_features(
        temporalized_evidence, clinical_stages
    )
    temporal_features, temporal_features_path = write_parquet(
        temporal_features.join(
            get_target_features(targets), on="target_id", how="left"
        ),
        "temporal_features",
    )
    static_features, static_features_path = write_parquet(
        static_features.join(get_target_features(targets), on="target_id", how="left"),
        "static_features",
    )

    def _validate_features(df: SparkDataFrame, path: Path) -> None:
        logger.info(f"Validating features at {str(path)!r}")
        try:
            validate_features(df)
        except:
            logger.error(f"Feature validation failed; review at {str(path)!r}")
            raise

    _validate_features(temporal_features, temporal_features_path)
    _validate_features(static_features, static_features_path)

    logger.info("Feature export complete")


###############################################################################
# Summaries
###############################################################################


def get_feature_advancement_statistics(
    features: SparkDataFrame,
    initial_phases: list[int],
    min_transition_year: int,
    max_transition_year: int,
) -> SparkDataFrame:
    return (
        features.filter(F.col("year").isNotNull())
        .withColumn(
            "target_disease__clinical__early_phase__reached",
            F.greatest(
                F.lit(0),
                *[
                    F.coalesce(
                        f"target_disease__clinical__phase{phase}__reached", F.lit(0)
                    )
                    for phase in initial_phases
                ],
            ),
        )
        .transform(
            spark_lambda(
                lambda df: (
                    df.groupby("target_id", "disease_id").agg(
                        F.min(
                            F.when(
                                F.col("target_disease__clinical__early_phase__reached")
                                > 0,
                                F.col("year"),
                            )
                        ).alias("phase__reached__first_year"),
                        F.min(
                            F.when(
                                (
                                    F.col(
                                        "target_disease__clinical__phase_max__reached"
                                    )
                                    > max(initial_phases)
                                )
                                & (
                                    F.col(
                                        "target_disease__clinical__early_phase__reached"
                                    )
                                    > 0
                                ),
                                F.col("year"),
                            )
                        ).alias("phase__surpassed__first_year"),
                        F.array(
                            *[
                                F.struct(
                                    F.min(F.when(F.col(c) > 0, F.col("year"))).alias(
                                        "first_year"
                                    ),
                                    F.lit("__".join(c.split("__")[1:3])).alias("name"),
                                )
                                for c in df.columns
                                if c.startswith("target_disease__")
                                and "__clinical__" not in c
                            ]
                        ).alias("features"),
                    )
                )
            )
        )
        .filter(
            F.col("phase__reached__first_year").between(
                min_transition_year, max_transition_year
            )
        )
        .select("*", F.explode("features").alias("feature"))
        .drop("features")
        .select(
            "*",
            F.col("feature.name").alias("feature_name"),
            F.col("feature.first_year").alias("feature_first_year"),
        )
        .drop("feature")
        .withColumn(
            "emerged",
            F.when(
                F.col("feature_first_year").isNull()
                | F.col("phase__reached__first_year").isNull(),
                F.lit("none"),
            )
            .when(
                F.col("feature_first_year") < F.col("phase__reached__first_year"),
                F.lit("before"),
            )
            .when(
                F.col("feature_first_year") >= F.col("phase__reached__first_year"),
                F.lit("after"),
            ),
        )
        .withColumn(
            "progress",
            F.when(F.col("phase__reached__first_year").isNull(), F.lit("none"))
            .when(F.col("phase__surpassed__first_year").isNull(), F.lit("stalled"))
            .when(
                F.col("phase__surpassed__first_year")
                >= F.col("phase__reached__first_year"),
                F.lit("advanced"),
            ),
        )
        .withColumn("pair", F.struct(F.col("target_id"), F.col("disease_id")))
        .groupby(
            "feature_name",
            "emerged",
            "progress",
        )
        .agg(F.count_distinct("pair").alias("n_pairs"))
    )


def get_feature_statistics(
    features: PandasDataFrame, therapeutic_areas: PandasDataFrame
) -> PandasDataFrame:
    all_features = features.filter(regex="^target_disease__").columns.tolist()
    evidence_features = features.filter(
        regex="^target_disease__(?!time__|clinical__|outcome__)"
    ).columns.tolist()
    non_evidence_features = list(set(all_features) - set(evidence_features))
    logger.info(
        "Target-disease features partitioned into:\n"
        f"evidence features={evidence_features}\n"
        f"non-evidence features={non_evidence_features}"
    )
    feature_statistics = (
        features.assign(
            pair=lambda df: df.apply(
                lambda r: (r["target_id"], r["disease_id"]), axis=1
            )
        )
        .assign(
            has_target_disease_evidence=lambda df: (
                (df[evidence_features] > 0).any(axis=1)
            )
        )
        .merge(
            therapeutic_areas,
            on="disease_id",
            how="left",
        )
        .pipe(assert_condition, lambda df: df["therapeutic_area_id"].notnull().all())
        .pipe(
            assert_condition,
            lambda df: not df[
                ["therapeutic_area_name", "split", "target_id", "disease_id"]
            ]
            .duplicated()
            .any(),
        )
        .groupby(["therapeutic_area_name", "split"])
        .agg(
            n_targets=("target_id", "nunique"),
            n_diseases=("disease_id", "nunique"),
            n_pairs=("pair", "nunique"),
            min_year=("transition_year", "min"),
            max_year=("transition_year", "max"),
            balance=("target_disease__outcome__advanced", "mean"),
            n_pairs_with_evidence=("has_target_disease_evidence", "sum"),
            fraction_pairs_with_evidence=("has_target_disease_evidence", "mean"),
        )
        .rename_axis("statistic", axis="columns")
    )
    return feature_statistics


def get_feature_presence(features: PandasDataFrame) -> PandasDataFrame:
    def get_feature_group_label(fg: str) -> str:
        entity, group = fg.split("__")
        if entity == "target_disease":
            entity = "pair"
        if "phase_2" in group:
            group = "prior phase 2+ trials"
        elif group == "outcome":
            group = "advanced beyond phase 2"
        elif group == "no_data":
            group = "no evidence"
        else:
            group = f"{group} evidence"
        return f"{entity} has {group.replace('_', ' ')}"

    # fmt: off
    return (
        features
        .assign(target_disease_id=lambda df: df[['target_id', 'disease_id']].apply(tuple, axis=1))
        .set_index('target_disease_id')
        .assign(
            target__phase_2=lambda df: df["target__clinical__phase_max__reached"] > 2,
            disease__phase_2=lambda df: df["disease__clinical__phase_max__reached"] > 2,
        )
        .filter(regex='^target__phase|^disease__phase|^target_disease__(?!clinical|time)')
        .assign(target_disease__no_data=lambda df: (
            df.filter(regex='^target_disease__(?!outcome)')
            .pipe(lambda df: df.fillna(0) <= 0)
            .all(axis=1)
        ))
        .pipe(lambda df: df.where(df > 0))
        .pipe(apply, lambda df: logger.info(
            f"The following features will be used to calculate presence: {df.columns.tolist()}"
        ))
        .rename_axis("feature", axis="columns")
        .stack().rename("value").reset_index()
        .assign(feature_group=lambda df: df['feature'].str.split('__').str[:2].str.join("__"))
        .groupby("feature_group")["target_disease_id"].unique()
        .rename("pairs").reset_index()
        .assign(n_pairs=lambda df: df['pairs'].apply(len))
        .assign(feature_group_label=lambda df: df['feature_group'].map(get_feature_group_label))
        .sort_values("n_pairs", ascending=True)
        [["feature_group", "feature_group_label", "n_pairs", "pairs"]]
    )
    # fmt: on


def get_feature_relative_risk(
    ds: Dataset,
    model_names: list[str] | None = None,
    model_limits: list[int] | None = None,
    confidence: float = 0.9,
) -> PandasDataFrame:
    if ds.dims["index"] == 0:
        raise ValueError(f"No rows present; shape = {ds.dims}")

    def _compute_relative_risk(g: PandasDataFrame) -> PandasSeries:
        mask = g["value"].notnull()
        n_exposed = mask.sum()
        n_exposed_true = g[mask]["outcome"].sum()
        n_control = (~mask).sum()
        n_control_true = g[~mask]["outcome"].sum()
        if n_exposed == 0 or n_control == 0:
            return pd.Series(dtype=float)
        rr = relative_risk(n_exposed_true, n_exposed, n_control_true, n_control)
        return pd.Series(
            {
                "fraction_exposed": n_exposed_true / n_exposed,
                "fraction_control": n_control_true / n_control,
                "n_exposed": n_exposed,
                "n_control": n_control,
                "n_total": n_exposed + n_control,
                "n_exposed_true": n_exposed_true,
                "relative_risk": rr.relative_risk,
                "relative_risk_low": rr.confidence_interval(
                    confidence_level=confidence
                ).low,
                "relative_risk_high": rr.confidence_interval(
                    confidence_level=confidence
                ).high,
            }
        )

    def _binarize_feature(
        values: PandasSeries,
        limits: list[int] | None = None,
        quantiles: list[float] | None = None,
        mode: Literal["values", "quantiles", "limits"] | None = None,
        values_condition: Literal["eq", "gte"] = "gte",
    ) -> PandasDataFrame:
        feature = values.name
        entity_values = values
        if feature.startswith("target__"):
            entity_values = values.groupby("target_id").max()
        elif feature.startswith("disease__"):
            entity_values = values.groupby("disease_id").max()
        if mode is None:
            mode = "quantiles" if entity_values.nunique() > 10 else "values"

        if mode == "quantiles":
            if quantiles is None:
                quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            return pd.concat(
                [
                    values.where(values >= entity_values.quantile(quantile)).rename(
                        f"{feature}__qtl_{int(100*quantile):03d}"
                    )
                    for quantile in quantiles
                ],
                axis=1,
            )
        elif mode == "limits":
            if limits is None:
                limits = [10, 30, 50, 100, 250, 500, 1000]
            return pd.concat(
                [
                    values.where(
                        values
                        >= entity_values.sort_values(ascending=False).head(limit).min()
                    ).rename(f"{feature}__limit_{limit:04d}")
                    for limit in limits
                ],
                axis=1,
            )
        else:
            is_int = (entity_values.dropna() == entity_values.dropna().round()).all()
            return pd.concat(
                [
                    values.where(
                        (values == threshold)
                        if values_condition == "eq"
                        else (values >= threshold)
                    ).rename(
                        f"{feature}__{values_condition}_{str(int(threshold) if is_int else threshold)}"
                    )
                    for threshold in sorted(entity_values.dropna().unique())
                ],
                axis=1,
            )

    features = ds.feature.to_series().unstack()
    features = pd.concat(
        [
            # Collapse 0 scores to NaN
            features.filter(regex="target_disease__").pipe(lambda df: df.where(df > 0)),
            *[
                _binarize_feature(features[c], mode="values", values_condition="eq")
                for c in features.filter(regex="^target_disease__time__")
            ],
            *[
                _binarize_feature(features[c], mode="limits", limits=[100, 250, 500])
                for c in features.filter(regex="^target_disease__literature__")
            ],
            *[
                _binarize_feature(features[c])
                for c in features.filter(regex="^target__|^disease__")
            ],
        ],
        axis=1,
    )
    predictions = (
        ds.prediction.sel(classes="positive")
        .squeeze()
        .to_series()
        .unstack("models")[
            model_names
            or [model for model in ds.models.values if not model.startswith("baseline")]
        ]
        .add_prefix("target_disease__model__")
    )
    predictions = pd.concat(
        [
            (v := predictions[c])
            .where(v >= v.sort_values(ascending=False).head(limit).min())
            .rename(f"{c}__limit_{limit:04d}")
            for c in predictions
            for limit in model_limits or [10, 25, 50, 75, 100, 250, 500]
        ],
        axis=1,
    )
    outcome = ds.outcome.squeeze().to_series()
    assert outcome.notnull().all()
    assert ds.dims["index"] == len(features) == len(predictions) == len(outcome)

    return (
        pd.concat([features, predictions, outcome], axis=1)
        .set_index(["outcome"], append=True)
        .rename_axis("method", axis="columns")
        .stack(dropna=False)
        .rename("value")
        .reset_index()
        .groupby("method")
        .apply(_compute_relative_risk)
        .unstack()
        .sort_values("relative_risk_low", ascending=False)
        .reset_index()
        .pipe(assert_condition, lambda df: not df["method"].duplicated().any())
    )


def enrich_feature_relative_risk(
    relative_risk: PandasDataFrame,
    primary_models: PandasDataFrame,
    primary_benchmarks: PandasDataFrame,
) -> PandasDataFrame:
    def get_model_name(method: str) -> str | None:
        m = re.findall(r"^target_disease__model__(.+)__limit_\d+$", method)
        assert len(m) < 2
        return m[0] if m else None

    def get_limit(method: str) -> int | None:
        m = re.findall(r".*__limit_(\d+)$", method)
        assert len(m) < 2
        return int(m[0]) if m else None

    return (
        relative_risk
        # Add model-specific properties
        .assign(model_name=lambda df: df["method"].apply(get_model_name))
        .assign(
            model_slug=lambda df: df["model_name"].map(
                primary_models.set_index("model")["model_slug"]
            )
        )
        .assign(is_model_feature=lambda df: df["model_name"].notnull())
        .assign(
            is_primary_model_feature=lambda df: df["model_name"].isin(
                primary_models["model"]
            )
        )
        # Add benchmark-specific properties
        .pipe(
            lambda df: df.assign(
                **{
                    c[0]: df["method"].map(
                        primary_benchmarks.set_index("feature")[c[1]]
                    )
                    for c in [
                        ("benchmark_name", "benchmark"),
                        ("benchmark_slug", "benchmark_slug"),
                        ("benchmark_disp", "benchmark_disp"),
                    ]
                }
            )
        )
        .assign(is_primary_benchmark_feature=lambda df: df["benchmark_name"].notnull())
        # Add generic properties
        .assign(limit=lambda df: df["method"].apply(get_limit).astype("Int64"))
        .pipe(
            assert_condition,
            lambda df: not df[
                ["is_primary_model_feature", "is_primary_benchmark_feature"]
            ]
            .all(axis=1)
            .any(),
        )
    )


def get_classification_metrics_by_therapeutic_area(
    ds: Dataset, therapeutic_areas: PandasDataFrame
) -> PandasDataFrame:
    def _metrics(
        ds: Dataset, therapeutic_area_id: str, therapeutic_area_name: str
    ) -> PandasDataFrame:
        disease_ids = (
            therapeutic_areas.pipe(
                lambda df: df[df["therapeutic_area_id"] == therapeutic_area_id]
            )["disease_id"]
            .drop_duplicates()
            .to_list()
        )
        ds = ds.sel(index=ds.disease_id.isin(disease_ids).values)
        if ds.dims["index"] == 0:
            return pd.DataFrame()
        if ds.outcome.to_series().nunique() <= 1:
            logger.info(
                f"Skipping therapeutic area {therapeutic_area_name!r} with only one outcome"
            )
            return pd.DataFrame()
        ks = list(np.arange(10, 100, 10)) + [150, 200, 250, 300, 400, 500, 1000]
        return (
            ds.pipe(xrml.score_classification, k=ks).score.to_series().unstack("scores")
        )

    return pd.concat(
        [
            _metrics(ds, **r.to_dict())
            .assign(**r.to_dict())
            .set_index(r.index.to_list(), append=True)
            for _, r in (
                therapeutic_areas[["therapeutic_area_id", "therapeutic_area_name"]]
                .drop_duplicates()
                .iterrows()
            )
        ],
        axis=0,
        ignore_index=False,
    )


def get_relative_risk_thresholds(
    relative_risk: PandasDataFrame, ds: Dataset
) -> PandasDataFrame:
    models = (
        relative_risk.pipe(lambda df: df[df["is_primary_model_feature"]])["model_name"]
        .unique()
        .tolist()
    )
    predictions = (
        ds.prediction.sel(classes="positive", models=models)
        .squeeze()
        .to_series()
        .reset_index()
        .rename(columns={"models": "model"})
    )

    def get_score_threshold(row: PandasSeries) -> float:
        model, limit = row["model"], row["limit"]
        if pd.isnull(limit):
            return float("nan")
        limit = int(limit)
        assert limit > 0
        values = predictions.pipe(lambda df: df[df["model"] == model])[
            "prediction"
        ].sort_values(ascending=False)
        if len(values) == 0:
            raise ValueError(f"No predictions found for model {model!r}")
        if limit > len(values):
            return float("nan")
        return float(values.head(limit).values[-1])

    # fmt: off
    model_relative_risk = (
        relative_risk.pipe(lambda df: df[df["is_primary_model_feature"]])[
            ["model_name", "limit", "relative_risk", "n_exposed"]
        ]
        .pipe(assert_condition, lambda df: df["limit"].notnull().all())
        .assign(limit=lambda df: df["limit"].astype(int))
        .pipe(assert_condition, lambda df: not df[["model_name", "limit"]].duplicated().any(),)
        .rename(
            columns={
                "model_name": "model",
                "relative_risk": "model_relative_risk",
                "n_exposed": "model_n_pairs",
            }
        )
    )
    benchmark_relative_risk = (
        relative_risk.pipe(lambda df: df[df["is_primary_benchmark_feature"]])[
            ["benchmark_name", "benchmark_slug", "relative_risk", "n_exposed"]
        ]
        .pipe(
            assert_condition, lambda df: not df[["benchmark_name"]].duplicated().any()
        )
        .rename(
            columns={
                "relative_risk": "benchmark_relative_risk",
                "n_exposed": "benchmark_n_pairs",
            }
        )
    )

    thresholds = (
        pd.merge(
            model_relative_risk,
            benchmark_relative_risk,
            how="cross",
        )
        .pipe(
            lambda df: pd.concat([
                    df.pipe(lambda df: df[df["model_relative_risk"] > df["benchmark_relative_risk"]])
                    .groupby(
                        key := ["model", "benchmark_name", "benchmark_slug", "benchmark_relative_risk", "benchmark_n_pairs",]
                    )["limit"]
                    .max().rename("limit_lower_bound"),
                    df.pipe(lambda df: df[df["model_relative_risk"] <= df["benchmark_relative_risk"]])
                    .groupby(key)["limit"]
                    .min()
                    .rename("limit_upper_bound"),
                ],
                axis=1,
            )
        )
        .assign(
            limit=lambda df: df[["limit_lower_bound", "limit_upper_bound"]].mean(
                axis=1, skipna=False
            )
        )
        .reset_index()
        .assign(
            model_prediction_threshold=lambda df: df.apply(get_score_threshold, axis=1)
        )
    )
    # fmt: on
    return thresholds


def get_relative_risk_by_therapuetic_area(
    ds: Dataset,
    therapeutic_areas: PandasDataFrame,
    model_names: list[str],
    model_limits: list[int] | None = None,
    min_rows: int = 2,
) -> PandasDataFrame:
    dfs = []
    for (ta_id, ta_name), ta_diseases in therapeutic_areas.groupby(
        ["therapeutic_area_id", "therapeutic_area_name"]
    ):
        ds_ta = ds.sel(
            index=ds["disease_id"]
            .isin(ta_diseases["disease_id"].unique().tolist())
            .values
        )
        if ds_ta.dims["index"] < min_rows:
            logger.info(f"Skipping TA {ta_name!r}; n_rows={ds_ta.dims['index']}")
            continue
        dfs.append(
            get_feature_relative_risk(
                ds_ta, model_names=model_names, model_limits=model_limits
            ).assign(therapeutic_area_id=ta_id, therapeutic_area_name=ta_name)
        )
    return pd.concat(dfs, axis=0, ignore_index=True).pipe(
        assert_condition,
        lambda df: not df[["therapeutic_area_name", "method"]].duplicated().any(),
    )


###############################################################################
# Modeling
###############################################################################

OUTCOME_COLUMN = "target_disease__outcome__advanced"
TRANSITION_TIME_COLUMN = "target_disease__time__transition"


def get_primary_benchmarks() -> PandasDataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": "omim",
                "benchmark_slug": "OMIM",
                "benchmark_disp": "OMIM",
                "feature": "target_disease__genetic_association__omim",
            },
            {
                "benchmark": "eva",
                "benchmark_slug": "EVA",
                "benchmark_disp": "EVA/ClinVar",
                "feature": "target_disease__genetic_association__eva",
            },
            {
                "benchmark": "ot_genetics_portal",
                "benchmark_slug": "OTG",
                "benchmark_disp": "OTG/GWAS",
                "feature": "target_disease__genetic_association__ot_genetics_portal",
            },
        ]
    )


def get_primary_models() -> PandasDataFrame:
    return pd.DataFrame(
        [
            {
                "model": "rdg__no_time__positive",
                "model_slug": "RDG",
                "display_color": "#1f77b4",  # blue
            },
            {
                "model": "rdg__all__positive",
                "model_slug": "RDG-T",
                "display_color": "#7f7f7f",  # grey
            },
            {
                "model": "rdg__no_tgc__positive",
                "model_slug": "RDG-X",
                "display_color": "#7f7f7f",  # grey
            },
            {
                "model": "gbm__all__unconstrained",
                "model_slug": "GBM-T",
                "display_color": "#ffffff",  # white
            },
            {
                "model": "ots__all",
                "model_slug": "OTS",
                "display_color": "#2ca02c",  # green
            },
        ]
    )


def _get_feature_names_by_kind(
    features: list[str], kind: Literal["static", "temporal"]
) -> list[str]:
    feature_info = get_feature_info(
        [c for c in features if c not in FEATURES_PRIMARY_KEY]
    )
    return [r.feature for r in feature_info.itertuples() if r.kind == kind]


def get_static_feature_names(features: list[str]) -> list[str]:
    return _get_feature_names_by_kind(features, "static")


def get_temporal_feature_names(features: list[str]) -> list[str]:
    return _get_feature_names_by_kind(features, "temporal")


def make_pandas_pipeline(*args: Any, **kwargs: Any) -> Pipeline:
    return make_pipeline(*args, **kwargs).set_output(transform="pandas")


def validate_modeling_features(features: SparkDataFrame) -> SparkDataFrame:
    compatible_features = features.withColumnRenamed(
        "transition_year", "year"
    ).transform(
        spark_lambda(
            lambda df: df.select(
                *FEATURES_PRIMARY_KEY,
                *get_feature_info(df.columns, raise_on_uknown=False)[
                    "feature"
                ].to_list(),
            )
        )
    )
    validate_features(compatible_features, has_timeseries=False)
    return features


def get_modeling_features(
    features: SparkDataFrame,
    phase: int,
    min_transition_year: int,
    max_transition_year: int,
    max_advancement_year: int,
) -> SparkDataFrame:
    static_feature_names = get_static_feature_names(features.columns)
    temporal_feature_names = get_temporal_feature_names(features.columns)
    logger.info("Using static features:\n" + "\n".join(static_feature_names))
    transition_years = (
        features.filter(F.col("year").isNotNull())
        .filter(F.col(f"target_disease__clinical__phase{phase}__reached") > 0)
        .filter(F.col("target_disease__clinical__phase_max__reached") <= phase)
        .groupby("target_id", "disease_id")
        .agg(F.min("year").alias("transition_year"))
        .filter(
            F.col("transition_year").between(min_transition_year, max_transition_year)
        )
    )
    feature_years = (
        features.filter(F.col("year").isNotNull())
        .select("target_id", "disease_id", "year")
        .withColumnRenamed("year", "feature_year")
        .join(transition_years, on=["target_id", "disease_id"], how="inner")
        .filter(F.col("feature_year") < F.col("transition_year"))
        .groupby("target_id", "disease_id")
        .agg(F.max("feature_year").alias("feature_year"))
    )
    advancement_years = (
        features.filter(F.col("year").isNotNull())
        .filter(F.col("target_disease__clinical__phase_max__reached") > phase)
        .groupby("target_id", "disease_id")
        .agg(F.min("year").alias("year_first_advanced"))
    )
    static_features = (
        features.join(transition_years, on=["target_id", "disease_id"], how="inner")
        .groupby("target_id", "disease_id", "transition_year")
        .agg(*[F.max(c).alias(c) for c in static_feature_names])
    )
    temporal_features = (
        features.select(*FEATURES_PRIMARY_KEY, *temporal_feature_names)
        .withColumnRenamed("year", "feature_year")
        .join(
            feature_years, on=["target_id", "disease_id", "feature_year"], how="inner"
        )
    )
    result = (
        static_features.join(
            temporal_features, on=["target_id", "disease_id"], how="left"
        )
        .join(advancement_years, on=["target_id", "disease_id"], how="left")
        .withColumn(
            OUTCOME_COLUMN,
            F.col("year_first_advanced").isNotNull()
            & (F.col("year_first_advanced") <= max_advancement_year)
            if max_advancement_year is not None
            else F.col("year_first_advanced").isNotNull(),
        )
        .withColumn(
            TRANSITION_TIME_COLUMN,
            F.lit(max_advancement_year) - F.col("transition_year"),
        )
        .drop("year_first_advanced")
    )
    assert transition_years.count() == result.count()
    return result


def create_modeling_dataset(features: pd.DataFrame) -> Dataset:
    index = ["target_id", "disease_id"]
    descriptors = [c for c in features if "__" not in c and c not in index]
    assert len([c for c in features if "__outcome__" in c]) == 1
    outcomes = [OUTCOME_COLUMN]
    return xrml.create_dataset(
        features,
        index=index,
        outcomes=outcomes,
        descriptors=descriptors,
    )


class RidgeRegression(BaseEstimator):  # type: ignore[misc]
    def __init__(
        self,
        alpha: float = 1.0,
        positive: bool = False,
        inference_features: list[str] | None = None,
        random_state: int = 0,
    ):
        self.alpha = alpha
        self.positive = positive
        self.inference_features = inference_features
        self.random_state = random_state

    def fit(self, X: PandasDataFrame, y: PandasSeries) -> "RidgeRegression":
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        self._check_feature_names(X, reset=True)
        inference_features = self.inference_features or X.columns.tolist()
        assert len(set(inference_features) - set(X.columns)) == 0
        self.model_ = Ridge(
            alpha=self.alpha, positive=self.positive, random_state=self.random_state
        )
        self.model_.fit(X, y)
        assert self.model_.coef_.ndim == 1
        assert self.model_.intercept_.ndim == 0
        self.coef_ = pd.Series(self.model_.coef_, index=X.columns).loc[
            inference_features
        ]
        self.intercept_ = self.model_.intercept_
        return self

    def explain(self, X: PandasDataFrame) -> PandasDataFrame:
        assert isinstance(X, pd.DataFrame)
        return X[self.coef_.index].mul(self.coef_, axis="columns")

    def predict(self, X: Any) -> Any:
        assert isinstance(X, pd.DataFrame)
        y = (0 if self.positive else self.intercept_) + self.explain(X).sum(axis=1)
        return np.clip(y, 0, 1)

    def predict_proba(self, X: Any) -> Any:
        p = self.predict(X)
        return np.column_stack((1 - p, p))


def get_feature_selector(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("selector", "passthrough", feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_imputer(feature_names: list[str]) -> FunctionTransformer:
    static_target_features = get_static_feature_names(
        [c for c in feature_names if c.startswith("target__")]
    )

    def _impute(X: PandasDataFrame) -> PandasDataFrame:
        means = (
            X[static_target_features].groupby("target_id").max().mean(axis=0).to_dict()
        )
        return X.fillna({c: means.get(c, 0) for c in X})

    return FunctionTransformer(
        func=_impute, validate=False, feature_names_out="one-to-one"
    )


def get_ridge_regression_pipeline(
    feature_names: list[str],
    inference_features: list[str] | None = None,
    positive: bool = False,
    add_scaler: bool = True,
    add_imputer: bool = True,
) -> list[BaseEstimator]:
    return [
        *([get_imputer(feature_names)] if add_imputer else []),
        *([MinMaxScaler(clip=True)] if add_scaler else []),
        RidgeRegression(
            positive=positive,
            inference_features=inference_features,
            alpha=1e-6,
            random_state=SEED,
        ),
    ]


def get_ridge_regression_weights(ds: Dataset, model: str) -> PandasDataFrame:
    def _evidence_type(feature: str) -> str:
        if "__genetic_association__" in feature or "__somatic_mutation__" in feature:
            return "3. human genetic"
        if "__literature__" in feature:
            return "4. literature"
        if "__clinical__" in feature:
            return "1. human clinical"
        if "__animal_model__" in feature:
            return "2. animal model"
        if feature.startswith("target__"):
            return "5. target properties"
        return "6. other"

    # fmt: off
    return (
        pd.concat(
            [
                (
                    get_ridge_regression_explanations(ds, model=model)
                    .pipe(lambda df: df.where(df > 0))
                    .drop(columns=["prediction", "outcome"])
                    .agg(["min", "max", "mean", "median"])
                    .T
                ), (
                    ds.model.sel(models=model)
                    .item(0).estimator[-1]
                    .coef_.rename("coefficient")
                ),
            ],
            axis=1,
        )
        .pipe(lambda df: df[df["mean"] > 0])
        .rename_axis("feature", axis="index")
        .pipe(apply, lambda df: display(
            df.sort_values("mean")
            .style.background_gradient(cmap="Blues")
        ))
        .filter(items=["mean", "coefficient"])
        .reset_index()
        .assign(feature=lambda df: pd.Categorical(
            df["feature"], ordered=True,
            categories=df.sort_values("mean")["feature"].values,
        ))
        .set_index("feature")
        .rename(columns={"mean": "effect mean", "coefficient": "effect max (i.e. coefficient)"})
        .rename_axis("stat", axis="columns")
        .stack()
        .rename("value")
        .reset_index()
        .assign(stat=lambda df: pd.Categorical(
            df["stat"], ordered=True,
            categories=df["stat"].drop_duplicates().sort_values(ascending=False).values,
        ))
        .assign(evidence_type=lambda df: df["feature"].apply(_evidence_type))
    )
    # fmt: on


def get_ridge_regression_explanations(ds: Dataset, model: str) -> PandasDataFrame:
    pipe = ds.model.sel(models=model).item(0).estimator
    features = pipe[:-1].transform(ds.feature.to_series().unstack())
    predictions = (
        ds.prediction.sel(models=model, classes="positive").squeeze().to_series()
    )
    outcomes = ds.outcome.squeeze().to_series()
    explanations = pipe[-1].explain(features)
    assert len(explanations) == len(features)
    assert explanations.index.equals(features.index)
    return pd.concat([explanations, predictions, outcomes], axis=1)


class OpenTargetsClassifier(BaseEstimator):  # type: ignore[misc]
    TYPES = {
        "affected_pathway",
        "animal_model",
        "genetic_association",
        "known_drug",
        "literature",
        "rna_expression",
        "somatic_mutation",
    }
    WEIGHTS = {
        # See https://platform-docs.opentargets.org/associations#data-source-weights
        "europepmc": 0.2,
        "expression_atlas": 0.2,
        "impc": 0.2,
        "progeny": 0.5,
        "slapenrich": 0.5,
        "cancer_biomarkers": 0.5,
        "sysbio": 0.5,
    }

    @classmethod
    def get_weight(cls, feature: str) -> float | None:
        parts = feature.split("__")
        if len(parts) != 3:
            return None
        if parts[1] not in cls.TYPES:
            return None
        return cls.WEIGHTS.get(parts[2], 1.0)

    def fit(self, X: Any, y: Any) -> "OpenTargetsClassifier":
        coef = pd.Series([self.get_weight(c) for c in X.columns], index=X.columns)
        if coef.isnull().any():
            raise ValueError(f"Found unsupported features: {coef[coef.isnull()].index}")
        self.coef_ = coef
        return self

    def explain(self, X: PandasDataFrame) -> PandasDataFrame:
        assert isinstance(X, pd.DataFrame)
        return X[self.coef_.index].mul(self.coef_, axis="columns")

    def predict(self, X: Any) -> Any:
        assert isinstance(X, pd.DataFrame)
        return self.explain(X).fillna(0).mean(axis=1)

    def predict_proba(self, X: Any) -> Any:
        p = self.predict(X)
        return np.column_stack((1 - p, p))


def get_models(feature_names: list[str]) -> list[xrml.Model]:
    core_features = [
        f for f in feature_names if not f.startswith("target_disease__clinical__")
    ]
    clinical_features = [f for f in feature_names if "__clinical__" in f]
    target_disease_features = [
        f for f in feature_names if f.startswith("target_disease__")
    ]
    time_features = [f for f in core_features if "__time__" in f]
    feature_sets = {
        "all": core_features,
        "no_time": [f for f in core_features if f not in time_features],
        "no_tgc": [
            f
            for f in core_features
            if f not in time_features
            and "__clinical__" not in f
            and "__genetic_association__" not in f
        ],
        "only_genetics": [f for f in core_features if "__genetic_association__" in f],
    }
    constraint_sets = ["positive", "unconstrained"]
    models = []
    for constraint_set in constraint_sets:
        for feature_set, features in feature_sets.items():
            gbm_constraints = {}
            positive_only = constraint_set == "positive"
            if positive_only:
                gbm_constraints = {
                    "monotone_constraints": [1] * len(features),
                    "monotone_constraints_method": "advanced",
                }
            models.extend(
                [
                    xrml.Model(
                        (name := f"gbm__{feature_set}__{constraint_set}"),
                        name,
                        make_pandas_pipeline(
                            get_feature_selector(features),
                            LGBMClassifier(random_state=SEED, **gbm_constraints),
                        ),
                    ),
                    xrml.Model(
                        (name := f"rdg__{feature_set}__{constraint_set}"),
                        name,
                        make_pandas_pipeline(
                            get_feature_selector(core_features),
                            *get_ridge_regression_pipeline(
                                feature_names=core_features,
                                inference_features=features,
                                positive=positive_only,
                            ),
                        ),
                    ),
                ]
            )
            if not positive_only:
                models.append(
                    xrml.Model(
                        (name := f"lrg__{feature_set}__{constraint_set}"),
                        name,
                        make_pandas_pipeline(
                            get_feature_selector(features),
                            SimpleImputer(
                                strategy="constant", fill_value=0, add_indicator=False
                            ),
                            StandardScaler(),
                            LogisticRegressionCV(random_state=SEED, cv=5),
                        ),
                    )
                )
    models.extend(
        [
            xrml.Model(
                "ots__all",
                "Open Targets Classifier",
                make_pandas_pipeline(
                    get_feature_selector(
                        [
                            f
                            for f in feature_names
                            if f in target_disease_features
                            and f not in clinical_features
                            and f not in time_features
                        ]
                    ),
                    OpenTargetsClassifier(),
                ),
            ),
            xrml.Model(
                "baseline__most_frequent",
                "Baseline Classifier (most frequent class)",
                make_pandas_pipeline(
                    DummyClassifier(strategy="most_frequent"),
                ),
            ),
            *[
                xrml.Model(
                    f"baseline__random_seed_{seed}",
                    "Baseline Classifier (random)",
                    make_pandas_pipeline(
                        DummyClassifier(strategy="stratified", random_state=seed),
                    ),
                )
                for seed in range(3)
            ],
        ]
    )
    return models


def get_training_dataset(features: PandasDataFrame) -> Dataset:
    assert len(features) > 0
    ds = features.pipe(create_modeling_dataset)
    assert ds.dims["outcomes"] == 1
    feature_names = ds.features.values.tolist()
    return (
        ds.pipe(xrml.add_single_split, split="train")
        .pipe(xrml.add_models, estimator_fn=lambda: get_models(feature_names))
        .pipe(xrml.fit, groups="target_id")
    )


def get_evaluation_dataset(
    features: PandasDataFrame,
    training_dataset: Dataset,
) -> Dataset:
    return (
        features.pipe(create_modeling_dataset)
        .merge(training_dataset[["model"]])
        .pipe(xrml.predict_proba)
    )


###############################################################################
# Opportunities
###############################################################################


def get_opportunity_dataset(
    features: SparkDataFrame,
    models: list[str],
    training_dataset: Dataset,
    time_since_transition: int = 5,
) -> Dataset:
    return (
        features.withColumnRenamed("year", "feature_year")  # type: ignore[attr-defined]
        .withColumn(OUTCOME_COLUMN, F.lit(True))
        .withColumn(TRANSITION_TIME_COLUMN, F.lit(time_since_transition))
        .toPandas()
        .pipe(create_modeling_dataset)
        .merge(training_dataset.sel(models=models)[["model"]])
        .pipe(xrml.predict_proba)
    )


def get_opportunity_dataframe(
    spark: SparkSession,
    training_dataset: Dataset,
    model: str,
    input_dir: Path,
    output_dir: Path | None,
) -> SparkDataFrame:
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_path = output_dir / "opportunities.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = spark.read.parquet(str(input_dir / "static_features.parquet"))
    dataset = get_opportunity_dataset(
        features=features,
        models=[model],
        training_dataset=training_dataset,
    )
    feature_names = [
        "target__clinical__phase_max__reached",
        "disease__clinical__phase_max__reached",
        "target_disease__clinical__phase_max__reached",
        "target_disease__known_drug__chembl",
        *[
            c
            for c in features.columns
            if c.startswith("target_disease__genetic_association__")
        ],
    ]
    df = pd.concat(
        [
            dataset.feature.sel(features=feature_names).to_series().unstack(),
            dataset.prediction.sel(classes="positive").squeeze().to_series(),
        ],
        axis=1,
    ).reset_index()
    logger.info(f"Writing opportunity dataframe to {str(output_path)!r}")
    df.to_parquet(output_path, index=False)
    return spark.read.parquet(str(output_path))


TRACTABILITY_BUCKETS = {
    # See https://github.com/chembl/tractability_pipeline_v2
    "HIGH": {
        "SM": (
            clin_buckets := ["Approved Drug", "Advanced Clinical", "Phase 1 Clinical"]
        ),
        "PR": clin_buckets,
        "AB": clin_buckets,
        "OC": ["Approved Drug"],
    },
    "MED": {
        "SM": ["Structure with Ligand", "High-Quality Ligand"],
        "PR": ["Literature", "UniProt Ubiquitination"],
        "AB": ["UniProt loc high conf", "GO CC high conf"],
        "OC": ["Advanced Clinical"],
    },
}

TRACTABILITY_MODALITY_LABELS = {
    # See https://github.com/chembl/tractability_pipeline_v2
    "PR": "PROTAC",
    "SM": "Small molecule",
    "AB": "Antibody",
    "OC": "Other",
}


def get_tractability_buckets(tractability: SparkDataFrame) -> PandasDataFrame:
    return (
        tractability.filter(F.col("tractability_value"))  #  type: ignore[attr-defined]
        .select(
            F.col("tractability_id").alias("evidence"),
            F.col("tractability_modality").alias("modality"),
        )
        .drop_duplicates()
        .toPandas()
        .merge(
            pd.DataFrame(
                [
                    {
                        "confidence": confidence,
                        "modality": modality,
                        "evidence": evidence,
                    }
                    for confidence, buckets in TRACTABILITY_BUCKETS.items()
                    for modality, evidences in buckets.items()
                    for evidence in evidences
                ]
            ),
            on=["evidence", "modality"],
            how="left",
        )
        .assign(confidence=lambda df: df["confidence"].fillna("LOW"))
        .pipe(assert_condition, lambda df: df["confidence"].notnull().all())
        .pipe(
            assert_condition,
            lambda df: df["confidence"].isin(["LOW", "MED", "HIGH"]).all(),
        )
        .assign(
            confidence=lambda df: pd.Categorical(
                df["confidence"], ordered=True, categories=["LOW", "MED", "HIGH"]
            )
        )
        .pipe(
            assert_condition,
            lambda df: df["modality"].isin(["AB", "OC", "PR", "SM"]).all(),
        )
        .assign(
            modality=lambda df: pd.Categorical(
                df["modality"], ordered=True, categories=["OC", "AB", "PR", "SM"]
            )
        )
        .sort_values(["modality", "confidence", "evidence"])
        .reset_index(drop=True)
    )


def get_opportunity_groupings(
    opportunities: SparkDataFrame,
    tractability: SparkDataFrame,
    therapeutic_areas: SparkDataFrame,
) -> SparkDataFrame:
    tractability = (
        tractability.filter("tractability_value")
        .select(
            "target_id",
            F.col("tractability_modality").alias("target__tractability__modality"),
            F.col("tractability_id").alias("target__tractability__evidence"),
        )
        .transform(
            spark_lambda(
                lambda df: (
                    df.unionByName(
                        df.withColumn("target__tractability__evidence", F.lit("ALL"))
                    ).unionByName(
                        df.withColumn(
                            "target__tractability__evidence",
                            F.coalesce(
                                functools.reduce(
                                    lambda a, b: F.coalesce(a, b),
                                    [
                                        F.when(
                                            (
                                                F.col("target__tractability__modality")
                                                == F.lit(modality)
                                            )
                                            & (
                                                F.col("target__tractability__evidence")
                                                == F.lit(e)
                                            ),
                                            F.lit(level),
                                        )
                                        for level, buckets in TRACTABILITY_BUCKETS.items()
                                        for modality, evidence in buckets.items()
                                        for e in evidence
                                    ],
                                    F.lit(None),
                                ),
                                F.lit("LOW"),
                            ),
                        )
                    )
                )
            )
        )
        .distinct()
    )
    therapeutic_areas = (
        therapeutic_areas.filter(F.col("therapeutic_area_name") != "all")
        .select(
            "disease_id",
            F.col("therapeutic_area_name").alias("disease__therapeutic_area"),
        )
        .distinct()
    )
    return (
        opportunities.withColumn(
            "target_disease__stage",
            F.when(
                (
                    F.coalesce(
                        col := F.col("target_disease__clinical__phase_max__reached"),
                        F.lit(0),
                    )
                    <= 0
                ),
                F.lit("NONE"),
            ).otherwise(F.concat(F.lit("Phase "), col.cast("int"))),
        )
        .transform(
            spark_lambda(
                lambda df: (
                    df.unionByName(df.withColumn("target_disease__stage", F.lit("ALL")))
                )
            )
        )
        .transform(
            spark_lambda(
                lambda df: (
                    df.withColumn("target__tractability__modality", F.lit("ALL"))
                    .withColumn("target__tractability__evidence", F.lit("ALL"))
                    .unionByName(
                        df.join(
                            tractability,
                            on="target_id",
                            how="inner",
                        )
                    )
                )
            )
        )
        .transform(
            spark_lambda(
                lambda df: (
                    df.withColumn(
                        "disease__therapeutic_area", F.lit("ALL")
                    ).unionByName(
                        df.filter(
                            F.col("target__tractability__evidence").isin(
                                ["ALL", "HIGH", "MED", "LOW"]
                            )
                        ).join(
                            therapeutic_areas,
                            on="disease_id",
                            how="inner",
                        )
                    )
                )
            )
        )
        .distinct()
    )


def get_opportunity_statistics(
    opportunity_groupings: SparkDataFrame, relative_risk_thresholds: PandasDataFrame
) -> SparkDataFrame:
    assert relative_risk_thresholds["model"].nunique() == 1
    return (
        opportunity_groupings.withColumn("pair", F.struct("target_id", "disease_id"))
        .groupby(
            "target_disease__stage",
            "target__tractability__modality",
            "target__tractability__evidence",
            "disease__therapeutic_area",
        )
        .agg(
            F.count_distinct(F.col("pair")).alias("n_pairs"),
            F.array(
                *(
                    [
                        F.struct(
                            F.lit(row.benchmark_name).alias("benchmark_name"),
                            F.lit(row.benchmark_slug).alias("benchmark_slug"),
                            F.count_distinct(
                                F.when(
                                    F.col("prediction")
                                    >= row.model_prediction_threshold,
                                    F.col("pair"),
                                )
                            ).alias("model__n_pairs__over_threshold"),
                            F.count_distinct(
                                F.when(
                                    F.col(
                                        f"target_disease__genetic_association__{row.benchmark_name}"
                                    )
                                    > 0,
                                    F.col("pair"),
                                )
                            ).alias("feature__n_pairs__exists"),
                        )
                        for row in (relative_risk_thresholds.itertuples())
                    ]
                    + [
                        F.struct(
                            F.lit("NONE").alias("benchmark_name"),
                            F.lit("NONE").alias("benchmark_slug"),
                            F.count_distinct(F.col("pair")).alias(
                                "model__n_pairs__over_threshold"
                            ),
                            F.count_distinct(F.col("pair")).alias(
                                "feature__n_pairs__exists"
                            ),
                        )
                    ]
                )
            ).alias("stats"),
        )
        .select(
            "*",
            F.explode("stats").alias("stat"),
        )
        .drop("stats")
        .select("*", "stat.*")
        .drop("stat")
    )


def display_opportunities_by_modality(
    opportunity_statistics: PandasDataFrame, benchmark_name: str
) -> PandasDataFrame:
    def _modality_categories(s: pd.Series) -> pd.Series:
        return pd.Categorical(
            s,
            ordered=True,
            categories=(cats := ["ALL", "OC", "AB", "SM", "PR"])
            + sorted([v for v in s.unique() if v not in cats]),
        )

    def _evidence_categories(s: pd.Series) -> pd.Series:
        return pd.Categorical(
            s,
            ordered=True,
            categories=(cats := ["ALL", "HIGH", "MED", "LOW"])
            + sorted([v for v in s.unique() if v not in cats]),
        )

    # fmt: off
    return (
        opportunity_statistics
        .pipe(lambda df: df[df["target_disease__stage"] == "NONE"])
        .pipe(lambda df: df[df["benchmark_name"] == benchmark_name])
        .pipe(lambda df: df[df["target__tractability__evidence"].isin(
            ["HIGH", "MED", "LOW", "ALL"]
        )])
        .assign(target__tractability__modality=lambda df: _modality_categories(
            df["target__tractability__modality"]
        ))
        .assign(target__tractability__evidence=lambda df: _evidence_categories(
            df["target__tractability__evidence"]
        ))
        .set_index([
            "disease__therapeutic_area",
            "target__tractability__modality",
            "target__tractability__evidence",
        ])["model__n_pairs__over_threshold"]
        .unstack("disease__therapeutic_area")
        .T.fillna(0).astype(int)
        .rename_axis("therapeutic area", axis="index")
        .rename_axis(("modality", "confidence"), axis="columns")
        .pipe(lambda df: df[df.sum(axis=1) > 0])
        .pipe(lambda df: df.loc[df.sum(axis=1).sort_values(ascending=False).index])
        .pipe(
                lambda df: (
                    df.style.format(na_rep="")
                    .background_gradient(cmap="Greys", axis=None, subset=[c for c in df if c[0] == "ALL"], vmax=(vmax := 1000))
                    .background_gradient(cmap="Greens", axis=None, subset=[c for c in df if c[0] == "AB"], vmax=vmax)
                    .background_gradient(cmap="Blues", axis=None, subset=[c for c in df if c[0] == "OC"], vmax=vmax)
                    .background_gradient(cmap="Reds", axis=None, subset=[c for c in df if c[0] == "SM"], vmax=vmax)
                    .background_gradient(cmap="Purples", axis=None, subset=[c for c in df if c[0] == "PR"], vmax=vmax)
                    .set_table_styles(
                        white_header_styles()
                        + [
                            {
                                "selector": "th:not(:first-child)",
                                "props": [
                                    ("background-color", "white"),
                                    ("border", "1px black solid !important"),
                                    ("text-align", "center"),
                                ],
                            },
                        ]
                    )
                    .set_properties(**{"border": "1px black solid !important"})
                )
            )
    )
    # fmt: on


def display_opportunities_by_category(
    opportunity_statistics: PandasDataFrame, benchmark_name: str
) -> PandasDataFrame:
    def _categories(row: pd.Series) -> tuple[tuple[str, str], ...]:
        benchmark = row["benchmark_name"]
        benchmark_slug = row["benchmark_slug"]
        stage = row["target_disease__stage"]
        modality = row["target__tractability__modality"]
        evidence = row["target__tractability__evidence"]

        categories = []
        if evidence == "ALL" and modality == "ALL" and benchmark == benchmark_name:
            categories.append((f"stage<br>[threshold={benchmark_slug}]", stage))
        if stage == "NONE" and evidence == "ALL" and modality == "ALL":
            categories.append(("threshold<br>[stage=NONE]", benchmark_slug))
        if stage == "NONE" and evidence == "HIGH" and benchmark == benchmark_name:
            categories.append(
                (
                    f"tractability<br>[stage=NONE, confidence=HIGH, threshold={benchmark_slug}]",
                    modality,
                )
            )
        return tuple(categories)

    def _validate(df: pd.DataFrame, index: list[str]) -> pd.DataFrame:
        cts = df.groupby(index).size()
        if len(duplicates := cts[cts > 1]) > 0:
            raise ValueError(
                f"Found multiple rows for the following groups:\n{duplicates}"
            )
        return df

    # fmt: off
    return (
        opportunity_statistics
        .pipe(lambda df: df[df["benchmark_name"] != "NONE"])
        .assign(categories=lambda df: df.apply(_categories, axis=1))
        .explode("categories")
        .dropna(subset="categories")
        .assign(category_name=lambda df: df["categories"].apply(lambda t: t[0]))
        .assign(category_value=lambda df: df["categories"].apply(lambda t: t[1]))
        .drop(columns="categories")
        .pipe(_validate, index=(index := ["disease__therapeutic_area", "category_name", "category_value"]))
        .set_index(index)["model__n_pairs__over_threshold"]
        .unstack(["category_name", "category_value"])
        .sort_index(axis="columns")
        .fillna(0).astype(int)
        .pipe(lambda df: df[df.sum(axis=1) > 0])
        .pipe(lambda df: df.loc[df.sum(axis=1).sort_values(ascending=False).index])
        .rename_axis("therapeutic area", axis="index")
        .rename_axis(("", ""), axis="columns")
        .pipe(
            lambda df: df.style
            .background_gradient(cmap="Reds", axis=0, subset=[c for c in df if c[0].startswith("threshold")], vmax=(vmax := 1000))
            .background_gradient(cmap="Greens", axis=0, subset=[c for c in df if c[0].startswith("tractability")], vmax=vmax)
            .background_gradient(cmap="Blues", axis=0, subset=[c for c in df if c[0].startswith("stage")], vmax=vmax)
            .set_table_styles(
                white_header_styles()
                + [
                    {
                        "selector": "th:not(:first-child)",
                        "props": [
                            ("background-color", "white"),
                            ("border", "1px black solid !important"),
                            ("text-align", "center"),
                        ],
                    },
                ]
            )
            .set_properties(**{"border": "1px black solid !important"})
        )
    )
    # fmt: on


###############################################################################
# Visualization
###############################################################################

FIGURE_WIDTH = 8


def rotate_column_styles(
    width_px: int = 20, height_px: int = 250
) -> list[dict[str, Any]]:
    return [
        {"selector": "th", "props": [("width", f"{width_px}px")]},
        {
            "selector": "th.col_heading",
            "props": [
                ("writing-mode", "vertical-rl"),
                ("transform", "rotateZ(180deg)"),
                ("height", f"{height_px}px"),
                ("vertical-align", "top"),
                ("text-align", "left"),
            ],
        },
    ]


def white_header_styles() -> list[dict[str, Any]]:
    return [{"selector": "th", "props": [("background-color", "white")]}]


def set_default_table_styles(styler: Styler, height_px: int = 250) -> Styler:
    return (
        styler.set_table_styles(
            rotate_column_styles(height_px=height_px) + white_header_styles()
        )
        .highlight_null(color="white")
        .set_properties(**{"border": "1px black solid !important"})
    )


def get_assets_directory() -> Path:
    path = Path(__file__).parent.resolve() / "paper" / "assets"
    if not path.exists():
        raise ValueError(f"Figure directory {path} does not exist")
    return path


def get_asset_path(filename: str) -> Path:
    return get_assets_directory() / filename


def save_figure(filename: str, enable: bool) -> Callable[[Any], Any]:
    def _fn(plot: Any) -> Any:
        if not enable:
            return plot
        path = get_assets_directory() / filename
        logger.info(f"Saving figure to {str(path)!r}")
        plot.save(str(path))
        return plot

    return _fn


def save_table(
    table: Styler | PandasDataFrame, filename: str, enable: bool, **kwargs: Any
) -> Styler | PandasDataFrame:
    if not enable:
        return table
    styler = (
        table
        if isinstance(table, Styler)
        else (
            table.style.format(escape="latex")
            .format_index(escape="latex", axis="index")
            .format_index(escape="latex", axis="columns")
        )
    )
    path = get_assets_directory() / filename
    logger.info(f"Saving table to {str(path)!r}")
    styler.to_latex(path, **kwargs)
    return table


def prepare_relative_risk_by_feature(  # noqa: C901
    relative_risk: PandasDataFrame,
) -> dict[str, PandasDataFrame]:
    # fmt: off
    def categorize_method(df: PandasDataFrame) -> PandasSeries:
        method = df["method"]
        dupes = method[method.duplicated()]
        if len(dupes) > 0:
            raise ValueError(f"Duplicate method names: {set(dupes)}")
        return pd.Categorical(method, ordered=True, categories=method.values)

    def get_core_feature_relative_risk() -> PandasDataFrame:
        def _method_group(row: pd.Series) -> str:
            if row["is_model_feature"]:
                return "1. model"
            method = row["method"]
            if "__genetic_association__" in method:
                return "2. genetic"
            if "__literature__" in method:
                return "4. literature"
            if "__time__" in method:
                return "3. time"
            return "5. other"


        def _method_sort(row: pd.Series) -> tuple[str, str | float]:
            if row["method_group"] in {"2. genetic", "5. other"}:
                return (row["method_group"], -row["relative_risk_low"],)
            elif row["method_group"] == "3. time":
                return (row["method_group"], -float(row["method"].split("_")[-1]))
            else:
                return (row["method_group"], row["method"])

        models = {"rdg__no_time__positive": "rdg"}
        return (
            relative_risk
            .pipe(lambda df: df[~df["is_model_feature"] | df["model_name"].isin(models)])
            .assign(model_name=lambda df: df["model_name"].map(models))
            .pipe(lambda df: df[df["method"].str.contains("^target_disease__")])
            .assign(method_group=lambda df: df.apply(_method_group, axis=1))
            .pipe(lambda df: df[~df["method"].str.contains("^target_disease__clinical__")])
            .assign(method=lambda df: (
                df["method"].str.replace("target_disease__", "")
                .str.replace("__limit_", "@")
            ))
            .assign(method=lambda df: df["method"].where(
                ~df["is_model_feature"],
                df["model_name"] + "@" + df["limit"].apply("{:03d}".format)
            ))
            .assign(method_sort=lambda df: df.apply(_method_sort, axis=1))
            .sort_values("method_sort", ascending=False)
            .assign(method=categorize_method)
        )

    def get_model_feature_relative_risk() -> PandasDataFrame:

        def _method_group(method: str) -> str:
            if "__all__" in method:
                return "1. all features"
            if "__no_time__" in method:
                return "2. no time"
            if "__no_tgc__" in method:
                return "3. no time/genetics/clinical"
            return "4. other"

        return (
            relative_risk
            .pipe(lambda df: df[df["is_model_feature"]])
            .pipe(lambda df: df[df["limit"].between(0, 100)])
            .assign(method_group=lambda df: df["method"].apply(_method_group))
            .pipe(lambda df: df[~df["method_group"].isin(["4. other"])])
            .assign(model_group=lambda df: df["model_name"].str.split("__").str[0])
            .pipe(lambda df: df[df["model_group"].isin(["gbm", "rdg"])])
            .assign(method=lambda df: df["model_name"].str.split("__").str[1:].str.join("__") + "@" + df["limit"].apply("{:03d}".format))
            .assign(method_sort=lambda df: df.apply(lambda r: (r["method_group"], r["model_name"], r["limit"]), axis=1))
            .sort_values("method_sort", ascending=False)
            .assign(method=lambda df: pd.Categorical(
                df["method"], ordered=True,
                # Deduplicate while preserving order
                categories=list(dict.fromkeys(df["method"]))
            ))
        )

    def get_static_feature_relative_risk() -> PandasDataFrame:

        def _method_group(method: str) -> str:
            if "__clinical__" in method:
                return "4. clinical evidence"
            if "__genetic_constraint__" in method:
                return "1. target constraint"
            if "__tissue_" in method:
                return "2. target expression"
            if "__mouse_ko_score__" in method:
                return "3. target essentiality"
            return "5. other"

        return (
            relative_risk
            .pipe(lambda df: df[df["method"].str.contains("^target__|^disease__")])
            .assign(method_group=lambda df: df["method"].apply(_method_group))
            .assign(method_sort=lambda df: df.apply(lambda r: (r["method_group"], r["method"]), axis=1))
            .sort_values("method_sort", ascending=False)
            .assign(method=categorize_method)
        )

    return {
        "core": get_core_feature_relative_risk(),
        "model": get_model_feature_relative_risk(),
        "static": get_static_feature_relative_risk(),
    }
    # fmt: on


def prepare_classification_metrics_by_therapeutic_area(
    classification_metrics: PandasDataFrame,
    primary_models: PandasDataFrame,
    primary_therapeutic_areas: list[str],
    ap_max_value: float = 0.3,
) -> PandasDataFrame:
    # fmt: off
    return (
        classification_metrics.filter(items=["roc_auc", "average_precision"])
        .reset_index()
        .pipe(lambda df: df[df["therapeutic_area_name"].isin(primary_therapeutic_areas)])
        .pipe(assert_condition, lambda df: (df["therapeutic_area_name"] == "all").any())
        .pipe(lambda df: df[df["therapeutic_area_name"] != "all"])
        .assign(
            model_slug=lambda df: df["models"].map(
                primary_models.set_index("model")["model_slug"]
            )
        )
        .assign(
            model_slug=lambda df: pd.Categorical(
                df["model_slug"], ordered=True, categories=["OTS", "RDG", "RDG-T"]
            )
        )
        .dropna(subset="model_slug")
        .set_index(["model_slug", "therapeutic_area_name"])[
            ["roc_auc", "average_precision"]
        ]
        .rename_axis("metric", axis="columns")
        .stack()
        .rename("value")
        .reset_index()
        .assign(metric=lambda df: pd.Categorical(
            df["metric"].str.replace("_", " "), ordered=True,
            categories=["roc auc", "average precision"]
        ))
        .pipe(assert_condition, lambda df: df["metric"].notnull().all())
        .assign(model_group=lambda df: (df["model_slug"] == "RDG-T").map({True: "secondary", False: "primary"}))
        .pipe(lambda df: (
            (
                df.assign(clipped=False)
            ) if ap_max_value is None else (
                df
                .assign(clipped=lambda df: np.where(df["metric"] == "average precision", df["value"] > ap_max_value, False))
                .assign(value=lambda df: np.where(df["metric"] == "average precision", df["value"].clip(0, ap_max_value), df["value"]))
            )
        ))
    )
    # fmt: on


def prepare_relative_risk_by_therapeutic_area(
    relative_risk: PandasDataFrame,
    primary_therapeutic_areas: list[str],
) -> PandasDataFrame:
    def _p_values(df: PandasDataFrame, base_model: str) -> pd.DataFrame:
        values = df.set_index(["therapeutic_area_name", "model_slug"])[
            "relative_risk"
        ].unstack("model_slug")
        return pd.DataFrame(
            [
                {
                    "model_slug": model,
                    "p_value": (
                        wilcoxon(
                            values[model],
                            values[base_model],
                            alternative="greater",
                            method="approx",
                        ).pvalue
                        if model != base_model
                        else 0
                    ),
                }
                for model in values.columns
            ]
        )

    relative_risk = relative_risk.pipe(
        lambda df: df[df["therapeutic_area_name"].isin(primary_therapeutic_areas)]
    )

    therapeutic_areas = [
        ta
        for ta in relative_risk["therapeutic_area_name"].unique()
        if ta not in {"all"}
    ]
    logger.info(
        "Computing RR means and p-values for TAs:\n" + "\n".join(therapeutic_areas)
    )
    # Find all fields that are unique to a method, not method+TA
    method_cols = (
        relative_risk.groupby("method")
        .nunique()
        .max(axis=0)
        .pipe(lambda s: s[s <= 1])
        .index.tolist()
    )
    # fmt: off
    return (
        relative_risk
        .pipe(lambda df: df[df["is_primary_benchmark_feature"] | df["is_model_feature"]])
        # Fill limit before using in grouping and invert fill after (it is dropped otherwise as Int64)
        .assign(limit=lambda df: df["limit"].fillna(0))
        # Compute summaries across therapeutic areas
        .pipe(lambda df: pd.concat([
            df,
            df
            .pipe(lambda df: df[df["therapeutic_area_name"].isin(therapeutic_areas)])
            .groupby(["method", "limit"])
            .agg(**{
                **{
                    "relative_risk": ("relative_risk", "mean"),
                    "n_therapeutic_areas": ("therapeutic_area_name", "nunique")
                },
                **{
                    c: (c, "first")
                    for c in method_cols
                    if c not in ["method", "limit"]
                }
            })
            .reset_index()
            .assign(therapeutic_area_name="average", n_total=np.nan)
        ], axis=0, ignore_index=True))
        .pipe(lambda df: df.merge(
            df
            .pipe(lambda df: df[df["therapeutic_area_name"].isin(therapeutic_areas)])
            .pipe(lambda df: df[df["is_primary_model_feature"]])
            .pipe(assert_condition, lambda df: df["model_slug"].notnull().all())
            .groupby("limit")
            .apply(_p_values, base_model="OTS"),
            on=["model_slug", "limit"],
            how='left'
        ))
        .assign(limit=lambda df: df["limit"].where(df["limit"] > 0).astype('Int64'))
        # Ensure that RRs and p-values are present for all primary models
        .pipe(apply, lambda df: (
            df[df["is_primary_model_feature"]]
            .pipe(assert_condition, lambda df: 
                df["relative_risk"].notnull().all() and
                df["p_value"].notnull().all()
            )
        ))
        # Also ensure that primary model mean RRs come from the same therapeutic areas
        .pipe(apply, lambda df: (
            df[df["is_primary_model_feature"] & (df["therapeutic_area_name"] == "average")]
            .pipe(assert_condition, lambda df: 
                (df["n_therapeutic_areas"] == len(set(primary_therapeutic_areas) - {"all"})).all()
            )
        ))
    )
    # fmt: on


def display_relative_risk_by_therapeutic_area(
    df: PandasDataFrame, limits: list[int] | None = None
) -> Styler:
    def split_cols(df: PandasDataFrame, char: str) -> tuple[list[str], list[str]]:
        cols = df.columns.tolist()
        return (has_char := [c for c in cols if char in c]), [
            c for c in cols if c not in has_char
        ]

    # fmt: off
    return (
        df
        .assign(
            method=lambda df: df["benchmark_disp"].where(
                df["is_primary_benchmark_feature"],
                df["model_slug"] + "@" + df["limit"].apply("{:03d}".format)
            )
        )
        .pipe(assert_condition, lambda df: df["method"].notnull().all())
        .pipe(lambda df: df[df["is_primary_benchmark_feature"] | df["is_primary_model_feature"]])
        .pipe(lambda df: df[df["model_slug"].isin(["RDG", "OTS"]) | df["model_slug"].isnull()])
        .pipe(lambda df: df if limits is None else df[df["limit"].isin(limits) | df["limit"].isnull()])
        .rename(columns={"n_total": "n_pairs", "therapeutic_area_name": "therapeutic_area"})
        .set_index(["method", "therapeutic_area", "n_pairs"])["relative_risk"]
        .unstack("method")
        # Reset index, sort, and re-index rather than sorting index to
        # avoid bug in `na_position` with `sort_index(level=...)`
        .reset_index()
        .sort_values("n_pairs", na_position="first", ascending=False)
        .set_index(["therapeutic_area", "n_pairs"])
        .rename_axis("", axis="columns")
        .pipe(lambda df: df[
            df[split_cols(df, "@")[1]].mean(axis=0)
            .sort_values(ascending=False).index.tolist()
            + sorted(split_cols(df, "@")[0])
        ])
        .pipe(
            lambda df: (
                df.style
                .format(precision=2, na_rep="")
                .format_index("{:.0f}", na_rep="", level="n_pairs")
                .background_gradient(cmap="Greys", axis=None, subset=split_cols(df, "@")[1], vmin=0, vmax=(vmax := 10))
                .background_gradient(cmap="Blues", axis=None, subset=[c for c in split_cols(df, "@")[0] if c.startswith("RDG")], vmin=0, vmax=vmax)
                .background_gradient(cmap="Greens", axis=None, subset=[c for c in split_cols(df, "@")[0] if c.startswith("OTS")], vmin=0, vmax=vmax)
                .pipe(set_default_table_styles, height_px=150)
                .apply_index(lambda v: ["white-space: nowrap;"] * len(v), level="therapeutic_area", axis="index")
            )
        )
    )
    # fmt: on


def combine_average_therapeutic_metrics(
    relative_risk: PandasDataFrame,
    classification_metrics: PandasDataFrame,
    primary_therapeutic_areas: list[str],
) -> PandasDataFrame:
    therapeutic_areas = set(primary_therapeutic_areas) - {"all"}
    # fmt: off
    return (
        pd.concat([
            (
                classification_metrics[["roc_auc", "average_precision"]]
                .rename_axis("metric", axis="columns")
                .stack().rename("value").reset_index()
                .rename(columns={"models": "model_name"})
                .pipe(assert_condition, lambda df: not df[["model_name", "therapeutic_area_name", "metric"]].duplicated().any())
                .pipe(lambda df: df[df["therapeutic_area_name"].isin(therapeutic_areas)])
                .groupby(["model_name", "metric"])
                .agg(n_therapeutic_areas=("therapeutic_area_name", "nunique"), value=("value", "mean"))
                .assign(limit=np.nan)
                .reset_index()
                [["model_name", "metric", "limit", "n_therapeutic_areas", "value"]]
            ), (
                relative_risk
                .pipe(lambda df: df[df["is_model_feature"]])
                .pipe(lambda df: df[df["therapeutic_area_name"] == "average"])
                .assign(limit=lambda df: df["limit"].astype(int))
                .assign(metric=lambda df: "rr@" + df["limit"].apply("{:03d}".format))
                .assign(value=lambda df: df["relative_risk"])
                .pipe(assert_condition, lambda df: not df[["model_name", "metric"]].duplicated().any())
                [["model_name", "metric", "limit", "n_therapeutic_areas", "value"]]
            )
        ], axis=0, ignore_index=True)
        .assign(n_therapeutic_areas=lambda df: df["n_therapeutic_areas"].astype(int))
        .assign(has_all_primary_therapeutic_areas=lambda df: df["n_therapeutic_areas"] == len(therapeutic_areas))
        .assign(model_algorithm=lambda df: df["model_name"].str.split("__").str[0].fillna(""))
        .assign(model_features=lambda df: df["model_name"].str.split("__").str[1].fillna(""))
        .assign(model_constraint=lambda df: df["model_name"].str.split("__").str[2].fillna(""))
        .pipe(assert_condition, lambda df: not df[["model_name", "metric"]].duplicated().any())
    )
    # fmt: on


def prepare_average_therapeutic_area_metrics(
    average_therapeutic_metrics: PandasDataFrame,
) -> PandasDataFrame:
    metrics = [
        "rr@010",
        "rr@020",
        "rr@030",
        "rr@040",
        "rr@050",
        "rr@100",
        "average_precision",
        "roc_auc",
    ]

    feature_group = "feature ablations\n([+] models)"
    algorithms_group = "model algorithms\n(core features)"
    df = average_therapeutic_metrics.pipe(
        lambda df: df[df["has_all_primary_therapeutic_areas"]]
    ).pipe(lambda df: df[df["metric"].isin(metrics)])
    # fmt: off
    return (
        pd.concat(
            [
                (
                    df
                    .pipe(lambda df: df[df["model_algorithm"].isin(["rdg", "ots"])])
                    .pipe(lambda df: df[df["model_features"].isin(["all", "no_time", "no_tgc"])])
                    .pipe(lambda df: df[(df["model_algorithm"] == "ots") | (df["model_constraint"] == "positive")])
                    .assign(
                        model=lambda df: (
                            df["model_algorithm"].str.upper() + 
                            np.where(
                                df["model_algorithm"] == "ots", "",
                                df["model_features"].map({
                                    "all": "-T",
                                    "no_time": "",
                                    "no_tgc": "-X",
                                })
                            )
                        )
                    )
                    .assign(group=feature_group)
                ),
                (
                    df
                    .pipe(lambda df: df[df["model_algorithm"].isin(["gbm", "rdg", "ots"])])
                    .pipe(lambda df: df[
                        (df["model_features"] == "no_time") | 
                        (df["model_algorithm"] == "ots")
                    ])
                    .assign(
                        model=lambda df: (
                            df["model_algorithm"].str.upper()
                            + df["model_constraint"].map({
                                "": "",
                                "unconstrained": "[+/-]",
                                "positive": "[+]",
                            })
                        )
                    )
                    .assign(group=algorithms_group)
                ),
            ],
            axis=0,
            ignore_index=True,
        )
        .assign(metric=lambda df: (
            df["metric"]
            .str.replace("rr@", "RR@")
            .str.replace("average_precision", "AP")
            .str.replace("roc_auc", "ROC")
        ))
        .pipe(assert_condition, lambda df: df["model"].notnull().all())
        .pipe(assert_condition, lambda df: (df.groupby(["group", "model"])["model_name"].nunique() == 1).all())
        .assign(norm_value=lambda df: (
            df.groupby(["group", "metric"], group_keys=False)["value"]
            .apply(lambda g: (g - g.min()) / (g.max() - g.min()))
        ))
        .assign(
            model=lambda df: pd.Categorical(
                df["model"],
                ordered=True,
                categories=(
                    df.groupby(["group", "model"])["norm_value"].mean()
                    .reset_index()
                    .sort_values(["group", "norm_value"], ascending=True)
                    ["model"].drop_duplicates().values
                ),
            )
        )
        .pipe(assert_condition, lambda df: df["model"].notnull().all())
        .assign(group=lambda df: pd.Categorical(
            df["group"], ordered=True, 
            categories=[feature_group, algorithms_group]
        ))
        .pipe(assert_condition, lambda df: df["group"].notnull().all())
        .assign(text=lambda df: df["value"].apply("{:.2f}".format))
        .pipe(assert_condition, lambda df: df.drop(columns="limit").notnull().all().all())
    )
    # fmt: on


def prepare_opportunity_funnel(
    opportunity_statistics: PandasDataFrame,
    relative_risk_thresholds: PandasDataFrame,
    color_fn: Callable[[str, float], str],
) -> PandasDataFrame:
    assert relative_risk_thresholds["model"].nunique() == 1

    def _get_level_colors(levels: list[str]) -> dict[str, str]:
        result = {}
        for level in levels:
            value = int(level.split(":")[1]) / 5.0
            if "tractability" in level.lower():
                color = color_fn("Blues", value)
            elif "development" in level.lower():
                color = color_fn("Reds", value)
            else:
                color = color_fn("Greens", value)
            result[level] = color
        return result

    # fmt: off
    df = (
        opportunity_statistics
        .pipe(lambda df: df[df["disease__therapeutic_area"] == "ALL"])
        .pipe(lambda df: df[df["target__tractability__evidence"].isin(["ALL", "LOW", "MED", "HIGH"])])
        .assign(benchmark=lambda df: df["benchmark_slug"])
        .assign(benchmark_relative_risk=lambda df: df["benchmark_slug"].map(
            relative_risk_thresholds.set_index("benchmark_slug")["benchmark_relative_risk"]
        ))
    )

    facets = {
        "stage": "Present-day\ndevelopment\nstatus",
        "threshold": "Clinical\nadvancement\nconfidence",
        "tractability": "Tractability\nconfidence\nby modality",
    }

    stage_data = (
        df
        .pipe(lambda df: df[df["target__tractability__evidence"] == "ALL"])
        .pipe(lambda df: df[df["target__tractability__modality"] == "ALL"])
        .pipe(lambda df: df[df["target_disease__stage"] != "ALL"])
        .pipe(lambda df: df[df["benchmark_slug"] == "NONE"])
        .pipe(assert_condition, lambda df: not df["target_disease__stage"].duplicated().any())
        .assign(size=lambda df: df["model__n_pairs__over_threshold"].pipe(lambda s: s / s.sum()))
        .assign(label=lambda df: df["target_disease__stage"].map({"NONE": "Undeveloped"}).fillna(df["target_disease__stage"]))
        .pipe(assert_condition, lambda df: df["label"].notnull().all())
        .assign(level=lambda df: df["target_disease__stage"].map({
            "NONE": 0, "Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Phase 4": 4,
        }))
        .pipe(assert_condition, lambda df: df["level"].notnull().all())
        .assign(tick="Stage")
        .assign(facet=facets["stage"])
    )
    threshold_data = (
        df
        .pipe(lambda df: df[df["target__tractability__evidence"] == "ALL"])
        .pipe(lambda df: df[df["target__tractability__modality"] == "ALL"])
        .pipe(lambda df: df[df["target_disease__stage"] == "NONE"])
        .pipe(lambda df: df[df["benchmark_slug"] != "NONE"])
        .pipe(assert_condition, lambda df: not df["benchmark_slug"].duplicated().any())
        .assign(size=lambda df: df["model__n_pairs__over_threshold"].pipe(lambda s: s / s.sum()))
        .assign(label=lambda df: np.array([
            {"OTG": "Medium", "EVA": "High", "OMIM": "Highest"}[r.benchmark_slug] 
            + f"\nRR={r.benchmark_relative_risk:.2f}"
            for r in df.itertuples()
        ]))
        .pipe(assert_condition, lambda df: df["label"].notnull().all())
        .assign(level=lambda df: df["benchmark_slug"].map({
            "OTG": 0, "EVA": 1, "OMIM": 2
        }))
        .pipe(assert_condition, lambda df: df["level"].notnull().all())
        .assign(tick="Threshold")
        .assign(facet=facets["threshold"])
    )
    tractability_data = (
        df
        .pipe(lambda df: df[df["target__tractability__evidence"].isin(["LOW", "MED", "HIGH"])])
        .pipe(lambda df: df[df["target__tractability__modality"] != "ALL"])
        .pipe(lambda df: df[df["benchmark_slug"] == "EVA"])
        .pipe(lambda df: df[df["target_disease__stage"] == "NONE"])
        .pipe(assert_condition, lambda df: not df[["target__tractability__modality", "target__tractability__evidence"]].duplicated().any().any())
        .assign(size=lambda df: df.groupby("target__tractability__modality")["model__n_pairs__over_threshold"].transform(lambda s: s / s.sum()))
        .assign(label=lambda df: df["target__tractability__evidence"].map({
            "LOW": "Low", "MED": "Medium", "HIGH": "High"
        }))
        .pipe(assert_condition, lambda df: df["label"].notnull().all())
        .assign(level=lambda df: df["target__tractability__evidence"].map({
            "LOW": 0, "MED": 1, "HIGH": 2
        }))
        .pipe(assert_condition, lambda df: df["level"].notnull().all())
        .assign(tick=lambda df: df["target__tractability__modality"].map(TRACTABILITY_MODALITY_LABELS))
        .pipe(assert_condition, lambda df: df["tick"].notnull().all())
        .assign(facet=facets["tractability"])
    )

    return (
        pd.concat([stage_data, threshold_data, tractability_data], axis=0, ignore_index=True)
        .assign(facet=lambda df: pd.Categorical(
            df["facet"], ordered=True,
            categories=list(facets.values())
        ))
        .pipe(assert_condition, lambda df: df["facet"].notnull().all())
        .assign(tick=lambda df: pd.Categorical(
            df["tick"], ordered=True,
            categories=["Stage", "Threshold"] + list(TRACTABILITY_MODALITY_LABELS.values())[::-1]
        ))
        .pipe(assert_condition, lambda df: df["tick"].notnull().all())
        .assign(label=lambda df: np.where(
            (df["facet"] != facets["stage"]) | (df["label"].isin(["Undeveloped", "Phase 2"])),
            df["label"] + "\n(" + df["model__n_pairs__over_threshold"].astype(int).apply("{:,}".format) + ")",
            ""
        ))
        .assign(level=lambda df: df["facet"].astype(str) + ":" + df["level"].astype(str))
        .assign(level_color=lambda df: df["level"].map(_get_level_colors(df["level"].drop_duplicates().to_list())))
        .assign(level=lambda df: pd.Categorical(
            df["level"], ordered=True, 
            categories=df["level"].drop_duplicates().sort_values(ascending=False)
        ))
        .pipe(assert_condition, lambda df: df["level"].notnull().all())
    )
    # fmt: on


###############################################################################
# Utils
###############################################################################

P = ParamSpec("P")
T = TypeVar("T")


def spark_lambda(
    fn: Callable[P, Any]
) -> Callable[[SparkDataFrame, P.args, P.kwargs], SparkDataFrame]:
    """
    Syntactic sugar to access lambda function as input to transform
    method.
    """
    return fn  # type:ignore[return-value]


def apply(
    obj: T, fn: Callable[Concatenate[T, P], T], *args: P.args, **kwargs: P.kwargs
) -> T:
    fn(obj, *args, **kwargs)
    return obj


def assert_condition(value: T, condition: Callable[[T], bool], msg: Any = None) -> T:
    if msg is None:
        assert condition(value)
    else:
        assert condition(value), msg
    return value


def get_gcs_path(version: str, dataset: str) -> str:
    if version not in DATASETS:
        raise NotImplementedError(f"Version {version!r} not supported")
    table = DATASETS[version][dataset]
    return f"gs://open-targets-data-releases/{version}/output/etl/parquet/{table}"


def _get_spark(
    *,
    project: str,
    driver_memory_limit: str | None = None,
    driver_host: str = "127.0.0.1",
    configuration: dict[str, Any] | None = None,
    enable_arrow: bool = True,
    mem_fraction: float = 0.8,
) -> SparkSession:
    conf = get_gcs_enabled_config(project=project)
    default_options = {
        "spark.driver.defaultJavaOptions": "-Duser.timezone=UTC",
        "spark.executor.defaultJavaOptions": "-Duser.timezone=UTC",
        "spark.sql.session.timeZone": "UTC",
    }
    for key, value in default_options.items():
        conf.set(key, value)
    if driver_memory_limit is not None:
        conf.set("spark.driver.memory", driver_memory_limit)

    conf.set("spark.driver.host", driver_host)

    if driver_memory_limit is None and "SPARK_DRIVER_MEMORY" not in os.environ:
        default_mem = math.ceil(mem_fraction * (psutil.virtual_memory().total >> 30))
        conf.set("spark.driver.memory", f"{default_mem}g")

    if "PYSPARK_PYTHON" not in os.environ:
        os.environ["PYSPARK_PYTHON"] = sys.executable
        conf.set("spark.pyspark.python", sys.executable)

    conf.set("spark.sql.execution.arrow.pyspark.enabled", str(enable_arrow).lower())

    configuration = configuration or {}
    for k, v in configuration.items():
        conf.set(k, v)

    return SparkSession.builder.config(conf=conf).getOrCreate()


def get_spark(project: str | None = None, mem_fraction: float = 0.8) -> SparkSession:
    if project is None:
        load_dotenv()
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project is None:
        raise ValueError("Project not provided and GOOGLE_CLOUD_PROJECT not set")
    return _get_spark(
        project=project,
        mem_fraction=mem_fraction,
        configuration={
            "spark.hadoop.fs.gs.requester.pays.mode": "CUSTOM",
            "spark.hadoop.fs.gs.requester.pays.project.id": project,
            "spark.hadoop.fs.gs.requester.pays.buckets": "open-targets-data-releases",
            "spark.driver.maxResultSize": "4G",
        },
    )


def to_snake_case(df: SparkDataFrame) -> SparkDataFrame:
    return df.select(
        *[
            F.col(c).alias(re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower())
            for c in df.columns
        ]
    )


def shorten_text(text: str, width: int, placeholder: str = "...") -> str:
    if len(text) <= width:
        return text
    return text[: (width - len(placeholder))] + placeholder


###############################################################################
# Execution
###############################################################################


def export_notebook(path: str) -> str:
    import nbformat
    from nbconvert import HTMLExporter

    input_path = Path(path)

    if not input_path.name.endswith(".ipynb"):
        raise ValueError(
            f"Input path {str(input_path)!r} does not end with extension .ipynb"
        )
    if not input_path.exists():
        raise ValueError(f"Input path {str(input_path)!r} does not exist")

    ouput_path = input_path.parent / f"{input_path.name.removesuffix('.ipynb')}.html"

    logger.info(f"Exporting notebook from {str(input_path)!r} to {str(ouput_path)!r}")
    with open(input_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.template_name = "classic"
        (body, _) = html_exporter.from_notebook_node(nb)
        with open(ouput_path, "w", encoding="utf-8") as f:
            f.write(body)
    logger.info(f"Notebook exported to {str(ouput_path)!r}")
    return str(ouput_path)


def run_analysis(
    output_path: str,
    version: str | float,
    max_training_transition_year: int = DEFAULT_MAX_TRAINING_TRANSITION_YEAR,
    clinical_advancement_window: int = DEFAULT_CLINICAL_ADVANCEMENT_WINDOW,
    notebook_name: str = "analysis",
) -> None:
    import papermill as pm

    if isinstance(version, float):
        version = str(version)
    output_dir = Path(output_path) / "results" / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(Path(__file__).parent / f"{notebook_name}.ipynb")
    parameters = {
        "open_targets_version": version,
        "clinical_advancement_window": clinical_advancement_window,
        "max_training_transition_year": max_training_transition_year,
        "validate_modeling_features": True,
        "enable_result_export": False,
        "enable_opportunity_analysis": False,
        "enable_meta_analysis": False,
    }
    output_path = str(
        output_dir
        / f"{notebook_name}__{version}__{max_training_transition_year}__{clinical_advancement_window}.ipynb"
    )
    logger.info(
        f"Running analysis with input={input_path!r}, output={output_path!r}, parameters={parameters}"
    )
    pm.execute_notebook(
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
    )
    output_path = export_notebook(output_path)
    logger.info(f"Analysis notebook exported to {output_path!r}")


def run_analysis_configs(output_path: str) -> None:
    import tqdm

    year = DEFAULT_MAX_TRAINING_TRANSITION_YEAR
    configs = [
        {
            "version": version,
            "max_training_transition_year": max_training_transition_year,
            "clinical_advancement_window": clinical_advancement_window,
        }
        for version in sorted(DATASETS.keys())
        for max_training_transition_year in [year - 2, year, year + 2]
        for clinical_advancement_window in [2, 4]
    ]
    for config in tqdm.tqdm(configs):
        run_analysis(output_path, **config)
    logger.info(f"Analysis notebooks exported to {output_path!r}")


###############################################################################
# Sensitivity
###############################################################################


def get_sensitivity_analysis_configs(data_dir: str) -> PandasDataFrame:
    return pd.DataFrame(
        [
            {
                "path": path,
                "filename": (filename := path.name.removesuffix(".html")),
                "version": filename.split("__")[1],
                "year": int(filename.split("__")[2]),
                "window": int(filename.split("__")[3]),
            }
            for path in list((Path(data_dir) / "results" / "notebooks").glob("*.html"))
        ]
    )


def get_sensitivity_data(configs: PandasDataFrame) -> dict[str, PandasDataFrame]:
    import tqdm
    from bs4 import BeautifulSoup

    def parse_table(html: str) -> PandasDataFrame:
        return pd.read_html(str(html))[0].pipe(
            lambda df: pd.DataFrame(
                df.values,
                columns=[
                    [e for e in c if not e.startswith("Unnamed:")][-1] for c in df
                ],
                index=df.index,
            ).set_index([c[1] for c in df.columns if not c[1].startswith("Unnamed:")])
        )

    def extract_table(path: Path, table_caption: str) -> pd.DataFrame:
        with open(path) as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            caption = table.find("caption")
            if caption and (table_caption.lower() == caption.text.lower()):
                return parse_table(str(table))
        raise ValueError(f"Table with caption {table_caption!r} not found")

    def extract_dataframe(table_caption: str) -> pd.DataFrame:
        return pd.concat(
            [
                extract_table(path=config.path, table_caption=table_caption)
                .reset_index()
                .assign(version=config.version, year=config.year, window=config.window)
                for config in tqdm.tqdm(configs.itertuples(), total=len(configs))
            ],
            axis=0,
            ignore_index=True,
        )

    return {
        table: extract_dataframe(table_caption=properties["caption"])
        for table, properties in PRIMARY_TABLES.items()
    }


def display_sensitivity_relative_risk_averages(
    sensitivity_data: dict[str, PandasDataFrame]
) -> PandasDataFrame:
    return (
        sensitivity_data["relative_risk_averages_by_ta"]
        .assign(limit=lambda df: df["limit"].astype(int))
        .set_index(["version", "year", "window", "limit"])
        .rename_axis("model", axis="columns")
        .pipe(
            apply,
            lambda df: display(
                df.assign(ratio=lambda df: df["RDG"] / df["OTS"])["ratio"]
                .unstack("limit")
                .sort_index()
                .pipe(
                    apply,
                    lambda df: display(
                        df.describe()
                        .T.style.format(precision=3)
                        .format(precision=0, subset="count")
                        .background_gradient(cmap="Blues", axis=0)
                        .set_caption(
                            "Relative risk ratio distribution across configurations"
                        )
                    ),
                )
                .style.format(precision=3)
                .background_gradient(cmap="Blues", axis=None)
                .set_caption("Relative risk ratios across configurations")
            ),
        )
        .stack()
        .rename("relative_risk")
        .reset_index()
        .pipe(
            apply,
            lambda df: display(
                df.groupby(["limit", "model"])["relative_risk"]
                .describe()
                .style.format(precision=3)
                .format(precision=0, subset="count")
                .background_gradient(cmap="Blues", axis=0)
                .set_caption("Relative risk average distribution across configurations")
            ),
        )
        .pipe(
            apply,
            lambda df: display(
                df.groupby(["limit", "model"])["relative_risk"]
                .mean()
                .unstack("model")
                .style.background_gradient(cmap="Blues", axis=None)
                .set_caption("Relative risk average across configurations")
            ),
        )
        .assign(
            limit=lambda df: pd.Categorical(
                df["limit"].astype(int),
                ordered=True,
                categories=df["limit"].drop_duplicates().sort_values(),
            )
        )
    )


def display_sensitivity_p_values(
    sensitivity_data: dict[str, PandasDataFrame]
) -> PandasDataFrame:
    return (
        sensitivity_data["relative_risk_p_values"]
        .assign(limit=lambda df: df["limit"].astype(int))
        .set_index(["version", "year", "window", "limit"])
        .rename_axis("model", axis="columns")
        .pipe(lambda df: -np.log10(df))
        .pipe(
            apply,
            lambda df: display(
                df["RDG"]
                .unstack("limit")
                .sort_index()
                .pipe(
                    apply,
                    lambda df: display(
                        df.describe()
                        .T.style.format(precision=3)
                        .format(precision=0, subset="count")
                        .background_gradient(cmap="Blues", axis=0)
                        .set_caption(
                            "P-value exponent (log10) distribution across configurations"
                        )
                    ),
                )
                .style.format(precision=3)
                .background_gradient(cmap="Blues", axis=None)
                .set_caption("P-value exponent (log10) across configurations")
            ),
        )
        .stack()
        .rename("neg_log10_p_value")
        .reset_index()
        .pipe(
            apply,
            lambda df: display(
                df.groupby(["limit", "model"])["neg_log10_p_value"]
                .describe()
                .style.format(precision=3)
                .format(precision=0, subset="count")
                .background_gradient(cmap="Blues", axis=0)
                .set_caption("P value (-log10) distribution across configurations")
            ),
        )
        .pipe(
            apply,
            lambda df: display(
                df.groupby(["limit", "model"])["neg_log10_p_value"]
                .mean()
                .unstack("model")
                .style.background_gradient(cmap="Blues", axis=None)
                .set_caption(
                    "P values (-log10) by model and limit across configurations"
                )
            ),
        )
        .assign(
            limit=lambda df: pd.Categorical(
                df["limit"].astype(int),
                ordered=True,
                categories=df["limit"].drop_duplicates().sort_values(),
            )
        )
    )


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    # Example invocation:
    # ./bin/run_py --output_mode=dev -- \
    # pipelines/analyses/publications/clinical_progression_forecasting/public/analysis.py \
    # export_features --version='23.12' --output-path=/home/eczech/repos/facets/local/data/open_targets_ml
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "export_features": export_features,
            "aggregate_features": aggregate_features,
            "run_analysis": run_analysis,
            "run_analysis_configs": run_analysis_configs,
        }
    )
