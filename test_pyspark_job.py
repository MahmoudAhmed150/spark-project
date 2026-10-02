import pytest
from pyspark.sql import SparkSession

from pyspark_job import clean_data


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("PySparkTests")
        .getOrCreate()
    )

    yield spark

    spark.stop()


def test_valid_records_are_kept(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, "Mona", 50.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df).collect()

    assert len(result) == 2


def test_non_positive_amounts_are_removed(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, "Mona", 0.0),
        (3, "Ali", -50.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df).collect()

    assert len(result) == 1
    assert result[0]["name"] == "Ahmed"


def test_null_names_are_removed(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, None, 50.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df).collect()

    assert len(result) == 1
    assert result[0]["name"] == "Ahmed"


def test_amount_with_tax_is_correct(spark):
    data = [
        (1, "Ahmed", 100.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df).collect()

    assert result[0]["amount_with_tax"] == pytest.approx(120.0)
