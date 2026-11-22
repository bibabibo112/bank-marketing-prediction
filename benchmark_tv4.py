"""Measure Decision Tree fit on the team's processed training split."""

import argparse
import csv
import os
import platform
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parent
DEPTH = 8
LEAF_SIZE = 20
SEED = 42


def measure_fit(fit, processes):
    def rss():
        return sum(process.memory_info().rss for process in processes)

    done = Event()
    peak = rss()

    def sample():
        nonlocal peak
        while not done.wait(0.02):
            peak = max(peak, rss())

    monitor = Thread(target=sample)
    monitor.start()
    try:
        start = perf_counter()
        fit()
        seconds = perf_counter() - start
    finally:
        done.set()
        monitor.join()
    return seconds, max(peak, rss()) / 2**20


def record(fit, processes, args, factor, n_rows, engine_version, java=""):
    records = []
    for run in range(1, args.runs + 1):
        seconds, memory = measure_fit(fit, processes)
        records.append({
            "engine": args.engine,
            "factor": factor,
            "train_rows": n_rows,
            "run": run,
            "fit_seconds": round(seconds, 6),
            "peak_rss_mib": round(memory, 3),
            "engine_version": engine_version,
            "java_version": java,
            "python_version": platform.python_version(),
            "os": platform.system(),
            "fit_cores": 1,
        })
        print(f"{args.engine} {factor}x run {run}: {seconds:.3f} s, {memory:.1f} MiB", flush=True)

    previous = []
    if args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as source:
            previous = [
                row for row in csv.DictReader(source)
                if (row["engine"], int(row["factor"])) != (args.engine, factor)
            ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(previous + records)
    temp.replace(args.output)


def scikit_benchmark(X, y, args):
    from sklearn.tree import DecisionTreeClassifier

    processes = [psutil.Process()]
    for factor in args.factors:
        inputs, labels = (X, y) if factor == 1 else (np.tile(X, (factor, 1)), np.tile(y, factor))

        def fit():
            DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=LEAF_SIZE, random_state=SEED).fit(
                inputs, labels,
            )

        if factor == args.factors[0]:
            fit()
        record(fit, processes, args, factor, len(labels), version("scikit-learn"))
        del inputs, labels


def spark_benchmark(path, args):
    from pyspark import StorageLevel
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import broadcast

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    local = ROOT / ".cache" / "tv4_spark"
    local.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[1]").appName("bank-marketing-tv4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "512m")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.dir", str(local))
        .config("spark.sql.warehouse.dir", (local / "warehouse").as_uri())
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        jvm = spark.sparkContext._jvm
        processes = [psutil.Process(), psutil.Process(jvm.java.lang.ProcessHandle.current().pid())]
        java = jvm.java.lang.System.getProperty("java.version")
        raw = spark.read.parquet(path.as_uri())
        names = ["label" if column == "y" else f"x{index}" for index, column in enumerate(raw.columns)]
        assembler = VectorAssembler(inputCols=[name for name in names if name != "label"], outputCol="features")
        base = assembler.transform(raw.toDF(*names)).select("features", "label")
        original_count = base.count()

        # ponytail: a local benchmark; use TV3's run table for any cluster comparison.
        for factor in args.factors:
            data = (
                spark.range(factor, numPartitions=min(factor, 32))
                .crossJoin(broadcast(base)).drop("id").persist(StorageLevel.MEMORY_AND_DISK)
            )
            try:
                n_rows = data.count()
                if n_rows != original_count * factor:
                    raise RuntimeError(f"Unexpected Spark row count: {n_rows}")

                def fit():
                    DecisionTreeClassifier(
                        maxDepth=DEPTH, minInstancesPerNode=LEAF_SIZE, seed=SEED,
                    ).fit(data)

                if factor == args.factors[0]:
                    fit()
                record(fit, processes, args, factor, n_rows, spark.version, java)
            finally:
                data.unpersist(blocking=True)
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["scikit-learn", "Spark"])
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "tv4" / "benchmark_runs.csv")
    args = parser.parse_args()
    args.factors = sorted(set(args.factors))
    if args.runs < 1 or args.factors[0] < 1:
        parser.error("Factors and runs must be positive.")
    shared = ROOT / "data" / "processed" / "train.parquet"
    if not shared.is_file():
        parser.error("Expected the team's data/processed/train.parquet.")
    train = pd.read_parquet(shared)
    if "y" not in train or "duration" in train or set(train["y"].unique()) != {0, 1}:
        parser.error("Expected the main training split, with labels 0/1 and without duration.")
    if not np.isfinite(train).all(axis=None):
        parser.error("Expected a finite numeric training frame.")
    if args.engine == "Spark":
        del train
        spark_benchmark(shared, args)
    else:
        X = train.drop(columns="y").to_numpy(dtype=np.float32, copy=True)
        y = train["y"].to_numpy(dtype=np.int8)
        del train
        scikit_benchmark(X, y, args)


if __name__ == "__main__":
    main()
