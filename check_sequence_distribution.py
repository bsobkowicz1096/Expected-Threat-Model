"""Check sequence length distribution with ALL carries (no filtering)."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, when, sqrt, expr
from pyspark.sql.functions import percentile_approx

spark = SparkSession.builder \
    .appName("sequence_distribution_check") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

df = spark.read.parquet('data/all_events_combined_2015_2016.parquet')

# Filter to Pass, Shot, Carry (no distance filter on carries)
df_movement = df.filter(col('type').isin(['Pass', 'Shot', 'Carry']))

# Exclude penalty shots
df_movement = df_movement.filter(
    (col("type") != "Shot") |
    (col("shot_type").isNull()) |
    (col("shot_type") != "Penalty")
)

print("=== Event counts (no carry filter) ===")
df_movement.groupBy('type').count().orderBy('count', ascending=False).show()

# Count events per possession
possession_lengths = df_movement.groupBy('match_id', 'possession') \
    .agg(count('*').alias('seq_len'))

total = possession_lengths.count()
print(f"Total possessions: {total:,}")

print("\n=== Sequence length distribution (no carry filter) ===")
possession_lengths.select('seq_len').describe().show()

print("=== Percentiles ===")
possession_lengths.selectExpr(
    "percentile_approx(seq_len, 0.10) as p10",
    "percentile_approx(seq_len, 0.25) as p25",
    "percentile_approx(seq_len, 0.50) as p50",
    "percentile_approx(seq_len, 0.75) as p75",
    "percentile_approx(seq_len, 0.85) as p85",
    "percentile_approx(seq_len, 0.90) as p90",
    "percentile_approx(seq_len, 0.95) as p95",
    "percentile_approx(seq_len, 0.99) as p99",
    "percentile_approx(seq_len, 1.00) as max"
).show()

# Histogram-style binned counts
print("=== Binned distribution ===")
possession_lengths.withColumn(
    "bin",
    when(col("seq_len") <= 3, "1-3")
    .when(col("seq_len") <= 6, "4-6")
    .when(col("seq_len") <= 10, "7-10")
    .when(col("seq_len") <= 15, "11-15")
    .when(col("seq_len") <= 20, "16-20")
    .when(col("seq_len") <= 30, "21-30")
    .when(col("seq_len") <= 50, "31-50")
    .otherwise("51+")
).groupBy("bin").agg(
    count("*").alias("count")
).orderBy("bin").show()

# Compare with pass-only baseline distribution
print("\n=== For comparison: Pass+Shot only (baseline) ===")
df_baseline = df.filter(col('type').isin(['Pass', 'Shot']))
df_baseline = df_baseline.filter(
    (col("type") != "Shot") |
    (col("shot_type").isNull()) |
    (col("shot_type") != "Penalty")
)
baseline_lengths = df_baseline.groupBy('match_id', 'possession') \
    .agg(count('*').alias('seq_len'))

baseline_lengths.selectExpr(
    "percentile_approx(seq_len, 0.10) as p10",
    "percentile_approx(seq_len, 0.25) as p25",
    "percentile_approx(seq_len, 0.50) as p50",
    "percentile_approx(seq_len, 0.75) as p75",
    "percentile_approx(seq_len, 0.85) as p85",
    "percentile_approx(seq_len, 0.90) as p90",
    "percentile_approx(seq_len, 0.95) as p95",
    "percentile_approx(seq_len, 0.99) as p99",
    "percentile_approx(seq_len, 1.00) as max"
).show()

# How much do carries inflate sequence lengths?
print("=== Carry ratio per possession ===")
carry_stats = df_movement.groupBy('match_id', 'possession').agg(
    count('*').alias('total'),
    count(when(col('type') == 'Carry', True)).alias('n_carries'),
    count(when(col('type') == 'Pass', True)).alias('n_passes'),
    count(when(col('type') == 'Shot', True)).alias('n_shots')
).withColumn("carry_pct", expr("round(n_carries / total * 100, 1)"))

carry_stats.selectExpr(
    "percentile_approx(carry_pct, 0.50) as median_carry_pct",
    "round(avg(carry_pct), 1) as mean_carry_pct",
    "round(avg(n_carries), 1) as mean_carries",
    "round(avg(n_passes), 1) as mean_passes",
    "round(avg(total), 1) as mean_total"
).show()

spark.stop()
print("\nDone!")
