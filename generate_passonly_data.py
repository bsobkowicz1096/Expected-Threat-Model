"""
Generate pass-only training data with configurable context length.
Mirror of generate_carry_data.py but without Carry events.

Usage:
    python generate_passonly_data.py --max_seq_len 8
"""

import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, struct, collect_list, desc,
    row_number, size, transform, round as spark_round, expr
)
from pyspark.sql.window import Window


def main(args):
    spark = SparkSession.builder \
        .appName("xT_passonly_preprocessing") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    MAX_SEQUENCE_LENGTH = args.max_seq_len

    print(f"=== Generating pass-only data with max_seq_len={MAX_SEQUENCE_LENGTH} ===")

    # Load raw data
    df = spark.read.parquet('data/all_events_combined_2015_2016.parquet')
    print(f"Total events: {df.count():,}")

    # Select needed columns
    df_filtered = df.select(
        'match_id', 'possession', 'team', 'possession_team', 'index',
        'type', 'location', 'pass_end_location',
        'shot_outcome', 'shot_type'
    )

    # Filter to Pass, Shot only (no Carry)
    df_movement = df_filtered.filter(col('type').isin(['Pass', 'Shot']))

    # Exclude penalty shots
    df_movement = df_movement.filter(
        (col("type") != "Shot") |
        (col("shot_type").isNull()) |
        (col("shot_type") != "Penalty")
    )

    print(f"\nEvent type distribution:")
    df_movement.groupBy('type').count().orderBy('count', ascending=False).show()

    # Flip coordinates for defensive team events
    df_movement = df_movement.withColumn(
        "location",
        when(
            col("team") != col("possession_team"),
            F.array(
                F.round(lit(120) - col("location")[0], 1),
                F.round(lit(80) - col("location")[1], 1)
            )
        ).otherwise(col("location"))
    ).withColumn(
        "pass_end_location",
        when(
            col("team") != col("possession_team"),
            F.array(
                F.round(lit(120) - col("pass_end_location")[0], 1),
                F.round(lit(80) - col("pass_end_location")[1], 1)
            )
        ).otherwise(col("pass_end_location"))
    )

    # Extract scalar coordinates
    df_with_coords = df_movement \
        .withColumn("x", col("location")[0].cast("double")) \
        .withColumn("y", col("location")[1].cast("double")) \
        .withColumn("end_x", col("pass_end_location")[0].cast("double")) \
        .withColumn("end_y", col("pass_end_location")[1].cast("double"))

    # Truncation: keep last MAX_SEQUENCE_LENGTH events
    n_before = df_with_coords.count()
    window_spec = Window.partitionBy('match_id', 'possession').orderBy(desc('index'))
    df_numbered = df_with_coords.withColumn('event_rank', row_number().over(window_spec))
    df_truncated = df_numbered.filter(col('event_rank') <= MAX_SEQUENCE_LENGTH)

    n_after = df_truncated.count()
    print(f"\nEvents before truncation: {n_before:,}")
    print(f"Events after truncation:  {n_after:,}")

    # Build event structs
    df_events = df_truncated.withColumn(
        'is_goal',
        when(col('shot_outcome') == 'Goal', 1).otherwise(0).cast('int')
    ).withColumn(
        'event',
        struct(
            col('event_rank').alias('rank'),
            col('type').alias('type'),
            col('x').alias('x'),
            col('y').alias('y'),
            col('end_x').alias('end_x'),
            col('end_y').alias('end_y'),
            col('is_goal').alias('is_goal')
        )
    )

    # Aggregate into sequences per possession
    sequences = df_events.groupBy('match_id', 'possession').agg(
        collect_list('event').alias('events')
    )

    # Sort chronologically
    sequences = sequences.withColumn(
        'events',
        expr("array_sort(events, (left, right) -> case when left.rank < right.rank then 1 else -1 end)")
    )

    # Detect goal possessions
    sequences = sequences.withColumn(
        'goal',
        expr("aggregate(events, 0, (acc, x) -> acc + x.is_goal)").cast('int')
    )

    # Add terminal token
    sequences = sequences.withColumn(
        'end_event',
        struct(
            lit(0).alias('rank'),
            when(col('goal') == 1, lit('GOAL')).otherwise(lit('NO_GOAL')).alias('type'),
            lit(None).cast('double').alias('x'),
            lit(None).cast('double').alias('y'),
            lit(None).cast('double').alias('end_x'),
            lit(None).cast('double').alias('end_y'),
            col('goal').alias('is_goal')
        )
    )
    sequences = sequences.withColumn('events', expr("concat(events, array(end_event))"))
    sequences = sequences.withColumn('sequence_length', size('events'))

    sequences_final = sequences.select('match_id', 'possession', 'events', 'sequence_length', 'goal')

    print(f"\n=== SEQUENCE STATISTICS ===")
    sequences_final.select('sequence_length').describe().show()
    sequences_final.groupBy('goal').count().show()
    total = sequences_final.count()
    print(f"Total possessions: {total:,}")

    # Train/Val/Test split (same seed as carry for comparability)
    train_sequences, val_sequences, test_sequences = sequences_final.randomSplit(
        [0.80, 0.10, 0.10], seed=42
    )

    n_train = train_sequences.count()
    n_val = val_sequences.count()
    n_test = test_sequences.count()
    print(f"\nTrain: {n_train:,}")
    print(f"Val:   {n_val:,}")
    print(f"Test:  {n_test:,}")

    # Balance train set (target ~5% goal rate)
    goals_train = train_sequences.filter(col('goal') == 1)
    non_goals_train = train_sequences.filter(col('goal') == 0)
    n_goals_train = goals_train.count()
    n_non_goals = non_goals_train.count()
    sample_fraction = (n_goals_train / 0.05 - n_goals_train) / n_non_goals
    train_balanced = goals_train.union(
        non_goals_train.sample(fraction=sample_fraction, seed=42)
    )

    print(f"\nTrain after balancing: {train_balanced.count():,}")
    train_balanced.groupBy('goal').count().show()

    # Normalize coordinates to [0, 1]
    def normalize_events(df):
        return df.withColumn(
            'events',
            transform('events', lambda e: struct(
                e.type.alias('type'),
                spark_round(e.x / 120.0, 5).alias('x'),
                spark_round(e.y / 80.0, 5).alias('y'),
                spark_round(e.end_x / 120.0, 5).alias('end_x'),
                spark_round(e.end_y / 80.0, 5).alias('end_y')
            ))
        )

    train_normalized = normalize_events(train_balanced)
    val_normalized = normalize_events(val_sequences)
    test_normalized = normalize_events(test_sequences)

    # Save
    ctx_suffix = f"_ctx{MAX_SEQUENCE_LENGTH}" if MAX_SEQUENCE_LENGTH != 12 else ""

    train_normalized.toPandas().to_parquet(
        f'data/sequences_passonly{ctx_suffix}_train_balanced.parquet', index=False
    )
    val_normalized.toPandas().to_parquet(
        f'data/sequences_passonly{ctx_suffix}_val_natural.parquet', index=False
    )
    test_normalized.toPandas().to_parquet(
        f'data/sequences_passonly{ctx_suffix}_test_natural.parquet', index=False
    )

    # Save vocab (pass-only, 5 tokens)
    type_vocab = {
        'Pass': 0,
        'Shot': 1,
        'GOAL': 2,
        'NO_GOAL': 3,
        '<pad>': 4,
    }
    with open('data/vocab_continuous_passonly.json', 'w') as f:
        json.dump(type_vocab, f, indent=2)

    print(f"\nSaved:")
    print(f"  data/sequences_passonly{ctx_suffix}_train_balanced.parquet")
    print(f"  data/sequences_passonly{ctx_suffix}_val_natural.parquet")
    print(f"  data/sequences_passonly{ctx_suffix}_test_natural.parquet")
    print(f"  data/vocab_continuous_passonly.json")

    spark.stop()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pass-only data for xT model")
    parser.add_argument("--max_seq_len", type=int, default=12,
                        help="Max events per possession before terminal token (default: 12)")
    args = parser.parse_args()
    main(args)
