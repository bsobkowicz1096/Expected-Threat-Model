"""
Generate training data for xT model with Carry events.

Three modes:
  Threshold: Hard filter carries by distance
    python generate_carry_data.py --carry_threshold 8

  Adaptive: Include ALL carries, but when truncating long possessions
            drop shortest carries first (no arbitrary threshold)
    python generate_carry_data.py --adaptive

  No filter: Include ALL carries without any distance filter
    python generate_carry_data.py --no_filter
"""

import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lit, when, struct, collect_list, desc,
    row_number, size, transform, round as spark_round, expr, sqrt
)
from pyspark.sql.window import Window


def main(args):
    spark = SparkSession.builder \
        .appName("xT_carry_preprocessing") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    adaptive = args.adaptive
    no_filter = args.no_filter
    threshold = args.carry_threshold
    MAX_SEQUENCE_LENGTH = args.max_seq_len

    if no_filter:
        print(f"=== Generating carry data with NO FILTER (all carries included) ===")
    elif adaptive:
        print(f"=== Generating carry data with ADAPTIVE truncation ===")
    else:
        print(f"=== Generating carry data with threshold >= {threshold} ===")

    # Load raw data
    df = spark.read.parquet('data/all_events_combined_2015_2016.parquet')
    print(f"Total events: {df.count():,}")

    # Select needed columns
    df_filtered = df.select(
        'match_id', 'possession', 'team', 'possession_team', 'index',
        'type', 'location', 'pass_end_location', 'carry_end_location',
        'shot_outcome', 'shot_type'
    )

    # Filter to Pass, Shot, Carry
    movement_types = ['Pass', 'Shot', 'Carry']
    df_movement = df_filtered.filter(col('type').isin(movement_types))

    # Exclude penalty shots
    df_movement = df_movement.filter(
        (col("type") != "Shot") |
        (col("shot_type").isNull()) |
        (col("shot_type") != "Penalty")
    )

    # Compute carry distance (needed for both modes)
    df_movement = df_movement.withColumn(
        "carry_dist",
        when(
            col("type") == "Carry",
            sqrt(
                (col("carry_end_location")[0] - col("location")[0]) ** 2 +
                (col("carry_end_location")[1] - col("location")[1]) ** 2
            )
        ).otherwise(lit(None))
    )

    if not adaptive and not no_filter:
        # Threshold mode: hard filter carries by distance
        df_movement = df_movement.filter(
            (col("type") != "Carry") | (col("carry_dist") >= threshold)
        )

    print(f"\nEvent type distribution after filtering:")
    df_movement.groupBy('type').count().orderBy('count', ascending=False).show()

    # Flip coordinates for defensive team events
    # StatsBomb gives coordinates from the acting team's perspective,
    # we need everything from the possession team's perspective
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
    ).withColumn(
        "carry_end_location",
        when(
            (col("team") != col("possession_team")) & col("carry_end_location").isNotNull(),
            F.array(
                F.round(lit(120) - col("carry_end_location")[0], 1),
                F.round(lit(80) - col("carry_end_location")[1], 1)
            )
        ).otherwise(col("carry_end_location"))
    )

    # Unified end_location: pass_end_location for passes, carry_end_location for carries
    df_movement = df_movement.withColumn(
        "end_location",
        when(col("type") == "Pass", col("pass_end_location"))
        .when(col("type") == "Carry", col("carry_end_location"))
        .otherwise(lit(None))
    )

    # Extract scalar coordinates
    df_with_coords = df_movement \
        .withColumn("x", col("location")[0].cast("double")) \
        .withColumn("y", col("location")[1].cast("double")) \
        .withColumn("end_x", col("end_location")[0].cast("double")) \
        .withColumn("end_y", col("end_location")[1].cast("double"))

    # Truncation
    n_before = df_with_coords.count()

    if adaptive:
        # Adaptive truncation: drop shortest carries first, then earliest events
        # Drop priority: 1=short carries (drop first), 2=passes/shots (drop last)
        # Within carries: shortest first. Within passes/shots: earliest first.
        from pyspark.sql.functions import count as spark_count

        df_with_priority = df_with_coords.withColumn(
            "drop_priority",
            when(col("type") == "Carry", lit(1)).otherwise(lit(2))
        ).withColumn(
            "carry_dist_sort",
            when(col("type") == "Carry", col("carry_dist")).otherwise(lit(9999.0))
        )

        # Rank events by drop order: short carries first, then earliest events
        window_drop = Window.partitionBy('match_id', 'possession').orderBy(
            col("drop_priority").asc(),
            col("carry_dist_sort").asc(),
            col("index").asc()
        )
        df_drop_ranked = df_with_priority.withColumn(
            'drop_rank', row_number().over(window_drop)
        )

        # Count total events per possession
        possession_counts = df_with_coords.groupBy('match_id', 'possession') \
            .agg(spark_count('*').alias('total_events'))

        df_with_counts = df_drop_ranked.join(
            possession_counts, on=['match_id', 'possession']
        )

        # Keep events that survive truncation:
        # drop_rank > (total_events - MAX_SEQUENCE_LENGTH) means "keep"
        # If total_events <= MAX, all events pass (n_to_drop <= 0)
        df_truncated_raw = df_with_counts.filter(
            col('drop_rank') > (col('total_events') - lit(MAX_SEQUENCE_LENGTH))
        )

        # Re-rank by index DESC for chronological ordering (same as baseline)
        window_chrono = Window.partitionBy('match_id', 'possession').orderBy(desc('index'))
        df_truncated = df_truncated_raw.withColumn(
            'event_rank', row_number().over(window_chrono)
        )
    else:
        # Standard truncation: keep last MAX_SEQUENCE_LENGTH events
        window_spec = Window.partitionBy('match_id', 'possession').orderBy(desc('index'))
        df_numbered = df_with_coords.withColumn('event_rank', row_number().over(window_spec))
        df_truncated = df_numbered.filter(col('event_rank') <= MAX_SEQUENCE_LENGTH)

    n_after = df_truncated.count()
    print(f"\nEvents before truncation: {n_before:,}")
    print(f"Events after truncation:  {n_after:,}")
    if adaptive:
        print(f"Truncation mode: adaptive (shortest carries dropped first)")

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

    # Sort chronologically (high rank = earlier event)
    sequences = sequences.withColumn(
        'events',
        expr("array_sort(events, (left, right) -> case when left.rank < right.rank then 1 else -1 end)")
    )

    # Detect goal possessions
    sequences = sequences.withColumn(
        'goal',
        expr("aggregate(events, 0, (acc, x) -> acc + x.is_goal)").cast('int')
    )

    # Add terminal token (GOAL or NO_GOAL)
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
    sequences_final.selectExpr(
        "percentile_approx(sequence_length, 0.50) as p50",
        "percentile_approx(sequence_length, 0.75) as p75",
        "percentile_approx(sequence_length, 0.90) as p90",
        "percentile_approx(sequence_length, 0.95) as p95",
        "percentile_approx(sequence_length, 0.99) as p99"
    ).show()
    sequences_final.groupBy('goal').count().show()
    total = sequences_final.count()
    print(f"Total possessions: {total:,}")

    # Train/Val/Test split (same seed as baseline for comparability)
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

    # Save via pandas (run in pyspark311 env for compatible pandas/numpy)
    threshold_suffix = "nofilter" if no_filter else ("adaptive" if adaptive else str(int(threshold)))
    ctx_suffix = f"_ctx{MAX_SEQUENCE_LENGTH}" if MAX_SEQUENCE_LENGTH != 14 else ""
    suffix = f"{threshold_suffix}{ctx_suffix}"

    train_normalized.toPandas().to_parquet(
        f'data/sequences_carry{suffix}_train_balanced.parquet', index=False
    )
    val_normalized.toPandas().to_parquet(
        f'data/sequences_carry{suffix}_val_natural.parquet', index=False
    )
    test_normalized.toPandas().to_parquet(
        f'data/sequences_carry{suffix}_test_natural.parquet', index=False
    )

    # Save vocab with Carry
    type_vocab = {
        'Pass': 0,
        'Shot': 1,
        'GOAL': 2,
        'NO_GOAL': 3,
        '<pad>': 4,
        'Carry': 5
    }
    with open('data/vocab_continuous_carry.json', 'w') as f:
        json.dump(type_vocab, f, indent=2)

    id_to_type = {v: k for k, v in type_vocab.items()}
    with open('data/id_to_type_continuous_carry.json', 'w') as f:
        json.dump(id_to_type, f, indent=2)

    print(f"\nSaved:")
    print(f"  data/sequences_carry{suffix}_train_balanced.parquet")
    print(f"  data/sequences_carry{suffix}_val_natural.parquet")
    print(f"  data/sequences_carry{suffix}_test_natural.parquet")
    print(f"  data/vocab_continuous_carry.json")

    spark.stop()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate carry data for xT model"
    )
    parser.add_argument(
        "--carry_threshold", type=float, default=0,
        help="Minimum carry distance in StatsBomb units (ignored if --adaptive)"
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Include all carries, drop shortest first during truncation"
    )
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Include all carries without any distance filter"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=14,
        help="Max number of events per possession before terminal token (default: 14)"
    )
    args = parser.parse_args()

    if not args.adaptive and not args.no_filter and args.carry_threshold <= 0:
        parser.error("Provide --carry_threshold, --adaptive, or --no_filter")

    main(args)
