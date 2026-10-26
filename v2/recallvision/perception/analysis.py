"""Standard-library summaries for persisted perception observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from recallvision.memory.schema import Observation


def summarize_observations(observations: Sequence[Observation]) -> dict[str, object]:
    """Build the required P1B analysis summary without model dependencies."""

    observations_by_class = dict(Counter(observation.class_name for observation in observations))
    track_ids_by_class = {
        class_name: sorted(
            {
                observation.track_id
                for observation in observations
                if observation.class_name == class_name
                and not observation.track_id.startswith("untracked:")
            }
        )
        for class_name in sorted(observations_by_class)
    }
    timestamp_ranges_by_class = {
        class_name: {
            "first": min(
                observation.timestamp
                for observation in observations
                if observation.class_name == class_name
            ),
            "last": max(
                observation.timestamp
                for observation in observations
                if observation.class_name == class_name
            ),
        }
        for class_name in sorted(observations_by_class)
    }
    bottle_observations = [
        observation for observation in observations if observation.class_name == "bottle"
    ]
    bottle_track_ids = sorted(
        {
            observation.track_id
            for observation in bottle_observations
            if not observation.track_id.startswith("untracked:")
        }
    )
    return {
        "total_observations": len(observations),
        "observations_by_class": observations_by_class,
        "unique_track_ids_by_class": track_ids_by_class,
        "timestamp_ranges_by_class": timestamp_ranges_by_class,
        "bottle_observation_count": len(bottle_observations),
        "unique_bottle_track_ids": bottle_track_ids,
        "unique_bottle_track_id_count": len(bottle_track_ids),
        "first_bottle_timestamp": (
            min(observation.timestamp for observation in bottle_observations)
            if bottle_observations
            else None
        ),
        "last_bottle_timestamp": (
            max(observation.timestamp for observation in bottle_observations)
            if bottle_observations
            else None
        ),
    }
