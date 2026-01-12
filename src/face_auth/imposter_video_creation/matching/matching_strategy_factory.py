"""Factory for creating video matching strategy."""
from typing import Dict, Callable

from face_auth.config.models import MatchingStrategyConfig
from face_auth.authentication.imposter_video_creation.matching.strategy.impl.all import AllVideosMatchingStrategy
from face_auth.authentication.imposter_video_creation.matching.strategy.impl.random import RandomSamplingMatchingStrategy
from face_auth.authentication.imposter_video_creation.matching.strategy.impl.scenario import ScenarioMatchingStrategy
from face_auth.authentication.imposter_video_creation.matching.strategy.video_matching_strategy import VideoMatchingStrategy


def create_scenario_strategy(config: dict) -> ScenarioMatchingStrategy:
    """Create scenario matching strategy."""
    return ScenarioMatchingStrategy()


def create_all_videos_strategy(config: dict) -> AllVideosMatchingStrategy:
    """Create all-videos matching strategy."""
    return AllVideosMatchingStrategy()


def create_random_sampling_strategy(config: dict) -> RandomSamplingMatchingStrategy:
    """Create random sampling matching strategy."""
    imposters_per_genuine = config.get("imposters_per_genuine", 10)
    random_seed = config.get("random_seed", 42)

    return RandomSamplingMatchingStrategy(
        imposters_per_genuine=imposters_per_genuine,
        random_seed=random_seed
    )


STRATEGY_REGISTRY: Dict[str, Callable[[dict], VideoMatchingStrategy]] = {
    "scenario": create_scenario_strategy,
    "all_videos": create_all_videos_strategy,
    "random_sampling": create_random_sampling_strategy,
}


def create_matching_strategy(strategy_config: MatchingStrategyConfig) -> VideoMatchingStrategy:
    """Create a video matching strategy from config.

    Args:
        strategy_config: Strategy configuration with type and config dict

    Returns:
        Video matching strategy instance

    Raises:
        ValueError: If strategy type is not recognized
    """
    if strategy_config.type not in STRATEGY_REGISTRY:
        supported = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unsupported matching strategy: {strategy_config.type}. "
            f"Supported strategy: [{supported}]"
        )

    factory_fn = STRATEGY_REGISTRY[strategy_config.type]
    return factory_fn(strategy_config.config)
