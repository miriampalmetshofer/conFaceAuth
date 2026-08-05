"""Factory for creating video matching strategy."""
from face_auth.config.models import MatchingStrategyConfig
from face_auth.imposter_video_creation.matching.strategy.impl.all import AllVideosMatchingStrategy
from face_auth.imposter_video_creation.matching.strategy.impl.random_match import RandomSamplingMatchingStrategy
from face_auth.imposter_video_creation.matching.strategy.impl.scenario import ScenarioMatchingStrategy
from face_auth.imposter_video_creation.matching.strategy.video_matching_strategy import VideoMatchingStrategy


def create_matching_strategy(strategy_config: MatchingStrategyConfig) -> VideoMatchingStrategy:
    """Create a video matching strategy from config.

    Args:
        strategy_config: Strategy configuration with type and strategy-specific settings

    Returns:
        Video matching strategy instance

    Raises:
        ValueError: If strategy type is not recognized
    """
    if strategy_config.type == "scenario":
        return ScenarioMatchingStrategy()

    if strategy_config.type == "all_videos":
        return AllVideosMatchingStrategy()

    if strategy_config.type == "random_sampling":
        return RandomSamplingMatchingStrategy(
            imposters_per_genuine=strategy_config.random_sampling.imposters_per_genuine,
            random_seed=strategy_config.random_sampling.random_seed,
        )

    raise ValueError(f"Unsupported matching strategy: {strategy_config.type}")
