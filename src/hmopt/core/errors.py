"""Custom exception hierarchy."""


class HMOptError(Exception):
    """Base error."""


class ConfigError(HMOptError):
    """Invalid configuration."""


class PipelineError(HMOptError):
    """Raised when a pipeline step fails."""
