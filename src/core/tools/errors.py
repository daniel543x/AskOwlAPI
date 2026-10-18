"""Errors shared by the ask and research pipelines."""

class ToolProtocolError(Exception):
    """The model did not answer through the required tool call."""
