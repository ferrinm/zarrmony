import pytest

from zarrmony.errors import (
    ExtractorWarning,
    OutputExistsError,
    ZarrmonyError,
)


def test_output_exists_error_is_zarrmony_error() -> None:
    assert issubclass(OutputExistsError, ZarrmonyError)


def test_extractor_warning_is_user_warning() -> None:
    assert issubclass(ExtractorWarning, UserWarning)


def test_errors_are_raisable() -> None:
    with pytest.raises(OutputExistsError):
        raise OutputExistsError("already exists")
