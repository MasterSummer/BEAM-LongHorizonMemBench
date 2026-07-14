"""Smoke test: verify the lhmsb package imports cleanly."""


def test_imports() -> None:
    """The package must import and expose __version__."""
    import lhmsb

    assert lhmsb.__version__ == "0.1.0"
