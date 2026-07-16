"""Tests for bounded incremental NDJSON framing."""

import pytest

from better_backgrounds.ndjson import NdjsonDecodeError, NdjsonDecoder


def test_partial_chunks_are_reassembled() -> None:
    """Support arbitrary stdout chunk boundaries."""
    decoder = NdjsonDecoder()

    assert decoder.feed(b'{"one":') == []
    assert decoder.feed(b'1}\n{"two":2}\r\n') == ['{"one":1}', '{"two":2}']


def test_final_unterminated_line_is_available() -> None:
    """Expose a complete final line even when newline is omitted."""
    decoder = NdjsonDecoder()
    decoder.feed(b'{"final":true}')

    assert decoder.finish() == ['{"final":true}']


def test_oversized_line_is_rejected() -> None:
    """Bound memory used by hostile or broken workers."""
    decoder = NdjsonDecoder(max_line_bytes=4)

    with pytest.raises(NdjsonDecodeError, match="exceeds"):
        decoder.feed(b"12345")


def test_invalid_utf8_is_rejected() -> None:
    """Require unambiguous UTF-8 protocol data."""
    decoder = NdjsonDecoder()

    with pytest.raises(NdjsonDecodeError, match="UTF-8"):
        decoder.feed(b"\xff\n")


def test_null_byte_is_rejected() -> None:
    """Reject ambiguous protocol framing before JSON validation."""
    decoder = NdjsonDecoder()

    with pytest.raises(NdjsonDecodeError, match="null byte"):
        decoder.feed(b'{"value":"\x00"}\n')
