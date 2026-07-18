"""Bounded incremental decoding for worker NDJSON streams."""

from __future__ import annotations

DEFAULT_MAX_LINE_BYTES = 64 * 1024


class NdjsonDecodeError(ValueError):
    """Raised for malformed or unbounded protocol framing."""


class NdjsonDecoder:
    """Split arbitrary byte chunks into strict UTF-8 JSON lines."""

    def __init__(self, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> None:
        """Create a decoder with a positive per-line byte limit."""
        if max_line_bytes < 1:
            msg = "max_line_bytes must be positive"
            raise ValueError(msg)
        self._buffer = bytearray()
        self._max_line_bytes = max_line_bytes

    def feed(self, chunk: bytes) -> list[str]:
        """Consume bytes and return every newly completed non-empty line."""
        if b"\x00" in chunk:
            msg = "Protocol output contains a null byte."
            raise NdjsonDecodeError(msg)
        self._buffer.extend(chunk)
        lines: list[str] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                self._check_bound()
                return lines
            raw = bytes(self._buffer[:newline]).rstrip(b"\r")
            del self._buffer[: newline + 1]
            if len(raw) > self._max_line_bytes:
                msg = "Protocol line exceeds the configured byte limit."
                raise NdjsonDecodeError(msg)
            if raw:
                lines.append(self._decode(raw))

    def finish(self) -> list[str]:
        """Decode a final unterminated line when the process exits."""
        if not self._buffer:
            return []
        self._check_bound()
        raw = bytes(self._buffer).rstrip(b"\r")
        self._buffer.clear()
        return [self._decode(raw)] if raw else []

    def _check_bound(self) -> None:
        if len(self._buffer) > self._max_line_bytes:
            msg = "Protocol line exceeds the configured byte limit."
            raise NdjsonDecodeError(msg)

    @staticmethod
    def _decode(raw: bytes) -> str:
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            msg = "Protocol output is not valid UTF-8."
            raise NdjsonDecodeError(msg) from error
