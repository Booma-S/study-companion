import logging
from typing import Final

import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_ENCODING_NAME: Final[str] = "cl100k_base"

# Encoding-aware cache: keyed by encoding name so each distinct encoding
# gets its own cached instance. Multiple calls with the same name reuse the
# same object; calls with different names get independent encoders.
_encodings: dict[str, tiktoken.Encoding] = {}



def get_token_encoder(encoding_name: str = DEFAULT_ENCODING_NAME) -> tiktoken.Encoding:
    """
    Get or initialize the tiktoken encoding instance.

    Each valid encoding is cached under its actual encoding name.
    If the requested encoding is invalid, return the default encoding
    without caching it under the invalid name.
    """

    if encoding_name in _encodings:
        return _encodings[encoding_name]

    try:
        encoder = tiktoken.get_encoding(encoding_name)

    except Exception as exc:
        logger.warning(
            "Failed to load tiktoken encoding '%s': %s. "
            "Falling back to %s.",
            encoding_name,
            exc,
            DEFAULT_ENCODING_NAME,
        )

        # Reuse the default encoder if it is already cached.
        if DEFAULT_ENCODING_NAME not in _encodings:
            _encodings[DEFAULT_ENCODING_NAME] = (
                tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
            )

        return _encodings[DEFAULT_ENCODING_NAME]

    # Cache only under the actual requested encoding name.
    _encodings[encoding_name] = encoder

    return encoder



def get_tokenizer_name() -> str:
    """Return the name of the default tokenizer encoding."""
    return DEFAULT_ENCODING_NAME


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING_NAME) -> int:
    """
    Count the number of tokens in a given text string.

    Safely handles empty strings and special tokens (<|endoftext|> etc.).
    If tiktoken raises unexpectedly, falls back to a character-length heuristic
    (~4 chars per token) which is an approximation and may not satisfy a strict
    token budget.
    """
    if not text:
        return 0

    try:
        encoder = get_token_encoder(encoding_name)
        # disallowed_special=() allows strings containing '<|endoftext|>' to be
        # encoded as plain text rather than raising an error.
        tokens = encoder.encode(text, disallowed_special=())
        return len(tokens)
    except Exception as exc:
        logger.error(
            "Token counting failed for text (len=%d): %s. "
            "Falling back to approximate character-based heuristic (~4 chars/token). "
            "Result may not satisfy a strict token budget.",
            len(text),
            exc,
        )
        return max(1, len(text) // 4)


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> list[str]:
    """
    Split a string into chunks where each chunk contains at most *max_tokens*,
    with an optional token overlap between consecutive chunks.

    The implementation tracks actual consumed token positions so that:
    - If the BPE boundary-safety loop removes tokens from a slice, the next
      chunk starts from the actual end position of the shortened slice minus
      overlap_tokens, ensuring no source token is silently skipped.
    - Each emitted chunk is verified to re-encode within max_tokens.

    Falls back to an approximate character-based split if tiktoken raises.
    The fallback output is explicitly logged as degraded and cannot guarantee
    the max_tokens limit.
    """
    if not text:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))

    try:
        encoder = get_token_encoder(encoding_name)
        tokens = encoder.encode(text, disallowed_special=())

        if len(tokens) <= max_tokens:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            slice_tokens = tokens[start:end]
            decoded = encoder.decode(slice_tokens).strip()

            # BPE boundary condition: decoding a suffix-stripped slice and then
            # re-encoding it in isolation can yield +1 token compared to its
            # position in the original stream. Trim until it fits.
            while decoded and len(encoder.encode(decoded, disallowed_special=())) > max_tokens:
                end -= 1
                slice_tokens = tokens[start:end]
                decoded = encoder.decode(slice_tokens).strip()

            if decoded:
                chunks.append(decoded)

            # Advance by the number of tokens actually consumed, then step back
            # by overlap_tokens so the next chunk shares context.
            actual_end = end  # position after last token included in this chunk
            next_start = actual_end - overlap_tokens

            # Guard: always advance by at least 1 to prevent an infinite loop
            # when the safety-trim reduces the slice to a single token.
            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    except Exception as exc:
        logger.error(
            "Token splitting failed: %s. "
            "Falling back to approximate character-based splitting (degraded mode). "
            "Output chunks may exceed max_tokens for some languages or content types.",
            exc,
        )
        char_max = max_tokens * 4
        char_overlap = overlap_tokens * 4
        char_step = max(1, char_max - char_overlap)
        return [
            text[i : i + char_max].strip()
            for i in range(0, len(text), char_step)
            if text[i : i + char_max].strip()
        ]


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> str:
    """
    Truncate *text* to at most *max_tokens* without breaking token boundaries.
    """
    if not text or max_tokens <= 0:
        return ""

    encoder = get_token_encoder(encoding_name)
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text

    end = max_tokens
    slice_tokens = tokens[:end]
    truncated = encoder.decode(slice_tokens).strip()
    while truncated and len(encoder.encode(truncated, disallowed_special=())) > max_tokens:
        end -= 1
        slice_tokens = tokens[:end]
        truncated = encoder.decode(slice_tokens).strip()
    return truncated
