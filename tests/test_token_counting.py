import pytest
import tiktoken

from app.services.token_counting_service import (
    _encodings,
    count_tokens,
    get_token_encoder,
    get_tokenizer_name,
    split_text_by_tokens,
    truncate_text_to_tokens,
)
import app.services.token_counting_service as token_counting_service

# ---------------------------------------------------------------------------
# Existing tests (retained unchanged)
# ---------------------------------------------------------------------------

def test_get_tokenizer_name():
    assert get_tokenizer_name() == "cl100k_base"


def test_count_tokens_empty_and_none():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_simple():
    text = "Hello world! This is a test for token counting."
    tokens = count_tokens(text)
    assert tokens > 0
    assert 9 <= tokens <= 13


def test_count_tokens_special_chars_and_tokens():
    # Ensure raw special tokens do not cause encoding errors
    special_text = "Here is some code with <|endoftext|> and special brackets [test]."
    count = count_tokens(special_text)
    assert count > 0


def test_split_text_by_tokens_under_limit():
    text = "Short text under limit."
    result = split_text_by_tokens(text, max_tokens=50)
    assert len(result) == 1
    assert result[0] == text


def test_split_text_by_tokens_oversized():
    long_text = "Word " * 100  # ~100 tokens
    chunks = split_text_by_tokens(long_text, max_tokens=30, overlap_tokens=0)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert count_tokens(chunk) <= 30


def test_split_text_by_tokens_with_overlap():
    long_text = (
        "Apple banana cherry date elderberry fig grape honeydew kiwi lemon "
        "mango nectarine orange papaya quince raspberry strawberry tangerine uva watermelon. "
    ) * 5
    chunks = split_text_by_tokens(long_text, max_tokens=25, overlap_tokens=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 25


def test_split_text_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        split_text_by_tokens("Test text", max_tokens=0)


def test_truncate_text_to_tokens():
    long_text = "The quick brown fox jumps over the lazy dog repeatedly until the limit is exceeded."
    truncated = truncate_text_to_tokens(long_text, max_tokens=5)
    assert count_tokens(truncated) <= 5
    assert len(truncated) < len(long_text)
    assert truncate_text_to_tokens("", 5) == ""
    assert truncate_text_to_tokens(long_text, 0) == ""


# ---------------------------------------------------------------------------
# Test A — Multiple encoding names (Fix 1: encoding-aware cache)
# ---------------------------------------------------------------------------

def test_multiple_encoding_names_return_distinct_encoders():
    """
    Loading two different encoding names must return two independent encoder objects.
    Re-requesting the first encoding must return the already-cached instance.
    """
    enc_a = get_token_encoder("cl100k_base")
    enc_b = get_token_encoder("p50k_base")

    # Both must be real tiktoken encoders
    assert isinstance(enc_a, tiktoken.Encoding)
    assert isinstance(enc_b, tiktoken.Encoding)

    # They must be distinct objects (different encodings, different instances)
    assert enc_a is not enc_b

    # Re-requesting cl100k_base must return the same cached instance (no re-load)
    enc_a_again = get_token_encoder("cl100k_base")
    assert enc_a_again is enc_a

    # Both names must be present in the cache dictionary
    assert "cl100k_base" in _encodings
    assert "p50k_base" in _encodings


def test_different_encodings_have_different_names():
    """Each cached encoder must report a name matching the requested encoding."""
    enc_cl100k = get_token_encoder("cl100k_base")
    enc_p50k = get_token_encoder("p50k_base")

    assert enc_cl100k.name == "cl100k_base"
    assert enc_p50k.name == "p50k_base"


def test_count_tokens_uses_requested_encoding():
    """
    count_tokens() with an explicit encoding_name must use that encoder.
    Both encoders must produce a positive count for the same input.
    (Exact counts differ by encoding, but both must be > 0.)
    """
    text = "Hello, world! This is a tokenizer test."
    count_cl = count_tokens(text, encoding_name="cl100k_base")
    count_p50 = count_tokens(text, encoding_name="p50k_base")

    assert count_cl > 0
    assert count_p50 > 0
    # Both are valid but may differ — just confirm each returns a sensible count
    assert count_cl <= len(text)
    assert count_p50 <= len(text)


def test_next_chunk_starts_from_actual_end_after_safety_trim(monkeypatch):
    """
    Verify that when boundary-safety trimming shortens a token slice,
    the next chunk starts from the actual consumed position minus overlap.
    """

    class FakeEncoder:
        def __init__(self):
            self.decoded_slices = []
            self.candidate_encode_calls = 0

        def encode(self, text, disallowed_special=()):
            if text == "source":
                return list(range(10))
            # Force the first decoded candidate to exceed max_tokens.
            self.candidate_encode_calls += 1
            if self.candidate_encode_calls == 1:
                return list(range(5))
            return [int(token) for token in text.split(",")]

        def decode(self, tokens):
            token_slice = tuple(tokens)
            self.decoded_slices.append(token_slice)
            return ",".join(str(token) for token in tokens)

    fake_encoder = FakeEncoder()

    monkeypatch.setattr(
        token_counting_service,
        "get_token_encoder",
        lambda encoding_name: fake_encoder,
    )

    chunks = split_text_by_tokens(
        "source",
        max_tokens=4,
        overlap_tokens=1,
    )

    # Confirm that the safety-trimming branch was actually triggered.
    assert fake_encoder.candidate_encode_calls > 0

    # First planned slice: (0, 1, 2, 3).
    # It is trimmed to (0, 1, 2).
    assert fake_encoder.decoded_slices[0] == (0, 1, 2, 3)
    assert fake_encoder.decoded_slices[1] == (0, 1, 2)

    # actual_end = 3; overlap = 1; next start must be 2.
    # Therefore, token 2 must appear at the start of the next slice.
    assert fake_encoder.decoded_slices[2] == (2, 3, 4, 5)

    assert chunks[0] == "0,1,2"
    assert chunks[1] == "2,3,4,5"



def test_invalid_encoding_falls_back_without_caching_invalid_name(monkeypatch):
    """Invalid encoding names must not be cached as the fallback encoder."""
    # Isolate the cache for this test
    monkeypatch.setattr(
        token_counting_service,
        "_encodings",
        {}
    )

    invalid_name = "invalid_encoding_name"
    encoder = get_token_encoder(invalid_name)
    # Must return the default fallback encoder
    assert encoder.name == "cl100k_base"
    # Invalid name must not be stored in the cache
    assert invalid_name not in token_counting_service._encodings
    # Fallback must be cached under its actual name
    assert "cl100k_base" in token_counting_service._encodings
    # Repeated invalid requests should return the same fallback instance
    encoder_again = get_token_encoder(invalid_name)
    assert encoder_again is encoder
