import pytest

from app.chunking import chunk_page, normalize_text


def test_chunk_page_preserves_overlap_and_metadata():
    chunks = chunk_page(
        text="one two three four five six seven",
        document_id="doc-1",
        filename="guide.pdf",
        page=3,
        chunk_words=4,
        overlap_words=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]
    assert chunks[0].page == 3
    assert chunks[1].chunk_id == "doc-1:p3:c1"


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_page(
            text="some words",
            document_id="doc",
            filename="guide.pdf",
            page=1,
            chunk_words=2,
            overlap_words=2,
        )


def test_normalize_text_removes_noise():
    assert normalize_text("hello\x00   world\n\n\nnext") == "hello world\n\nnext"
