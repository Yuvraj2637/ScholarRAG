"""Unit tests for structure-aware extractor and chunker."""

from src.ingestion.pdf_extractor import ParsedDocument, ParsedSection
from src.ingestion.chunker import StructureAwareChunker

def test_structure_aware_chunker():
    mock_doc = ParsedDocument(
        paper_id="test_paper",
        title="Test Attention Paper",
        authors=["Author One", "Author Two"],
        year=2023,
        abstract="This is a test abstract discussing attention efficiency and memory hierarchies.",
        sections=[
            ParsedSection(
                section_title="1 Introduction",
                section_number="1",
                text="Transformers require quadratic memory. We propose an exact tiling method. It splits attention into blocks.",
                paragraphs=["Transformers require quadratic memory.", "We propose an exact tiling method. It splits attention into blocks."]
            ),
            ParsedSection(
                section_title="2 Method",
                section_number="2",
                text="Our algorithm computes softmax without materializing full N x N matrix in HBM. It maintains running statistics.",
                paragraphs=["Our algorithm computes softmax without materializing full N x N matrix in HBM.", "It maintains running statistics."]
            )
        ],
        total_pages=5
    )

    chunker = StructureAwareChunker(max_tokens=20, min_tokens=5, overlap_tokens=5)
    chunks = chunker.chunk_document(mock_doc, primary_topic="Attention Optimization")

    assert len(chunks) >= 3
    # Check abstract chunk
    assert chunks[0].section_title == "Abstract"
    assert chunks[0].chunk_id == "test_paper_s0_c0"

    # Check section integrity
    section_titles = {c.section_title for c in chunks}
    assert "Abstract" in section_titles
    assert "1 Introduction" in section_titles
    assert "2 Method" in section_titles

    # Verify all metadata preserved
    for c in chunks:
        assert c.paper_id == "test_paper"
        assert c.paper_title == "Test Attention Paper"
        assert c.year == 2023
        assert len(c.authors) == 2
        assert c.token_count > 0
