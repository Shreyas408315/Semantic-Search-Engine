import tempfile
import unittest
from pathlib import Path

import fitz

from semantic_search.ingestion import (
    chunk_text,
    extract_document,
    pages_to_chunks,
)


class IngestionTests(unittest.TestCase):
    def test_chunk_text_respects_size_and_overlap(self):
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = chunk_text(text, chunk_size=400, overlap=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))

    def test_txt_extraction_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("First paragraph.\n\nSecond paragraph about search.", encoding="utf-8")
            pages = extract_document(path)
            chunks = pages_to_chunks(pages)
            self.assertEqual(pages[0].file_type, "txt")
            self.assertEqual(chunks[0].metadata["source"], "sample.txt")
            self.assertEqual(chunks[0].metadata["page"], 1)

    def test_pdf_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), "A small PDF about vector search.")
                pdf.save(path)
            pages = extract_document(path)
            self.assertEqual(len(pages), 1)
            self.assertIn("vector search", pages[0].text)


if __name__ == "__main__":
    unittest.main()
