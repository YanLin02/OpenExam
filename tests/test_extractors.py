from __future__ import annotations

import pytest

from openexam.extractors.docx import extract_docx
from openexam.extractors.pdf import extract_pdf
from openexam.extractors.pptx import extract_pptx
from openexam.extractors.text import extract_text_file
from openexam.config import AppConfig
from openexam.ingest import ingest_directory


def test_text_extractor_preserves_paragraphs(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("first paragraph\n\nsecond paragraph with CNN", encoding="utf-8")

    sections = extract_text_file(path)

    assert [section.paragraph_index for section in sections] == [1, 2]
    assert sections[1].location_label == "para.2"
    assert "CNN" in sections[1].text


def test_txt_extractor_preserves_paragraphs(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first paragraph\n\nsecond paragraph with CNN", encoding="utf-8")

    sections = extract_text_file(path)

    assert [section.paragraph_index for section in sections] == [1, 2]
    assert sections[0].location_label == "para.1"
    assert sections[1].location_type == "paragraph"


def test_answer_bank_md_uses_section_parser(tmp_path) -> None:
    path = tmp_path / "深度学习简答题_开卷检索版.md"
    path.write_text(
        """## 第二章 生成模型

### 5．请简述 GAN 的训练过程。

GAN 由生成器和判别器组成。

训练过程：
1. 固定 G，训练 D；
2. 固定 D，训练 G。

### 6．请简述 Dropout。

Dropout 是一种正则化方法。
""",
        encoding="utf-8",
    )

    sections = extract_text_file(path)

    assert len(sections) == 2
    assert sections[0].location_type == "section"
    assert sections[0].location_label == "section.1"
    assert "请简述 GAN" in sections[0].text
    assert "生成器" in sections[0].text
    assert "固定 G" in sections[0].text
    assert "Dropout 是一种正则化方法" in sections[1].text


def test_regular_md_keeps_paragraph_split(tmp_path) -> None:
    path = tmp_path / "regular.md"
    path.write_text(
        """### 5．请简述 GAN 的训练过程。

GAN 由生成器和判别器组成。
""",
        encoding="utf-8",
    )

    sections = extract_text_file(path)

    assert len(sections) == 2
    assert sections[0].location_type == "paragraph"
    assert sections[0].text.startswith("###")
    assert "生成器" in sections[1].text


def test_docx_extractor_preserves_paragraphs(tmp_path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("regularization")
    document.add_paragraph("optimization")
    document.save(path)

    sections = extract_docx(path)

    assert [section.paragraph_index for section in sections] == [1, 2]
    assert sections[0].location_type == "paragraph"


def test_pptx_extractor_preserves_slides(tmp_path) -> None:
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "sample.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "GAN"
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "diffusion"
    presentation.save(path)

    sections = extract_pptx(path)

    assert [section.slide_number for section in sections] == [1, 2]
    assert sections[0].location_label == "slide.1"


def test_pdf_extractor_preserves_pages(tmp_path) -> None:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "CNN convolution")
    page = document.new_page()
    page.insert_text((72, 72), "Transformer attention")
    document.save(path)
    document.close()

    sections = extract_pdf(path)

    assert [section.page_number for section in sections] == [1, 2]
    assert sections[1].location_label == "p.2"


def test_empty_directory_does_not_crash(tmp_path) -> None:
    config = AppConfig(index_dir=tmp_path / ".openexam")

    stats = ingest_directory(tmp_path, config=config, rebuild=True)

    assert stats.scanned_files == 0
    assert stats.indexed_files == 0
    assert stats.failed_files == 0


def test_damaged_pdf_is_recorded_as_failure(tmp_path) -> None:
    path = tmp_path / "坏 文件+OCR.pdf"
    path.write_bytes(b"not a real pdf")
    config = AppConfig(index_dir=tmp_path / ".openexam")

    stats = ingest_directory(tmp_path, config=config, rebuild=True)

    assert stats.scanned_files == 1
    assert stats.failed_files == 1
    assert stats.errors


def test_empty_pdf_is_recorded_as_no_extractable_text(tmp_path) -> None:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "empty.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()
    config = AppConfig(index_dir=tmp_path / ".openexam")

    stats = ingest_directory(tmp_path, config=config, rebuild=True)

    assert stats.scanned_files == 1
    assert stats.failed_files == 1
    assert stats.errors
    assert "no extractable text" in stats.errors[0][1]
