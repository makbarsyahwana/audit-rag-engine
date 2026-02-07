"""Docling DocumentConverter wrapper for all supported formats."""

import logging
from pathlib import Path
from typing import Optional, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)


def create_converter(
    do_ocr: bool = True,
    do_table_structure: bool = True,
    ocr_lang: Optional[list[str]] = None,
) -> DocumentConverter:
    """Create a Docling DocumentConverter with PDF pipeline options.

    Args:
        do_ocr: Enable OCR for scanned PDFs and images.
        do_table_structure: Enable table structure recognition.
        ocr_lang: OCR languages (default: ["en"]).

    Returns:
        Configured DocumentConverter instance.
    """
    ocr_options = EasyOcrOptions(lang=ocr_lang or ["en"])
    table_options = TableStructureOptions(do_cell_matching=True)

    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_options=ocr_options,
        do_table_structure=do_table_structure,
        table_structure_options=table_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
        }
    )

    logger.info(
        "DocumentConverter created (ocr=%s, table_structure=%s)",
        do_ocr,
        do_table_structure,
    )
    return converter


def convert_document(
    source: Union[str, Path],
    converter: Optional[DocumentConverter] = None,
) -> DoclingDocument:
    """Convert a document file to a DoclingDocument.

    Args:
        source: Path to the document file.
        converter: Optional pre-configured converter. Creates default if None.

    Returns:
        Parsed DoclingDocument with full structure and provenance.
    """
    if converter is None:
        converter = create_converter()

    source_path = Path(source) if isinstance(source, str) else source
    logger.info("Converting document: %s", source_path.name)

    result = converter.convert(str(source_path))
    doc: DoclingDocument = result.document

    logger.info(
        "Conversion complete: %s — pages=%s",
        source_path.name,
        getattr(doc, "num_pages", "?"),
    )
    return doc
