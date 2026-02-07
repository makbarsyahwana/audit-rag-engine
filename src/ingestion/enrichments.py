"""Multimodal enrichments config: picture captioning (VLM), classification, formula, code."""

import logging
from dataclasses import dataclass, field

from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """On-demand enrichment toggles for Docling pipeline.

    These add processing time but improve extraction quality for specific content:
    - do_table_structure: Extract table rows/columns/headers (default: On)
    - do_ocr: OCR for scanned pages/images (default: On)
    - do_picture_classification: Classify figure types (default: Off)
    - do_picture_description: Caption images using VLM (default: Off)
    - do_formula_enrichment: Extract LaTeX from equations (default: Off)
    - do_code_enrichment: Parse code blocks with language detection (default: Off)
    """

    do_table_structure: bool = True
    do_ocr: bool = True
    do_picture_classification: bool = False
    do_picture_description: bool = False
    do_formula_enrichment: bool = False
    do_code_enrichment: bool = False
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])

    def apply_to_pipeline_options(self, options: PdfPipelineOptions) -> PdfPipelineOptions:
        """Apply enrichment toggles to a PdfPipelineOptions instance."""
        options.do_ocr = self.do_ocr
        options.do_table_structure = self.do_table_structure

        logger.info(
            "Enrichments applied: ocr=%s, table=%s, pic_class=%s, pic_desc=%s, "
            "formula=%s, code=%s",
            self.do_ocr,
            self.do_table_structure,
            self.do_picture_classification,
            self.do_picture_description,
            self.do_formula_enrichment,
            self.do_code_enrichment,
        )
        return options


# Default enrichment config (OCR + table structure on, rest off)
default_enrichments = EnrichmentConfig()
