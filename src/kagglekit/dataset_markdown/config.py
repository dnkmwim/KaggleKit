from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeGuards:
    """Provisional conservative runtime values, never dataset-md-v1.0 constants."""

    max_full_download_bytes_per_file: int = 1_500_000
    max_expanded_bytes_per_file: int = 4_000_000
    max_header_bytes: int = 65_536
    max_footer_bytes: int = 65_536
    request_timeout: float = 15.0
    inspection_timeout_per_file: float = 15.0
    max_total_transfer_bytes_per_dataset: int = 5_000_000
    max_files_inspected_per_dataset: int = 10
    max_subresources_per_file: int = 32
    max_fields_per_subresource: int = 256
    max_xlsx_xml_bytes: int = 8_000_000
    max_xlsx_xml_member_bytes: int = 4_000_000
    max_xlsx_compression_ratio: float = 50.0
    max_xlsx_header_rows: int = 8
    max_field_label_chars: int = 200
    max_file_examples: int = 10

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("All runtime guards must be positive")
