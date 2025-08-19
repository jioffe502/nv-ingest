# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for saving images from ingestion results to disk as actual image files.

This module provides enhanced utilities for saving base64-encoded images from ingestion
results to local disk, with configurable filtering by image type and descriptive naming.
"""

import base64
import io
import logging
import os
from typing import Any, Dict, List
from PIL import Image

from nv_ingest_client.client.util.processing import get_valid_filename

logger = logging.getLogger(__name__)


def save_images_to_disk(
    response_data: List[Dict[str, Any]],
    output_directory: str,
    save_charts: bool = True,
    save_tables: bool = True,
    save_infographics: bool = True,
    save_page_images: bool = False,
    save_raw_images: bool = False,
    count_images: bool = True,
    organize_by_type: bool = True,
) -> Dict[str, int]:
    """
    Save base64-encoded images from ingestion results to disk as actual image files.

    This utility extracts images from ingestion response data and saves them to disk
    with descriptive filenames that include the image subtype and page information.
    It provides granular control over which types of images to save.

    Parameters
    ----------
    response_data : List[Dict[str, Any]]
        List of document results from ingestion, each containing metadata with base64 images.
    output_directory : str
        Base directory where images will be saved.
    save_charts : bool, optional
        Whether to save chart images. Default is True.
    save_tables : bool, optional
        Whether to save table images. Default is True.
    save_infographics : bool, optional
        Whether to save infographic images. Default is True.
    save_page_images : bool, optional
        Whether to save page-as-image files. Default is False.
    save_raw_images : bool, optional
        Whether to save raw/natural images. Default is False.
    count_images : bool, optional
        Whether to count and log image statistics. Default is True.
    organize_by_type : bool, optional
        Whether to organize images into subdirectories by type. Default is True.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.

    Examples
    --------
    >>> from nv_ingest_client.util.image_disk_utils import save_images_to_disk
    >>>
    >>> # Save only charts and tables
    >>> counts = save_images_to_disk(
    ...     response_data,
    ...     "./output/images",
    ...     save_charts=True,
    ...     save_tables=True,
    ...     save_page_images=False
    ... )
    >>> print(f"Saved {counts['chart']} charts and {counts['table']} tables")
    """

    if not response_data:
        logger.warning("No response data provided")
        return {}

    # Initialize counters
    image_counts = {"chart": 0, "table": 0, "infographic": 0, "page_image": 0, "image": 0, "total": 0}

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    for doc_idx, document in enumerate(response_data):
        try:
            metadata = document.get("metadata", {})
            doc_type = document.get("document_type", "unknown")

            # Skip if no image content
            image_content = metadata.get("content")
            if not image_content:
                continue

            # Get document info for naming
            source_metadata = metadata.get("source_metadata", {})
            source_id = source_metadata.get("source_id", f"document_{doc_idx}")
            clean_source_name = get_valid_filename(os.path.basename(source_id))

            content_metadata = metadata.get("content_metadata", {})
            subtype = content_metadata.get("subtype", "image")
            page_number = content_metadata.get("page_number", 0)

            # Apply filtering based on subtype
            should_save = False
            if subtype == "chart" and save_charts:
                should_save = True
            elif subtype == "table" and save_tables:
                should_save = True
            elif subtype == "infographic" and save_infographics:
                should_save = True
            elif subtype == "page_image" and save_page_images:
                should_save = True
            elif (
                doc_type == "image"
                and subtype not in ["chart", "table", "infographic", "page_image"]
                and save_raw_images
            ):
                should_save = True
                subtype = "image"  # Normalize for counting

            if not should_save:
                continue

            # Determine image format
            image_type = "png"  # Default
            if doc_type == "image":
                image_metadata = metadata.get("image_metadata", {})
                image_type = image_metadata.get("image_type", "png").lower()

            # Create descriptive filename
            if organize_by_type:
                type_dir = os.path.join(output_directory, subtype)
                os.makedirs(type_dir, exist_ok=True)
                image_filename = f"{clean_source_name}_p{page_number}_{doc_idx}.{image_type}"
                image_path = os.path.join(type_dir, image_filename)
            else:
                image_filename = f"{clean_source_name}_{subtype}_p{page_number}_{doc_idx}.{image_type}"
                image_path = os.path.join(output_directory, image_filename)

            # Save the image
            try:
                # Decode base64 content
                image_data = base64.b64decode(image_content)
                image = Image.open(io.BytesIO(image_data))

                # Save with proper format
                image_ext = "jpg" if image_type == "jpeg" else image_type
                image.save(image_path, format=image_ext.upper())

                # Update counters
                image_counts[subtype] += 1
                image_counts["total"] += 1

                logger.debug(f"Saved {subtype} image: {image_path}")

            except Exception as e:
                logger.error(f"Failed to save {subtype} image for {clean_source_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to process document {doc_idx}: {e}")
            continue

    # Log summary if requested
    if count_images and image_counts["total"] > 0:
        logger.info(f"Saved {image_counts['total']} images to {output_directory}")
        for img_type, count in image_counts.items():
            if img_type != "total" and count > 0:
                logger.info(f"  - {img_type}: {count}")
    elif count_images:
        logger.info("No images were saved")

    return image_counts


def save_images_from_response(response: Dict[str, Any], output_directory: str, **kwargs) -> Dict[str, int]:
    """
    Convenience function to save images from a full API response.

    Parameters
    ----------
    response : Dict[str, Any]
        Full API response containing a "data" field with document results.
    output_directory : str
        Directory where images will be saved.
    **kwargs
        Additional arguments passed to save_images_to_disk().

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.
    """

    if "data" not in response or not response["data"]:
        logger.warning("No data found in response")
        return {}

    return save_images_to_disk(response["data"], output_directory, **kwargs)


def save_images_from_ingestor_results(
    results: List[List[Dict[str, Any]]], output_directory: str, **kwargs
) -> Dict[str, int]:
    """
    Save images from Ingestor.ingest() results.

    Parameters
    ----------
    results : List[List[Dict[str, Any]]]
        Results from Ingestor.ingest(), where each inner list contains
        document results for one source file.
    output_directory : str
        Directory where images will be saved.
    **kwargs
        Additional arguments passed to save_images_to_disk().

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.
    """

    # Flatten results from multiple documents
    all_documents = []
    for doc_results in results:
        if isinstance(doc_results, list):
            all_documents.extend(doc_results)
        else:
            all_documents.append(doc_results)

    return save_images_to_disk(all_documents, output_directory, **kwargs)
