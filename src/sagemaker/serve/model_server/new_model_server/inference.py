"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import os
import io
import json
import cloudpickle
import shutil
import platform
from pathlib import Path
from sagemaker.serve.validations.check_integrity import perform_integrity_check
import logging

logger = logging.getLogger(__name__)

schema_builder = None
SHARED_LIBS_DIR = Path(__file__).parent.parent.joinpath("shared_libs")
SERVE_PATH = Path(__file__).parent.joinpath("serve.pkl")
METADATA_PATH = Path(__file__).parent.joinpath("metadata.json")


def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    raise NotImplementedError


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    raise NotImplementedError


def _run_preflight_diagnostics():
    _py_vs_parity_check()
    _pickle_file_integrity_check()


def _py_vs_parity_check():
    container_py_vs = platform.python_version()
    local_py_vs = os.getenv("LOCAL_PYTHON")

    if not local_py_vs or container_py_vs.split(".")[1] != local_py_vs.split(".")[1]:
        logger.warning(
            f"The local python version {local_py_vs} differs from the python version "
            f"{container_py_vs} on the container. Please align the two to avoid unexpected behavior"
        )


def _pickle_file_integrity_check():
    with open(SERVE_PATH, "rb") as f:
        buffer = f.read()

    perform_integrity_check(buffer=buffer, metadata_path=METADATA_PATH)


def _set_up_schema_builder():
    """Sets up the schema_builder object."""
    raise NotImplementedError


def _set_up_shared_libs():
    """Sets up the shared libs path."""
    raise NotImplementedError


# on import, execute
_run_preflight_diagnostics()
_set_up_schema_builder()
_set_up_shared_libs()
