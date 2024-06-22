"""Module for artifacts preparation for tensorflow_serving"""

from __future__ import absolute_import
from pathlib import Path
import shutil
from typing import List, Dict, Any

from sagemaker.serve.model_format.mlflow.utils import (
    _get_saved_model_path_for_tensorflow_and_keras_flavor,
    _move_contents,
)
from sagemaker.serve.detector.dependency_manager import capture_dependencies
from sagemaker.serve.validations.check_integrity import (
    generate_secret_key,
    compute_hash,
)
from sagemaker.remote_function.core.serialization import _MetaData


def prepare_for_new_model_server(
    model_path: str,
    shared_libs: List[str],
    dependencies: Dict[str, Any],
) -> str:
    """Prepares the model for serving.

    Args:
        model_path (str): Path to the model directory.
        shared_libs (List[str]): List of shared libraries.
        dependencies (Dict[str, Any]): Dictionary of dependencies.

    Returns:
        str: Secret key.
    """

    # TODO: Add logic to package model artifacts based on new model server
    """
        General Steps:
        1. Move inference script to code folder (or some other folder names required 
            by model server).
        2. Move shared_libs, dependencies etc.
        3. Generate hash key for pickled file.
        4. Write hash key into metadata.json file.
    
    """
    raise NotImplementedError
