from tests.models.llama.test_modeling_llama import *
from transformers import LlamaAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class LlamaAdapterModelTest(AdapterModelTesterMixin, LlamaModelTest):
    all_model_classes = (LlamaAdapterModel,)
    fx_compatible = False
