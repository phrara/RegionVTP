"""LLaVA-OneVision + QADP support (opt-in, isolated from the LLaVA-1.5 path).

This package lives under ``llava.onevision`` — NOT ``llava.model`` — deliberately:
``llava/model/__init__.py`` imports ``llava_mpt`` which does ``from transformers import
MptForCausalLM`` (removed in transformers >= 4.40). Importing anything under
``llava.model`` in the OneVision env (transformers >= 4.45) would therefore crash. This
package is a sibling, so it imports without triggering ``llava/model/__init__.py``.

Nothing in here is imported by the LLaVA-1.5 path; the OneVision scripts import it
directly (``from llava.onevision.llava_onevision_qadp import ...``).
"""
