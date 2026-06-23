"""Tokenizer package — Phase 1: 16k byte-level BPE + phonological init.

Public API
----------
``BPETokenizer`` — byte-level BPE tokenizer, vocab = 16,384.
``phonological_init_fn`` — phonological embedding initialiser (exported for
    scripts that want to call it directly with a tokenizer vocab).

Registration side-effect
------------------------
Importing this package (or ``graph_llm.tokenizer.phonological_init``) registers
the ``"phonological"`` embedding init under
:func:`graph_llm.models.registry.register_embedding_init`.  This import happens
automatically when ``model.embedding_init = "phonological"`` is resolved by
build code that does ``import graph_llm.tokenizer``.
"""

from graph_llm.tokenizer.bpe import TOKENIZER_VERSION, VOCAB_SIZE, BPETokenizer
from graph_llm.tokenizer.phonological_init import (
    compute_phonological_vectors,
    phonological_init_fn,
)

__all__ = [
    "BPETokenizer",
    "VOCAB_SIZE",
    "TOKENIZER_VERSION",
    "compute_phonological_vectors",
    "phonological_init_fn",
]
