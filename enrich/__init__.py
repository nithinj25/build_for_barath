"""Enrich stage (TECHNICAL_SPEC.md §6): narrative embeddings and MO extraction.

Model providers live here, never in linkage/ (CLAUDE.md rule 2): linkage/
stays pure numpy over plain records. Bedrock was the spec's provider; these
run locally (sentence-transformers on the GPU, Ollama) because Bedrock access
is not available. Swapping provider changes this package only.
"""
