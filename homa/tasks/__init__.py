"""The tasks, in two families.

``homa.tasks.diagnostic``   controlled tasks that isolate interaction order:
                            PARITY-k / MAJORITY-k and MATCH2 / MATCH3.
                            Needs only numpy and torch.
``homa.tasks.protein``      the TAPE protein-sequence benchmarks: secondary
                            structure, contact prediction and fluorescence.
                            Needs scipy and lmdb as well.

Neither is imported here, so the diagnostic tasks stay usable in an
environment without the protein data stack.
"""
