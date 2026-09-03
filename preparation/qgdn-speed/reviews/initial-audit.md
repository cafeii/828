# Initial QGDN speed audit

- No QGDN training job is active or resumed.
- Existing full-run logs show mean throughput of 148,698 and 149,311 token/s for GDN versus 122,997 and 123,470 token/s for QGDN on seeds 3407 and 42. QGDN is about 82.7% of GDN throughput.
- The current QGDN production backend constructs a 2T virtual DPLR sequence and then discards every Recall-position output.
- Its DPLR call leaves `chunk_size` unspecified, selecting 16. The first low-risk experiment is an exact same-operator comparison at chunk sizes 16/32/64.
- Algebraic review found an exact physical-T rank-two affine representation, including the Recall-to-Delta cross term. A FP64 oracle and all-input gradient parity test are being added before any production-kernel change.
