#!/usr/bin/env python3
"""
Wrapper for /fairseq/examples/speech_recognition/infer.py that fixes the
CriterionType NameError when flashlight is not installed.
"""
import sys
import warnings

# Ensure fairseq source is importable
if "/fairseq" not in sys.path:
    sys.path.insert(0, "/fairseq")

# Pre-import w2l_decoder and inject CriterionType if flashlight is missing
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import examples.speech_recognition.w2l_decoder as _w2l

if not hasattr(_w2l, "CriterionType"):
    from enum import Enum
    class CriterionType(Enum):
        CTC = 1
        ASG = 2
    _w2l.CriterionType = CriterionType

# CpuViterbiPath (flashlight) is also missing — replace W2lViterbiDecoder.decode
# with a pure PyTorch greedy CTC decoder (argmax per frame = viterbi with zero transitions)
import torch
def _viterbi_decode_pytorch(self, emissions):
    B, T, N = emissions.size()
    viterbi_path = emissions.argmax(dim=-1).int()  # (B, T)
    return [
        [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
        for b in range(B)
    ]
_w2l.W2lViterbiDecoder.decode = _viterbi_decode_pytorch

# Run the original infer.py as __main__
import runpy
runpy.run_path(
    "/fairseq/examples/speech_recognition/infer.py",
    run_name="__main__",
)
