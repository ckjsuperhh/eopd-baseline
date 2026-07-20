# Local pure-Torch shim for `flash_attn.bert_padding`.
#
# Purpose: on machines where `flash_attn` cannot be installed (e.g. glibc < 2.32,
# which blocks the prebuilt wheel, and no network/GitHub access to build from
# source), verl's `use_remove_padding=True` (varlen packing) path still imports
# `flash_attn.bert_padding` for the 4 functions below. We provide pure-Torch
# equivalents so remove-padding works with the SDPA attention backend
# (verl's actual attention uses torch SDPA, not flash_attn, for the sdpa backend).
#
# The run/eval launchers inject this directory onto PYTHONPATH ONLY when a real
# `flash_attn.bert_padding` import fails, so it never shadows a real install.
