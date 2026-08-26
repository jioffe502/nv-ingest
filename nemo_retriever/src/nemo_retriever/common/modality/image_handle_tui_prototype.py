# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE TUI for the lossless image-handle state model.

Run from ``nemo_retriever`` with:

    PYTHONPATH=src uv run --no-sync python -m nemo_retriever.common.modality.image_handle_tui_prototype

The prototype uses an in-memory blob map. It models ownership and failure
transitions only; it is not storage or throughput evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from nemo_retriever.common.modality.image_handle_state_prototype import PrototypeState
from nemo_retriever.common.modality.image_handle_state_prototype import commit_receipt
from nemo_retriever.common.modality.image_handle_state_prototype import corrupt_blob
from nemo_retriever.common.modality.image_handle_state_prototype import delete_blob
from nemo_retriever.common.modality.image_handle_state_prototype import embed
from nemo_retriever.common.modality.image_handle_state_prototype import initial_state
from nemo_retriever.common.modality.image_handle_state_prototype import persist
from nemo_retriever.common.modality.image_handle_state_prototype import publish
from nemo_retriever.common.modality.image_handle_state_prototype import rehydrate
from nemo_retriever.common.modality.image_handle_state_prototype import release

_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _render(state: PrototypeState) -> None:
    print("\033[2J\033[H", end="")
    handle = asdict(state.handle) if state.handle is not None else None
    view = {
        "phase": state.phase,
        "inline_bytes_present": state.inline_b64 is not None,
        "handle": handle,
        "stored_blob_uris": sorted(state.blobs),
        "rehydrated_bytes_present": state.rehydrated_b64 is not None,
        "embedding_digest": state.embedding_digest,
        "receipt_committed": state.receipt_committed,
        "failure": state.failure,
    }
    print(f"{_BOLD}Lossless image-handle transport prototype{_RESET}\n")
    print(json.dumps(view, indent=2, default=str))
    print(
        "\n"
        f"{_BOLD}[p]{_RESET} {_DIM}persist/idempotent store{_RESET}  "
        f"{_BOLD}[t]{_RESET} {_DIM}publish handle{_RESET}  "
        f"{_BOLD}[r]{_RESET} {_DIM}rehydrate+verify{_RESET}\n"
        f"{_BOLD}[e]{_RESET} {_DIM}embed{_RESET}  "
        f"{_BOLD}[w]{_RESET} {_DIM}commit write receipt{_RESET}  "
        f"{_BOLD}[g]{_RESET} {_DIM}release blob{_RESET}\n"
        f"{_BOLD}[c]{_RESET} {_DIM}corrupt blob{_RESET}  "
        f"{_BOLD}[d]{_RESET} {_DIM}delete blob{_RESET}  "
        f"{_BOLD}[x]{_RESET} {_DIM}reset{_RESET}  "
        f"{_BOLD}[q]{_RESET} {_DIM}quit{_RESET}"
    )


def main() -> None:
    state = initial_state()
    actions = {
        "p": persist,
        "t": publish,
        "r": rehydrate,
        "e": embed,
        "w": commit_receipt,
        "g": release,
        "c": corrupt_blob,
        "d": delete_blob,
        "x": lambda _state: initial_state(),
    }
    while True:
        _render(state)
        choice = input("\nAction: ").strip().lower()[:1]
        if choice == "q":
            return
        action = actions.get(choice)
        if action is not None:
            state = action(state)


if __name__ == "__main__":
    main()
