"""LoRA action head — unit tests.

Verifies the three guarantees that the previous CommandParser system did NOT
provide:

  1. Constrained output: every decoded action is in the vocabulary.
  2. Explicit failure: empty / refused goals raise ActionDecodingError.
  3. Trainable: a single gradient step on the LoRA matrices actually moves
     the loss (i.e. the head is genuinely wired up, not a dummy).

Run::

    .venv/bin/python tests/test_lora_action_head.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from cadenza.actions.library import ActionCall
from cadenza.parser.lora_action_head import (
    ActionDecodingError,
    HashTextEncoder,
    LoRAActionDecoder,
    LoRAActionHead,
    LoRALinear,
)
from cadenza.stack.vocabulary import build_vocabulary


def test_constrained_output_is_in_vocab() -> None:
    """Every decoded action name must be a real library action."""
    vocab = build_vocabulary("go1")
    head = LoRAActionHead(vocab, hidden_dim=32, max_seq_len=8, lora_rank=4)
    torch.manual_seed(0)
    hidden = torch.randn(4, 8, 32)
    batches = head.decode(hidden)
    for batch in batches:
        for call in batch:
            assert call.action_name in vocab, (
                f"decoded action {call.action_name!r} not in vocabulary "
                f"{vocab.names()}"
            )
    print("  [ok] decoded actions are all in vocabulary")


def test_param_ranges_respect_schema() -> None:
    """Decoded params must lie inside ParamSchema [min, max] for that action."""
    vocab = build_vocabulary("go1")
    head = LoRAActionHead(vocab, hidden_dim=32, max_seq_len=8, lora_rank=4)
    torch.manual_seed(1)
    hidden = torch.randn(2, 8, 32)
    for batch in head.decode(hidden):
        for call in batch:
            desc = vocab.get(call.action_name)
            schema_by_name = {p.name: p for p in desc.params}
            for field_name, value in [
                ("speed", call.speed),
                ("extension", call.extension),
                ("distance_m", call.distance_m),
                ("rotation_rad", call.rotation_rad),
                ("duration_s", call.duration_s),
            ]:
                p = schema_by_name.get(field_name)
                if p is None or p.min is None or p.max is None:
                    continue
                assert p.min - 1e-4 <= value <= p.max + 1e-4, (
                    f"{call.action_name}.{field_name}={value} out of "
                    f"[{p.min}, {p.max}]"
                )
            assert call.repeat >= 1, f"repeat={call.repeat} should be >= 1"
    print("  [ok] decoded params lie inside ParamSchema ranges")


def test_empty_goal_raises() -> None:
    """Empty / whitespace goals must raise ActionDecodingError, not default to stand."""
    vocab = build_vocabulary("go1")
    decoder = LoRAActionDecoder(vocab, hidden_dim=64, max_seq_len=8, lora_rank=4)
    for bad in ("", "   ", "\n\t"):
        try:
            decoder.decode(bad)
        except ActionDecodingError:
            continue
        raise AssertionError(f"empty goal {bad!r} did not raise")
    print("  [ok] empty goals raise ActionDecodingError (no silent stand fallback)")


def test_unknown_action_in_training_target_raises() -> None:
    """Training targets that name a non-vocab action must fail loudly."""
    vocab = build_vocabulary("go1")
    head = LoRAActionHead(vocab, hidden_dim=16, max_seq_len=4, lora_rank=2)
    try:
        head.encode_targets([[ActionCall(action_name="fly_to_moon")]])
    except ActionDecodingError:
        print("  [ok] unknown training action raises ActionDecodingError")
        return
    raise AssertionError("unknown training action did not raise")


def test_only_lora_matrices_have_grads() -> None:
    """Frozen base weights must have requires_grad=False; A/B must have True."""
    layer = LoRALinear(8, 4, rank=2)
    assert not layer.W.requires_grad, "base W should be frozen"
    assert not layer.bias.requires_grad, "base bias should be frozen"
    assert layer.A.requires_grad, "LoRA A should be trainable"
    assert layer.B.requires_grad, "LoRA B should be trainable"
    print("  [ok] base weights frozen; only A/B trainable")


def test_training_step_reduces_loss() -> None:
    """One Adam step on the LoRA matrices must reduce the supervised loss.

    If this fails, the head isn't actually wired into the gradient graph and
    'training' would be a no-op — which is exactly the kind of silent
    failure we're trying to eliminate.
    """
    torch.manual_seed(7)
    vocab = build_vocabulary("go1")
    head = LoRAActionHead(vocab, hidden_dim=32, max_seq_len=6, lora_rank=4)
    hidden = torch.randn(3, 4, 32)
    target_seqs = [
        [ActionCall(action_name="stand"), ActionCall(action_name="walk_forward",
                                                      distance_m=1.0)],
        [ActionCall(action_name="jump")],
        [ActionCall(action_name="sit"), ActionCall(action_name="stand_up")],
    ]
    idx, params, mask = head.encode_targets(target_seqs)
    # encode_targets returns T = max_seq_len(real)+1. Pad hidden to match.
    if hidden.shape[1] < idx.shape[1]:
        pad = torch.zeros(hidden.shape[0], idx.shape[1] - hidden.shape[1],
                          hidden.shape[2])
        hidden = torch.cat([hidden, pad], dim=1)
    else:
        hidden = hidden[:, : idx.shape[1]]
    opt = torch.optim.Adam(head.trainable_parameters(), lr=1e-2)
    initial = head.training_loss(hidden, idx, params, mask).item()
    for _ in range(50):
        opt.zero_grad()
        loss = head.training_loss(hidden, idx, params, mask)
        loss.backward()
        opt.step()
    final = head.training_loss(hidden, idx, params, mask).item()
    assert final < initial - 0.05, (
        f"loss did not decrease meaningfully: initial={initial:.3f}, "
        f"final={final:.3f}"
    )
    print(f"  [ok] LoRA training reduces loss ({initial:.3f} -> {final:.3f})")


def test_stop_token_truncates_sequence() -> None:
    """When the head emits STOP, decoding stops cleanly and reports it."""
    vocab = build_vocabulary("go1")
    head = LoRAActionHead(vocab, hidden_dim=16, max_seq_len=8, lora_rank=2)
    # Force the action projection to always emit STOP by zeroing its base W
    # and biasing toward the last logit. (We don't touch A/B — irrelevant.)
    with torch.no_grad():
        head.action_proj.W.zero_()
        head.action_proj.bias.zero_()
        head.action_proj.bias[head._stop_index] = 10.0
    hidden = torch.randn(1, 8, 16)
    reports = head.decode_report(hidden)
    assert reports[0].calls == [], "expected empty calls when STOP fires first"
    assert reports[0].stop_emitted, "stop_emitted flag should be True"
    assert not reports[0].truncated, "STOP at slot 0 is not truncation"
    print("  [ok] STOP token cleanly ends the sequence")


def test_decoder_handles_long_goal_with_truncation_report() -> None:
    """Goals exceeding the encoder's slot budget surface truncation in the report."""
    vocab = build_vocabulary("go1")
    decoder = LoRAActionDecoder(
        vocab, hidden_dim=64, max_seq_len=4, lora_rank=2,
        encoder=HashTextEncoder(hidden_dim=64, max_seq_len=4),
    )
    report = decoder.decode_report("walk then jump then sit then stand then more")
    # max_seq_len=4 and the (untrained) head never emits STOP, so we expect
    # 4 calls plus the truncated flag.
    assert len(report.calls) == 4, f"expected 4 calls, got {len(report.calls)}"
    assert report.truncated or report.stop_emitted, (
        "report should flag truncation or stop"
    )
    print(f"  [ok] long goal -> {len(report.calls)} calls, "
          f"truncated={report.truncated}, stop={report.stop_emitted}")


def test_encoder_reproducible_across_processes() -> None:
    """HashTextEncoder must embed identically across processes.

    The encoder buckets trigrams with a stable hash; a LoRA adapter trained
    in one process has to decode the same way when reloaded in another. This
    starts a subprocess with a different PYTHONHASHSEED and asserts the
    embedding matches bit-for-bit — a salted builtin hash() would diverge.
    """
    import subprocess
    text = "walk forward then turn left into the debris"
    enc = HashTextEncoder(hidden_dim=48, max_seq_len=8)
    here = [round(float(x), 6) for x in enc(text).flatten().tolist()]
    snippet = (
        "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';"
        "from cadenza.parser.lora_action_head import HashTextEncoder;"
        "e=HashTextEncoder(hidden_dim=48, max_seq_len=8);"
        f"print(','.join(f'{{round(float(x),6)}}' for x in e({text!r}).flatten().tolist()))"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", snippet],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    there = [float(v) for v in out.stdout.strip().split(",")]
    assert here == there, "encoder differs across processes (hash not stable)"
    print(f"  [ok] encoder reproducible across processes ({len(here)} dims match)")


def main() -> int:
    tests = [
        test_constrained_output_is_in_vocab,
        test_param_ranges_respect_schema,
        test_empty_goal_raises,
        test_unknown_action_in_training_target_raises,
        test_only_lora_matrices_have_grads,
        test_training_step_reduces_loss,
        test_stop_token_truncates_sequence,
        test_decoder_handles_long_goal_with_truncation_report,
        test_encoder_reproducible_across_processes,
    ]
    failures = 0
    for t in tests:
        print(f"- {t.__name__}")
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"  [FAIL] {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
