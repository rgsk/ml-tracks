"""
Profiling a training step with torch.profiler.

torch.profiler wraps a region of code and records every dispatched op's CPU (and,
on GPU, CUDA) time, input shapes, and optionally memory. The classic uses:
  - find the HOT ops — in a transformer these are the matmul family
    (aten::addmm for Linear-with-bias, aten::mm for bias-free, aten::bmm for the
    batched attention scores/values). If those dominate, you're compute-bound.
  - spot OVERHEAD — many tiny ops, kernel-launch latency, or dataloading stalls
    mean you're NOT math-bound. This is the same question MFU answers from the
    other side: low MFU + matmuls not dominating the trace => fix the overhead.
  - export a chrome trace and view the timeline in chrome://tracing or perfetto.

Key API:
    from torch.profiler import profile, schedule, ProfilerActivity
    with profile(activities=[...], schedule=sched, record_shapes=True) as prof:
        for ... :
            <one training step>
            prof.step()          # advance the schedule, once per iteration

THE SCHEDULE matters. The first iterations are cold (lazy init, cudnn autotune,
allocator warmup), so blaming them on your model is misleading. schedule(wait=w,
warmup=u, active=a, repeat=r) skips `w` iters, traces-but-discards `u`, then
RECORDS `a` — so you measure steady state. You must call prof.step() every
iteration for the schedule to advance.

Run `python profile_run.py` from the llm/ dir to self-check.
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from config import GPTConfig
from model import GPT


def make_activities(device_type: str) -> list[ProfilerActivity]:
    """CPU always; add CUDA only when actually running on a GPU."""
    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def train_step(model, xb, yb, optimizer) -> None:
    """One full optimizer step: the region we want the profiler to attribute."""
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(xb, yb)
    loss.backward()
    optimizer.step()


def profile_training(model, batches, optimizer, sched, activities,
                     record_shapes: bool = True):
    """Profile a sequence of training steps and return the profiler object.

    `batches` is an iterable of (xb, yb); supply enough to cover the whole
    schedule window (wait + warmup + active). Each iteration runs one train_step
    and then advances the profiler schedule via prof.step().
    """
    with profile(
        activities=activities,
        schedule=sched,
        record_shapes=record_shapes,
    ) as prof:
        for xb, yb in batches:
            train_step(model, xb, yb, optimizer)
            prof.step()
    return prof


MATMUL_OPS = {"aten::addmm", "aten::mm", "aten::bmm", "aten::matmul", "aten::linear"}


def _self_time(event, device_type: str) -> float:
    """The op's own time on the relevant device (CUDA if on GPU, else CPU)."""
    if device_type == "cuda":
        # attribute name moved across torch versions; try the modern one first
        return getattr(event, "self_device_time_total", None) or \
            getattr(event, "self_cuda_time_total", 0)
    return event.self_cpu_time_total


def summarize(prof, device_type: str) -> float:
    """Print and return the matmul time share — one number for 'compute-bound?'.

    share = (time in matmul-family ops) / (time across all aten:: ops). A high
    share means the step is dominated by matrix multiplies = compute-bound, the
    regime where MFU is meaningful and FlashAttention later pays off.

    We restrict to aten:: ops (skipping container rows like ProfilerStep* and the
    raw CUDA kernels those ops dispatch) so we stay at ONE abstraction layer and
    don't double-count device time against the kernels it already includes.
    """
    aten = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    total = sum(_self_time(e, device_type) for e in aten) or 1.0
    matmul = sum(_self_time(e, device_type) for e in aten if e.key in MATMUL_OPS)
    share = matmul / total
    print(f"matmul time share ({device_type}): {share:.1%}  "
          f"({matmul:.0f}us of {total:.0f}us across aten ops)")
    return share


# ---------------------------------------------------------------------------
# Self-check: run `python profile_run.py` from the llm/ dir. Don't edit below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import tempfile

    torch.manual_seed(1337)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPTConfig(vocab_size=65, block_size=64, n_embd=96, n_head=4, n_layer=4)
    model = GPT(cfg).to(device_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # schedule: skip 1 cold iter, warm up 1, then RECORD 3 -> need 5 batches.
    wait, warmup, active = 1, 1, 3
    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    n_steps = wait + warmup + active

    B, T = 8, cfg.block_size
    batches = [(torch.randint(0, cfg.vocab_size, (B, T), device=device_type),
                torch.randint(0, cfg.vocab_size, (B, T), device=device_type))
               for _ in range(n_steps)]

    prof = profile_training(model, batches, optimizer, sched, make_activities(device_type))

    # --- 1. the profiler captured ops from the active window -----------------
    events = prof.key_averages()
    assert len(events) > 0, "profiler captured no events — did you call prof.step() each iter?"

    sort_key = "self_cuda_time_total" if device_type == "cuda" else "self_cpu_time_total"
    print(events.table(sort_by=sort_key, row_limit=12))

    # --- 2. matmuls must show up: a transformer step is matmul-dominated ------
    names = {e.key for e in events}
    matmul_ops = {"aten::addmm", "aten::mm", "aten::bmm", "aten::matmul", "aten::linear"}
    found = names & matmul_ops
    assert found, f"expected matmul-family ops in the trace, saw none of {matmul_ops}"
    print(f"matmul-family ops present: {sorted(found)} ✅")

    # --- 2b. matmuls should be the MAJORITY of aten op time (compute-bound) ---
    share = summarize(prof, device_type)
    assert share > 0.5, f"expected matmuls to dominate aten time, got {share:.1%}"

    # --- 3. a chrome trace exports and is valid JSON with events -------------
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as fh:
        trace = json.load(fh)
    trace_events = trace["traceEvents"] if isinstance(trace, dict) else trace
    assert len(trace_events) > 0, "chrome trace has no events"
    print(f"chrome trace exported: {len(trace_events)} events -> {trace_path} ✅")

    print("all checks passed ✅")
