"""Tandem reproduction trainer (card 2dd3400f) — mixed M+R stream, repo pipeline.

Drives the REPO :class:`~graph_llm.models.delta_memory_lm.DeltaMemoryLM` (tandem
ON) on the mixed memory+reasoning stream to reproduce the scratchpad tandem result
through the repo model + config, resolving the scratchpad's integration debts:

* the reasoner is the CAUSAL v3 WIN (card e4e8a4dc — CausalGRUEncoder + clause-END
  locate-then-walk), run per position and leak-free (no bidir future peek);
* routing is UNSUPERVISED (card 31fe6b00 — load-balance + commitment annealed +
  gate-logit noise + forced-mix warmup; NO type labels in training — labels used at
  EVAL only for per-type accuracy + gate-by-type);
* the reasoner is supervised only by the R-SYNTHETIC aux labels (locate-CE at the
  clause-END + per-hop walk-aux along the chain trajectory), applied ONLY on the
  reasoning rows.

Each training step: a mixed M+R batch of ``n_segments`` segments is processed left
to right — the non-answer segments build the cross-segment delta-memory state
(``memory_forward``, no reasoner), the LAST (answer) segment runs ``tandem_step``
(reasoner walk + gate fusion + forced-mix + aux collection).  Loss = answer-CE
(locate-first ramp) + locate-CE (R rows) + walk-aux (R rows) + the unsupervised
gate losses (skipped during the forced-mix warmup).

Sharding: ``--seeds`` splits the seeds across two GPU-pinned workers; ``merge``
combines the per-seed JSONs.  Result JSONs are written incrementally so a killed
run resumes from disk.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from graph_llm.config import Config, DataConfig, ModelConfig, TrainConfig
from graph_llm.data.reasoning_tasks import ReasoningBatch, make_batch
from graph_llm.models import build_model


@dataclass
class TandemConfig:
    """Repro run configuration (mirrors the scratchpad tandem_harness knobs)."""

    train_r_depths: tuple[int, ...] = (4, 5, 6)     # SHALLOW R training depth
    test_r_depths: tuple[int, ...] = (4, 6, 16, 32)  # eval R in-dist {4,6} + extrap {16,32}
    k_train: int = 8                                 # reasoner K during training (>= max train depth)
    train_steps: int = 3000
    batch_size: int = 48
    seg_len: int = 512
    n_segments: int = 2
    n_chains: int = 2
    d_model: int = 96
    # VERIFIED Stage-3 default (card 2dd3400f comment 159/169): a SINGLE delta layer
    # (like the scratchpad's bare MemoryPathway) so the memory PROVABLY fails the
    # multi-hop chain (dissociation) while still solving the 1-hop cross-segment M
    # retrieval.  A deeper stack learns 2-chain R during the forced-mix and collapses
    # the routing pressure.  (Override with --layers only for ablations.)
    delta_layers: int = 1
    delta_n_heads: int = 4
    delta_head_dim: int = 32
    eval_batches: int = 4
    eval_batch: int = 96
    lr: float = 1e-3
    lr_warmup: int = 200
    log_every: int = 500
    # Reasoner aux + gate recipe (the model reads gate/reasoning knobs from its cfg;
    # these are the loss weights the TRAINER applies).
    locate_warmup: int = 600
    locate_weight: float = 2.0
    walk_weight: float = 1.0
    gate_balance_weight: float = 2.0
    gate_commit_weight: float = 1.0
    gate_commit_anneal: int = 900
    gate_mix_warmup: int = 600
    gate_noise_std: float = 0.5
    # Gate bias init for the REPRO: 0.0 (g~=0.5, scratchpad-faithful) so the balance
    # loss does not have to lift the mean far and pick the direction arbitrarily.  The
    # model DEFAULT (-3, memory-favoring) is the right safe default for a real LM /
    # text8, but the controlled two-type repro routes cleanest from a neutral gate.
    gate_bias_init: float = 0.0
    hard_gate: bool = False   # straight-through hard fusion gate (routing-direction fix)
    gate_scalar: bool = False  # scalar per-position gate (whole position routes to one pathway)
    # Curriculum warmup (rung-3 fix for the M-in-tandem gap): for the first
    # ``type_warmup`` steps, ROUTE each example to its specialist (M->memory, R->
    # reasoner) so each pathway reaches its isolated competence on clean, stable
    # gradient BEFORE the gate learns routing.  Uses the type labels for the WARMUP
    # fusion only (a competence scaffold); the GATE still learns to route unsupervised
    # (no supervised gate loss).  0 == off (use the label-free forced-mix instead).
    type_warmup: int = 0
    # VERIFIED Stage-3 default = FALSE (do NOT flip back to True).  card 2dd3400f
    # comments 168/169 (the TIED-HEAD-WARP finding): with a TIED head (lm_head ==
    # embed^T), the answer-CE decoding the REASONER's features on R rows REWRITES the
    # byte embeddings the MEMORY's associative binding reads -> warps M's input space
    # every batch, collapsing acc_M to ~0.08.  Untying (own head weights) is the fix
    # that reaches the accepted numbers (acc_M=1.0, gate sep +0.99, 3 seeds).  The
    # isolated memory never has reasoner features hit its head, so tying is harmless
    # there — this footgun is SPECIFIC to multi-pathway models sharing an embedding.
    tie_embeddings: bool = False


def build_tandem_model(cfg: TandemConfig, device: torch.device) -> torch.nn.Module:
    """Build the repo DeltaMemoryLM with the tandem ON + the validated memory config.

    The MEMORY pathway uses the validated cross-segment retrieval config (forget gate
    OFF + silu_l2 feature map — card 61f900ca; with the default remember-by-default
    forget alpha~0.98 the k->v binding still decays over a long segment, so the M task
    needs no decay).  The REASONER window == ``seg_len`` (one window per segment).
    """
    m = ModelConfig(
        name="delta_memory_lm",
        vocab_size=256,
        d_model=cfg.d_model,
        delta_layers=cfg.delta_layers,
        delta_n_heads=cfg.delta_n_heads,
        delta_head_k_dim=cfg.delta_head_dim,
        delta_head_v_dim=cfg.delta_head_dim,
        delta_conv_width=4,
        delta_chunk_size=32,
        delta_feature_map="silu_l2",
        delta_use_forget_gate=False,
        delta_dropout=0.0,
        delta_scan="chunkwise",
        delta_ff_mult=4,
        dropout=0.0,
        max_seq_len=cfg.seg_len,
        tie_embeddings=cfg.tie_embeddings,
        tandem_enabled=True,
        reasoning_segment_len=cfg.seg_len,
        causal_reasoner_steps=cfg.k_train,
        causal_reasoner_gamma_floor=2.0,
        causal_reasoner_key_dim=cfg.d_model // 2,
        causal_reasoner_conv_kernel=5,
        causal_reasoner_gru_layers=1,
        causal_reasoner_query_window=12,
        causal_reasoner_hard_seed=True,
        gate_balance_weight=cfg.gate_balance_weight,
        gate_commit_weight=cfg.gate_commit_weight,
        gate_commit_anneal_steps=cfg.gate_commit_anneal,
        gate_noise_std=cfg.gate_noise_std,
        # With the curriculum, disable the model's internal label-free forced-mix
        # (the trainer drives per-type-directed routing during ``type_warmup`` instead).
        gate_mix_warmup_steps=(0 if cfg.type_warmup > 0 else cfg.gate_mix_warmup),
        reasoning_locate_warmup=cfg.locate_warmup,
        tandem_gate_bias_init=cfg.gate_bias_init,
        tandem_hard_gate=cfg.hard_gate,
        tandem_gate_scalar=cfg.gate_scalar,
    )
    full = Config(
        model=m,
        data=DataConfig(seq_len=cfg.seg_len, batch_size=cfg.batch_size),
        train=TrainConfig(lr=cfg.lr, max_steps=cfg.train_steps, mixed_precision="no"),
    )
    return build_model(full).to(device)


def _to_dev(batch: ReasoningBatch, device: torch.device) -> dict:
    return {
        "seg": torch.from_numpy(batch.seg_tokens).to(device),
        "answer_pos": torch.from_numpy(batch.answer_pos).to(device),
        "answer": torch.from_numpy(batch.answer).to(device),
        "is_r": torch.from_numpy(batch.kind_is_r).to(device),
        "locate": torch.from_numpy(batch.locate_offset).to(device),
        "walk": torch.from_numpy(batch.walk_traj).to(device),
        "answer_seg": batch.answer_seg,
    }


def _build_state(model: torch.nn.Module, seg: torch.Tensor, answer_seg: int):
    """Carry the delta-memory state over the non-answer segments (memory only)."""
    states = None
    for si in range(answer_seg):
        _, _, states = model.memory_forward(seg[:, si], states, return_states=True)
    return states


def _answer_gather(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    b = t.shape[0]
    return t[torch.arange(b, device=t.device), pos]


def _train_one(cfg: TandemConfig, seed: int, device: torch.device, verbose: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_tandem_model(cfg, device)
    params = model.num_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.lr_warmup))
    )
    rng = random.Random(1000 + seed)
    model.train()
    t0 = time.time()
    for step in range(cfg.train_steps):
        kinds = [rng.choice(("M", "R")) for _ in range(cfg.batch_size)]
        depth = rng.choice(cfg.train_r_depths)
        batch = make_batch(rng, kinds, depth, cfg.seg_len, cfg.n_segments, cfg.n_chains, cfg.k_train)
        d = _to_dev(batch, device)
        states = _build_state(model, d["seg"], d["answer_seg"])
        is_r = d["is_r"]
        # LEAK FIX: predict the answer at answer_pos from PRED_POS = answer_pos-1 (the
        # answer byte is IN the input at answer_pos; gathering there would let the model
        # copy it).  Query the reasoner + gather the gate at the same PRED_POS.
        # INVARIANT: answer_pos >= 1 (the reasoning_tasks generators always place >=1
        # prefix byte before the answer).  pred_pos MUST be answer_pos-1 (leak-free); if
        # answer_pos were 0 the clamp would gather logits[0], which depends on token[0]
        # (the answer byte) -> reintroduces the copy leak.  Assert rather than clamp-hide.
        assert bool((d["answer_pos"] >= 1).all()), "answer_pos==0 would reintroduce the copy leak"
        pred_pos = d["answer_pos"] - 1
        tf = torch.where(is_r, d["locate"], torch.full_like(d["locate"], -1))
        # Curriculum: during type_warmup, force per-row specialist routing (R->reasoner
        # g=1, M->memory g=0) so each pathway trains on its own type cleanly.
        force = is_r.to(torch.float32) if (cfg.type_warmup > 0 and step < cfg.type_warmup) else None
        out = model.tandem_step(
            d["seg"][:, d["answer_seg"]], None, states, return_states=False,
            aux_query_pos=pred_pos, tf_seed=tf, steps=cfg.k_train, force_gate=force,
        )
        gstep = out["step"]
        ans_logits = _answer_gather(out["logits"], pred_pos)
        ans_loss = F.cross_entropy(ans_logits, d["answer"])
        ans_w = min(1.0, (gstep + 1) / max(1, cfg.locate_warmup))
        loss = ans_w * ans_loss

        if bool(is_r.any()):
            seed_logits = out["aux"]["seed_logits"]        # (B, L)
            loss = loss + cfg.locate_weight * F.cross_entropy(seed_logits[is_r], d["locate"][is_r])
            walk_w = out["aux"]["walk_w"]                   # (B, K, L)
            traj = d["walk"]                               # (B, K+1)
            wr, tr = walk_w[is_r], traj[is_r]
            k = wr.shape[1]
            nll = wr.new_zeros(())
            for j in range(k):
                p = wr[:, j].gather(1, tr[:, j + 1].clamp_min(0).unsqueeze(1)).squeeze(1)
                nll = nll + (-(p.clamp_min(1e-9).log())).mean()
            loss = loss + cfg.walk_weight * (nll / max(1, k))

        # Unsupervised gate losses at the prediction position (skip during forced mix).
        gate_ans = _answer_gather(out["gate"], pred_pos)  # (B,)
        if out["gate_mix"] is None:
            if cfg.gate_balance_weight > 0:
                loss = loss + cfg.gate_balance_weight * (gate_ans.mean() - 0.5) ** 2
            if cfg.gate_commit_weight > 0:
                commit_w = cfg.gate_commit_weight
                if cfg.gate_commit_anneal > 0:
                    release = cfg.type_warmup if cfg.type_warmup > 0 else cfg.gate_mix_warmup
                    since = gstep - release
                    commit_w = commit_w * min(1.0, max(0, since) / max(1, cfg.gate_commit_anneal))
                loss = loss + commit_w * (gate_ans * (1.0 - gate_ans)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % cfg.log_every == 0 or step == cfg.train_steps - 1):
            # Live gate-by-type from THIS batch (routing direction, no extra compute):
            # g -> 1 = reasoner, g -> 0 = memory; want gate on R > gate on M.
            with torch.no_grad():
                gr = float(gate_ans[is_r].mean()) if bool(is_r.any()) else float("nan")
                gm = float(gate_ans[~is_r].mean()) if bool((~is_r).any()) else float("nan")
            mixflag = "MIX" if out["gate_mix"] is not None else "   "
            print(f"    [tandem s{seed}] step {step:>5d}/{cfg.train_steps} {mixflag} "
                  f"loss {loss.item():.4f} (ans {ans_loss.item():.3f}) "
                  f"gate[R={gr:.3f} M={gm:.3f} sep={gr - gm:+.3f}]", flush=True)

    erng = random.Random(7777 + seed)
    accM, gM = _eval_M(model, cfg, erng, device)
    accR: dict[int, float] = {}
    gR: dict[int, float] = {}
    for depth in cfg.test_r_depths:
        a, g = _eval_R_depth(model, cfg, depth, erng, device)
        accR[depth] = a
        gR[depth] = g
    # Dissociation probe (the interpretable proof): each pathway in isolation.
    d0 = cfg.train_r_depths[0]
    mem_M, _ = _eval_M(model, cfg, erng, device, force_gate=0.0)      # memory-only on M
    mem_R, _ = _eval_R_depth(model, cfg, d0, erng, device, force_gate=0.0)   # memory-only on R
    rea_M, _ = _eval_M(model, cfg, erng, device, force_gate=1.0)      # reasoner-only on M
    rea_R, _ = _eval_R_depth(model, cfg, d0, erng, device, force_gate=1.0)   # reasoner-only on R
    dissoc = {"memory_only": {"M": mem_M, "R": mem_R}, "reasoner_only": {"M": rea_M, "R": rea_R}}
    dt = time.time() - t0
    if verbose:
        rstr = " ".join(f"R{depth}={accR[depth]:.3f}" for depth in cfg.test_r_depths)
        gstr = " ".join(f"R{depth}={gR[depth]:.3f}" for depth in cfg.test_r_depths)
        print(f"    [tandem s{seed}] params={params:,} ({dt:.0f}s) accM={accM:.3f} {rstr} "
              f"| gate[M={gM:.3f} {gstr}]", flush=True)
        print(f"      DISSOCIATION memory_only[M={mem_M:.3f} R={mem_R:.3f}] "
              f"reasoner_only[M={rea_M:.3f} R={rea_R:.3f}]", flush=True)
    return {
        "seed": seed, "params": params, "acc_M": accM,
        "acc_R": {str(depth): accR[depth] for depth in cfg.test_r_depths},
        "gate_M": gM, "gate_R": {str(depth): gR[depth] for depth in cfg.test_r_depths},
        "dissociation": dissoc, "wall_seconds": dt,
    }


@torch.no_grad()
def _eval_type(model, cfg, kinds, depth, k_eval, rng, device, force_gate=None):
    model.eval()
    correct = total = 0
    gates: list[float] = []
    for _ in range(cfg.eval_batches):
        batch = make_batch(rng, kinds, depth, cfg.seg_len, cfg.n_segments, cfg.n_chains, cfg.k_train)
        d = _to_dev(batch, device)
        states = _build_state(model, d["seg"], d["answer_seg"])
        # answer_pos-1 is the leak-free prediction position (see _train_one); answer_pos
        # is never 0 for any exercised config (reasoning_tasks always has a prefix byte).
        assert bool((d["answer_pos"] >= 1).all()), "answer_pos==0 would reintroduce the copy leak"
        pred_pos = d["answer_pos"] - 1
        out = model.tandem_step(
            d["seg"][:, d["answer_seg"]], None, states, return_states=False,
            aux_query_pos=None, tf_seed=None, steps=k_eval, collect_aux=False,
            force_gate=force_gate,
        )
        ans_logits = _answer_gather(out["logits"], pred_pos)  # LEAK FIX: predict from answer_pos-1
        gates.append(float(_answer_gather(out["gate"], pred_pos).mean().item()))
        correct += int((ans_logits.argmax(-1) == d["answer"]).sum().item())
        total += len(kinds)
    return correct / total, float(np.mean(gates))


def _eval_M(model, cfg, rng, device, force_gate=None):
    kinds = ["M"] * cfg.eval_batch
    return _eval_type(model, cfg, kinds, cfg.train_r_depths[0], cfg.k_train, rng, device, force_gate)


def _eval_R_depth(model, cfg, depth, rng, device, force_gate=None):
    kinds = ["R"] * cfg.eval_batch
    return _eval_type(model, cfg, kinds, depth, max(depth, cfg.k_train), rng, device, force_gate)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(seed_results: list[dict], cfg: TandemConfig) -> dict:
    """Aggregate per-seed results -> per-type table + gate routing + verdict."""
    test_depths = [str(x) for x in cfg.test_r_depths]

    def ms(xs):
        return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))

    accM = [r["acc_M"] for r in seed_results]
    accR = {d: [r["acc_R"][d] for r in seed_results] for d in test_depths}
    gM = [r["gate_M"] for r in seed_results]
    gR = {d: [r["gate_R"][d] for r in seed_results] for d in test_depths}
    deepest = test_depths[-1]

    mM = ms(accM)
    routing = {"gate_M": ms(gM)[0], f"gate_R@{deepest}": ms(gR[deepest])[0]}
    routing["separation"] = routing[f"gate_R@{deepest}"] - routing["gate_M"]

    # In-distribution (<= max train depth) vs extrapolation.
    in_dist = [d for d in test_depths if int(d) <= max(cfg.train_r_depths)]
    indist_R = float(np.mean([np.mean(accR[d]) for d in in_dist])) if in_dist else float("nan")
    both_indist = mM[0] >= 0.9 and indist_R >= 0.9
    routes = routing["separation"] >= 0.5 and routing[f"gate_R@{deepest}"] > routing["gate_M"]

    print("\n" + "=" * 78)
    print("TANDEM REPRO (repo pipeline) — per-type accuracy (mean+/-std over seeds)")
    print(f"  seeds={[r['seed'] for r in seed_results]}  train R {cfg.train_r_depths} -> test {cfg.test_r_depths}")
    print("=" * 78)
    print(f"  acc_M = {mM[0]:.3f}+/-{mM[1]:.3f}")
    for d in test_depths:
        m, s = ms(accR[d])
        print(f"  acc_R@{d:<3} = {m:.3f}+/-{s:.3f}   gate_R@{d} = {ms(gR[d])[0]:.3f}")
    print(f"  gate_M = {routing['gate_M']:.3f}  gate_R@{deepest} = {routing[f'gate_R@{deepest}']:.3f}  "
          f"separation = {routing['separation']:+.3f}")
    verdict = (
        "POSITIVE: both types >=0.9 in-dist AND gate routes (sep>=0.5)"
        if both_indist and routes else
        f"PARTIAL/REPORT: both_indist={both_indist} routes={routes}"
    )
    print(f"  VERDICT: {verdict}")
    return {
        "config": asdict(cfg), "seeds": seed_results,
        "acc_M_mean": mM[0], "acc_R_indist_mean": indist_R,
        "acc_R": {d: ms(accR[d]) for d in test_depths},
        "gate_routing": routing, "both_indist_ge_0.9": both_indist,
        "gate_routes_ge_0.5": routes, "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class _Args:
    seeds: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2))


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        ap = argparse.ArgumentParser()
        ap.add_argument("merge")
        ap.add_argument("shards", nargs="+")
        ap.add_argument("--out", default="")
        args = ap.parse_args()
        merged_seeds: list[dict] = []
        cfg_dict: dict = {}
        for p in args.shards:
            sh = json.load(open(p))
            cfg_dict = sh.get("config", cfg_dict)
            merged_seeds.extend(sh.get("seeds", sh.get("seed_results", [])))
        # report() only needs the depth grids; reconstruct those (typed) and default rest.
        cfg = TandemConfig(
            train_r_depths=tuple(int(x) for x in cfg_dict.get("train_r_depths", (4, 5, 6))),
            test_r_depths=tuple(int(x) for x in cfg_dict.get("test_r_depths", (4, 6, 16, 32))),
        )
        out = report(merged_seeds, cfg)
        if args.out:
            json.dump(out, open(args.out, "w"), indent=2, default=str)
            print(f"wrote {args.out}")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None, help="delta_layers (ablation; default 1)")
    ap.add_argument("--tie-embeddings", dest="tie", action="store_true",
                    help="TIE the head to the embedding (known-broken for the tandem — ablation only)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""), flush=True)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    if args.smoke:
        # Inherits the VERIFIED defaults (delta_layers=1, tie_embeddings=False); only the
        # cheap-run knobs are overridden.
        cfg = TandemConfig(train_steps=120, batch_size=16, seg_len=256, k_train=8,
                           d_model=64, eval_batches=2, eval_batch=64,
                           test_r_depths=(4, 6, 16), log_every=40, locate_warmup=40,
                           gate_mix_warmup=30, gate_commit_anneal=40, lr_warmup=20)
        print(">>> SMOKE", flush=True)
    else:
        cfg = TandemConfig()
        print(">>> FULL", flush=True)
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.steps is not None:
        cfg.train_steps = args.steps
    if args.layers is not None:
        cfg.delta_layers = args.layers
    if args.tie:
        cfg.tie_embeddings = True
    print(f"    delta_layers={cfg.delta_layers} tie_embeddings={cfg.tie_embeddings}", flush=True)

    seed_results: list[dict] = []
    for seed in seeds:
        seed_results.append(_train_one(cfg, seed, device, verbose=True))
        if args.out:  # write incrementally so a killed run resumes from disk
            json.dump({"config": asdict(cfg), "seeds": seed_results}, open(args.out, "w"),
                      indent=2, default=str)
            print(f"  wrote {args.out} ({len(seed_results)} seeds)", flush=True)

    report(seed_results, cfg)
    print("done", flush=True)


if __name__ == "__main__":
    main()
