"""Pynini port of mimi2kana/aligner/src/decoding.rs.

Labels are tokenizer IDs (0 is epsilon). Weights are negative log probabilities.
Use log weights until the final shortest-path operation so alternative CTC
alignments contribute their summed probability, rather than only their maximum.
"""

from dataclasses import dataclass
from loguru import logger
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pynini
import torch


@dataclass(frozen=True)
class PruneConfig:
    beam: float | None = 10.0
    sausage_rank: int | None = None

    def __post_init__(self) -> None:
        if self.beam is not None and (math.isnan(self.beam) or self.beam <= 0):
            raise ValueError("beam must be positive, or None to disable pruning")
        if self.sausage_rank is not None and self.sausage_rank < 1:
            raise ValueError("sausage_rank must be positive, or None")


def _weight(fst: pynini.Fst, value: float) -> pynini.Weight:
    return pynini.Weight(fst.weight_type(), float(value))


def _compose(left: pynini.Fst, right: pynini.Fst) -> pynini.Fst:
    return pynini.compose(
        left.copy().arcsort("olabel"), right.copy().arcsort("ilabel")
    )


def optimize_lattice(fst: pynini.Fst) -> pynini.Fst:
    """Explicit epsilon removal, determinization and minimization of an acceptor."""
    fst = fst.copy().connect()
    if fst.start() == -1:
        return fst
    fst.rmepsilon()
    fst = pynini.determinize(fst)
    fst.minimize()
    return fst


def prune_acyclic(fst: pynini.Fst, beam: float | None) -> pynini.Fst:
    """Keep arcs with forward + arc + backward < total + beam.

    Unlike OpenFst's tropical prune, log shortest distances sum path mass.
    This reproduces the Rust aligner's posterior-mass pruning criterion.
    The input is copied; disconnected/empty lattices are safe.
    """
    fst = fst.copy().connect()
    if fst.start() == -1:
        return fst
    if not fst.properties(pynini.ACYCLIC, True):
        raise ValueError("prune_acyclic requires an acyclic lattice")
    if beam is None or math.isinf(beam):
        return fst
    fst.topsort()
    fwd = pynini.shortestdistance(fst, queue_type="top")
    bwd = pynini.shortestdistance(fst, reverse=True, queue_type="top")
    threshold = float(bwd[fst.start()]) + beam
    for state in fst.states():
        arcs = [
            arc for arc in fst.arcs(state)
            if float(fwd[state]) + float(arc.weight) + float(bwd[arc.nextstate])
            < threshold
        ]
        fst.delete_arcs(state)
        for arc in arcs:
            fst.add_arc(state, arc)
    return fst.connect()


def _ctc_fsa(
    logits: torch.Tensor,
    masks: torch.Tensor | None,
    blank_id: int,
    config: PruneConfig,
    suppressed_ids: Sequence[int],
    arc_type: str = "log",
) -> pynini.Fst:
    """Expand only reachable frame/previous-label states of sausage o dectc.

    This is the output projection of the Rust composition, constructed directly
    to avoid materializing a vocabulary-squared CTC transducer for the text head.
    """
    logits = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float32)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [steps, classes]")
    if not 0 <= blank_id < logits.shape[1]:
        raise ValueError("blank_id is outside the vocabulary")
    if masks is not None:
        masks = torch.as_tensor(masks, device="cpu")
        if masks.dtype != torch.bool or masks.shape != logits.shape[:1]:
            raise ValueError("masks must be boolean with shape [steps]")
        logits = logits[masks]
    if not torch.isfinite(logits).all():
        raise ValueError("unmasked logits must be finite")
    logits = logits.clone()
    suppressed = set(suppressed_ids) - {blank_id}
    if any(i < 0 or i >= logits.shape[1] for i in suppressed):
        raise ValueError("suppressed_ids are outside the vocabulary")
    if blank_id != 0 and 0 not in suppressed:
        raise ValueError("token ID 0 is epsilon; suppress it or use it as blank")
    if suppressed:
        logits[:, sorted(suppressed)] = -10000.0
    logprobs = logits.log_softmax(-1)
    fst = pynini.Fst(arc_type=arc_type)
    start = fst.add_state()
    fst.set_start(start)
    previous = {blank_id: start}
    one = _weight(fst, 0)
    for row in logprobs:
        threshold = -math.inf if config.beam is None else float(row.max()) - config.beam
        if config.sausage_rank is not None and config.sausage_rank < len(row):
            # Rust indexes rank r (zero-based), then applies a strict threshold.
            # Ties at the boundary are excluded, not arbitrarily split.
            rank_threshold = float(row.topk(config.sausage_rank + 1).values[-1])
            threshold = max(threshold, rank_threshold + torch.finfo(torch.float32).eps)
        labels = [
            i for i in (row > threshold).nonzero().flatten().tolist()
            if i not in suppressed
        ]
        current = {label: fst.add_state() for label in labels}
        for last_label, src in previous.items():
            for label, dst in current.items():
                output = 0 if label == blank_id or label == last_label else label
                fst.add_arc(src, pynini.Arc(output, output, _weight(fst, -float(row[label])), dst))
        previous = current
    for state in previous.values():
        fst.set_final(state, one)
    return fst.connect()


def make_ctc_lattice(
    logits: torch.Tensor,
    masks: torch.Tensor | None = None,
    blank_id: int = 4,
    prune_config: PruneConfig = PruneConfig(),
    suppressed_ids: Sequence[int] = (0, 1, 2, 3),
) -> pynini.Fst:
    fst = _ctc_fsa(logits, masks, blank_id, prune_config, suppressed_ids)
    return optimize_lattice(prune_acyclic(fst, prune_config.beam))


def load_lexicon(path: str | Path) -> pynini.Fst:
    """Read an OpenFst binary, including a rustfst text2kana.fst (token IDs)."""
    fst = pynini.Fst.read(str(path))
    if fst.arc_type() != "standard":
        raise ValueError("text2kana must use standard (tropical) arcs")
    if fst.start() == -1:
        raise ValueError("text2kana dictionary is empty")
    # Labels must be tokenizer IDs, not Unicode code points. Symbol tables are
    # optional metadata and must not alter numeric composition.
    fst.set_input_symbols(None)
    fst.set_output_symbols(None)
    return fst.arcsort("ilabel")


def make_ctc_lattice_from_text(
    text2kana: pynini.Fst,
    logits: torch.Tensor,
    masks: torch.Tensor | None = None,
    blank_id: int = 4,
    prune_config: PruneConfig = PruneConfig(),
    suppressed_ids: Sequence[int] = (0, 1, 2, 3),
) -> pynini.Fst:
    if text2kana.arc_type() != "standard":
        raise ValueError("text2kana must use standard (tropical) arcs")
    text = _ctc_fsa(logits, masks, blank_id, prune_config, suppressed_ids, "standard")
    kana = _compose(text, text2kana)
    kana = pynini.arcmap(kana, map_type="to_log").project("output")
    return optimize_lattice(prune_acyclic(kana, prune_config.beam))


def allow_accent_errors(fst: pynini.Fst, vocab: Mapping[str, int]) -> pynini.Fst:
    """Add the accented/unaccented counterpart of each mora, as in lexicon.rs."""
    pairs = {}
    for token, label in vocab.items():
        counterpart = token.rstrip("'") if token.endswith("'") else token + "'"
        if label != 0 and counterpart in vocab and vocab[counterpart] != 0:
            pairs[label] = vocab[counterpart]
    fst = fst.copy()
    for state in fst.states():
        for arc in list(fst.arcs(state)):
            if arc.olabel in pairs:
                label = pairs[arc.olabel]
                fst.add_arc(state, pynini.Arc(label, label, arc.weight, arc.nextstate))
    return fst


def _add_start_weight(fst: pynini.Fst, penalty: float) -> None:
    if fst.start() == -1:
        return
    # Match Rust: apply to outgoing arcs (not the start state's final weight).
    arcs = list(fst.arcs(fst.start()))
    fst.delete_arcs(fst.start())
    for arc in arcs:
        arc.weight = _weight(fst, float(arc.weight) + penalty)
        fst.add_arc(fst.start(), arc)


def _normalize_outgoing_weights(fst: pynini.Fst) -> None:
    # Preserve the source algorithm's local normalization, excluding finals.
    for state in fst.states():
        arcs = list(fst.arcs(state))
        if not arcs:
            continue
        minimum = min(float(arc.weight) for arc in arcs)
        bias = -minimum + math.log(sum(math.exp(minimum - float(a.weight)) for a in arcs))
        fst.delete_arcs(state)
        for arc in arcs:
            arc.weight = _weight(fst, float(arc.weight) + bias)
            fst.add_arc(state, arc)


def shortest_path_ids(fst: pynini.Fst) -> list[int]:
    """Decode an acceptor after its equivalent label sequences were summed."""
    if fst.start() == -1:
        raise ValueError("No accepting path remains in the lattice")
    tropical = pynini.arcmap(fst, map_type="to_std") if fst.arc_type() != "standard" else fst
    best = pynini.shortestpath(tropical)
    state = best.start()
    if state == -1:
        raise ValueError("No accepting path remains in the lattice")
    labels = []
    while True:
        arcs = list(best.arcs(state))
        if not arcs:
            return labels
        if len(arcs) != 1:
            raise RuntimeError("Expected a single shortest path")
        arc = arcs[0]
        if arc.olabel:
            labels.append(arc.olabel)
        state = arc.nextstate


def merge_and_decode(
    main_lattice: pynini.Fst,
    side_lattice: pynini.Fst,
    main_lattice_penalty: float = 0.0,
) -> list[int]:
    logger.info(f"Merging two FSTs: # main state = {main_lattice.num_states()}, # side state = {side_lattice.num_states()}")
    
    if not math.isfinite(main_lattice_penalty):
        raise ValueError("main_lattice_penalty must be finite")
    main = main_lattice.copy()

    # put weights on the side's edges.
    side = _compose(side_lattice, main)
    if main_lattice_penalty > 0:
        _add_start_weight(main, main_lattice_penalty)
    elif main_lattice_penalty < 0:
        _add_start_weight(side, -main_lattice_penalty)

    side = optimize_lattice(side)
    _normalize_outgoing_weights(side)
    logger.info(f"side after preprocessing: #state = {side.num_states()}")

    # Remove union's epsilon arcs explicitly before log determinization.
    merged = optimize_lattice(pynini.union(main, side))
    return shortest_path_ids(merged)


def decode_with_lexicon(
    kana_logits: torch.Tensor,
    text_logits: torch.Tensor,
    text2kana: pynini.Fst,
    *,
    kana_blank_id: int,
    text_blank_id: int,
    masks: torch.Tensor | None = None,
    prune_config: PruneConfig = PruneConfig(),
    kana_vocab: Mapping[str, int] | None = None,
    main_lattice_penalty: float = 0.0,
) -> list[int]:
    """Decode a single utterance's [T, V] tensors to kana token IDs."""
    main = make_ctc_lattice(kana_logits, masks, kana_blank_id, prune_config)
    side = make_ctc_lattice_from_text(text2kana, text_logits, masks, text_blank_id, prune_config)
    if kana_vocab is not None:
        side = allow_accent_errors(side, kana_vocab)
        symbols = pynini.SymbolTable()
        symbols.add_symbol("<eps>", 0)
        for token, token_id in kana_vocab.items():
            if token_id != 0:
                symbols.add_symbol(token, token_id)
        for fst in (main, side):
            fst.set_input_symbols(symbols)
            fst.set_output_symbols(symbols)
    return merge_and_decode(main, side, main_lattice_penalty)
