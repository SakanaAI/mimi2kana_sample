"""Build a token-ID text-to-kana dictionary usable by fst_decoder.

The CSV adapter follows aligner/src/lexicon.rs's UniDic columns and accents.
Existing rustfst/OpenFst dictionaries can also be loaded directly.
"""

import csv
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Sequence

import pynini
from tokenizers import Tokenizer


def build_lexicon(entries: Iterable[tuple[Sequence[int], Sequence[int]]]) -> pynini.Fst:
    """Compile alternative word readings, allowing zero or more words.

    Share identical (input, output) prefixes without determinizing the potentially
    non-functional transducer. Distinct readings of the same spelling survive.
    Duplicate identical entries are collapsed in the tropical semiring.
    Empty sides and ID 0 are rejected to avoid input-epsilon cycles in decoding.
    """
    fst = pynini.Fst()
    root = fst.add_state()
    fst.set_start(root)
    one = pynini.Weight.one("tropical")
    edges: dict[tuple[int, int, int], int] = {}
    count = 0
    for text, kana in entries:
        text, kana = tuple(text), tuple(kana)
        if not text or not kana or any(i <= 0 for i in (*text, *kana)):
            raise ValueError("Dictionary entries require nonempty sequences of positive token IDs")
        state = root
        for ilabel, olabel in zip_longest(text, kana, fillvalue=0):
            key = state, ilabel, olabel
            if key not in edges:
                edges[key] = fst.add_state()
                fst.add_arc(state, pynini.Arc(ilabel, olabel, one, edges[key]))
            state = edges[key]
        fst.set_final(state, one)
        count += 1
    if not count:
        raise ValueError("No usable dictionary entries")
    return fst.closure().arcsort("ilabel")


def _special_ids(tokenizer: Tokenizer) -> set[int]:
    return {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}


def construct_char2kana(
    csv_path: str | Path,
    text_tokenizer: Tokenizer,
    kana_tokenizer: Tokenizer,
    *,
    ignore_accent: bool = False,
) -> pynini.Fst:
    """Read headerless UniDic lex.csv (surface=0, pronunciation=13, accent=28)."""
    text_special = _special_ids(text_tokenizer)
    kana_special = _special_ids(kana_tokenizer)

    def entries():
        with Path(csv_path).open(newline="", encoding="utf-8") as source:
            for line, row in enumerate(csv.reader(source), 1):
                if len(row) < 29:
                    raise ValueError(f"UniDic CSV line {line}: expected at least 29 columns")
                surface, reading, accent_spec = row[0], row[13], row[28]
                if not reading or reading.strip() == "*":
                    continue
                text = text_tokenizer.encode(surface, add_special_tokens=False).ids
                kana = kana_tokenizer.encode(reading, add_special_tokens=False).ids
                if not text or not kana or text_special.intersection(text) or kana_special.intersection(kana):
                    continue
                accents = [0] if ignore_accent or accent_spec in ("", "*") else [int(a) for a in accent_spec.split(",")]
                for accent in accents:
                    variant = kana.copy()
                    if 0 < accent <= len(variant):
                        token = kana_tokenizer.id_to_token(variant[accent - 1])
                        accented = kana_tokenizer.token_to_id(token + "'")
                        if accented is not None:
                            variant[accent - 1] = accented
                    yield text, variant

    return build_lexicon(entries())


def main(
    lex_csv: Path,
    text_tokenizer: Path,
    kana_tokenizer: Path,
    output: Path,
    ignore_accent: bool = False,
) -> None:
    """Compile a UniDic CSV with the same tokenizers as the inference model."""
    fst = construct_char2kana(
        lex_csv,
        Tokenizer.from_file(str(text_tokenizer)),
        Tokenizer.from_file(str(kana_tokenizer)),
        ignore_accent=ignore_accent,
    )
    fst.write(str(output))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
