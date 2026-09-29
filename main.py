"""Transcribe one audio file without depending on s5stt."""

import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import torch
import tyro
from huggingface_hub import hf_hub_download
from loguru import logger
from torch import nn
from transformers import LlamaConfig, LlamaModel, MimiModel, PreTrainedTokenizerFast
from fst_decoder import PruneConfig, decode_with_lexicon, load_lexicon

class Mimi2Kana(nn.Module):
    """Inference-only version of the model defined in s5stt's mimi2kana_modules."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        c = SimpleNamespace(**config)
        self.encoder = None
        if c.encoder_layers > 0:
            self.encoder = LlamaModel(
                LlamaConfig(
                    hidden_size=c.d_model,
                    num_hidden_layers=c.encoder_layers,
                    intermediate_size=c.encoder_ffn_dim,
                    num_attention_heads=c.encoder_attention_heads,
                    max_position_embeddings=c.max_source_positions,
                    attention_dropout=c.encoder_attention_dropout,
                    initializer_range=c.encoder_initializer_range,
                )
            )
            self.encoder.embed_tokens = None
        self.input_adapter = (
            nn.Linear(c.input_dim or c.d_model, c.d_model)
            if c.use_input_adapter
            else nn.Identity()
        )
        self.embed_norm = nn.Identity() if c.skip_embed_norm else nn.RMSNorm(c.d_model)
        self.kana_head = nn.Linear(c.d_model, c.kana_vocab_size)
        self.text_head = (
            nn.Linear(c.d_model, c.text_vocab_size) if c.text_vocab_size > 0 else None
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.embed_norm(self.input_adapter(features))
        if self.encoder is not None:
            hidden = self.encoder(
                inputs_embeds=hidden, use_cache=False
            ).last_hidden_state
        logits = {"kana": self.kana_head(hidden)}
        if self.text_head is not None:
            logits["text"] = self.text_head(hidden)
        return logits


def decode(
    logits: torch.Tensor, tokenizer: PreTrainedTokenizerFast, blank_id: int
) -> str:
    ids = logits.argmax(dim=-1)[0].unique_consecutive().tolist()
    return tokenizer.decode([i for i in ids if i != blank_id], skip_special_tokens=True)


@torch.inference_mode()
def main(
    audio: Path,
    model_id: str = "yotarokubo/mimi2kana_csj",
    device: str = "cpu",
    text2kana: Path | None = None,
    fst_beam: float = 5.0,
    fst_rank: int | None = None,
    main_lattice_penalty: float = 0.0,
    allow_accent_errors: bool = True,
) -> None:
    """Transcribe an audio file and print text/kana as JSON.

    Args:
        audio: Input audio file (WAV, FLAC, etc.); converted to mono at 24 kHz.
        model_id: Hugging Face repository containing the mimi2kana checkpoint.
        device: PyTorch device, e.g. cpu or cuda.
        text2kana: OpenFst token-ID dictionary. If omitted, download
            tokenizers/text2kana.fst from yotarokubo/mimi2kana_csj.
        fst_beam: Positive pruning beam for frame candidates and lattice arcs.
        fst_rank: Optional per-frame candidate rank cutoff.
        main_lattice_penalty: Penalty applied to the kana-only branch before union.
        allow_accent_errors: Include accented/unaccented dictionary alternatives.
    """
    if text2kana is None:
        logger.info("Loading default text2kana dictionary from Hugging Face")
        text2kana = Path(hf_hub_download(
            "yotarokubo/mimi2kana_csj", "tokenizers/text2kana.fst"
        ))

    logger.info("Loading audio: {}", audio)
    waveform, _ = librosa.load(audio, sr=24000, mono=True)
    if waveform.size == 0:
        raise ValueError("Input audio is empty")

    logger.info("Loading {} and kyutai/mimi on {}", model_id, device)
    config = json.loads(
        Path(hf_hub_download(model_id, "model_config.json")).read_text()
    )
    if config.get("text_vocab_size", 0) <= 0:
        raise ValueError("FST dictionary decoding requires a model with a text head")
    checkpoint = torch.load(
        hf_hub_download(model_id, "checkpoint.pt"),
        map_location="cpu",
        weights_only=True,
    )

    weights = {
        k: v
        for k, v in checkpoint["model_state_dict"].items()
        if not k.startswith("f0_loss.")
    }
    model = Mimi2Kana(config)
    model.load_state_dict(weights)
    model.to(device).eval()
    del checkpoint, weights
    mimi = (
        MimiModel.from_pretrained("kyutai/mimi", dtype=torch.bfloat16).to(device).eval()
    )

    logger.info("Transcribing {:.2f} seconds of audio", waveform.size / 24000)
    signals = torch.from_numpy(waveform).to(device=device, dtype=torch.bfloat16)[
        None, None
    ]
    # Use continuous Mimi features before downsampling and quantization.
    features = mimi.encoder(signals, padding_cache=None).transpose(1, 2)
    features = mimi.encoder_transformer(
        features, use_cache=False, return_dict=True
    ).last_hidden_state
    outputs = model(features.float())
    result = {}
    tokenizers = {}
    for name, logits in outputs.items():
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(
                model_id,
                f"tokenizers/{name}/tokenizer.json",
            )
        )
        tokenizers[name] = tokenizer
        blank_id = config["blank_token_id" if name == "kana" else "text_blank_token_id"]
        result[name] = decode(logits, tokenizer, blank_id)

    prune_config = PruneConfig(fst_beam, fst_rank)
    lexicon = load_lexicon(text2kana)

    logger.info("Decoding kana/text lattices with {}", text2kana)
    ids = decode_with_lexicon(
        outputs["kana"][0],
        outputs["text"][0],
        lexicon,
        kana_blank_id=config["blank_token_id"],
        text_blank_id=config["text_blank_token_id"],
        prune_config=prune_config,
        kana_vocab=tokenizers["kana"].get_vocab() if allow_accent_errors else None,
        main_lattice_penalty=main_lattice_penalty,
    )
    result["kana_refined"] = tokenizers["kana"].decode(ids, skip_special_tokens=True)
    logger.success("Transcription complete")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(main)
