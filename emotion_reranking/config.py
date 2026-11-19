"""Experiment configuration and domain constants.

Every tunable value of the experiment lives here, so a run is fully described
by a single :class:`ExperimentConfig` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 28 labels of the GoEmotions taxonomy, in the order used for the emotion vectors.
GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)

# Valence of each emotion. ``None`` means the emotion has no polarity, so
# rating-based relevance is not defined for queries targeting it.
# NOTE: the insertion order matters, as data-driven queries are generated
# by iterating over this mapping.
EMOTION_VALENCE: dict[str, str | None] = {
    "admiration": "positive", "amusement": "positive", "approval": "positive",
    "caring": "positive", "curiosity": "positive", "desire": "positive",
    "excitement": "positive", "gratitude": "positive", "joy": "positive",
    "love": "positive", "optimism": "positive", "pride": "positive",
    "relief": "positive", "realization": "positive", "surprise": "positive",
    "anger": "negative", "annoyance": "negative", "confusion": "negative",
    "disappointment": "negative", "disapproval": "negative", "disgust": "negative",
    "embarrassment": "negative", "fear": "negative", "grief": "negative",
    "nervousness": "negative", "remorse": "negative", "sadness": "negative",
    "neutral": None,
}

# Product / service aspect lexicon used by the hybrid relevance judgment
# (matched as substrings of the cleaned review text).
ASPECT_TERMS: tuple[str, ...] = (
    "bateria", "tela", "som", "câmera", "teclado", "mouse", "carregador",
    "entrega", "embalagem", "qualidade", "material", "tamanho", "cor",
    "peso", "conexão", "bluetooth", "wifi", "aplicativo", "software",
    "atendimento", "suporte", "garantia", "reembolso", "troca", "devolução",
    "prazo", "atraso", "defeito", "funcionamento", "instalação", "manual",
    "preço", "custo", "valor", "desconto", "frete", "envio",
)

# Domain words removed in addition to scikit-learn's English stop words.
EXTRA_STOPWORDS: tuple[str, ...] = ("amazon", "prime")

QUERY_MODES = ("synthetic", "data_driven", "both")


@dataclass(frozen=True)
class HybridRelevanceWeights:
    """Weights of the hybrid relevance: ``w_e·emotion + w_i·informativeness + w_a·aspect``."""

    emotion: float = 0.5
    informativeness: float = 0.3
    aspect: float = 0.2


@dataclass
class ExperimentConfig:
    """All parameters of an experiment run."""

    # Data
    data_path: Path = Path("data/Amazon_Reviews.csv")
    kaggle_dataset: str = "dongrelaxman/amazon-reviews-dataset"
    n_docs: int = 4000
    min_docs_per_class: int = 150
    seed: int = 42

    # Output
    output_dir: Path = Path("results")

    # Encoders
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    text_model: str = "all-MiniLM-L6-v2"
    embed_batch_size: int = 16
    max_words_emotion: int = 400

    # First-stage retrieval (BM25)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 100

    # Re-ranking
    rl_sim_iterations: int = 2
    rl_sim_neighbors: int = 10
    reciprocal_k: int = 15
    reciprocal_lambda: float = 0.3

    # Fusion
    alpha_values: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    rrf_k: int = 60

    # Queries
    query_mode: str = "both"
    n_data_driven_queries: int = 100

    # Evaluation
    eval_k: int = 10
    n_bootstrap: int = 2000
    relevance_weights: HybridRelevanceWeights = field(default_factory=HybridRelevanceWeights)

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {self.query_mode!r}")

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"
