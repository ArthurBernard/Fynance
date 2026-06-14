# Plan — §1.2 Transformer financier

> Plan jetable à la racine (à archiver). Tâche roadmap §1.2.

## Livré
`fynance/models/transformer.py` :
- `PositionalEncoding` : encodage sinusoïdal **absolu** (Vaswani 2017).
- `_TransformerBlock` : self-attention causale (réutilise `MultiHeadAttention`
  de `attention.py`, qui fait déjà résiduel+LayerNorm) + FFN position-wise.
- `Transformer(BaseNeuralNet)` : proj. entrée N→d_model, +PE, `num_layers` blocs,
  read-out d_model→M. `forward` (L,N)→(L,M).
- **Masking causal** : `torch.tril` (L,L) passé à la MHA → position t n'attend que ≤t.

Intègre set_optimizer/train_on/predict ; compatible MSE et SharpeLoss. Exporté
(`fynance.models.Transformer`, `PositionalEncoding`).

## Tests (10) — `tests/models/test_transformer.py`
forme · dims · num_layers · d_model%heads≠0 lève · PE ajoute du signal ·
**non-lookahead strict** (masking causal vérifié) · gradient flow ·
train MSE · train SharpeLoss (poids bougent) · predict détaché.

## Choix / suite
- Positional encoding **absolu** retenu (standard, bien défini). Le **relatif**
  (roadmap « relatif vs absolu ») est laissé en option future — non bloquant.
- `RollTransformer` (walk-forward) calquable sur RollMLP au besoin.

## Vérif
- 10 tests verts ; suite 317 ; ruff + mypy 0 ; doctest + sphinx -W OK.
