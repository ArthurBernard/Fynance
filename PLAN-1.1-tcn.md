# Plan — §1.1 Temporal Convolutional Network (TCN)

> Plan jetable à la racine (à archiver). Tâche roadmap §1.1.

## Livré
`fynance/models/tcn.py` : `TemporalConvNet(BaseNeuralNet)` — stack de blocs
résiduels `_TemporalBlock` (2 convs 1D dilatées causales + skip), dilation
doublée par bloc (1,2,4,…), read-out `Linear` vers M. `forward` : (L,N)→(1,N,L)
→ TCN → (L,M). Causalité par `_Chomp1d` (padding gauche tronqué) → output[t]
ne dépend que de input[≤t].

Intégration framework : hérite de `BaseNeuralNet` (`set_data`/`set_optimizer`/
`train_on`/`predict`). Compatible MSELoss **et** SharpeLoss. Exporté
(`fynance.models.TemporalConvNet`).

## Tests (8) — `tests/models/test_tcn.py`
forme forward · construction depuis dims · dilation 1/2/4 · gradient flow ·
**non-lookahead strict** (perturber X[t:] ne change pas output[:t]) ·
train_on MSELoss (loss finie) · train_on SharpeLoss (poids bougent) · predict détaché.

## Suite éventuelle (non bloquant)
`RollTemporalConvNet(TemporalConvNet, _RollingBasis)` calqué sur `RollMLP` si on
veut le walk-forward sur TCN — trivial, à faire au besoin.

## Vérif
- 8 tests verts ; suite 306 ; ruff + mypy 0 ; doctest tcn.py + sphinx -W OK.
