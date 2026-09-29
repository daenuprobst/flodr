---
title: Choosing the input
sidebar_position: 2
---

# Choosing the input

FloDR builds its neighbour graph and its distance order on the input exactly as given. Rescaling a column changes
both, and with them the layout. Pass the representation in which you want distances to be read.

## Single-cell data

Use the same matrix you would give UMAP, such as the first 50 principal components of normalised, log-transformed
and scaled highly variable genes. The single-cell results of the FloDR paper used 50 components, each scaled to
unit variance.

An integrated latent from scVI, scPoli or a similar model can be passed as it is. Do not rescale its dimensions
first. The model that produced the latent defines its geometry, and standardising the dimensions moves both the
neighbour graph and the distance order.

## Binary fingerprints

If the first 1,000 rows contain only zeros and ones, FloDR treats the input as binary and uses Jaccard distances
for the graph and the ordinal term. Molecular and reaction fingerprints fall into this case.

## Narrow inputs

Inputs with fewer than 30 columns, such as a 10-dimensional scVI latent, are padded with columns of random noise up
to 30. The noise is drawn once per point at fit and gives the flow more room to route the layout through. The
neighbour graph and the ordinal term still use the input alone.

Padding has three visible effects.

- `transform` fills the padding of new points with zeros, so a training point passed to `transform` lands near
  its position in `embedding_`, not on it.
- `score_samples` is the density of the padded coordinates, with the padding of the evaluated points set to zero.
- Decoded inputs from `inverse_transform` and `inverse_latent` drop the padding and have the input's width.

`FloDR(advanced={"pad": 0})` turns padding off.

## Wide inputs

The flow carries one coordinate per input column, so time and memory grow with the width. Reduce inputs with
thousands of columns first, for example with PCA.

## Missing values

Rows with missing or infinite values raise a `ValueError`. Impute or drop them before fitting.
