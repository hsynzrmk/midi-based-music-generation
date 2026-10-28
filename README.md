# MIDI-Based Music Generation via Gated Recurrent Units


This study investigates symbolic piano music generation using a
bidirectional Gated Recurrent Unit (BiGRU) network with self-attention.
The work focuses on a reproducible MIDI-to-piano-roll-to-token pipeline
and on evaluating generated sequences with objective musical
descriptors. The study does not propose a fundamentally new network
architecture. Instead, it examines the structure captured by the
recurrent model and the influence of different initialization
strategies.

## Dataset

The experiments use the **MAESTRO v1.0.0** dataset.

-   Dataset page: https://magenta.tensorflow.org/datasets/maestro#v100
-   Data format: MIDI performances.
-   Selected subset: 100 performances, sampled randomly rather than
    selected manually.
-   Split: 80 training performances, 10 validation performances, and 10
    test performances.
-   Random seed: 666.
-   When the official MAESTRO metadata CSV is available, the code first
    respects its train, validation, and test partitions. It then
    randomly samples the specified number of performances within each
    partition. If the metadata CSV is unavailable, the code uses a
    seeded random split of the file list.
-   The tokenizer is fitted on the training performances only to avoid
    using validation or test data to define the vocabulary.

The subset and split procedure are intended to make the experimental
setup reproducible. The selected files and their characteristics should
be reported from the generated dataset summary rather than inferred from
the dataset as a whole.

## MIDI representation and tokenization

MIDI performances are converted into piano-roll representations.
At each frame, simultaneously active pitches are represented as a token. 
A token may therefore represent a single note or a chord. 
Frames without active notes are represented by a rest token.

The model uses sequences of 50 consecutive frames as input and predicts
the next token. At 5 fps, the input context corresponds to 10 seconds.
Tokens are assigned indices using a vocabulary learned from the training
performances. Tokens that are absent from the training vocabulary are
tracked during validation and testing; unseen target tokens are excluded
from next-token evaluation, while unseen tokens in the input context are
mapped to the rest token.

The frame-based representation simplifies the symbolic sequence but also
introduces limitations. It quantizes note onsets to a 200 ms grid and
may omit notes that do not survive the piano-roll conversion.

## Model architecture and training

The main model consists of an embedding layer, three bidirectional GRU
recurrent layers, and two self-attention layers. Dropout is applied
after the attention layers and the final recurrent layer. A dense layer
with LeakyReLU activation precedes the softmax output layer.

  Parameter                         Setting
  --------------------------------- ----------------------------------
  Recurrent cell                    GRU
  Recurrent layers                  3 bidirectional layers
  Units per direction               128
  Embedding dimension               100
  Self-attention layers             2
  Attention type                    Multiplicative self-attention
  Attention width                   50
  Dropout                           0.3
  Dense layer                       64 units with LeakyReLU
  Optimizer                         Nadam
  Loss                              Sparse categorical cross-entropy
  Training epochs                   50
  Sequence length                   50 frames
  Batch size for gradient updates   96 sequences
  Random seed                       666

The model with the lowest validation loss is retained for evaluation.
The software and hardware environment should be reported using the
values exported by the experiment script, because these depend on the
machine on which the code is run.

## Baselines

The proposed model is compared with three baselines. All use the same
frame-token representation and the same piece-level train, validation,
and test split.

-   **Unigram sampling:** tokens are sampled independently according to
    their frequencies in the training token stream.
-   **First-order Markov chain:** transition probabilities are estimated
    from consecutive training tokens and smoothed using the unigram
    distribution.
-   **LSTM variant:** the recurrent cells are changed from GRU to LSTM
    while retaining the same overall model design and data pipeline.

These comparisons help distinguish structure learned by the recurrent
model from patterns that can be reproduced using token frequencies or
first-order transitions.

## Generation experiments

Generation is evaluated under two initialization conditions:

-   **Single-note initialization:** generation begins with a specified
    single-note token.
-   **Random-sequence initialization:** the initial context is randomly
    initialized.

The experiment script also includes sampling-based generation from the
training token distribution as a control. Ten independent generations
are produced per condition, with a maximum length of 300 generated
tokens. A temperature of 1.0 is used. The same playback frame rate is
applied to all generated files to avoid confounding differences in
playback speed with differences in initialization.

## Objective evaluation

Generated MIDI files are assessed using objective descriptors rather
than relying only on visual inspection of piano rolls. The descriptors
include:

-   pitch and pitch-class distributions;
-   pitch range and pitch variability;
-   pitch and pitch-class entropy;
-   note-duration statistics and duration diversity;
-   inter-onset intervals and their variability;
-   onset density and note density;
-   rest ratio and estimated tempo;
-   repeated short patterns; and
-   divergence between generated and reference pitch distributions.
