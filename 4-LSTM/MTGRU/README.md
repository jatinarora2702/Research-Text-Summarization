# Sequence-to-Sequence Model and MTGRU Units

Here, we explore **MTGRU (Multiple Timescale Gated Recurrent Unit)** units in place of LSTMs. The meaning of a sentence is determined by the way its words are combined with each other. Similarly, to capture the entire meaning conveyed by a research text, the positioning of various paragraphs in important, and MTGRU cells have an temporal hierarchy component which helps them in capturing such compositionality.

Each paragraph in the scientific article is summarised by computing TF-IDF (Luhn, 1958) scores to extract one salient sentence. Such sentences are then concatenated to get the input for the model. The model is tested against the abstract of the paper.

Reference:

1. Kim, Minsoo, Moirangthem Dennis Singh, and Minho Lee. "Towards Abstraction from Extraction: Multiple Timescale Gated Recurrent Unit for Summarization." arXiv preprint arXiv:1607.00718 (2016).

2. Cho, Kyunghyun, et al. "[Learning phrase representations using RNN encoder-decoder for statistical machine translation](http://arxiv.org/abs/1406.1078)." arXiv preprint arXiv:1406.1078 (2014).