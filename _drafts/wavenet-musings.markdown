The most realistic audio generation models in the recent past, namely,
[WaveNet](https://arxiv.org/abs/1609.03499) and
[SampleRNN](https://arxiv.org/abs/1612.07837) have some interesting (and surprising)
properties:

- they model audio at the PCM sample level, and not at the FFT frame level,
which the vast majority of previous work has preferred
- they are auto-regressive and recurrent models, respectively, ultimately
meaning that each successive sample is conditioned on all previous samples.
Sample generation is thus serial, and not parallelizeable (later work,
especially on WaveNet makes this not entirely true)
- the problem is modelled in both cases as a *classification* problem, and not
a regression problem.  Samples are modeled as discrete classes, and the
relationships between those classes must be learned, from scratch, by the model