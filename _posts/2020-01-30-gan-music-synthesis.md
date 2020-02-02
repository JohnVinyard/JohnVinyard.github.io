---
layout: post
title:  "Music Synthesis Using Conditional and Unconditional Generative Adversarial Networks"
date:   2020-02-02 07:00:00 -0500
categories: synthesis
published: false
---

The last couple of posts have been all about audio analysis and search but in
this one, I'll return to some work that gets me a just a little closer to my
ultimate goal, which is to build synthesizers with high-level parameters
allowing the production of audio ranging from convincingly-real renderings of
traditional acoustic instruments to novel synthetic textures and sounds.

I think there's a lot of value in symbolic approaches to music generation but
I feel there's incredible promise and flexibility in producing audio directly.  
Some incredibly moving sequences of sound are also very difficult to score, and
not every valid or valuable piece of music can be expressed as a collection of
discrete notes or events.

#  A Break in the Case

I had been exploring the use of generative adversarial networks to produce
very short and noisy snippets of Bach piano music when I read the WaveGAN paper,
which was the first published application of GANs to raw audio that I'm aware
of.  This was a very exciting and insightful paper, but I got *really* excited
last year when I read a pair of papers that had something very cool in common.  
Those were:

- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
- [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/abs/1909.11646)

Both papers focused primarily on speech, although the MelGAN paper included
piano samples as well, but the interesting feature that both shared is that they
were both strongly-conditioned GANs that used low-frequency audio features as
input to the generator, which would then output high-frequency raw audio samples.

What really excited me about these papers is that they seemed to lay the
foundation for a very flexible two-stage music synthesis pipeline.  At its
simplest, two trained networks would be required:

1. A strongly-conditioned generator that can produce convincing audio from low-frequency features
1. Another network, unconditioned or perhaps conditioned on yet higher-level features, that could produce plausible sequences of these low-frequency features


I chose to use 256-bin Mel-frequency spectrograms as my low-frequency
conditioning feature.  So, to get a little more specific, I'd be performing an
experiment in two phases, training against the [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset initially.

1. Train a generator to produce 11025hz audio conditioned on spectrograms of real
classical music pieces.
2. Train a second generator to produce unconditioned, novel spectrograms

# Phase One: Adapting the MelGAN Paper

The MelGAN paper uses a single generator that makes use of dilated convolutions
to greatly expand the receptive field of its layers, as well as multiple discriminators that operate at a few different
time scales/sampling rates.  They use the hinge loss version of the GAN training
objective, which I also adopted.

TODO: HINGE LOSS FORMULA HERE

They use an additional feature matching objective for the generator, where the
generator gets to peek inside the discriminator's inner feature maps and tries
to match the internal feature maps it produces to those of real examples.

TODO: FEATURE MATCHING FORMULA HERE

## Generating Audio at Sampling Rates Appropriate to the Frequency Band

I diverged from the generator and discriminator architectures in the MelGAN
paper, however, as I saw an opportunity to try an idea for multi-resolution
analysis and synthesis idea that mirrors multi-resolution wavelet transforms or
the [Non-stationary Gabor Transform](https://grrrr.org/research/software/nsgt/).

In most raw audio-producing neural network architectures I've encountered thus far, all audio and thus all frequency bands are produced at the same sampling rate.  I wanted to try an architecture where groups of frequency
bands were produced at an appropriate sampling rate, such that low frequency
bands could be faithfully rendered with significantly fewer samples and only up-sampled at the last possible moment.

More concretely, for audio at 11025 hertz, the Nyquist frequency, or the highest
frequency we can accurately render would be 5512 hertz.

- One second of audio for frequencies from 2756 hertz to 5512 hertz would require 11025 samples
- One second of audio for frequencies from 1378 hertz to 2756 hertz would require
5512 samples
- One second of audio for frequencies from 689 hertz to 1378 hertz would require
2756 samples
- One second of audio for frequencies from 344 hertz to 689 hertz would require
1378 samples
- One second of audio for frequencies from 0 hertz to 344 hertz would require 689 samples

TODO: Schematic of multi-scale resolution

## Conditioned Audio Examples

# Phase Two: An Unconditioned GAN for Spectrogram Production

## Unconditioned Music Generation Examples

# Benefits of the Two-Stage Approach

# Other Datasets
