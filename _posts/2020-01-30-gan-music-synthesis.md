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
experiment in two phases, using the [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset initially.

1. Train a generator to produce 11025hz audio from spectrograms of real
classical music pieces.

# Phase One: Adapting the MelGAN Paper to my Needs

## Conditioned Audio Examples

# Phase Two: An Unconditioned GAN for Spectrogram Production

## Unconditioned Music Generation Examples
