---
layout: post
title:  "Perceptual Audio Loss"
date:   2018-06-07 07:00:00 -0500
categories: zounds synthesis neural-networks pytorch
---


Today, I perform a small experiment to investigate whether a carefully designed
loss function can help a very low-capacity neural network "spend" that capacity
only on perceptually relevant features.  If we can design audio codecs like
 [Ogg Vorbis](https://xiph.org/vorbis/doc/Vorbis_I_spec.html) that allocate bits
 according to perceptual relevance, then we should be able to design a loss
 function that penalizes perceptually relevant errors, and doesn't bother much
 with those that fall near or below the threshold of human awareness.

Given enough capacity, data, and time, a model trained using mean squared error
 will eventually produce pleasing samples, but can we produce those same samples
 with *less* of all three?

I used [`zounds`](https://github.com/JohnVinyard/zounds) and
[`pytorch`](https://pytorch.org/) to build a small experiment where I dip the
first half of my pinkie toe into these waters.
# The Experiment

In this experiment, I'll train a "generator" network to transform a fixed noise
vector (128-dimensional, drawn from a gaussian distribution with zero mean,
unit variance, and a diagonal covariance matrix) into a *single* fixed audio
sample of dimension 8192 (~.75 seconds at 11025hz sampling rate).  I'll perform
this same experiment with five different audio samples, and two different loss
functions:

- [mean squared error](https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss)
- [a perceptually-inspired loss function](https://github.com/JohnVinyard/zounds/blob/master/zounds/learn/loss.py#L12)

## The Perceptually-Inspired Loss
The [`PerceptualLoss`](https://github.com/JohnVinyard/zounds/blob/master/zounds/learn/loss.py#L12)
is very rudimentary, but hopefully captures some characteristics of early stages
in the human auditory processing pipeline, namely:

- an FIR filter bank, whose filters' center frequencies lie along the [Bark scale](https://en.wikipedia.org/wiki/Bark_scale)
- half-wave rectification (AKA RELU)
- a logarithmic, or decibel-like amplitude scaling

The network architecture, as well as the weight initialization *scheme*
(but not the exact initialized weights) is held constant as we subjectively
evaluate the performance of our two loss functions on five different audio
files containing:

- [Richard Nixon speaking](https://archive.org/details/Greatest_Speeches_of_the_20th_Century)
- [a Bach piano piece](https://archive.org//details/AOC11B)
- [A song from the *Top Gun* soundtrack](https://archive.org/details/TopGunAnthem)
- [A Kevin Gates song](https://archive.org/details/TopGunAnthem)
- [A drumkit sample](http://www.phatdrumloops.com/beats.php)

For each `(audio sample, loss)` pair, the generator network is given 1000
iterations to learn to transform the fixed noise vector into the given audio
sample, and "checkpoint" audio samples are recorded from the network every 250
iterations.

## Inspiration and Previous Work
This approach was inspired by the
[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) paper.
While that paper sought to understand biases inherent in neural network
architectures by keeping the loss fixed, and playing with architectural choices,
this experiment holds the architecture fixed, and tries to understand the
contribution of different losses.

Both
[Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)
and [Generating Images with Perceptual Similarity Metrics based on Deep Networks](https://arxiv.org/abs/1602.02644)
explore losses that go beyond simple per-pixel (or per-sample) metrics.

## The Code
 The [code for this experiment can be found on GitHub](https://github.com/JohnVinyard/experiments/blob/master/audio-loss/audio_loss.py).


# The Results

In each section below, you can hear generations from the network every 250
iterations.

## Richard Nixon Speaking

### Original
<audio src="https://drive.google.com/open?id=1A7R9W-UH9lDP1h9vl62ola52kz3TRBdv&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1qwcY0SGKc8jQQGdWHp2snvDynRHL7FPo&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1V6JFl4f1x1yHavO0gojaAntcJEZrbZwK&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1T0W92I8HtvMTStBev8PJ6NIXz2arq8mI&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1ntYaPth9mmTfpvYC44C6u84HDyDMCCCe&authuser=0&export=download" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1NAi1MamTrVXCsaKZoWZ5SaVu1DGNBe46&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1eBmyq-_KP9bGsbsfqxgtLSIosRGRpr7D&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=19A9stv_GK4XdnhZd5xdQniDhTygkyiV5&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1nU6SBKGWlf8Q3QV06ZiM2CH1RPLuNX5R&authuser=0&export=download" controls></audio>

### Conclusions
For this sample, MSE at 250 noisy and unintelligible, while the perceptual loss
is already *starting* to be intelligible at the same point.  By the end of the
experiment, the samples generated by the network trained with MSE are fairly
intelligible, but completely missing the sibilant at the end of the word
"practice".  The network trained with perceptual loss captures this plainly and
simply.

## Bach Piano

### Original
<audio src="https://drive.google.com/uc?id=1E5MWSREqmDxCMCfG0P0GTaaeQLoQHftj&authuser=0&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=13Ny6OZc9t5cZlOwNquFhJbT5qQpdaBlZ&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1IKUrYGvf4IBC2xYswG_WoaExCmRQGuFY&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1DetNCEakbb1LG28DrfYLKG51K0d3wS9M&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1QIzDeRo0fRawEvuL4D4Zv6xpYVLpmY5O&authuser=0&export=download" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1ju4wLVD926QyOcX3Y0ycKMGTYHujvoqb&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1k3w5sI5DV-89_j1RyRU9NDlxAF76CZeD&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1-QphPZh7IQbeNWkQgDgLAyyNflwpD9hX&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1gOppnGPpWuxzr0ffcx1UjMdhGBd0iEfC&authuser=0&export=download" controls></audio>

### Conclusions
Even at the final iteration, generations from the network trained with MSE are
"blurry", and noisy.  Generations from the network trained with perceptual loss
 are also a bit noisy, but overall, the final result is significantly clearer.

## Top Gun Soundtrack

### Original
<audio src="https://drive.google.com/uc?id=1s23PoeVHJD9hyxCLcCK1GAwzt0G9cYFP&authuser=0&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=11vHj4MeaAWuRw7sNg9KmbkK_ouYiKbpF&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1VjrvvqhR53316iDjxH4emoIMzsEg_d4n&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1p29_bagg5Opb5w2klTa2aAHH6hhmj8Y1&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=19oHd2JsFPatweJmmuukWCrdIBh0ciuBv&authuser=0&export=download" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=18ZLPitxQPMjgqHcXQ73tcRo5rTjtZ1uW&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1A8xLjSvGpTvIB_JvqUyFSWywPcrCmAL1&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1WNzy71oDgN7kMUCYFhOgbkbVWrauxLjl&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1Uf9zbp-6t6hI2jGW0ltHRcu-uRQTCkVM&authuser=0&export=download" controls></audio>

### Conclusions

This sample has a pretty broad frequency range, and a lot's going on:
bass guitar, synth, female vocals, and a snare drum.  At 250 iterations,
generations from the network trained with MSE are a noisy, low frequency mess,
while generations from the network trained with perceptual loss are beginning to
be intelligible.

The network trained with MSE does OK by the final iteration, but the network
trained with perceptual loss does a much better job of generating the crisp
high end, especially notable in the crack of the snare drum.

## Kevin Gates

### Original
<audio src="https://drive.google.com/uc?id=14ooyrEn0yWstLK4G-6I6zMTfuY6F3j3a&authuser=0&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1eRIxPCYERS6yahhOgTQBcVoV4ufqLNyY&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1FBX76pDL31QgoUwCD7xVUFaqKqB9Wqiy&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1Ga3uk14qMgz2ZEmbR6R_hEBrr3-W1A9r&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1-Na5cNgUP1ldyXyLEAc4ut-4_gR9lRx9&authuser=0&export=download" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1-40tvz6kac8xP1g3SB335eBJ-q4_8yl8&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1yiCnpd3krXL1L6LYDpam6ou__ICrkFPH&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1fk-5wvkXPuen-DZA3ziLKMi1QE9-IrqR&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1L5QkDHwzK8ZtfxLrmTPU9QTNkMQki9R3&authuser=0&export=download" controls></audio>

### Conclusions

While this sample covers a frequencies similar to those from the "Top Gun"
soundtrack, both networks seem to struggle a bit more.  Unsurprisingly at this
point, the network trained with perceptual loss has managed to capture the
sharp attack of the snare drum, as well as make the words "I get..." at the end
 of the sample almost intelligibly.  Generations from both networks are noisy,
 but the MSE generations are much more so.

## Drumkit

### Original
<audio src="https://drive.google.com/uc?id=1APjAiOYRXeZ0imfduI6Wv0G2C5ODHrdK&authuser=0&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1gbNem3ldRcwYEO6L-p7-S41y9KwvBBai&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1bnWoGbWy1LYqs9WLY1bGreN6GwkS_Ri1&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1-ksrFiSjZdMCZmTk9tPLQwllVFuooUX8&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1MlIQpXZSa33YpcjiNpaItqBznYksup_z&authuser=0&export=download" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/uc?id=1NAOlxyZ3x3yDMEWeO-z1VRcgw_7o-yt9&authuser=0&export=download" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/uc?id=1N6rHaoq8GvyrX0r4Dvm2lNrsO1szEwPH&authuser=0&export=download" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/uc?id=1cGa3rxRKarF_lOgVLJeVIoetL7dBTWWY&authuser=0&export=download" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/uc?id=1IxT8uEzbbcnT-Uwc7dHl6WLrgIa8rlv2&authuser=0&export=download" controls></audio>

### Conclusions
The difference between the two losses is not as apparent here, although there's
certainly more definition in the mid range of the tom from the network trained
with perceptual loss.  Interestingly, the perceptual loss network seems to
plateau (or close to it) by iteration 250, while the MSE network starts pretty
terribly and makes its way slowly to the same point.


# Final Thoughts

While the results aren't terribly dramatic, it does seem that the
perceptually-inspired loss function has helped to use network capacity on more
salient (from a human perspective) features of each sound.  In general, this
loss promoted:

- crisper mid and high frequencies
- less noise
- quicker convergence on something intelligible

Obviously, my "perceptual model" is dead-simple, and certainly dead-wrong in
many cases.  Could we make even better use of model capacity by adding things
that perceptual audio codecs use to save bits, like tonal and temporal masking?
Our current model will penalize the generator for incorrectly producing a
frequency that may be inaudible to the listener!

# Addendum
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

The question of whether it's reasonable to expect to generate audio samples in
parallel, as if they were all independent of one another is an interesting one,
but isn't really the topic of this post.

What's most interesting in this context is the re-framing of the problem as a
classification problem.  This apparently has a couple benefits:

- While using MSE to do regression assumes a gaussian probability distribution,
this approach allows us to learn any arbitratily complex distribution
- With MSE and naive per-raw-sample regression, it's possible to do pretty
well by modelling low frequencies well (since that's where most of the energy
lies), and making a lot of noise.  By framing the problem as classification, it
becomes a lot harder for the model to get "partial credit" and coast along
happily

This is very clever, but doesn't totally sit right with me, because audio samples
*are* continuous, so why can't we model them as such.  Is the problem a missing
perceptual model that