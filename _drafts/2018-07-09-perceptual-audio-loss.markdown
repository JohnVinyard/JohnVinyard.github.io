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

## Richard Nixon Speaking

### Original
<audio src="https://drive.google.com/open?id=1A7R9W-UH9lDP1h9vl62ola52kz3TRBdv&export=download" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1qwcY0SGKc8jQQGdWHp2snvDynRHL7FPo" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1V6JFl4f1x1yHavO0gojaAntcJEZrbZwK" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1T0W92I8HtvMTStBev8PJ6NIXz2arq8mI" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1ntYaPth9mmTfpvYC44C6u84HDyDMCCCe" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1NAi1MamTrVXCsaKZoWZ5SaVu1DGNBe46" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1eBmyq-_KP9bGsbsfqxgtLSIosRGRpr7D" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=19A9stv_GK4XdnhZd5xdQniDhTygkyiV5" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1nU6SBKGWlf8Q3QV06ZiM2CH1RPLuNX5R" controls></audio>

### Conclusions

## Bach Piano

### Original
<audio src="https://drive.google.com/open?id=1E5MWSREqmDxCMCfG0P0GTaaeQLoQHftj" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/open?id=13Ny6OZc9t5cZlOwNquFhJbT5qQpdaBlZ" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1IKUrYGvf4IBC2xYswG_WoaExCmRQGuFY" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1DetNCEakbb1LG28DrfYLKG51K0d3wS9M" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1QIzDeRo0fRawEvuL4D4Zv6xpYVLpmY5O" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1ju4wLVD926QyOcX3Y0ycKMGTYHujvoqb" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1k3w5sI5DV-89_j1RyRU9NDlxAF76CZeD" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1-QphPZh7IQbeNWkQgDgLAyyNflwpD9hX" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1gOppnGPpWuxzr0ffcx1UjMdhGBd0iEfC" controls></audio>

### Conclusions

## Top Gun Soundtrack

### Original
<audio src="https://drive.google.com/open?id=1s23PoeVHJD9hyxCLcCK1GAwzt0G9cYFP" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/open?id=11vHj4MeaAWuRw7sNg9KmbkK_ouYiKbpF" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1VjrvvqhR53316iDjxH4emoIMzsEg_d4n" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1p29_bagg5Opb5w2klTa2aAHH6hhmj8Y1" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=19oHd2JsFPatweJmmuukWCrdIBh0ciuBv" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/open?id=18ZLPitxQPMjgqHcXQ73tcRo5rTjtZ1uW" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1A8xLjSvGpTvIB_JvqUyFSWywPcrCmAL1" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1WNzy71oDgN7kMUCYFhOgbkbVWrauxLjl" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1Uf9zbp-6t6hI2jGW0ltHRcu-uRQTCkVM" controls></audio>

### Conclusions

## Kevin Gates

### Original
<audio src="https://drive.google.com/open?id=14ooyrEn0yWstLK4G-6I6zMTfuY6F3j3a" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1eRIxPCYERS6yahhOgTQBcVoV4ufqLNyY" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1FBX76pDL31QgoUwCD7xVUFaqKqB9Wqiy" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1Ga3uk14qMgz2ZEmbR6R_hEBrr3-W1A9r" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1-Na5cNgUP1ldyXyLEAc4ut-4_gR9lRx9" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1-40tvz6kac8xP1g3SB335eBJ-q4_8yl8" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1yiCnpd3krXL1L6LYDpam6ou__ICrkFPH" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1fk-5wvkXPuen-DZA3ziLKMi1QE9-IrqR" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1L5QkDHwzK8ZtfxLrmTPU9QTNkMQki9R3" controls></audio>

### Conclusions

## Drumkit

### Original
<audio src="https://drive.google.com/open?id=1APjAiOYRXeZ0imfduI6Wv0G2C5ODHrdK" controls></audio>

### Mean Squared Error
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1gbNem3ldRcwYEO6L-p7-S41y9KwvBBai" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1bnWoGbWy1LYqs9WLY1bGreN6GwkS_Ri1" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1-ksrFiSjZdMCZmTk9tPLQwllVFuooUX8" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1MlIQpXZSa33YpcjiNpaItqBznYksup_z" controls></audio>

### Perceptual Loss
#### 250 Epochs
<audio src="https://drive.google.com/open?id=1NAOlxyZ3x3yDMEWeO-z1VRcgw_7o-yt9" controls></audio>
#### 500 Epochs
<audio src="https://drive.google.com/open?id=1N6rHaoq8GvyrX0r4Dvm2lNrsO1szEwPH" controls></audio>
#### 750 Epochs
<audio src="https://drive.google.com/open?id=1cGa3rxRKarF_lOgVLJeVIoetL7dBTWWY" controls></audio>
#### 1000 Epochs
<audio src="https://drive.google.com/open?id=1IxT8uEzbbcnT-Uwc7dHl6WLrgIa8rlv2" controls></audio>

### Conclusions

# Final Thoughts
