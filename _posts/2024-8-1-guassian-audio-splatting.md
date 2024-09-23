---
layout: post
title: "Gaussian/Gamma Splatting for Music"
date: 2024-8-1 07:00:00 -0500
categories: machine-learning
published: true
---

In this work, we apply a Gaussian Splatting-like approach to audio to produce a lossy, sparse, interpretable, and manipulatable representation of audio. We use a source-excitation model for each audio "atom", implemented by convolving a burst of band-limited noise with a variable-length "resonance", which is built using a number of exponentially decaying harmonics, meant to mimic the resonance of physical objects. Envelopes are built in both the time and frequency domain using gamma and/or gaussian distributions. Sixty-four atoms are randomly initialized and then fitted (3000 iterations) to a short segment of audio via a loss using multiple STFT resolutions. A sparse solution, with few active atoms, is encouraged by a second, weighted loss term.


[Read more, and listen to demos here!](/gamma-audio-splat.html)