---
layout: post
title: "Playable State-Space Model From a Single Audio Sample"
date: 2024-9-23 07:00:00 -0500
categories: machine-learning
published: true
---

This work attempts to reproduce a short segment of "natural" (i.e., produced by acoustic
instruments or physical objects in the world) audio by decomposing it into two distinct pieces:

1. A state-space model simulating the resonances of the system
2. a sparse control signal, representing energy injected into the system.

The control signal can be thought of as roughly corresponding to a musical score, and the state-space model
can be thought of as the dynamics/resonances of the musical instrument and the room in which it was played.

You can (read more and listen to a demo here!)[/ssm.html]