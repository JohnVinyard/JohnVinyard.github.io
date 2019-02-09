# Audio Query-By-Example via Unsupervised Embeddings

A couple months ago, I gave a talk at the 
[Austin Deep Learning Meetup](https://www.meetup.com/Austin-Deep-Learning/) 
about building [Cochlea](https://cochlea.xyz), a prototype audio similarity 
search I recently built.  There was a lot to cover in an hour, and much that was
glossed over, so I decided to write a blog post covering the process in a little
more detail.

[Slides from the talk](https://docs.google.com/presentation/d/1EB-B7WI42gOEKozXIkDvNUWaVjKQb_bqk5M_mUiueS0/edit?usp=sharing)


## Why Audio Search?
There's a ton of unlabelled audio data out there.  What if musicians and sound
designers could access large portions of it in a quick and intuitive way without
relying on manual tagging?

## Unsupervised Learning of Semantic Audio Representations
Train a network to produce representations of sound in a way that's similar to 
how word embeddings are produced.

https://arxiv.org/abs/1711.02209

## Deformations
We'd like our model to produce representations that are invariant to
 
- pitch shifts
- time stretches
- additive noise

Also, and probably most importantly, temporally proximal audio segments should
lie close together in the embedding space.

## Triplet Loss

## Log-Scaled Mel Spectrograms

## Within-Batch Semi-Hard Negative Mining

## Training

## Building an Index with Random Projections 

