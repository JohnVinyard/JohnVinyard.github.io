---
layout: post
title:  "Audio Query-By-Example via Unsupervised Embeddings"
date:   2019-02-13 07:00:00 -0500
categories: zounds search embeddings neural-networks pytorch
published: false
---

# Audio Query-By-Example via Unsupervised Embeddings

A couple months ago, I gave a talk at the 
[Austin Deep Learning Meetup](https://www.meetup.com/Austin-Deep-Learning/) 
about building [Cochlea](https://cochlea.xyz), a prototype audio similarity 
search I recently built.  There was a lot to cover in an hour, and much that was
glossed over, so I decided to write a blog post covering the process in a little
more detail.

[You can find slides from the original talk here](https://docs.google.com/presentation/d/1EB-B7WI42gOEKozXIkDvNUWaVjKQb_bqk5M_mUiueS0/edit?usp=sharing)


## Why Audio Search?
There are countless hours of audio out there on the internet, and much of it is
either not indexed at all, or is searchable only via subjective and relatively
low-bandwidth textual tags.  What if it was possible for musicians and sound
designers to navigate this data in an intuitive way, without depending on manual
annotations.

## Unsupervised Learning of Semantic Audio Representations
The first challenge in building this kind of index is producing a 
representation of short audio segments that captures important perceptual 
qualities of the sound.  This is of course subjective, and somewhat 
task-dependent (e.g., am I searching for audio with a similar timbre, or just
the same pitch, or maybe a similar loudness envelope), but in this case, we'd 
like to find some embedding space where elements are near one another if a 
human would likely assign the two sounds to the same class.  For example, two 
segments of classical solo piano music should fall closer together in the space 
than a segment of solo classical piano music and a segment of rock music. 

There's a great paper called 
[Unsupervised Learning of Semantic Audio Representations](https://arxiv.org/abs/1711.02209) 
from [Dan Ellis'](https://ai.google/research/people/DanEllis) research group at 
Google that develops one such representation by first noting a few things:

- certain types of transformations (e.g. pitch shifts and time dilation) don't
typically change the general sound class
- sounds that are temporally proximal (occur near in time) tend to be assigned
to the same sound class
- mixtures of sounds inherit the sound classes of the original elements

They leverage these observations to train a neural network that produces dense, 
128-dimensional embeddings of short (~one second) spectrograms such that 
perceptually similar sounds have a low cosine distance, or angle on the unit 
sphere.  What's neat is that this insight allows them to train in an 
unsupervised fashion, not requiring a large, labelled audio dataset to get 
started.

## Deformations
First, to prove to ourselves that the observations are valid, we can listen to 
some example deformations to get a feel for some of the transformations we'll
be applying.

### Original file
We'll choose the sound of a violin string being bowed, and use that as our 
anchor, applying different deformations to understand what those sound like, and
how they change our perception of the sound.

<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/original.ogg"></audio>

### Pitch Shift
Relatively small changes in pitch change the sound class we assign.

#### Higher
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/pitch-shift-up.ogg"></audio>

#### Lower
<audio control src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/pitch-shift-down.ogg"></audio>

### Time Stretch

Similarly, small changes in the duration of a sound shouldn't change the class
we assign either.

#### Longer
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/time-stretch-longer.ogg"></audio>

#### Shorter
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/time-stretch-faster.ogg"></audio>

### Temporal Proximity

If we're listening to an unedited audio recording (i.e., not spliced together
from many different sources), sounds that occur near one another in time are 
typically assigned to the same sound class.

#### Beginning
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/original-beginning.ogg"></audio>

#### Middle
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/original-middle.ogg"></audio>

#### End
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/original-end.ogg"></audio>

### Additive Noise

Small amounts of noise should not affect the sound class assigned.

<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/additive-noise.ogg"></audio>

### Mixtures of Sounds
Mixtures of two sounds should inherit the classes of their "parent" sounds.

#### Drums
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/drums.ogg"></audio>

#### Flute
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/flute.ogg"></audio>

#### Flute and Drums
<audio controls src="https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/drums-and-flute.ogg"></audio>


## Triplet Loss
So, we'd like to learn a function or mapping that simple cosine distance in the
target or embedding space corresponds to some of the highly complex geometric
relationships between the anchor and deformed sounds we've just listened to.
  
For this experiment, that function will take the form of a deep convolutional
neural network with learn-able parameters:

{% raw %}
$$g: \mathbb{R}^{F \times T} \rightarrow \mathbb{R}^d$$
{% endraw %}

    
 
## Log-Scaled Mel Spectrograms

## Within-Batch Semi-Hard Negative Mining

## Training

## Building an Index with Random Projections 

## Future Directions