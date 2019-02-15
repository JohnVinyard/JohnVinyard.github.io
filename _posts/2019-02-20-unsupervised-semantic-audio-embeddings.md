---
layout: post
title:  "Audio Query-By-Example via Unsupervised Embeddings"
date:   2019-02-13 07:00:00 -0500
categories: zounds search embeddings neural-networks pytorch
published: true
---

A couple months ago, I gave a talk at the 
[Austin Deep Learning Meetup](https://www.meetup.com/Austin-Deep-Learning/events/256293686/) 
about building [Cochlea](https://cochlea.xyz), a prototype audio similarity 
search I recently built.  There was a lot to cover in an hour, and much that was
glossed over, so I decided to write a blog post covering the process in a little
more detail.

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
So, we'd like to learn a function or mapping such that simple cosine distance 
in the target or embedding space corresponds to some of the highly complex 
geometric relationships between the anchor and deformed sounds we've just 
listened to.
  
For this experiment, that function will take the form of a deep convolutional
neural network with learn-able parameters:

{% raw %}
$$g: \mathbb{R}^{F \times T} \rightarrow \mathbb{R}^{embedding\_dim}$$
{% endraw %}

The original space is expressed as having dimensionality $F \times T$ because
our input representation will be a spectrogram, with the $F$ and $T$ 
representing the frequency and time dimensions, respectively.  We'll go into 
more detail about how this representation will be computed in the next section.

Our dataset will consist of $N$ "triplets" of data:

{% raw %}
$$\tau  = \{ t_i\}_{i=1}^N$$
{% endraw %}


Each triplet will look something like this:

{% raw %}
$$t_i = (x_{a}^{(i)}, x_{p}^{(i)}, x_{n}^{(i)})$$
{% endraw %}

and each member of the triplet will be a spectogram:
 
{% raw %}
$$x_a^{(i)}, x_p^{(i)}, x_n^{(i)} \in \mathbb{R}^{F \times T}  $$
{% endraw %}

Each $x_a$ represents an *anchor* audio segment, $a_p$ represents our 
*positive* example, i.e., the anchor with one of our audio deformations applied, 
or another audio segment that occurs near in time to the anchor, and $a_n$ 
represents our negative example, which we'll choose by picking any other audio 
segment from our dataset at random.  Given a large enough  dataset, our 
random strategy for choosing the negative example is probably fairly safe, 
but we will take some care  to not accidentally choose negative examples that 
are actually more perceptually similar to the anchor than the positive example.

We'll minimize a loss with respect to our network parameters that will try to 
push the anchor and positive examples closer together in our embedding space, 
while it also pushes our anchor and negative examples further apart:

{% raw %}
$$\mathcal{L}(\tau ) = \sum_{i=1}^{N}\left [  \left \|  g(x_a^{(i)}) - g(x_p^{(i)})\right \|_2^2 - \left \| g(x_a^{(i)}) - g(x_n^{(i)})\right \|_2^2   + \delta \right]_+$$
{% endraw %}

So, we'll try to ensure that the distance from anchor to positive examples is 
less than the distance from anchor to negative examples by some positive margin 
$\delta$, and the $_+$ here indicates the hinge loss, which really just means
that our loss goes to zero if it's less than this margin.  As usual, since we 
can't optimize over this term over all $N$ triplets, we'll optimize the 
learn-able parameters using minibatches of data.  Additionally, we'll see later
that because of the way we're sampling these triplets, our dataset is 
effectively unbounded.
 
## Log-Scaled Mel Spectrograms

The paper's authors compute their input spectrograms by:

1. using short time fourier transforms
1. discarding phase
1. mapping the resulting linear-spaced frequency bins onto an approximately log-spaced mel scale
1. applying a logarithmic scaling to the magnitudes

This is a fairly standard pipeline for computing a perceptually-motivated 
audio representation.  They go on to apply the various deformations mentioned 
above in this frequency domain.  There are some problems with this approach:

- mapping onto the Mel scale is done after the fact, from linear-spaced 
frequency bins that don't have the frequency resolution we'd like in lower 
frequency ranges and are needlessly precise in higher frequency ranges
- the representation isn't invertible, as phase is discarded and many frequency
bins in the higher ranges are averaged together, meaning that it's impossible to
know if the transformations actually _sound_ plausible.

For this project, I decided to perform all deformations in the time domain, 
mostly to convince myself that all of them continued to sound plausible.  I also 
opted to compute the time-frequency representation using a bank of log-spaced 
morlet wavelets, so that perceptually-inspired frequency spacing and 
time-frequency resolution trade-offs could be accounted for from the outset, 
instead of being something of an afterthought.
 
One nice side effect of this decision is that now the computation of the 
time-frequency representation happens in our neural network, or as part of our 
function $g$ with just another convolutional layer (with frozen, or 
non-learnable parameters), meaning that our input representation is now 
just $\mathbb{R}^T$, with $^T$ being the number of audio samples in each segment.
While the filter bank is not included in the network's learn-able parameters for 
this initial experiment, a learn-able time frequency transform is an open 
possibility in future iterations of this work.

You can see the implementation of our 
[learn-able $g$ function here](https://github.com/JohnVinyard/experiments/blob/master/unsupervised-semantic-audio-embeddings/network.py), 
which uses the [`zounds.learn.FilterBank`](https://zounds.readthedocs.io/en/latest/learn.html#zounds.learn.FilterBank)
module to compute the time frequency representation before passing it on to a 
stack of convolutional layers.  Also, [this jupyter notebook](https://github.com/JohnVinyard/experiments/blob/master/unsupervised-semantic-audio-embeddings/mel-scale-log-spectrogram.ipynb)
offers a more in-depth exploration of the motivations behind using a bank of 
log-spaced morlet wavelets to compute our log-frequency represnetation.

## Within-Batch Semi-Hard Negative Mining

## Training

## Building an Index with Random Projections 

## Future Directions

## Notes
[You can find slides from the original talk here](https://docs.google.com/presentation/d/1EB-B7WI42gOEKozXIkDvNUWaVjKQb_bqk5M_mUiueS0/edit?usp=sharing), 
and watch a [video here](https://www.youtube.com/watch?v=hKYuEZ0dEu0&feature=youtu.be).