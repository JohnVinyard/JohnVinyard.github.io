---
layout: post
title:  "Cochlea: A RESTful API for Annotating Audio"
date:   2019-10-22 07:00:00 -0500
categories: api search annotation
published: true
---

[Cochlea](https://github.com/JohnVinyard/annotate-api) is an early-stage, 
RESTful API that allows users to annotate audio files on the internet.  
Segments or time intervals can be annotated with text tags or other arbitrary 
data.  This may not sound very exciting on its own, but I believe that these 
simple primitives, make possible incredibly diverse applications tailored to 
the needs of electronic musicians, sound designers and other folks interested 
in playing with sound.  

Before starting to dream aloud about the endless possibilities, a little about 
how I arrived here...

# False Starts

I'm interested in building tools that allow musicians, sound designers 
and machine learning researchers to explore libraries of audio samples in new 
and intuitive ways that go far beyond traditional tag-based text searches.  Text 
can be a great starting point, but indexes based on perceptual audio similarity 
or other features such as pitch or timbre offer much more exciting possibilities.  
Whereas text-based approaches require painstaking manual tagging of vast 
quantities of audio, indexes that organize sound based on features derived 
directly from the audio samples themselves make it feasible to imagine the 
entire internet as your sample library!

With this ideal in mind, I've started and discarded several audio similarity 
search applications due to overly-rigid approaches.  I've often settled on a 
single feature or similarity metric I think will work well and based the 
entire application around it.  Inevitably, the search works well in some contexts 
and not so well in others.  In addition, I've failed time and again to make 
indexing _new_ sounds painless and I've eschewed more basic but necessary 
features, like allowing users to tag audio and search across those tags.

The common theme in all these ventures has been a lack of flexibility due to 
assumptions I've baked in much too early in the process.  The RESTful API I 
introduced above is one possible answer to this problem, providing a simple 
platform on which all sorts of diverse applications might be built.

Now, to dig into the details...


# API Resources

The experimental Cochlea API consists of just three simple resources:

## `/sounds`

![Sound](https://cochlea-example-app-images.s3.amazonaws.com/waveform.png)

`sound` resources are really just pointers to audio that's hosted somewhere out 
there on the internet.  While it's not totally necessary, ideally the servers 
hosting the audio content will conform to a basic interface:

- The servers should support 
[byte-range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests), 
making partial downloads of the files possible
- The servers should allow for cross-origin requests by setting appropriate 
[CORS headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS), making 
it possible for front-end applications to request the audio and play it using 
the [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)

It's also worth noting that `sound` identifiers are ordered according to the 
time they were created, which will come in handy when we discuss `featurebot` 
and `aggregator` users in a bit.

## `/annotations`

![Annotations](https://cochlea-example-app-images.s3.amazonaws.com/annotation-tags.png)

`annotations` describe all or part of a sound using text tags or any other 
arbitrary piece of data.  The Cochlea API natively supports the creation 
and storage of text tags, but other arbitrary data describing sound segments can 
be hosted elsewhere.  We might create annotations tagging a segment of audio 
as containing female speech or create an annotation that highlights an 
interesting segment of a longer audio sequence.  Finally, we might compute 
dense numerical features from the raw audio samples (such as short-time 
Fourier transform data, chroma, or MFCC data) and store them as NumPy arrays 
in an S3 bucket.

In this way, annotations again become pointers to arbitrary data hosted on the 
internet, just as `sound` resources are.  Just as servers hosting audio data 
should conform to a particular interface, servers hosting dense features or 
other arbitrary annotation data would ideally support byte-range and 
CORS requests.

`annotation` identifiers are also ordered according to the time they were 
created, which will again come into play when we discuss `featurebot` and 
`aggregator` users a little later.

## `/users`

The third and final resource type we'll discuss is the `user` type.  There are 
a few different types to cover, and I think that this is where the platform 
really starts to get interesting.

### Humans

Humans are the first and most obvious user type.  These users can read 
`sound` and `annotation` resources and can create `annotation` resources of 
their own, usually using textual tags added using some graphical user interface.

### Datasets

`dataset` users represent some sound collection or repository on the internet.  
Some examples might include well-known audio datasets used by the machine 
learning community, such as 
[NSynth](https://magenta.tensorflow.org/datasets/nsynth) or 
[MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html).  
Since these datasets often include structured data about their audio files, 
`dataset` users will generally create both `sounds` and `annotations`.  For 
example, the MusicNet dataset includes detailed information about each note 
played in each piece, include onset time, duration, instrument and pitch.  
The NSynth dataset tags each note with certain characteristics such as 
`acoustic`, `percussive` or `electronic`.

### Featurebots

![Chroma](https://cochlea-example-app-images.s3.amazonaws.com/chroma_bot.png)

`featurebot` users can listen to some or all `sound` or `annotation` resources, 
optionally filtering by the user that created the resource or tags applied to it, 
and can compute features, such as onset times, 
[chroma](https://en.wikipedia.org/wiki/Chroma_feature) or 
[MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) features and can 
point to the computed/derived data using new `annotation` resources.  A few 
example applications might include:

- a bot that computes onset times using 
[`librosa`'s](https://librosa.github.io/librosa/generated/librosa.onset.onset_detect.html) 
onset detection functionality
- a bot that computes short-time Fourier transform data for each sound
- a bot that listens for annotations from the short-time Fourier transform bot 
and computes [chroma](https://en.wikipedia.org/wiki/Chroma_feature) or 
[MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) data, 
thus beginning to form a distributed computation graph that transforms the raw
audio

Since sounds and annotations have identifiers ordered according to the time 
they were created, bots need only remember the last id they processed and poll 
against `sound` or `annotation` resources to continually compute new features 
for incoming resources.

### Aggregators
![Hyperplane Tree](https://s3-us-west-1.amazonaws.com/unsupervised-audio-embeddings-talk/tree.gv.svg)

`aggregator` users are similar to `featurebot` users in that they listen to a 
stream of some or all `sound` or `annotation` resources, but these users 
only have read access and generally create alternative indexes over resources, 
making them searchable in novel ways.  A few example applications might include:

- a bot that listens for any `annotation` with a tag and creates a more 
full-featured text search including fuzzy matching or using sound/music-related 
word embeddings for high-quality semantic searches
- a bot that computes low-dimensional embeddings (using a technique 
[similar to this](http://johnvinyard.github.io/zounds/search/embeddings/neural-networks/pytorch/2019/02/22/unsupervised-semantic-audio-embeddings.html)) 
from audio or other derived features, making visual exploration possible in a 
user interface

These indexes might be published via a private or public REST API, allowing 
their consumers to navigate some or all of the audio available via the Cochlea 
API in interesting ways.

# An Example User Interface

![User Interface](https://exampleapp.cochlea.xyz/visual_explorer.jpg)

An early, alpha-stage user interface built atop the Cochlea API and the concepts 
outlined above can be found [here](https://exampleapp.cochlea.xyz/).  All the 
code for the example app can be found in the 
[Github repo](https://github.com/JohnVinyard/annotate-api).  It pulls together 
data contributed by several `user`, `featurebot` and `aggregator` users into 
an interface for sound discovery.  The users leveraged include:

- `dataset` users that contribute sounds from several datasets, including 
[NSynth](https://magenta.tensorflow.org/datasets/nsynth) and [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) 
- `featurebot` users that compute alternate visualizations of audio, 
including [short-time fourier transforms](https://en.wikipedia.org/wiki/Short-time_Fourier_transform), 
[chroma](https://en.wikipedia.org/wiki/Chroma_feature) and 
[MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) features.
- an `aggregator` user that embeds short segments of audio onto a 
three-dimensional sphere based on perceptual similarity, allowing users to 
navigate sound "space" using a Google Maps-like interface.

The API and web app are invite-only (for now), 
[but please reach out](mailto:john.vinyard@gmail.com) if you're interested in 
giving them a spin (in exchange for some feedback, of course)!









