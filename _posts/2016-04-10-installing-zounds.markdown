---
layout: post
title:  "Getting Started with Zounds"
date:   2016-04-10 18:00:00 -0500
categories: zounds installation
---
[Zounds](https://github.com/JohnVinyard/zounds) is a python library for building audio feature extraction pipelines in a delcarative way, from re-usable building blocks.  It provides many of the primitives you'll need to start experimenting with audio, like [Short-Time fourier and discrete cosine transforms, chroma, and bark-frequency cepstral coefficients](https://github.com/JohnVinyard/zounds/blob/master/zounds/basic/audiograph.py#L28).

My goal is to make installing zounds as painless as possible, but as of this writing, there are a few small hurdles you'll need to clear before you can `pip install zounds`.

# Numpy and Scipy

Zounds depends on numpy and scipy.  If you don't already have them installed (if you do any kind of scientific or numerical computing with python, you surely do), I highly reccomemd [Continuum Analytics' Anaconda distribution](https://www.continuum.io/downloads).  This will get you up and running much more quickly than building numpy and scipy from source yourself.

# `libsndfile 1.0.26`

Zounds depends on the [most recent release of libsndfile](https://github.com/erikd/libsndfile/releases), which, as of this writing, [is not yet available from Ubuntu's default apt sources](https://launchpad.net/ubuntu/+source/libsndfile).  To get this latest version of libsndfile installed prior to installing zounds, [there's a bash script in the zounds git repository you can run](https://github.com/JohnVinyard/zounds/blob/master/setup.sh).  

# Installing From PyPI

Once you've got all the dependencies out of the way, your best bet will be to simply `pip install zounds`.  

# Installing From Source

Of course, if you need to be on the bleeding edge, you can [clone the repository from github](https://github.com/JohnVinyard/zounds), or [download the zipped source](https://github.com/JohnVinyard/zounds/archive/master.zip) and run `python setup.py install` from the source directory.

# Zounds Repl

For quick experiments, zounds provides a `zounds-quickstart` script, which you can run from anywhere after installing the package.  This will start an interactive repl that runs in your browser (any modern browser with WebAudio support will do).  The repl should behave just like the one you're accustomed to running from the console, but it also allows you to playback audio and view spectrograms, for a richer experience.  To start the server, just run:

```bash
zounds-quickstart --datadir data --port 9999
```

Then, in your browser, try the following:

```python
>>> _id = Document.process(meta='http://phatdrumloops.com/audio/wav/youregettn.wav')
>>> doc = Document(_id)
>>> doc.ogg # this should cause an html5 audio element to appear
>>> doc.centroid # this should cause a graph of spectral centroid to appear
```

Then, let's try sorting the audio frames by spectral centroid, ascending:

```python
>>> import numpy as np
>>> indices = np.argsort(doc.centroid)
>>> synth.synthesize(doc.dct[indices]) # a new html5 audio element should appear.  listen to it!
```
