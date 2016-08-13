---
layout: post
title:  "Building a Simple Audio Encoder"
date:   2017-08-13 07:00:00 -0500
categories: zounds synthesis
---

Previously, we covered extracting some commonly used features from audio, and
[building a search](https://johnvinyard.github.io/zounds/indexing/search/2016/06/04/timbre-based-similarity-search.html).  In this post, we'll use a similar workflow to build
a very rudimentary audio encoder and decoder.  If you want to jump ahead, the
[complete code for this example is here](https://github.com/JohnVinyard/zounds/blob/master/examples/mdct_synth.py).

# The Modified Discrete Cosine Transform
We'll be using the mofified discrete cosine transform (or MDCT for short), as
the basis for our encoder.  [The Ogg Vorbis audio encoding also takes the MDCT as its starting point](https://xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-230001.3.2)

A cosine transform uses a set of cosine functions,
oscillating at different frequencies as its basis functions, and expresses a
time-domain signal as a linear combination of these frequencies.

To get an idea of what's happening here, we can fire up zounds interactive,
in-browser REPL and run the following:

```python
>>> synth = zounds.SineSynthesizer(zounds.SR22050())
>>> signal = zounds.synthesize(zounds.Seconds(1), freqs_in_hz=[440., 660., 880.])
>>> from scipy.fftpack import dct
>>> dct(signal)
```

This will display an image where you can clearly see that the three sine waves
we created are now represented as large coefficients for the cosine basis
functions at those frequencies.

![Cosine Transform](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/DiscreteCosineTransform.png)

While we can perfectly reconstruct the original signal given these coefficients
, we have poor localization in time.  The MDCT solves this problem by
applying the same transformation to short, overlapping blocks of a signal.

If we use the processing graph [defined here](https://github.com/JohnVinyard/zounds/blob/master/examples/mdct_synth.py#L10) to instead extract the MDCT, we end up with something a bit different.  Running...

```python
>>> _id = Document.process(meta=signal.encode())
>>> doc = Document(_id)
>>> doc.mdct[:, :100] # only look at lower frequencies, for brevity
```

will display something like this:

![Modified Discrete Cosine Transform](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/MDCT.png)

Here, you can see the same peaks, but instead of a single vector representing
coefficients for the entire signal, we see a vector of coefficients for each
short slice of time.

We can recover the original audio by doing the following:

```python
>>> mdct_synth = zounds.MDCTSynthesizer()
>>> mdct_synth.synthesize(doc.mdct)
```
