---
layout: post
title:  "Building a Timbre-Based Similarity Search"
date:   2016-06-04 18:00:00 -0500
categories: zounds indexing search
---
For quite some time, I've been fascinated with the idea of indexing large corpuses of audio, based on some perceptual similarity metric.  This means that it isn't necessary for some human to listen to and tag each audio sample manually.  Searches are by example ([rather like the "search by image" feature that you've probably encountered](https://images.google.com/)), and it's possible to explore "neighborhoods" of similar sounds.

Prototyping these kinds of systems can be cumbersome and time-consuming.  When experimenting with these kinds of systems in the past, I've spent way too much time dealing with the mundane, but totally necessary details of:

- how to ingest lots of audio for a test bed
- how to extract common audio features
- how to use unsupervised learning to find good representations of the audio
- how to index and search those representations
- how and where to persist all of this data

[Zounds](https://github.com/JohnVinyard/zounds) was built with a mind to solving all of these common problems so that you have more time to focus on your idea, and spend less on bookkeeping.

# The Idea
Today, we'll be [working from this example from the zounds repository](https://github.com/JohnVinyard/zounds/blob/master/examples/timbre.py), and building a very simple, timbre-based similarity search.  Once you've read through this post, you shouldn't need to write a single line of code; just run [this example from the zounds repository](https://github.com/JohnVinyard/zounds/blob/master/examples/timbre.py), and it will run through each of the steps described below.

We'll be glossing over a lot of details; the goal is to get a high-level idea of what it takes to build such a search.  We'll explore some of the building blocks in much more detail in later posts. 

> **tim·bre**
>
> the character or quality of a musical sound or voice as distinct from its pitch and intensity.

# Defining our Baseline Features
Given that we want to group audio roughly by timbre (which is admittedly a very subjective thing), we'll need to transform audio in such a way that we achieve some translation invariance in the frequency domain.  Put another way, we'd like to understand something about the "shape" of the audio spectrum, or the relationships therein, ignoring the exact positions or frequencies of these shapes.

One of the most common ways to achieve this is to use a feature like [MFCC, or Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).  Roughly speaking, this is a spectrum of a spectrum, discarding phase, so we get an idea of what kinds of intervals, or relationships are present between dominant frequencies in the spectrum, without caring about the exact frequencies.

The first step will be to define a class [`WithTimbre`](https://github.com/JohnVinyard/zounds/blob/master/examples/timbre.py#L18) that will first extract [bark bands](https://en.wikipedia.org/wiki/Bark_scale) from the short-time fourier transform frames, and then compute the cepstral coefficients of those frames.

```python
import featureflow as ff
import zounds

class Settings(ff.PersistenceSettings):
    id_provider = ff.UuidProvider()
    key_builder = ff.StringDelimitedKeyBuilder()
    database = ff.LmdbDatabase(path='timbre', key_builder=key_builder)


windowing = zounds.HalfLapped()
STFT = zounds.stft(resample_to=zounds.SR22050(), wscheme=windowing)


class WithTimbre(STFT, Settings):
    bark = zounds.ConstantRateTimeSeriesFeature(
            zounds.BarkBands,
            needs=STFT.fft,
            store=True)

    bfcc = zounds.ConstantRateTimeSeriesFeature(
            zounds.BFCC,
            needs=bark,
            store=True)
```

Now calling..

```python
>>> _id = WithTimbre.process(meta=some_file_like_object_or_url)
```
...will extract the `bark` and `bfcc` features and persist them in an [LMDB database](https://symas.com/products/lightning-memory-mapped-database/) on your local filesystem.  

For a cello sound like this:
<audio controls="controls">
  Your browser does not support the <code>audio</code> element.
  <source src="http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/Cello.ogg" type="audio/ogg">
</audio>

Calling...
```python
>>> WithTimbre(_id).bark
```
...will return a [`ConstantRateTimeSeries`](https://github.com/JohnVinyard/zounds/blob/master/zounds/timeseries/timeseries.py#L172) instance, which is really just a subclass of [`numpy.ndarray`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.html) that knows about the frequency and duration of its samples.  It'll look like this, for an audio file containing a short snippet of a cello playing:

![Cello Bark Bands](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/CelloBarkBands.png)

Then, calling...
```python
>>> WithTimbre(_id).bfcc
```
...will return another `ConstantRateTimeSeries` that looks something like this, for the same cello sound:

![Cell Bark Frequency Cepstral Coefficients](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/Bfcc.png)

# Getting a Corpus of Data
There a tons of great places to find free audio samples online, but I happened upon this [ page](http://wiki.laptop.org/go/Free_sound_samples), hosted by the [One Laptop Per Child organization](http://wiki.laptop.org/go/The_OLPC_Wiki).  After browsing the sample sets available there, I decided that [this set](https://archive.org/details/FlavioGaete) looked like it had enough samples, and enough variety to make this search interesting.

We're going to download this file, and use the `WithTimbre` class we defined above to extract bark-frequency cepstral coefficients by streaming each audio file from the zip archive:

```python
# Download the zip archive
url = 'https://archive.org/download/FlavioGaete/FlavioGaete22.zip'
filename = os.path.split(urlparse(url).path)[-1]

resp = requests.get(url, stream=True)

print 'Downloading {url} -> {filename}...'.format(**locals())

with open(filename, 'wb') as f:
    for chunk in resp.iter_content(chunk_size=4096):
        f.write(chunk)

# stream all the audio files from the zip archive
print 'Processing Audio...'
for zf in ff.iter_zip(filename):
    if '._' in zf.filename:
        continue
    print zf.filename
    WithTimbre.process(meta=zf)
```

Notice the call to `WithTimbre.process(meta=zf)` in the very last line there.

# Learning a Good Representation
Now that we've processed all that audio, and persisted the extracted features to disk, we'd like to learn a representation of the features that is binary, making it possible to pack those features into bitarrays, and perform a brute-force, but fast bitwise hamming distance computation against every example in our dataset.

We'll start by doing  [K-Means clustering](https://en.wikipedia.org/wiki/K-means_clustering) on our `WithTimbre.bfcc` feature.  First, we'll define the pipeline for extracting a model from our dataset:

```python
@zounds.simple_settings
class BfccKmeans(ff.BaseModel):
    docs = ff.Feature(
            ff.IteratorNode,
            store=False)

    shuffle = ff.NumpyFeature(
            zounds.ReservoirSampler,
            nsamples=1e6,
            needs=docs,
            store=True)

    unitnorm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=shuffle,
            store=False)

    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=128,
            needs=unitnorm,
            store=False)

    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(unitnorm, kmeans),
            store=True)
```

This pipeline will:

- randomly sample from, and shuffle the dataset it receives
- give each example [unit norm](https://en.wikipedia.org/wiki/Unit_vector), because again, we're interested in the _shape_ of these examples, and not their magnitude
- Learn 128 means from the data
- Produce a pipeline that is now capable of transforming our `WithTimbre.bfcc` feature into a [one-hot encoding](https://en.wikipedia.org/wiki/One-hot): a 128-dimensional vector with a single non-zero entry
- 

To learn and persist our model, we just need to pass `BfccKmeans` an iterator over the examples we'd like it to learn from:

```python
>>> BfccKmeans.process(docs=(wt.bfcc for wt in WithTimbre))
```

# Building an Index
Now that we've learned a good representation of our `WithTimbre.bfcc` feature, we'd like to store that encoding.  Additionally, we know our choice of feature gives us translation invariance in the frequency domain, but we'd also like to introduce some invariance in the time domain, so in addition to storing our one-hot encoding, we'll also pool the feature together over short spans of time, so that, given our 128-dimensional one-hot encoding, pooled over say, 30 frames, we'd have a new 128-dimensional binary vector with at most 30 "on" entries, in the case where each of the thirty frames has a different code.

Concretely, for the same cello sound from above, we'd like to see our encoding looking something like this:

![K-means code for BFCC](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/CelloKmeans.png)

And we'd like to see the pooled version looking something like this:

![Pooled K-means codes](http://ec57ca2a108ec3bc8dd1-4304b0dba8021a8b61951b8806b1581c.r24.cf1.rackcdn.com/CelloKmeansPooled.png)

To achieve this, we'll define a new class, derived from our original `WithTimbre` class, that adds these new features:

```python
class WithCodes(WithTimbre):
    bfcc_kmeans = zounds.ConstantRateTimeSeriesFeature(
            zounds.Learned,
            learned=BfccKmeans(),
            needs=WithTimbre.bfcc,
            store=True)

    sliding_bfcc_kmeans = zounds.ConstantRateTimeSeriesFeature(
            zounds.SlidingWindow,
            needs=bfcc_kmeans,
            wscheme=windowing * zounds.Stride(frequency=30, duration=30),
            store=False)

    bfcc_kmeans_pooled = zounds.ConstantRateTimeSeriesFeature(
            zounds.Max,
            needs=sliding_bfcc_kmeans,
            axis=1,
            store=True)
```

It's interesting to note that these new stored features we've introduce will be computed lazily, on-demand.  The first time I do something like:

```python
doc = WithCodes(_id)
print doc.bfcc_kmeans_pooled
```

Both the `WithCodes.bfcc_kmeans` and `WithCodes.bfcc_kmeans_pooled` features will be computed and stored (the former will be computed because the latter depends upon it).

With that in mind, all we have to do now is to define and build our index over the `WithCodes.bfcc_kmeans_pooled` feature.  The encodings we need will be computed and stored on-demand in the course of building the index.

```python
BaseIndex = zounds.hamming_index(WithCodes, WithCodes.bfcc_kmeans_pooled)

@zounds.simple_settings
class BfccKmeansIndex(BaseIndex):
    pass

# build the index
BfccKmeansIndex.build()

# get an instance of the index
index = BfccKmeansIndex()
```

# Searching the Data
Finally, if we'd like to get an idea of whether we've sucessfully built an index that finds perceptually similar sounds, we can play around in the in-browser interactive repl.  There's a `random_search` method on the index we've built that chooses a random sample from the already-indexed data as a query, and then returns the n most similar or relavent results.

```python
>>> index.random_search()
```
Here's an example group of similar results:
<audio controls="controls">
  Your browser does not support the <code>audio</code> element.
  <source src="data:audio/ogg;base64,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" type="audio/ogg">
</audio>
