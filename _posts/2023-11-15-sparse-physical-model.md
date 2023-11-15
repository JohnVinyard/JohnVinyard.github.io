---
layout: post
title: "Sparse Interpretable Audio Model"
date: 2023-11-15 07:00:00 -0500
categories: machine-learning
published: true
---

This post covers a model I've recently developed that encodes audio as a high-dimensional and sparse
tensor, inspired by algorithms such as matching pursuit and dictionary learning.

Its decoder borrows techniques such as waveguide synthesis and convolution-based reverb to
pre-load inductive biases about the physics of sound into the model, hopefully allowing it to
spend capacity elsewhere.

The goal is to arrive at a sparse, interpretable,
and hopefully easy-to-manipulate representation of musical sound. All of the model training and inference
code can be found in the [`matching-pursuit`](https://github.com/JohnVinyard/matching-pursuit) github repository,
which has become my home-base for audio machine learning research.

The current set of experiments is performed using the [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA).

> ⚠️ This post contains sounds that may be loud. Please be sure to start your headphones or speakers at a lower volume to avoid
    unpleasant surprises!

## Motivations

I've formed several hunches/intuitions in my [recent years working on machine learning models for audio synthesis](https://github.com/JohnVinyard/matching-pursuit/tree/main/experiments) that all find expression in the work being covering today.

### Frame-Based Representations Are Limited

Almost all codecs and machine learning models that perform audio analysis and/or synthesis are frame-based; they chop audio up into
generally fixed-size frames, without much regard for the content, encode each framem and represent audio as a sequence of these encodings.

The model discussed today takes some baby-steps away from a frame-based representation, and toward something that might be derived using [matching pursuit
or dictionary learning](https://en.wikipedia.org/wiki/Matching_pursuit).

While a sequence of encoded frames may reproduce sound very faithfully, it is not easily _interpretable_; each frame
is a mixture of physical or musical events that have happened before that moment, and continue to reverberate.

My sense is that a high-dimensional and _very_ sparse representation maps more closely onto the way we think about sound, especially musical audio.

### Audio Synthesis Models Must Learn Physics

While there are certainly classes of audio where the laws of physics are unimportant (synthesized sounds/music), those laws
are integral to the sounds made by acoustic instruments, and even many synthesized instruments simply aim to mimic these sounds.

Do audio models waste capacity learning some of the invariant laws of sound from scratch? Could it be advantageous to bake
some inductive biases around the physics of sound into the model from the outset, so that it can instead focus on understanding and extracting
higher-level "events" from a given piece of audio?

Synthesis techniques such as [waveguide synthesis](https://ccrma.stanford.edu/~jos/swgt/) and the use of long convolution kernels
to model material transfer functions and room impulse responses suggest a different direction that might be fundamentally better (for audio) than the classic upsample-then-convolve-with-short-kernel approaches that have dominated both image and audio generation to-date.

## Quick Tour

Before diving into the details, it might be good to get an intuitive feel for what the high-level components of the model
are doing. We grab a random segment of audio from the MusicNet dataset...



```python
from conjure import audio_conjure

@audio_conjure(conjure_storage)
def get_audio(identifier):
    from data.audioiter import AudioIterator
    from util import playable
    import zounds

    stream = AudioIterator(
        1,
        2**15,
        zounds.SR22050(),
        normalize=True,
        overfit=False,
        step_size=1,
        pattern='*.wav')
    chunk = next(iter(stream))
    chunk = chunk.to('cpu').view(1, 1, stream.n_samples)
    audio = playable(chunk, stream.samplerate)
    bio = audio.encode()
    return bio.read()

example_6 = get_audio('example_7')
_ = get_audio.meta('example_7')
```

<div id="conjure-id-645cebe5670a6f4f96103cc2eea08cf8c67cbacf_db17dd96d44651bcc95d5e9ca5827d6465c3214b" data-conjure='{"key": "645cebe5670a6f4f96103cc2eea08cf8c67cbacf_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/645cebe5670a6f4f96103cc2eea08cf8c67cbacf_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "content_type": "audio/wav", "feed_uri": "/feed/645cebe5670a6f4f96103cc2eea08cf8c67cbacf", "func_name": "get_audio", "func_identifier": "645cebe5670a6f4f96103cc2eea08cf8c67cbacf"}'></div>

You can click on the waveform to play the sound.

## The Encoding


The network encodes the audio, and we end up with something like this.

> ⚠️ Because the representation is very high-dimensional, `(4096, 128)` to be exact, we downsample for easier viewing



```python
from conjure import numpy_conjure, SupportedContentType

@numpy_conjure(
    conjure_storage,
    content_type=SupportedContentType.Spectrogram.value)
def get_encoding(identifier):
    from models.resonance import model
    import zounds
    from io import BytesIO
    import torch
    from torch.nn import functional as F

    n_samples = 2**15
    bio = BytesIO(get_audio(identifier))
    chunk = torch.from_numpy(zounds.AudioSamples.from_file(bio)).float()
    chunk = chunk[...,:n_samples].view(1, 1, n_samples)

    encoded = model.sparse_encode(chunk)

    # downsample for easier viewing
    encoded = encoded[:, None, :, :]
    encoded = F.max_pool2d(encoded, (8, 8), (8, 8))
    encoded = encoded.view(1, *encoded.shape[2:])
    encoded = encoded[0].T
    return encoded.data.cpu().numpy()

encoding_1 = get_encoding('example_7')
_ = get_encoding.meta('example_7')
```

<div id="conjure-id-d58810f9f504e43d753ec182ab3b6a94b20bc6ce_db17dd96d44651bcc95d5e9ca5827d6465c3214b" data-conjure='{"key": "d58810f9f504e43d753ec182ab3b6a94b20bc6ce_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/d58810f9f504e43d753ec182ab3b6a94b20bc6ce_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "content_type": "application/spectrogram+octet-stream", "feed_uri": "/feed/d58810f9f504e43d753ec182ab3b6a94b20bc6ce", "func_name": "get_encoding", "func_identifier": "d58810f9f504e43d753ec182ab3b6a94b20bc6ce"}'></div>

The dark-blue portions represent zeros, and the yellow-orange portions represent events.

## Reconstructed Audio

We can also perform a full pass through the network, both encoding and decoding.  The reconstructions don't sound amazing, but are clearly the same musical sequence as the original.



```python
from conjure import audio_conjure

@audio_conjure(conjure_storage)
def reconstruct_audio(identifier):
    from models.resonance import model
    from util import playable
    from modules.normalization import max_norm
    import torch
    import zounds
    from io import BytesIO

    n_samples = 2**15
    bio = BytesIO(get_audio(identifier))
    chunk = torch.from_numpy(zounds.AudioSamples.from_file(bio)).float()
    chunk = chunk[...,:n_samples].view(1, 1, n_samples)
    recon, _ = model.forward(chunk)
    recon = torch.sum(recon, dim=1, keepdim=True)
    recon = max_norm(recon)
    recon = playable(recon, zounds.SR22050())
    encoded = recon.encode()
    return encoded.read()

result = reconstruct_audio('example_7')
_ = reconstruct_audio.meta('example_7')
```

<div id="conjure-id-9819f428c9940c09d6521b4cc88417342366041a_db17dd96d44651bcc95d5e9ca5827d6465c3214b" data-conjure='{"key": "9819f428c9940c09d6521b4cc88417342366041a_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/9819f428c9940c09d6521b4cc88417342366041a_db17dd96d44651bcc95d5e9ca5827d6465c3214b", "content_type": "audio/wav", "feed_uri": "/feed/9819f428c9940c09d6521b4cc88417342366041a", "func_name": "reconstruct_audio", "func_identifier": "9819f428c9940c09d6521b4cc88417342366041a"}'></div>

### Another Reconstruction Example

original:



```python
result = get_audio('example_8')
_ = get_audio.meta('example_8')
```

<div id="conjure-id-645cebe5670a6f4f96103cc2eea08cf8c67cbacf_c79446d9ab305d83a6b976087b3623a4a9d68d39" data-conjure='{"key": "645cebe5670a6f4f96103cc2eea08cf8c67cbacf_c79446d9ab305d83a6b976087b3623a4a9d68d39", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/645cebe5670a6f4f96103cc2eea08cf8c67cbacf_c79446d9ab305d83a6b976087b3623a4a9d68d39", "content_type": "audio/wav", "feed_uri": "/feed/645cebe5670a6f4f96103cc2eea08cf8c67cbacf", "func_name": "get_audio", "func_identifier": "645cebe5670a6f4f96103cc2eea08cf8c67cbacf"}'></div>

reconstruction:



```python
result = reconstruct_audio('example_8')
_ = reconstruct_audio.meta('example_8')
```

<div id="conjure-id-9819f428c9940c09d6521b4cc88417342366041a_c79446d9ab305d83a6b976087b3623a4a9d68d39" data-conjure='{"key": "9819f428c9940c09d6521b4cc88417342366041a_c79446d9ab305d83a6b976087b3623a4a9d68d39", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/9819f428c9940c09d6521b4cc88417342366041a_c79446d9ab305d83a6b976087b3623a4a9d68d39", "content_type": "audio/wav", "feed_uri": "/feed/9819f428c9940c09d6521b4cc88417342366041a", "func_name": "reconstruct_audio", "func_identifier": "9819f428c9940c09d6521b4cc88417342366041a"}'></div>

## Model and Training Details

![model architecture](https://zounds-blog-media.s3.amazonaws.com/sparse-physical-model.jpg)

The model's encoder portion is fairly standard and uninteresting, while the decoder includes some novel features. Once again, code
for the experiment [can be found on github](https://github.com/JohnVinyard/matching-pursuit/blob/main/experiments/e_2023_10_2/experiment.py).

### Encoder

Starting with `32768 samples` at `22050 hz`, or about 1.5 seconds of audio, we first perform a short-time fourier transform
with a [window size of `2048 samples` and a step size of `256 samples`](https://github.com/JohnVinyard/matching-pursuit/blob/main/experiments/e_2023_10_2/experiment.py#L331). This means that we begin with a representation of shape `(batch, 1024, 128)`, where `1024`
is the number of FFT coefficients and `128` the number of frames.

The spectrogram is then processed by a U-Net architecture with skip connections.

Finally, the reprsentation is projected into a high-dimensional space and sparsified by choosing the `N` top-k elements in the `(batch, 4096, 128)` tensor.

> ⚠️ The astute reader will notice that we're working with the very frame-based representation we earlier warned against. There are still steps that need to be taken to fully free ourselves from the fixed-size grid.

### Decoder

Beginning with our highly-sparse `(batch, 4096, 128)` representation, we "break apart" this tensor into `N` one-hot vectors. The current
experiment sets [`N = 64`](https://github.com/JohnVinyard/matching-pursuit/blob/main/experiments/e_2023_10_2/experiment.py#L36).

Put another way, we now have `N` or `64` one-hot vectors, ideally representing "events" for each element in the batch of examples. Next, we embed these high-dimensional one-hot vectors into lower-dimensional, dense vectors, and generate the audio for each event
independently from there.

First, we generate an impulse, which amounts to band-limited noise, representing the attack, breath, or some other injection of energy into a system.

Next we create a linear combination of a number of pre-generated resonance patterns, which amount to four different waveforms (sine, square, sawtooth, triangle) sampled at many different frequencies within the range of fundamental frequencies for most acoustic instruments, in this case, from `40 hz` to `4000 hz`.  This represents the transfer function or impulse response of whatever system, or instrument, the impulse's energy is injected into.

We also choose a decay value, which determines the _envelope_ of the resonance and apply the envelope to the resonance we've generated.

Each impulse and resonance are then convolved to produce a single event.

The `(batch, 4096, 128)` encoding is also summed along its time-axis, such that it becomes `(batch, 4096)`, and is then embedded into
a lower-dimensional, dense vector. This vector is passed to a sub-network which chooses a linear combination of impulse responses for
convolution-based reverb. Each event is also convolved with this room response kernel.

We end with a tensor of shape `(batch, n_events, 32768)`, with each channel along the `n_events` dimension representing

### Training

Instead of using MSE loss, each "event channel" loss is computed independently.  There's much more work to do to both refine this approach and understand better why it helps, but it does seem to encourage events to cover independent/orthogonal parts of the time-frequency plane.  The model used to produce the encodings and audio in this post was trained for about 24 hours.


## Random Generation

Finally, we can understand the model a little better (and have some fun) by generating random sparse tensors and listening to the results.




```python
from conjure import audio_conjure

@audio_conjure(conjure_storage)
def random_generation(identifier):
    import torch
    from models.resonance import model
    from modules import sparsify2
    from util import playable
    import zounds
    from torch.nn import functional as F

    n_samples = 2**15
    encoding = torch.zeros(1, 4096, 128).uniform_(0, 3)
    encoded, packed, one_hot = sparsify2(encoding, n_to_keep=64)
    audio = model.generate(encoded, one_hot, packed)
    audio = torch.sum(audio, dim=1, keepdim=True)
    audio = playable(audio, zounds.SR22050())[..., :n_samples]
    bio = audio.encode()
    return bio.read()

result = random_generation('random_1')
_ = random_generation.meta('random_1')
```

<div id="conjure-id-333fe6671f0708af5ad96dd913d074cef0443b47_b9afce4f9c508c512a0cda11dfbce50743df19af" data-conjure='{"key": "333fe6671f0708af5ad96dd913d074cef0443b47_b9afce4f9c508c512a0cda11dfbce50743df19af", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/333fe6671f0708af5ad96dd913d074cef0443b47_b9afce4f9c508c512a0cda11dfbce50743df19af", "content_type": "audio/wav", "feed_uri": "/feed/333fe6671f0708af5ad96dd913d074cef0443b47", "func_name": "random_generation", "func_identifier": "333fe6671f0708af5ad96dd913d074cef0443b47"}'></div>

Another random generation result:



```python
result = random_generation('random_3')
_ = random_generation.meta('random_3')
```

<div id="conjure-id-333fe6671f0708af5ad96dd913d074cef0443b47_ebcc881d43ebe4cde7386ae408380c6ad58c5754" data-conjure='{"key": "333fe6671f0708af5ad96dd913d074cef0443b47_ebcc881d43ebe4cde7386ae408380c6ad58c5754", "public_uri": "https://zounds-blog-media.s3.amazonaws.com/333fe6671f0708af5ad96dd913d074cef0443b47_ebcc881d43ebe4cde7386ae408380c6ad58c5754", "content_type": "audio/wav", "feed_uri": "/feed/333fe6671f0708af5ad96dd913d074cef0443b47", "func_name": "random_generation", "func_identifier": "333fe6671f0708af5ad96dd913d074cef0443b47"}'></div>

## Next Steps

- A single, low-dimensional and dense "context" vector for each audio segment might allow similar sparse vectors to represent related musical sequences played on different instruments, or in different rooms
- The "physical model" I've developed is very rudimentary.  More study into existing physical modelling techniques might produce a far superior model


## Thanks for Reading!

If you'd like to cite this article

```
@misc{vinyard2023audio,
  author = {Vinyard, John},
  title = {Diffusion language models},
  url = {https://JohnVinyard.github.io/machine-learning/2023/11/15/sparse-physical-model.html},
  year = {2023}
}
```
