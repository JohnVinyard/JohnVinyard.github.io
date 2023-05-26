---
layout: post
title:  "Conjure Testing"
date:   2023-04-13 07:00:00 -0500
categories: testing
published: false
---

# Add a New First Paragraph

Here is the content.

# Testing Conjure

Here's the first paragraph.

## Audio

The first thing I'll do is include some audio.



```python
from conjure import audio_conjure

@audio_conjure(conjure_storage)
def musicnet_segment(url):
    import requests
    from io import BytesIO
    import zounds
    from librosa import resample

    resp = requests.get(url)
    bio = BytesIO(resp.content)
    original_audio = zounds.AudioSamples.from_file(bio).mono
    target_sr = zounds.SR11025()
    samples = resample(
        original_audio,
        orig_sr=int(original_audio.samplerate),
        target_sr=int(target_sr))
    samples = zounds.AudioSamples(samples, target_sr)

    n_samples = 2 ** 15
    resampled = zounds.AudioSamples(samples[:n_samples], target_sr)

    output = BytesIO()
    resampled.encode(output)
    output.seek(0)
    return output.read()

url = 'https://music-net.s3.amazonaws.com/1919'
result = musicnet_segment(url)
_ = musicnet_segment.meta(url)
```

<div id="conjure-id-3b0048cb84ec3a5f3b86d44ac532b2169a659499_85c36ef0e77285a9889df49a838172d8d54c24dd" data-conjure='{"key": "3b0048cb84ec3a5f3b86d44ac532b2169a659499_85c36ef0e77285a9889df49a838172d8d54c24dd", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/3b0048cb84ec3a5f3b86d44ac532b2169a659499_85c36ef0e77285a9889df49a838172d8d54c24dd", "content_type": "audio/wav", "feed_uri": "/feed/3b0048cb84ec3a5f3b86d44ac532b2169a659499", "func_name": "musicnet_segment", "func_identifier": "3b0048cb84ec3a5f3b86d44ac532b2169a659499"}'></div>

## A Spectrogram

Below, find a spectrogram



```python
from conjure import numpy_conjure, SupportedContentType

@numpy_conjure(conjure_storage, content_type=SupportedContentType.Spectrogram.value)
def spec(url):
    import numpy as np
    import zounds
    from io import BytesIO
    audio_bytes = musicnet_segment(url)
    bio = BytesIO(audio_bytes)
    samples = zounds.AudioSamples.from_file(bio)
    s = np.abs(zounds.spectral.stft(samples))
    return s

url = 'https://music-net.s3.amazonaws.com/1919'
result = spec(url)
_ = spec.meta(url)
```

<div id="conjure-id-f11325af6f149d80c75ae651e82505ecae4f413b_85c36ef0e77285a9889df49a838172d8d54c24dd" data-conjure='{"key": "f11325af6f149d80c75ae651e82505ecae4f413b_85c36ef0e77285a9889df49a838172d8d54c24dd", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/f11325af6f149d80c75ae651e82505ecae4f413b_85c36ef0e77285a9889df49a838172d8d54c24dd", "content_type": "application/spectrogram+octet-stream", "feed_uri": "/feed/f11325af6f149d80c75ae651e82505ecae4f413b", "func_name": "spec", "func_identifier": "f11325af6f149d80c75ae651e82505ecae4f413b"}'></div>

## A Tensor



```python
from conjure import numpy_conjure

@numpy_conjure(conjure_storage)
def tensor(shape):
    return np.random.normal(0, 1, shape).astype(np.float32)

shape = (5, 5, 5)
t = tensor(shape)
_ = tensor.meta(shape)
```

<div id="conjure-id-f615e2a9547cd47b4505c7a718c4b6165a13b499_9fa6f724fc861f44d619a4200879b93605038e87" data-conjure='{"key": "f615e2a9547cd47b4505c7a718c4b6165a13b499_9fa6f724fc861f44d619a4200879b93605038e87", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/f615e2a9547cd47b4505c7a718c4b6165a13b499_9fa6f724fc861f44d619a4200879b93605038e87", "content_type": "application/tensor+octet-stream", "feed_uri": "/feed/f615e2a9547cd47b4505c7a718c4b6165a13b499", "func_name": "tensor", "func_identifier": "f615e2a9547cd47b4505c7a718c4b6165a13b499"}'></div>



## Text Output

Let's start easy, with text output. But where is it?



```python
from conjure import text_conjure
import numpy as np

@text_conjure(conjure_storage)
def array_shape(arr):
    return str(arr.shape)

arr = np.random.normal(0, 1, (3, 4, 5)).astype(np.float32)
result = array_shape(arr)
_ = array_shape.meta(arr)
```

<div id="conjure-id-40e6234681cc2ed57c0b80787f646fd35e3f686b_ce6510312c911b53639a9e63a707e87124835758" data-conjure='{"key": "40e6234681cc2ed57c0b80787f646fd35e3f686b_ce6510312c911b53639a9e63a707e87124835758", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/40e6234681cc2ed57c0b80787f646fd35e3f686b_ce6510312c911b53639a9e63a707e87124835758", "content_type": "text/plain", "feed_uri": "/feed/40e6234681cc2ed57c0b80787f646fd35e3f686b", "func_name": "array_shape", "func_identifier": "40e6234681cc2ed57c0b80787f646fd35e3f686b"}'></div>


## A Line



```python
from conjure import time_series_conjure

@time_series_conjure(conjure_storage, name='line2')
def line(size):
    return (np.random.normal(0, 1, (2, size)) ** 2).astype(np.float32)

size = 100
l = line(size)
_ = line.meta(size)
```

<div id="conjure-id-baf34dde1f53f100f59edbe6129b5942b81ada8f_line2" data-conjure='{"key": "baf34dde1f53f100f59edbe6129b5942b81ada8f_line2", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/baf34dde1f53f100f59edbe6129b5942b81ada8f_line2", "content_type": "application/time-series+octet-stream", "feed_uri": "/feed/baf34dde1f53f100f59edbe6129b5942b81ada8f", "func_name": "line", "func_identifier": "baf34dde1f53f100f59edbe6129b5942b81ada8f"}'></div>


## An AIM-Like Feature

A spectrogram-like feature that includes periodicity



```python
from conjure import numpy_conjure, SupportedContentType
from io import BytesIO
import zounds

@numpy_conjure(conjure_storage, content_type=SupportedContentType.TensorMovie.value)
def scattering_like_transform(arr: np.ndarray):
    import torch
    import zounds
    import numpy as np

    signal = torch.from_numpy(arr).float().view(1, 1, arr.shape[-1])
    sr = zounds.SR22050()
    band = zounds.FrequencyBand(20, sr.nyquist)

    channels = 128

    scale = zounds.MelScale(band, channels)
    fb = zounds.learn.FilterBank(
        sr, 256, scale, 0.1, normalize_filters=True, a_weighting=False)
    spec = fb.forward(signal, normalize=False)

    window_size = 512
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1

    windowed = spec.unfold(-1, window_size, step_size)

    scatter = torch.abs(torch.fft.rfft(windowed, dim=-1, norm='ortho'))
    scatter = torch.log(1 + scatter)
    scatter += scatter.min()

    result = scatter.view(channels, -1, n_coeffs).permute(1, 2, 0).float()

    data = result.data.cpu().numpy()
    data = data / (np.abs(data.max()) + 1e-12)  # normalize

    return data.astype(np.float32)

url = 'https://music-net.s3.amazonaws.com/1919'
result = musicnet_segment(url)
io = BytesIO(result)
samples = zounds.AudioSamples.from_file(io)

aim = scattering_like_transform(samples)
_ = scattering_like_transform.meta(samples)
```

<div id="conjure-id-2c34dac54500c80696935e61e8056368160b8dbb_aa7b1c4900235098e0fb38457a0ae7dd16721948" data-conjure='{"key": "2c34dac54500c80696935e61e8056368160b8dbb_aa7b1c4900235098e0fb38457a0ae7dd16721948", "public_uri": "https://conjure-blog-test.s3.amazonaws.com/2c34dac54500c80696935e61e8056368160b8dbb_aa7b1c4900235098e0fb38457a0ae7dd16721948", "content_type": "application/tensor-movie+octet-stream", "feed_uri": "/feed/2c34dac54500c80696935e61e8056368160b8dbb", "func_name": "scattering_like_transform", "func_identifier": "2c34dac54500c80696935e61e8056368160b8dbb"}'></div>

Why isn't anything showing up here?

## Conclusion

Thanks for reading!