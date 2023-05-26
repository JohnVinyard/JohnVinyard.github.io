---
layout: post
title:  "Conjure Testing"
date:   2023-04-13 07:00:00 -0500
categories: testing
published: true
---

# Add a New First Paragraph

Here is the content.

# Testing Conjure

Here's the first paragraph.

## Audio

The first thing I'll do is include some audio.

```python:
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

## A Spectrogram

Below, find a spectrogram

```python:
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

## A Tensor

```python:
from conjure import numpy_conjure

@numpy_conjure(conjure_storage)
def tensor(shape):
    return np.random.normal(0, 1, shape).astype(np.float32)

shape = (5, 5, 5)
t = tensor(shape)
_ = tensor.meta(shape)
```



## Text Output

Let's start easy, with text output. But where is it?

```python:
from conjure import text_conjure
import numpy as np

@text_conjure(conjure_storage)
def array_shape(arr):
    return str(arr.shape)

arr = np.random.normal(0, 1, (3, 4, 5)).astype(np.float32)
result = array_shape(arr)
_ = array_shape.meta(arr)
```


## A Line

```python:
from conjure import time_series_conjure

@time_series_conjure(conjure_storage, name='line2')
def line(size):
    return (np.random.normal(0, 1, (2, size)) ** 2).astype(np.float32)

size = 100
l = line(size)
_ = line.meta(size)
```


## An AIM-Like Feature

A spectrogram-like feature that includes periodicity

```python:
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

Why isn't anything showing up here?

## Conclusion

Thanks for reading!