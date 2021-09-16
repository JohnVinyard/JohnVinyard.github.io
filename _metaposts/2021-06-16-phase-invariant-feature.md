---
layout: post
title:  "Phase Invariant Feature"
date:   2021-06-16 07:00:00 -0500
categories: synthesis
published: true
---

# Here is a Title


## First a matrix

Here it is...

```python:
import numpy as np
from matplotlib import pyplot as plt

x = np.random.normal(0, 1, (128, 128))
_ = plt.matshow(x)
```

## Here's some math

{% raw %}
$$g: \mathbb{R}^{F \times T} \rightarrow \mathbb{R}^{embedding\_dim}$$
{% endraw %}


## This will display a plot

```python:
x = np.geomspace(0.01, 30, 101)
_ = plt.plot(x)
```

```python:
def a(f):
    return f

@a
def x(y):
    return y
```

## This will render an audio control

```python:
import zounds

synth = zounds.NoiseSynthesizer(zounds.SR22050())
_ = synth.synthesize(zounds.Seconds(10))
```

And here is the final paragraph.
