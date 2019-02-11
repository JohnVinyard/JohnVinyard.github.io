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


https://arxiv.org/abs/1711.02209

## Deformations
We'd like our model to produce representations that are invariant to
 
- pitch shifts
- time stretches
- additive noise

Also, and probably most importantly, temporally proximal audio segments should
lie close together in the embedding space.

## Triplet Loss

## Log-Scaled Mel Spectrograms

## Within-Batch Semi-Hard Negative Mining

## Training

## Building an Index with Random Projections 

## Future Directions