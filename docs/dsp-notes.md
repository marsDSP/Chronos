# DSP Notes

## Diffuser — section length cache

`Diffuser::computeSectionLens` runs a prime scan over a 65536-entry table.
The scan is expensive and runs more than once per prepare: the arena size
query (`ringStorageFloats`) and the prepare path (`prepareImpl_`) both call
it. A file-scope cache holds the result per sample rate so the scan runs
once.

The cache uses a `std::bitset<65536>` instead of the old `bool used[65536]`
stack array. The bitset lives inside the cache struct, not on the stack.

The prepare path is single-threaded at plugin scope. The cache needs no
mutex. If the engine ever prepares on multiple threads, add a lock.

## Modulation — OU step and depth calibration

`OrnsteinUhlenbeck` advances with the exact discrete step: `a = exp(-dt /
tau)`, `s = sqrt(1 - a^2)`, `x = a*x + s*g`. The Gaussian step `g` is a sum
of four uniforms, scaled to unit variance by sqrt(3). This choice needs no
branch and no cached second value, unlike Box-Muller. The state is clamped
to ±4 sigma so the delay read guard has a finite bound.

The depth parameter is in cents. The modulation maps it to an RMS delay
slope, not to an absolute delay offset. The delay slope sets the pitch
deviation: `f_out = f_in * (1 - d')`, so `cents = (1200 / ln 2) * d'` for
small slopes. A pitch reading on a tone averages the slope over the tone
period, and the OU increment RMS depends on the window length because the
process is not smooth. The scale `k` solves `k * windowedIncrementRms(tRef)
= cents * ln(2) / 1200`, where `windowedIncrementRms(w) = sqrt(2 * (1 -
a^w)) / w` and `tRef` is a 1 ms reference window (the period of a 1 kHz
tone). An absolute sample deviation would make the pitch deviation depend
on the delay time and on the rate. The slope mapping keeps the measured
pitch deviation equal to the depth at every delay time. The window choice
pins the depth reading to a 1 ms analysis window.

Both the chunked path and the reference path advance the OU states once per
sample per active channel, in the same order. The generators are seeded
from one constant plus a stream index, so two instances of the same
configuration produce identical streams.
