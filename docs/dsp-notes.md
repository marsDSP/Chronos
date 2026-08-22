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

## TanhAntiderivatives — regional minimax kernels for F1 and F2

`f1Tanh` and `f2Tanh` are regional double minimax kernels for the tanh
antiderivatives F1 = ln cosh(x) and F2 = integral of ln cosh. They replace
the dilogarithm closed form, whose relative error was unbounded near zero
(about 4.6e8 at x = 1e-8) because three 0.4-sized terms cancel down to
x^3/6. The factored forms carry no cancellation near zero.

Regions use a0 = 1 and a1 = 19 as crossovers.

Region I, |x| <= a0: F2 = x*u*P(u), F1 = u*S(u), u = x^2. No transcendental.
F2(0) and F1(0) are exact by construction. Parity is bit-exact: the sign of
F2 rides the leading x, and F1 depends on x only through u.

Region II, a0 < |x| < a1: F2 = 0.5*h^2 + C2 - 0.5*t*psi(t), F1 = h + t*L(t),
with h = a - ln2 and t = e^(-2a). The completed square keeps the
cancellation condition small (kappa <= 1.9). ln2 is split into a hi/lo pair so
its rounding does not enter h.

Region III, |x| >= a1: F2 = 0.5*h^2 + C2, F1 = h. The dropped terms are
below 1e-17 relative by the choice of a1.

Both region-I fits and both region-II fits interpolate the true value at the
a0 seam, so the seam discontinuity is evaluation rounding (under 1 ulp), not
the sum of two fit errors.

Evaluation uses Estrin, not Horner. A degree-14 Horner is a 14-deep serial
FMA chain; the Estrin split has dependency depth 4. The op order pairs on u,
then combines on u^2, u^4, u^8, all with fused multiply-adds. Coefficients
are padded with implicit zeros to a power-of-two count.

The coefficients are derived and regression-checked by the python scripts
(relative-error Remez in mpmath at 45 digits). The scripts exit non-zero if a
fresh derivation drifts from the header values.

## Trigonometry — removed approximations

`Trigonometry.h` previously carried `fasterExp`, `fasterLog`, `fasterTan`,
the Pade tan and exp approximants, and `boundToPi` / `boundToPiSIMD`. None
had a caller in the source or a test, so they were removed for hygiene. The
surviving `mmSin`, `mmCos`, `mmTan` minimax kernels keep their coefficients
and evaluation order unchanged.
