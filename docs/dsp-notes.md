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

## OutputFilterStage — crossfade

The mode crossfade uses one signed position `fadePos_` in `[0, 1]`. The
value 0 is the Digital terminal and 1 is the Analog terminal. Each sample
steps the position by `1 / (fadeLengthSamples_ - 1)` toward the target
mode, and the step saturates at the terminals.

A signed position makes a mid-fade reversal continuous. A reversal only
changes the step direction. The position returns to the source mode with
no jump, because the source path stayed warm and kept processing samples.

The stage resets the incoming path only on a saturated-edge departure.
Leaving `fadePos_ == 0` toward Analog resets the Sallen-Key filters and
their ADAA states. Leaving `fadePos_ == 1` toward Digital resets the SVFs.
A mid-fade reversal does not reset the path it returns to, because that
path kept its state.

`setModeImmediate` snaps the position to a terminal and resets the
now-inactive path. The engine calls it from `resetParams`, so a saved
Analog preset opens in Analog with no fade and no Digital-topology audio.

## OutputFilterStage — analog coefficient ramp

The Analog Sallen-Key coefficients are an instantaneous `setParams` that
replaces the scattering matrix in one step. A knob sweep made the
coefficient trajectory a staircase, and the staircase clicked.

The stage now smooths each cutoff with a 10 ms one-pole. The smoother
advances once per 32-sample sub-block in closed form. The stage calls
`setParams` only when the smoothed cutoff moved more than `1e-4` relative
since the last solve. The `1e-4` guard is a numeric no-op, not an audible
deadband: the 10 ms trajectory crosses it while moving and settles below
it at rest, so the solve cost at rest is zero.

## BrigadeLine — split-step transport

The BBD timing compensation subtracts a group-delay term from the effective
delay before it maps to the clock:

    dEff = d + mod - satLatency - fade * baseT - gdBank

where `gdBank = getBankGroupDelayAtDC(fs) + kSplitStepOffset`. The bank term
is the input plus output anti-aliasing pole-bank group delay at DC, in
samples. The split-step offset `kSplitStepOffset = -1.0` is the
read-before-write hand-off between `readTap` and `writeSample`.

The FeedbackDelay BBD path calls `readTap()` then `writeSample()` per sample
(read before write). `writeSample(u_n)` stores `u_n` in `lastIn_`. The even
phase of the next `readTap()` consumes `lastIn_` and writes the charge into
the bucket register, so the charge written at sample n enters the register
one audio sample after the write call. The ring-read (Digital) core writes
the input to the ring, then reads the tap `readDelay` samples back, so the
write and the read it feeds share the same sample step. The split step
therefore shifts the BBD line delay by one sample relative to the ring core.

The sign and magnitude come from the loop-period identity, not from a fit.
The loop period is the round trip from `writeSample(w_n)` to the tap that
returns `w_n`. `bbd_loop_check` measures the BBD repeat centroids against
the `n * d` grid at crossfeed 0. With the offset at -1.0 the centroids land
within the gate with no systematic drift. A +1.0 offset would shift the loop
period by two samples and fail the gate at the second repeat, so the
derivation and the measurement agree on -1.0. The constant replaces the
former undocumented `-1.0f` fudge; the numeric value is unchanged.

## FeedbackDelay — BBD clock authority

The BBD branch programs the bucket-brigade clock per sample. A BBD clock
change retimes all in-flight charge: the achieved delay of a sample is the
transit integral of the clock history over its whole traversal, not the
instantaneous effective delay at read time. The two channels ran independent
clocks, so each channel accumulated a random-walk timing offset from its own
Ornstein-Uhlenbeck history. The crossfeed rotation then summed two taps whose
offsets differed, so the repeats flammed and left the `n * d` grid.

The fix shares one clock base across the channel pair and scales only the
differential wobble by the crossfeed amount. Per sample:

    dBase   = d - satLatency - fade * baseT - gdBank
    modMean = 0.5 * (modL + modR)           (stereo; mono collapses to modL)
    modMix  = clamp(crossCos^2 - crossSin^2, 0, 1)   (= cos 2*theta)
    dEffL   = dBase + modMix * (modL - modMean) + modMean
    dEffR   = dBase + modMix * (modR - modMean) + modMean

At crossfeed 0, `modMix = 1` and each channel keeps its independent wobble,
so the behaviour matches the single-clock path bit for bit. At crossfeed 1,
`modMix = 0` and both channels run one clock (the mean), so the two lines
stay time-aligned and the rotation sums coherent taps. Between, the
differential component scales down exactly as fast as the rotation mixes the
channels.

The Ornstein-Uhlenbeck draw order does not change. The generators advance
once per sample per channel in the same order as before, so the Digital core
stays bit-exact and the zero-crossfeed BBD path stays bit-identical.

The stereo wobble decorrelation collapses to a common wobble at full
ping-pong. This is the authentic behaviour of a two-channel bucket-brigade
line driven by one clock.

## FeedbackDelay — BBD mode-flip priming

A digital-to-bbd mode flip drops the in-flight repeats unless the bucket
register holds the recent audio. The ring write runs in both modes, so the
ring carries the most recent samples at the flip. `setParams` detects the
0-to-1 edge and copies the most recent `min(kStages, writeIdx_)` ring
samples into each `BrigadeLine` through `primeFrom`.

`primeFrom` fills the bucket storage oldest-to-newest, so the newest sample
lands at `bufferPtr_ - 1`. It resets the pole banks and `yBBD_old_` but
leaves the clock untouched. The primed audio plays at the bucket clock, not
the audio rate. A real bucket-brigade line that receives a clock change plays
its existing charge at the new clock, so no sample-rate conversion is
performed. The first bbd repeats continue the material.

The reverse bbd-to-digital edge needs no work. The ring write runs in both
modes, so the ring already carries the audio when the mode flips back.
