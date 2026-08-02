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
