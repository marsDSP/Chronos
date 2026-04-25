#pragma once

// =====================================================================
//  Chronos SPSC queue
// ---------------------------------------------------------------------
//  A single-producer / single-consumer wait-free queue used as the
//  interlink between the audio (DSP) thread and the editor (UI) thread.
//
//  Layout
//  ------
//  The queue is a circular linked list of fixed-size "blocks".  Every
//  block is a power-of-two ring buffer; the queue therefore behaves as
//  a (potentially) growing chain of small ring buffers.  The head/tail
//  indices inside each block, and the head/tail block pointers in the
//  outer ring, are split between the two participating threads:
//
//      producer thread  (e.g. processBlock on the audio thread)
//          owns:  block.tail (write index)
//                 outer tailBlock pointer
//          reads: block.front, outer frontBlock
//
//      consumer thread  (e.g. juce::Timer callback on the message thread)
//          owns:  block.front (read index)
//                 outer frontBlock pointer
//          reads: block.tail, outer tailBlock
//
//  A block is full when (tail+1)&mask == front, empty when tail==front.
//  Because the producer never moves the consumer's variables and vice
//  versa, the steady-state operations need only release/acquire fences;
//  no CAS, no spinning, no lock.
//
//  Allocation policy
//  -----------------
//  enqueue() may allocate a fresh block when both the current tail
//  block is full and the next block in the ring is the consumer's
//  front block.  try_enqueue() / try_emplace() never allocate, returning
//  false instead.  Audio-thread producers should always use the try_*
//  variants; the queue should be sized at construction time so the
//  steady-state hot path is allocation-free.  Blocks are released only
//  when the queue itself is destroyed.
//
//  Threading contract
//  ------------------
//      * exactly one thread may call any of:
//            try_enqueue / enqueue / try_emplace / emplace
//      * exactly one thread may call any of:
//            try_dequeue / pop / peek
//      * size_approx() and max_capacity() may be called from either.
//      * move construction / assignment / destruction must be done
//        with no concurrent access.
// =====================================================================

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>

namespace Chronos::Concurrency
{

namespace SpscDetail
{
    // Cache-line guess.  Picked to dodge false sharing on every CPU we
    // currently care about (most x86 / ARM cores use 64-byte lines;
    // recent Apple Silicon uses 128, but 64 here only costs a little
    // padding and stays correctness-safe).
    inline constexpr std::size_t kCacheLineBytes = 64;

    // Round x up to the next power of two.  Returns 1 for x <= 1.
    constexpr std::size_t ceilToPowerOfTwo(std::size_t x) noexcept
    {
        if (x <= 1) return 1;
        --x;
        for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1)
            x |= x >> shift;
        return x + 1;
    }

    template <typename U>
    inline char* alignForwards(char* ptr) noexcept
    {
        constexpr std::size_t alignment = alignof(U);
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        const auto pad = (alignment - address % alignment) % alignment;
        return ptr + pad;
    }
} // namespace SpscDetail


// ---------------------------------------------------------------------
//  SpscQueue<T, MaxBlockSize>
// ---------------------------------------------------------------------
//  T            - element type
//  MaxBlockSize - largest individual block size; must be a power of two
//                 and >= 2.  Smaller blocks reduce wasted space if the
//                 queue is rarely full; larger blocks reduce pointer
//                 chasing in the dequeue path.
// ---------------------------------------------------------------------
template <typename T, std::size_t MaxBlockSize = 512>
class alignas(SpscDetail::kCacheLineBytes) SpscQueue
{
    static_assert(MaxBlockSize >= 2,
                  "SpscQueue: MaxBlockSize must be at least 2");
    static_assert((MaxBlockSize & (MaxBlockSize - 1)) == 0,
                  "SpscQueue: MaxBlockSize must be a power of two");

public:
    using value_type = T;

    // Construct a queue that can hold at least `initialCapacity`
    // elements without further allocation.  Larger requests are split
    // across several MaxBlockSize-sized blocks (with one spare block so
    // the producer is never blocked on the block the consumer is
    // currently draining).
    explicit SpscQueue(std::size_t initialCapacity = 15)
    {
        Block* firstBlock = nullptr;

        // We need one spare slot per block to disambiguate full from
        // empty (front == tail means empty, full is therefore
        // (tail + 1) & mask == front).
        largestBlockSize = SpscDetail::ceilToPowerOfTwo(initialCapacity + 1);

        if (largestBlockSize > MaxBlockSize * 2)
        {
            // Cap any single block at MaxBlockSize and chain enough
            // blocks for the requested capacity.  Solving the
            // "(blockSize - 1) * (numBlocks - 1) >= capacity" inequality
            // for numBlocks gives the closed form below.
            const std::size_t initialBlockCount =
                (initialCapacity + MaxBlockSize * 2 - 3) / (MaxBlockSize - 1);

            largestBlockSize = MaxBlockSize;
            Block* lastBlock = nullptr;
            for (std::size_t i = 0; i != initialBlockCount; ++i)
            {
                Block* block = makeBlock(largestBlockSize);
                if (block == nullptr) throw std::bad_alloc();

                if (firstBlock == nullptr) firstBlock = block;
                else                       lastBlock->next.store(block, std::memory_order_relaxed);

                lastBlock = block;
                block->next.store(firstBlock, std::memory_order_relaxed);
            }
        }
        else
        {
            firstBlock = makeBlock(largestBlockSize);
            if (firstBlock == nullptr) throw std::bad_alloc();
            firstBlock->next.store(firstBlock, std::memory_order_relaxed);
        }

        frontBlock.store(firstBlock, std::memory_order_relaxed);
        tailBlock .store(firstBlock, std::memory_order_relaxed);

        // Make sure the reader and writer threads see the fully
        // initialised structure when they start.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    SpscQueue(const SpscQueue&)            = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    // Move ops.  Caller guarantees no concurrent access during the move.
    SpscQueue(SpscQueue&& other) noexcept
        : frontBlock(other.frontBlock.load(std::memory_order_relaxed)),
          tailBlock (other.tailBlock .load(std::memory_order_relaxed)),
          largestBlockSize(other.largestBlockSize)
    {
        // Hand the moved-from queue a single fresh empty block so it
        // remains usable (and destructible) as a hollow shell.
        other.largestBlockSize = 32;
        Block* fresh = other.makeBlock(other.largestBlockSize);
        if (fresh == nullptr) throw std::bad_alloc();
        fresh->next.store(fresh, std::memory_order_relaxed);
        other.frontBlock.store(fresh, std::memory_order_relaxed);
        other.tailBlock .store(fresh, std::memory_order_relaxed);
    }

    SpscQueue& operator=(SpscQueue&& other) noexcept
    {
        Block* tmp = frontBlock.load(std::memory_order_relaxed);
        frontBlock.store(other.frontBlock.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        other.frontBlock.store(tmp, std::memory_order_relaxed);

        tmp = tailBlock.load(std::memory_order_relaxed);
        tailBlock.store(other.tailBlock.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
        other.tailBlock.store(tmp, std::memory_order_relaxed);

        std::swap(largestBlockSize, other.largestBlockSize);
        return *this;
    }

    ~SpscQueue()
    {
        // Make sure we see the final state of every variable across
        // the producer and consumer threads.
        std::atomic_thread_fence(std::memory_order_seq_cst);

        Block* startBlock = frontBlock.load(std::memory_order_relaxed);
        if (startBlock == nullptr) return;

        Block* block = startBlock;
        do
        {
            Block* nextBlock  = block->next.load(std::memory_order_relaxed);
            const std::size_t blockFront = block->front.load(std::memory_order_relaxed);
            const std::size_t blockTail  = block->tail .load(std::memory_order_relaxed);

            for (std::size_t i = blockFront; i != blockTail; i = (i + 1) & block->sizeMask)
            {
                T* element = reinterpret_cast<T*>(block->data + i * sizeof(T));
                element->~T();
            }

            char* rawAllocation = block->rawAllocation;
            block->~Block();
            std::free(rawAllocation);

            block = nextBlock;
        } while (block != startBlock);
    }

    // ------------------------------------------------------------- API

    // Try to enqueue without ever allocating.  Returns false when the
    // queue would need to grow.  Audio-thread producers should always
    // use this variant.
    bool try_enqueue(const T& element)
    {
        return innerEnqueue<NoAllocation>(element);
    }

    bool try_enqueue(T&& element)
    {
        return innerEnqueue<NoAllocation>(std::move(element));
    }

    template <typename... Args>
    bool try_emplace(Args&&... args)
    {
        return innerEnqueue<NoAllocation>(std::forward<Args>(args)...);
    }

    // Enqueue, growing the queue (i.e. allocating) if necessary.  Only
    // returns false if the underlying allocator fails.  NEVER call this
    // from a real-time thread.
    bool enqueue(const T& element)
    {
        return innerEnqueue<MayAllocate>(element);
    }

    bool enqueue(T&& element)
    {
        return innerEnqueue<MayAllocate>(std::move(element));
    }

    template <typename... Args>
    bool emplace(Args&&... args)
    {
        return innerEnqueue<MayAllocate>(std::forward<Args>(args)...);
    }

    // Try to move the front element into result.  Returns false if the
    // queue is empty.  Must be called only from the consumer thread.
    template <typename U>
    bool try_dequeue(U& result)
    {
        Block*       block      = frontBlock.load(std::memory_order_relaxed);
        std::size_t  blockTail  = block->localTail;
        std::size_t  blockFront = block->front.load(std::memory_order_relaxed);

        if (blockFront != blockTail
            || blockFront != (block->localTail = block->tail.load(std::memory_order_acquire)))
        {
            std::atomic_thread_fence(std::memory_order_acquire);

        nonEmptyFront:
            T* element = reinterpret_cast<T*>(block->data + blockFront * sizeof(T));
            result = std::move(*element);
            element->~T();

            blockFront = (blockFront + 1) & block->sizeMask;
            block->front.store(blockFront, std::memory_order_release);
            return true;
        }

        if (block != tailBlock.load(std::memory_order_acquire))
        {
            // Re-read the front block after the acquire on tailBlock so
            // we cannot miss writes the producer made before advancing
            // the outer tail pointer.
            std::atomic_thread_fence(std::memory_order_acquire);
            block      = frontBlock.load(std::memory_order_relaxed);
            blockTail  = block->localTail = block->tail.load(std::memory_order_acquire);
            blockFront = block->front.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            if (blockFront != blockTail) goto nonEmptyFront;

            // Front block is empty but the producer has moved on.
            Block* nextBlock = block->next.load(std::memory_order_relaxed);

            const std::size_t nextBlockFront = nextBlock->front.load(std::memory_order_relaxed);
            nextBlock->localTail = nextBlock->tail.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_acquire);

            // The producer only advances the outer tail after writing
            // at least one element, so the next block must be non-empty.
            assert(nextBlockFront != nextBlock->localTail);

            // Publish any pending updates to the (now drained) old front
            // block before handing it back to the producer.
            std::atomic_thread_fence(std::memory_order_release);
            frontBlock.store(nextBlock, std::memory_order_relaxed);
            std::atomic_signal_fence(std::memory_order_release);

            block = nextBlock;
            T* element = reinterpret_cast<T*>(block->data + nextBlockFront * sizeof(T));
            result = std::move(*element);
            element->~T();

            const std::size_t advanced = (nextBlockFront + 1) & block->sizeMask;
            block->front.store(advanced, std::memory_order_release);
            return true;
        }

        return false;
    }

    // Drop the front element on the floor.  Returns true on success.
    bool pop()
    {
        Block*       block      = frontBlock.load(std::memory_order_relaxed);
        std::size_t  blockTail  = block->localTail;
        std::size_t  blockFront = block->front.load(std::memory_order_relaxed);

        if (blockFront != blockTail
            || blockFront != (block->localTail = block->tail.load(std::memory_order_acquire)))
        {
            std::atomic_thread_fence(std::memory_order_acquire);

        nonEmptyFront:
            T* element = reinterpret_cast<T*>(block->data + blockFront * sizeof(T));
            element->~T();

            blockFront = (blockFront + 1) & block->sizeMask;
            block->front.store(blockFront, std::memory_order_release);
            return true;
        }

        if (block != tailBlock.load(std::memory_order_acquire))
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            block      = frontBlock.load(std::memory_order_relaxed);
            blockTail  = block->localTail = block->tail.load(std::memory_order_acquire);
            blockFront = block->front.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            if (blockFront != blockTail) goto nonEmptyFront;

            Block* nextBlock = block->next.load(std::memory_order_relaxed);
            const std::size_t nextBlockFront = nextBlock->front.load(std::memory_order_relaxed);
            nextBlock->localTail = nextBlock->tail.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_acquire);

            assert(nextBlockFront != nextBlock->localTail);

            std::atomic_thread_fence(std::memory_order_release);
            frontBlock.store(nextBlock, std::memory_order_relaxed);
            std::atomic_signal_fence(std::memory_order_release);

            block = nextBlock;
            T* element = reinterpret_cast<T*>(block->data + nextBlockFront * sizeof(T));
            element->~T();

            const std::size_t advanced = (nextBlockFront + 1) & block->sizeMask;
            block->front.store(advanced, std::memory_order_release);
            return true;
        }

        return false;
    }

    // Peek at the front element without consuming it.  Returns nullptr
    // when the queue is empty.  Consumer-thread only.
    T* peek() const
    {
        Block*       block      = frontBlock.load(std::memory_order_relaxed);
        std::size_t  blockTail  = block->localTail;
        std::size_t  blockFront = block->front.load(std::memory_order_relaxed);

        if (blockFront != blockTail
            || blockFront != (block->localTail = block->tail.load(std::memory_order_acquire)))
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            return reinterpret_cast<T*>(block->data + blockFront * sizeof(T));
        }

        if (block != tailBlock.load(std::memory_order_acquire))
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            block      = frontBlock.load(std::memory_order_relaxed);
            blockTail  = block->localTail = block->tail.load(std::memory_order_acquire);
            blockFront = block->front.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            if (blockFront != blockTail)
                return reinterpret_cast<T*>(block->data + blockFront * sizeof(T));

            Block* nextBlock = block->next.load(std::memory_order_relaxed);
            const std::size_t nextBlockFront = nextBlock->front.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            return reinterpret_cast<T*>(nextBlock->data + nextBlockFront * sizeof(T));
        }

        return nullptr;
    }

    // Approximate occupancy. May briefly disagree with the true count
    // by an amount bounded by what the other thread can produce or
    // consume between the loads.  Safe from either thread.
    std::size_t size_approx() const
    {
        std::size_t total = 0;
        Block* startBlock = frontBlock.load(std::memory_order_acquire);
        Block* block = startBlock;
        do
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            const std::size_t f = block->front.load(std::memory_order_relaxed);
            const std::size_t t = block->tail .load(std::memory_order_relaxed);
            total += (t - f) & block->sizeMask;
            block = block->next.load(std::memory_order_relaxed);
        } while (block != startBlock);
        return total;
    }

    // Total number of elements that can be held without allocating
    // when the queue is empty.
    std::size_t max_capacity() const
    {
        std::size_t total = 0;
        Block* startBlock = frontBlock.load(std::memory_order_acquire);
        Block* block = startBlock;
        do
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            total += block->sizeMask;
            block = block->next.load(std::memory_order_relaxed);
        } while (block != startBlock);
        return total;
    }

private:
    enum AllocationMode { MayAllocate, NoAllocation };

    template <AllocationMode Mode, typename... Args>
    bool innerEnqueue(Args&&... args)
    {
        Block*      block       = tailBlock.load(std::memory_order_relaxed);
        std::size_t blockFront  = block->localFront;
        std::size_t blockTail   = block->tail.load(std::memory_order_relaxed);

        std::size_t nextTail = (blockTail + 1) & block->sizeMask;
        if (nextTail != blockFront
            || nextTail != (block->localFront = block->front.load(std::memory_order_acquire)))
        {
            std::atomic_thread_fence(std::memory_order_acquire);

            // Tail block has room.
            char* slot = block->data + blockTail * sizeof(T);
            new (slot) T(std::forward<Args>(args)...);

            block->tail.store(nextTail, std::memory_order_release);
            return true;
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        Block* nextBlock = block->next.load(std::memory_order_acquire);
        if (nextBlock != frontBlock.load(std::memory_order_acquire))
        {
            // The next block in the ring is not the consumer's current
            // front block, so it must be empty (because we never skip
            // blocks).  Use it.
            std::atomic_thread_fence(std::memory_order_acquire);

            const std::size_t nextFront = nextBlock->front.load(std::memory_order_relaxed);
            const std::size_t nextTailL = nextBlock->tail .load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            assert(nextFront == nextTailL);
            (void)nextTailL;
            nextBlock->localFront = nextFront;

            char* slot = nextBlock->data + nextFront * sizeof(T);
            new (slot) T(std::forward<Args>(args)...);

            nextBlock->tail.store((nextFront + 1) & nextBlock->sizeMask,
                                  std::memory_order_release);

            std::atomic_thread_fence(std::memory_order_release);
            tailBlock.store(nextBlock, std::memory_order_relaxed);
            return true;
        }

        if constexpr (Mode == MayAllocate)
        {
            // Allocate a fresh block, splice it into the ring just
            // ahead of the current tail block, then publish.  Block
            // sizes grow up to MaxBlockSize.
            const std::size_t newSize =
                largestBlockSize >= MaxBlockSize ? largestBlockSize : largestBlockSize * 2;

            Block* newBlock = makeBlock(newSize);
            if (newBlock == nullptr) return false;
            largestBlockSize = newSize;

            new (newBlock->data) T(std::forward<Args>(args)...);
            assert(newBlock->front.load(std::memory_order_relaxed) == 0);
            newBlock->tail.store(1, std::memory_order_relaxed);
            newBlock->localTail = 1;

            newBlock->next.store(block->next.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
            block->next.store(newBlock, std::memory_order_relaxed);

            std::atomic_thread_fence(std::memory_order_release);
            tailBlock.store(newBlock, std::memory_order_relaxed);
            return true;
        }
        else
        {
            // Cannot allocate: the queue is genuinely full.
            return false;
        }
    }

    // ------------------------------------------------------------ Block

    struct Block
    {
        // Contended pair owned by the consumer thread.
        std::atomic<std::size_t> front{0};
        std::size_t              localTail{0};
        char pad0[SpscDetail::kCacheLineBytes
                  - sizeof(std::atomic<std::size_t>)
                  - sizeof(std::size_t)];

        // Contended pair owned by the producer thread.
        std::atomic<std::size_t> tail{0};
        std::size_t              localFront{0};
        char pad1[SpscDetail::kCacheLineBytes
                  - sizeof(std::atomic<std::size_t>)
                  - sizeof(std::size_t)];

        // Outer ring linkage; rarely contended but kept off the
        // tail's cache line just in case.
        std::atomic<Block*>      next{nullptr};

        char*                    data;
        char*                    rawAllocation;
        const std::size_t        sizeMask;

        Block(std::size_t size, char* rawPtr, char* dataPtr) noexcept
            : data(dataPtr), rawAllocation(rawPtr), sizeMask(size - 1) {}
    };

    static Block* makeBlock(std::size_t capacity)
    {
        // One slab for both the Block header and the element storage.
        // Padding is generous so we can independently align both pieces
        // with alignForwards<>().
        const std::size_t headerBytes  = sizeof(Block) + alignof(Block) - 1;
        const std::size_t elementBytes = sizeof(T) * capacity + alignof(T) - 1;
        const std::size_t totalBytes   = headerBytes + elementBytes;

        char* raw = static_cast<char*>(std::malloc(totalBytes));
        if (raw == nullptr) return nullptr;

        char* alignedHeader = SpscDetail::alignForwards<Block>(raw);
        char* alignedData   = SpscDetail::alignForwards<T>(alignedHeader + sizeof(Block));
        return new (alignedHeader) Block(capacity, raw, alignedData);
    }

    std::atomic<Block*> frontBlock{nullptr};   // owned by consumer
    char pad0[SpscDetail::kCacheLineBytes - sizeof(std::atomic<Block*>)];
    std::atomic<Block*> tailBlock{nullptr};    // owned by producer

    std::size_t largestBlockSize{MaxBlockSize};
};

} // namespace Chronos::Concurrency
