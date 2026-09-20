#include "attention_internal.cuh"

namespace slopfab::cuda::attention_detail {
// --- fused backend ----------------------------------------------------------
//
// The blocked path above is FlashAttention's algorithm with cuBLAS doing the two
// GEMMs, which means the score tile is written to HBM by the QK GEMM, read back
// by the softmax, and read a third time by the PV GEMM. At seq 37710 that round
// trip is ~12 bytes per score element and it is 86% of a denoising step. This
// kernel exists to delete it: S and P never leave shared memory.
//
// **Shape of the decomposition.** One block owns one query tile of `kBr` rows
// for one head; `kWarps` warps split it so that **each warp owns 16 whole rows
// and every column of them**. That is the load-bearing choice: a row's softmax
// never spans two warps, so the running max, the correction factor and the sum
// are warp-local and need no `__syncthreads` and no shared reduction. Only the
// K/V staging and the O rescale are block-wide.
//
// **Why bf16 fragments and not the fp16 the blocked path uses.** The fp16
// conversion of k and v exists only because cuBLAS will not take bf16 in and
// fp16 out; `mma.sync` takes bf16 directly with an fp32 accumulator, so the two
// full-sequence conversion buffers and their HBM traffic disappear. The
// probabilities stay fp16 -- P feeds a second mma whose operands must share a
// type, so V is converted to fp16 during SMEM staging, which costs no HBM
// traffic and keeps the 11 mantissa bits the blocked path deliberately chose
// over bf16's 8.
//
// **Accumulator in registers, via `mma.sync` rather than `wmma`.** O is
// rescaled by a *per-row* factor every key block, and `nvcuda::wmma` does not
// say which row an accumulator element belongs to -- the mapping is
// unspecified, so a register-resident O could not be scaled correctly through
// that API. Raw `mma.sync.aligned.m16n8k16` does specify it (PTX ISA 9.7.14),
// and the specification is what makes this kernel possible:
//
//     groupID = lane >> 2      tig = lane & 3
//     c0,c1 -> row groupID      cols tig*2, tig*2+1
//     c2,c3 -> row groupID + 8  cols tig*2, tig*2+1
//
// Three consequences, and each one deletes a shared-memory buffer:
//
//   * A thread's four accumulator registers span exactly **two** rows, so the
//     per-row rescale is four multiplies on registers -- not an 8192-element
//     read-modify-write over a shared tile.
//   * The four threads of a group hold all eight columns of one row, and they
//     are consecutive lanes, so a row reduction is `shfl_xor` by 1 then 2.
//     Rows `groupID` and `groupID+8` reduce simultaneously in separate
//     registers, and all 32 lanes work rather than 16.
//   * The QK accumulator tiles at n-offset 0 and 8 supply **exactly** the four
//     A-fragment registers the PV `mma` wants. S becomes P by a register
//     permute and an `f32 -> f16x2` convert, with no transpose and no store.
//
// So `st`, `ps`, `os` and `cs` are all gone, along with the fp32 shared tiles
// whose 2-way bank conflict was unfixable: `wmma` requires an fp32 `ldm` that
// is a multiple of 4 floats, and for any stride divisible by 4, rows 8 apart
// land in the same bank. Only K and V staging remain in shared memory.
//
// **V is staged transposed.** The PV `mma` wants B in column-major, and V
// arrives row-major. Writing V^T during staging turns each B fragment load
// into one aligned 32-bit read instead of two scattered 16-bit reads. Its row
// stride is `kBc + 2` halves = 17 four-byte words, and 17 is coprime with 32,
// so the transposing writes are conflict-free.
namespace fused {

// Eight warps, not four. Four warps is exactly one warp per sub-partition, so
// every shared-load and MUFU latency in the inner loop is fully exposed -- the
// kernel was ~14% of the tensor-pipe ceiling with nothing co-resident to cover
// it. Occupancy *percentage* is the wrong metric here (a well-tuned FA2 kernel
// runs at 12.5% and still reaches 70% of peak); what matters is independent
// instruction streams per scheduler, and eight warps gives two.
//
// kBc doubles with it. Everything proportional to kBr*D -- the rescale, the
// epilogue -- is paid once per key block, so its cost per FLOP falls as 1/kBc,
// and kBr=128 halves how many times K and V are re-read per head.
//
// --- REGISTER BUDGET: four instantiations now, in different regimes ---------
//
// Measured on this file with `-Xptxas -v`, sm_120a, at the commit that added
// this comment. Re-measure rather than trusting the numbers; the *ceiling* is
// what does not move.
//
//   D = 128, kBanded=false : 173 registers -> 1 block/SM
//   D = 128, kBanded=true  : 188 registers -> 1 block/SM
//   D =  64, kBanded=false : 125 registers -> 2 blocks/SM, three of margin
//   D =  64, kBanded=true  : 128 registers -> 2 blocks/SM, **zero margin**
//
// **`D=64, kBanded=true` sits exactly on the ceiling.** 128 * 256 threads * 2
// blocks is 65536, precisely the register file, so it gets its second block
// with nothing to spare -- any change to the banded path at that head dim
// breaks 2 blocks/SM on the first register it spends. That is a far tighter
// constraint than the unbanded path's three, and it is not what the next
// person will expect from a line that reads "128, fine".
//
// The unbanded numbers are the ones to defend hardest: banding is off by
// default, so `kBanded=false` is what production runs. It is instruction-for-
// instruction identical to the pre-banding kernel -- 1792 SASS opcodes at
// D=128, 1280 at D=64, zero differences in the stream -- and that is the
// property the template exists to hold, not a coincidence to be preserved by
// luck. The detail below on what it took to get there is not history for its
// own sake; it is the failure mode.
//
//   D = 128 : 173 registers, 0 spills, 34304 B smem -> 1 block/SM.
//             No cliff to fall off, but note *why* has changed: deleting the
//             shared Q tile took smem from 69120 to 34304 B, so two blocks now
//             fit in shared memory (68608 <= 102400) and are stopped by
//             registers alone (173 * 256 * 2 = 88576 > 65536). It reads like
//             an occupancy win and is not one. `o[kOTiles][4]` alone is 64
//             registers of irreducible accumulator, so 2 blocks/SM here would
//             need <=128 and is unreachable. Instruction count is still the
//             only currency; spending a register costs nothing.
//
//   D = 64  : 125 registers, 0 spills -> 2 blocks/SM, with **three registers
//             of margin**. The ceiling is 128 and it is exact: 128 * 256
//             threads is precisely half the 65536-register file. At 129 this
//             path drops to 1 block/SM -- a 2x occupancy loss.
//
// The trap: **no test in this suite can see that happen.** The tests check
// numbers, and this failure only moves the clock. A change that is bit-exact,
// digest-identical and obviously correct can still halve D=64's occupancy, and
// the only thing that reports it is `-Xptxas -v`. So if you touch this kernel,
// read the register count for **all four** instantiations, not just D=128 --
// the one with headroom is the one people quote, and the template doubled how
// many ways there are to miss.
//
// Deliberately not a `__launch_bounds__` / `minBlocksPerMultiprocessor` cap --
// and this has now been measured rather than assumed.
// `__launch_bounds__(kThreads, D == 64 ? 2 : 1)` *does* hold D=64 to 128
// registers and keep 2 blocks/SM. It pays 8 B stack frame, 12 B spill stores
// and 8 B spill loads **in the inner loop**, on the very path it rescues, and
// pushes D=128 to 202 registers (harmless -- no cliff there). So the trade is
// 2x occupancy against spills in the hot loop, which settles on a clock and
// not on an argument. Measured, understood, not applied: nothing in this port
// takes D=64, so the clock that would decide it has never been worth running.
//
// **These counts are branch-local, and this comment merges silently.** They
// were measured on the commit that wrote them. A merge will not conflict on
// them and will not update them, so this block can arrive in a tree where it
// is false -- which has already happened once, cleanly, with no diff to
// review. Therefore: **whoever merges a change to this file re-measures all
// four instantiations and edits these numbers in the same commit.** Nothing
// checks this. It is not optional and it is not the author's job, it is the
// merger's. This paragraph has now been honoured twice and corrected once --
// the "both" it used to say predated the template, which is the same staleness
// it exists to prevent, in the sentence that prevents it.
//
// Three measurements from the campaign that established the rule, all of which
// would have fooled a careful person:
//
//   * Two branches each started from D=64 = 125. One spent 3 registers, the
//     other 2, each correctly under the ceiling on its own branch. This was
//     first written as a prediction that the merge would land at 129-130 by
//     adding the two spends. **That prediction was wrong and understated its
//     own case: measured after the merge, D=64 came out at 132.** ptxas
//     reallocates across the whole function body rather than summing
//     per-branch costs, so a merge can overshoot what its parents spent.
//     Per-branch numbers are not additive and therefore not predictive:
//     **only a measurement on the integration branch, after the second change
//     lands, tells you anything.**
//   * A variant that derived `vt` from `ks` instead of tracking it **saved a
//     register at D=128 and cost one at D=64** -- ptxas rescheduled and the
//     trade inverted. D=128 is 1 block/SM at any count in this range, so the
//     saving bought nothing while the cost was the entire margin. Optimising
//     on the D=128 number alone would have shipped at the ceiling and called
//     it an improvement.
//   * **A flag charged the path that does not use it.** Frame banding is off by
//     default, so its cost should have landed entirely on callers who ask for
//     it. Three intermediates said otherwise, each producing correct output and
//     passing every test: the natural nested range-then-block loop took D=64
//     from 125 to **139**, through the ceiling to 1 block/SM, because the outer
//     loop kept the inner one's state live across it; flattening it fixed D=64
//     and pushed D=128 to **199**; templating on `kBanded` fixed the banded
//     path but left unbanded at **187/128**, because the loop's *shape* had
//     changed and a member had been added to `KvStage` for every caller,
//     including the ones that never call it. Only `advance(delta)` in place of
//     a stateful `seek`, and a `while` whose unbanded form is literally the
//     original `k0 < seq`, returned unbanded to 173/125.
//
//     The lesson is narrower than "check both instantiations": **an off-by-
//     default feature is not free merely because it is guarded.** Its cost
//     hides in shared state and in loop structure, neither of which the guard
//     covers, and the only proof is a SASS comparison against the kernel before
//     the flag existed.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarp;
constexpr int kBr = 16 * kWarps;  // query rows per block, 16 per warp
constexpr int kBc = 64;           // key rows per step
constexpr int kMmaN = 8;          // n extent of one mma tile

// 16-bit tiles only, so the bank argument is the honest one: at stride 136
// halves the QK operand addresses reduce to `groupID*4 + tig`, which is a
// permutation of 0..31. kPadV = 2 makes V^T's stride 17 words, coprime with 32.
constexpr int kPadH = 8;
constexpr int kPadV = 2;

// C = A*B + C on one m16n8k16 tile. Separate bf16 and f16 forms because the QK
// product consumes bf16 straight from the checkpoint while the PV product
// consumes fp16 probabilities -- the 11 mantissa bits the blocked path
// deliberately chose over bf16's 8.
__device__ inline void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline uint32_t ld32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }

// --- K/V staging ------------------------------------------------------------
//
// Staging one key block moves kBc*D halves of K and the same of V from global
// into shared. It sits between two `__syncthreads`, so no `mma` can issue while
// any of it is in flight, and that made it the kernel's largest single cost.
//
// **Nothing about a thread's share of it depends on the key block.** Its row,
// its column, both shared destinations and both global sources are fixed by the
// thread index; from one key block to the next the sources advance by exactly
// `kBc * kvld` and the destinations do not move at all. The previous version
// derived all of it from a flat index on every element -- `r = i / D`,
// `c = i % D`, then a 64-bit `row*kvld + kv_head*D + c`, reloading the kernel
// parameters from the constant bank as it went -- and cost **172 SASS
// instructions per trip over 8 trips, 1376 per warp per key block**, of which
// 128 were the loads and 128 the stores. `KvStage` hoists the lot: the staging
// call inside the `k0` loop does not reference `k0`.
//
// **Each thread takes eight contiguous columns**, one 16-byte access. K is
// staged row-major, so that is one `LDG.128` and one `STS.128`. V is staged
// transposed on purpose (see above), so its eight halves land in eight
// different rows of `vt` and its stores stay scalar `STS.U16`; only its load
// widens.
//
// **The lane -> row map is permuted, and that is load-bearing.** A warp covers
// 8 rows x 32 columns as (row rr, 16-byte column cc). K's 16-byte store starts
// in bank `4*rr + 4*cc + 16*cg (mod 32)`, and a 128-bit store is issued in
// phases of eight lanes whose starting banks must be eight distinct multiples
// of four. The natural `rr = lane/4` gives a phase `rr + cc` of
// 0,1,2,3,1,2,3,4 -- three collisions, a 2-way conflict on every K store.
// `rr = lane/8 + 4*((lane/4) & 1)` gives it {0,4} x {0,1,2,3}, which is
// 0,4,8,...,28. V's scatter is indifferent to the permutation: `kVStride` is 66
// halves, so 33 words, so its bank reduces to `col + e + row/2 (mod 32)` and a
// warp covers 16 distinct words with the two lanes of each pair writing the two
// halves of one word.
constexpr int kVec = 8;                          // halves in a 16-byte access
constexpr int kStageCols = 4;                    // 16-byte columns one warp covers
constexpr int kStageRows = kWarp / kStageCols;   // rows one warp covers

// Hoisted staging state for one thread. Constructed once; `run` stages the
// current key block and advances to the next, so the loop body carries no
// addressing of its own. Keeping it a value rather than inline code is what
// lets a second buffer be staged ahead of the one being consumed without
// touching any of the arithmetic.
template <int D>
struct KvStage {
  static constexpr int kColGroups = D / (kVec * kStageCols);          // 4 at D=128, 2 at D=64
  static constexpr int kRowsPerPass = kWarps / kColGroups * kStageRows;
  static constexpr int kPasses = kBc / kRowsPerPass;
  static constexpr int kKStride = D + kPadH;
  static constexpr int kVStride = kBc + kPadV;

  static_assert(kStageCols == 4 && kStageRows == 8, "the bank argument above assumes 4x8 warps");
  static_assert(D % (kVec * kStageCols) == 0, "D must tile into whole 16-byte column groups");
  static_assert(kWarps % kColGroups == 0, "warps must split evenly over the column groups");
  static_assert(kPasses * kRowsPerPass == kBc, "the passes must tile the key block exactly");
  // 16-byte shared stores need 16-byte-aligned rows. `ks` itself is at offset 0
  // of the dynamic allocation, which `__align__(16)` pins, so only the row
  // stride is left to check.
  static_assert((kKStride * 2) % 16 == 0, "K rows must start on a 16-byte boundary");

  const __nv_bfloat16* kp;  // this thread's K source for pass 0 of the current key block
  const __nv_bfloat16* vp;
  size_t pass_stride;       // halves between one pass and the next
  int ks_off;               // halves into ks
  int vt_off;               // halves into vt
  int row;                  // absolute key row of pass 0

  __device__ KvStage(const __nv_bfloat16* k, const __nv_bfloat16* v, int tid, int kv_head,
                     size_t kvld) {
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int cc = lane & (kStageCols - 1);
    const int rr = (lane >> 3) + 4 * ((lane >> 2) & 1);
    const int cg = warp % kColGroups;
    const int rg = warp / kColGroups;
    const int col = (cg * kStageCols + cc) * kVec;
    row = rg * kStageRows + rr;
    ks_off = row * kKStride + col;
    vt_off = col * kVStride + row;
    const size_t off = static_cast<size_t>(row) * kvld + static_cast<size_t>(kv_head) * D + col;
    kp = k + off;
    vp = v + off;
    pass_stride = static_cast<size_t>(kRowsPerPass) * kvld;
  }

  // Skip forward by `delta` key rows. Frame banding gives a query tile two
  // disjoint key ranges -- the text/audio prefix and its own band -- so the
  // staging has to move between them without re-deriving the per-thread
  // addressing that F1 exists to hoist.
  //
  // The caller passes the delta rather than a destination, deliberately: it
  // already knows both ends, and remembering this thread's own row offset here
  // would cost a live register on the unbanded path too, which never calls this
  // at all.
  __device__ void advance(int delta, size_t kvld) {
    const ptrdiff_t step = static_cast<ptrdiff_t>(delta) * static_cast<ptrdiff_t>(kvld);
    kp += step;
    vp += step;
    row += delta;
  }

  // Rows past the end of the sequence are zeroed exactly as before -- the
  // softmax excludes dead *columns* itself and must keep seeing zeros here, not
  // stale shared memory. Barriers are the caller's.
  __device__ void run(__nv_bfloat16* ks, __half* vt, int seq) {
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      uint4 kw = {0u, 0u, 0u, 0u};
      uint4 vw = {0u, 0u, 0u, 0u};
      if (row < seq) {
        kw = *reinterpret_cast<const uint4*>(kp);
        vw = *reinterpret_cast<const uint4*>(vp);
      }
      *reinterpret_cast<uint4*>(ks + ks_off + p * kRowsPerPass * kKStride) = kw;
      const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(&vw);
      __half* dst = vt + vt_off + p * kRowsPerPass;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        dst[e * kVStride] = __float2half(__bfloat162float(src[e]));
      }
      kp += pass_stride;
      vp += pass_stride;
      row += kRowsPerPass;
    }
  }
};

__device__ inline uint32_t pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Q is not here. It is read straight from global into the A-fragments at block
// entry and never staged -- see the load in `fused_kernel`.
template <int D>
__host__ __device__ inline size_t smem_bytes() {
  size_t n = 0;
  n += sizeof(__nv_bfloat16) * kBc * (D + kPadH);  // K
  n += sizeof(__half) * D * (kBc + kPadV);         // V transposed
  return n;
}

template <int D, bool kBanded>
__global__ __launch_bounds__(kThreads) void fused_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int seq, int heads,
    int num_kv_heads, float scale, const int4* __restrict__ band, int query_rows) {
  constexpr int kKStride = D + kPadH;
  constexpr int kVStride = kBc + kPadV;
  constexpr int kDSteps = D / 16;      // k-steps of the QK product
  constexpr int kSTiles = kBc / kMmaN; // n-tiles of S
  constexpr int kOTiles = D / kMmaN;   // n-tiles of O
  constexpr int kPSteps = kBc / 16;    // k-steps of the PV product

  // 16-byte aligned because the K stage stores `uint4`. The dynamic allocation
  // is suitably aligned already; saying so keeps it true if the declaration
  // ever moves.
  extern __shared__ __align__(16) char raw_smem[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw_smem);
  __half* vt = reinterpret_cast<__half*>(ks + kBc * kKStride);

  const int q0 = blockIdx.x * kBr;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int gid = lane >> 2;   // 0..7, selects the row pair
  const int tig = lane & 3;    // 0..3, selects the column pair

  const int group = heads / num_kv_heads;
  const int kv_head = head / group;
  const size_t qld = static_cast<size_t>(heads) * D;
  const size_t kvld = static_cast<size_t>(num_kv_heads) * D;

  // The two output rows this thread owns, block-relative.
  const int row_a = warp * 16 + gid;
  const int row_b = row_a + 8;

  // Q A-fragments, straight from global, loaded once and reused across every
  // key block.
  //
  // Q used to be staged through a kBr x D shared tile that existed for exactly
  // this one read: 34816 B alive for the kernel's whole lifetime, written once,
  // read once, dead from here on. The `mma` A-fragment layout says which two
  // elements of which row each lane wants, so the tile was only ever a
  // transpose-free reshuffle of bytes the thread could address itself --
  // 32 four-byte loads per thread, once per block, amortised over 590 key
  // blocks against a barrier and a kBr*D staging loop it also removes.
  //
  // Deleting it is the precondition for double-buffering K/V rather than a
  // saving in its own right: two K/V buffers are 68608 B, and
  // 68608 + 34816 = 103424 exceeds the 102400 B a block can hold. With Q gone
  // the pair fits with room to spare.
  //
  // Rows past the end of the sequence read zero, exactly as the zero-filled
  // shared tile gave them. They are discarded by the guard in the epilogue,
  // never by the arithmetic.
  const bool q_live_a = q0 + row_a < query_rows;
  const bool q_live_b = q0 + row_b < query_rows;
  const __nv_bfloat16* qsrc_a =
      q + static_cast<size_t>(q0 + row_a) * qld + head * D + tig * 2;
  const __nv_bfloat16* qsrc_b =
      q + static_cast<size_t>(q0 + row_b) * qld + head * D + tig * 2;
  uint32_t qa[kDSteps][4];
#pragma unroll
  for (int t = 0; t < kDSteps; ++t) {
    qa[t][0] = q_live_a ? ld32(qsrc_a + t * 16) : 0u;
    qa[t][1] = q_live_b ? ld32(qsrc_b + t * 16) : 0u;
    qa[t][2] = q_live_a ? ld32(qsrc_a + t * 16 + 8) : 0u;
    qa[t][3] = q_live_b ? ld32(qsrc_b + t * 16 + 8) : 0u;
  }

  float o[kOTiles][4];
#pragma unroll
  for (int j = 0; j < kOTiles; ++j) {
    o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
  }
  float m_a = kHostNegInf, m_b = kHostNegInf;
  float l_a = 0.0f, l_b = 0.0f;

  // Every address the staging needs, computed once. See KvStage.
  KvStage<D> stage(k, v, tid, kv_head, kvld);

  // Which keys this query tile may see. Without a band that is the whole
  // sequence in one range; with one it is the text/audio prefix plus this
  // tile's frame band, which are disjoint and in order. Both bounds are already
  // key-block aligned by the host, so the loop below is unchanged in shape --
  // banding moves the *bound*, not the body.
  //
  // Nothing else needs to know. The online softmax is a running max and sum
  // over whatever columns it is shown, so a subset is not a special case, and
  // it already excludes columns past `seq` rather than zeroing keys.
  // Flattened to one loop over key *blocks* rather than nested range-then-block
  // loops. The nested form cost 14 registers at D=64 -- 125 to 139, straight
  // through the 128 ceiling and down to 1 block/SM -- because the outer loop
  // kept the inner one's state live across it.
  //
  // `kBanded` is a template parameter and not a runtime test, because banding is
  // off by default and the default path must stay the kernel it already was.
  // With it false every line below compiles away and the loop is the original
  // walk from 0 to seq; the register counts say so -- 173/125 unbanded against
  // 199/128 banded, i.e. the cost lands only on the path that opted in.
  int k0 = 0;
  int k_stop = seq;
  int seam_lo = 0, seam_stop = 0;
  if constexpr (kBanded) {
    const int4 r = band[blockIdx.x];
    k0 = r.x;
    k_stop = r.y;
    seam_lo = r.z;
    seam_stop = r.w;
    stage.advance(k0, kvld);  // from row 0 to the first range
  }

  while (true) {
    if (k0 >= k_stop) {
      if constexpr (kBanded) {
        if (seam_stop <= seam_lo) break;
        stage.advance(seam_lo - k0, kvld);
        k0 = seam_lo;
        k_stop = seam_stop;
        seam_lo = seam_stop = 0;
      } else {
        break;
      }
    }
    __syncthreads();  // last iteration's mma has finished reading ks/vt
    stage.run(ks, vt, seq);
    __syncthreads();

    // S = Q K^T. K is row-major in shared memory and the B operand is
    // column-major, which is exactly K^T -- the same free transpose the
    // blocked path gets from cuBLAS.
    float s[kSTiles][4];
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
      for (int t = 0; t < kDSteps; ++t) {
        const __nv_bfloat16* kb = ks + (j * kMmaN + gid) * kKStride + t * 16 + tig * 2;
        uint32_t b[2] = {ld32(kb), ld32(kb + 8)};
        mma_bf16(s[j], qa[t], b);
      }
    }

    // Online softmax, entirely in registers. Columns past the end of the
    // sequence are excluded here rather than by zeroing k: a zero key scores 0,
    // whose exp is 1, which would inflate the denominator.
    const int col0 = k0 + tig * 2;
    float ma = kHostNegInf, mb = kHostNegInf;
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      const int c = col0 + j * kMmaN;
      if (c < seq) {
        ma = fmaxf(ma, s[j][0] * scale);
        mb = fmaxf(mb, s[j][2] * scale);
      }
      if (c + 1 < seq) {
        ma = fmaxf(ma, s[j][1] * scale);
        mb = fmaxf(mb, s[j][3] * scale);
      }
    }
    // The four lanes of a group hold all eight columns of a row and are
    // consecutive, so xor by 1 then 2 reduces both rows at once.
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, off));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, off));
    }

    const float new_a = fmaxf(m_a, ma);
    const float new_b = fmaxf(m_b, mb);
    const float c_a = (m_a == kHostNegInf || new_a == kHostNegInf) ? 0.0f : __expf(m_a - new_a);
    const float c_b = (m_b == kHostNegInf || new_b == kHostNegInf) ? 0.0f : __expf(m_b - new_b);

    float sum_a = 0.0f, sum_b = 0.0f;
#pragma unroll
    for (int j = 0; j < kSTiles; ++j) {
      const int c = col0 + j * kMmaN;
      const bool l0 = c < seq && new_a != kHostNegInf;
      const bool l1 = c + 1 < seq && new_a != kHostNegInf;
      const bool r0 = c < seq && new_b != kHostNegInf;
      const bool r1 = c + 1 < seq && new_b != kHostNegInf;
      s[j][0] = l0 ? __expf(s[j][0] * scale - new_a) : 0.0f;
      s[j][1] = l1 ? __expf(s[j][1] * scale - new_a) : 0.0f;
      s[j][2] = r0 ? __expf(s[j][2] * scale - new_b) : 0.0f;
      s[j][3] = r1 ? __expf(s[j][3] * scale - new_b) : 0.0f;
      sum_a += s[j][0] + s[j][1];
      sum_b += s[j][2] + s[j][3];
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      sum_a += __shfl_xor_sync(0xffffffffu, sum_a, off);
      sum_b += __shfl_xor_sync(0xffffffffu, sum_b, off);
    }
    l_a = l_a * c_a + sum_a;
    l_b = l_b * c_b + sum_b;
    m_a = new_a;
    m_b = new_b;

    // Four multiplies, because c0/c1 and c2/c3 are one row each.
#pragma unroll
    for (int j = 0; j < kOTiles; ++j) {
      o[j][0] *= c_a;
      o[j][1] *= c_a;
      o[j][2] *= c_b;
      o[j][3] *= c_b;
    }

    // S -> P with no data movement: the n-offset 0 and 8 tiles are precisely
    // the A-fragment's four registers.
    uint32_t pa[kPSteps][4];
#pragma unroll
    for (int t = 0; t < kPSteps; ++t) {
      const int j0 = t * 2;
      pa[t][0] = pack_h2(s[j0][0], s[j0][1]);
      pa[t][1] = pack_h2(s[j0][2], s[j0][3]);
      pa[t][2] = pack_h2(s[j0 + 1][0], s[j0 + 1][1]);
      pa[t][3] = pack_h2(s[j0 + 1][2], s[j0 + 1][3]);
    }

#pragma unroll
    for (int j = 0; j < kOTiles; ++j) {
#pragma unroll
      for (int t = 0; t < kPSteps; ++t) {
        const __half* vb = vt + (j * kMmaN + gid) * kVStride + t * 16 + tig * 2;
        uint32_t b[2] = {ld32(vb), ld32(vb + 8)};
        mma_f16(o[j], pa[t], b);
      }
    }
    k0 += kBc;
  }

  const float inv_a = l_a > 0.0f ? 1.0f / l_a : 0.0f;
  const float inv_b = l_b > 0.0f ? 1.0f / l_b : 0.0f;
  const int out_a = q0 + row_a;
  const int out_b = q0 + row_b;
  __nv_bfloat16* base = out + head * D;
#pragma unroll
  for (int j = 0; j < kOTiles; ++j) {
    const int c = j * kMmaN + tig * 2;
    if (out_a < query_rows) {
      base[static_cast<size_t>(out_a) * qld + c] = __float2bfloat16(o[j][0] * inv_a);
      base[static_cast<size_t>(out_a) * qld + c + 1] = __float2bfloat16(o[j][1] * inv_a);
    }
    if (out_b < query_rows) {
      base[static_cast<size_t>(out_b) * qld + c] = __float2bfloat16(o[j][2] * inv_b);
      base[static_cast<size_t>(out_b) * qld + c + 1] = __float2bfloat16(o[j][3] * inv_b);
    }
  }
}

// >48 KB of shared memory per block is opt-in and the opt-in is per function.
// Doing it once per (function, thread) rather than per launch keeps it off the
// hot path; the call is idempotent so a race is harmless.
template <int D>
void ensure_smem_optin() {
  static thread_local bool done = false;
  if (done) return;
  SLOPFAB_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<D, false>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes<D>())));
  SLOPFAB_CUDA_CHECK(cudaFuncSetAttribute(fused_kernel<D, true>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes<D>())));
  done = true;
}

template <int D>
void launch(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
            const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
            int num_kv_heads, int query_rows) {
  ensure_smem_optin<D>();
  const dim3 grid(static_cast<unsigned>((query_rows + kBr - 1) / kBr),
                  static_cast<unsigned>(cfg.num_heads));
  // Grid order matters and is load-bearing: x is the query tile and y is the
  // head, and CUDA dispatches x fastest, so the resident blocks share a head
  // and walk one K/V stream together. That stream is 19.3 MB at production
  // shape, which lives in L2. Swapping the dimensions would turn L2 hits into
  // DRAM traffic.
  const int4* band = reinterpret_cast<const int4*>(cfg.band_ranges);
  if (band != nullptr) {
    fused_kernel<D, true><<<grid, kThreads, smem_bytes<D>(), stream>>>(
        q, k, v, out, cfg.seq_len, cfg.num_heads, num_kv_heads, cfg.effective_scale(), band, query_rows);
  } else {
    fused_kernel<D, false><<<grid, kThreads, smem_bytes<D>(), stream>>>(
        q, k, v, out, cfg.seq_len, cfg.num_heads, num_kv_heads, cfg.effective_scale(), nullptr, query_rows);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

bool supported(const AttentionConfig& cfg) { return cfg.head_dim == 64 || cfg.head_dim == 128; }

}  // namespace fused

void run_fused(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
               const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
               int num_kv_heads, int query_rows) {
  check_config(cfg, num_kv_heads);
  if (!fused::supported(cfg)) {
    throw std::runtime_error("attention: kFused supports head_dim 64 or 128, got " +
                             std::to_string(cfg.head_dim));
  }
  // K and V are staged with 16-byte vector loads, and Q's A-fragments are read
  // from global as 4-byte pairs. Every element offset the kernel forms is a
  // multiple of eight halves for K/V and of two for Q -- head_dim is 64 or 128
  // and each thread starts on an eight-column boundary -- so only the base
  // pointers can break it, and every allocator here returns at least 256 bytes
  // of alignment. Check rather than fault: a misaligned address is a kernel
  // abort with no indication of which pointer was wrong.
  const uintptr_t bases = reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                          reinterpret_cast<uintptr_t>(v);
  if ((bases & 15u) != 0) {
    throw std::runtime_error("attention: kFused needs 16-byte aligned q, k and v");
  }
  if (cfg.head_dim == 64) {
    fused::launch<64>(stream, q, k, v, out, cfg, num_kv_heads,
                      query_rows > 0 ? query_rows : cfg.seq_len);
  } else {
    fused::launch<128>(stream, q, k, v, out, cfg, num_kv_heads,
                       query_rows > 0 ? query_rows : cfg.seq_len);
  }
}

}  // namespace slopfab::cuda::attention_detail

namespace slopfab::cuda {
int attention_fused_query_tile() { return attention_detail::fused::kBr; }
int attention_fused_key_align() { return attention_detail::fused::kBc; }
}  // namespace slopfab::cuda
