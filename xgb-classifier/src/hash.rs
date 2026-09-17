//! FNV-1a, 64-bit.
//!
//! The one hash both sides of the build have to agree on: `fnv1a64` in
//! `scripts/build_model.py` keys the vocabulary and the two target encodings
//! with it, and [`crate::blob`] looks them up with this. Changing either
//! without the other silently turns every lookup into a miss, which shows up
//! as the estimator answering from `GLOBAL_MEDIAN` and an empty embedding.
//!
//! Unfolded, unlike `llm-classifier`'s 32-bit variant: 50,000 vocabulary terms
//! would expect a collision at 32 bits, and a collision here maps one term's
//! tokens onto another term's basis row. At 64 bits it is a ~1e-10 event, and
//! `emit_blob` asserts there is none in the vocabulary it ships.

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = OFFSET;
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    hash
}
