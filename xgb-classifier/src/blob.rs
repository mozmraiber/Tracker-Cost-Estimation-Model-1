//! Reader for the `model.bin` payload `scripts/build_model.py` writes.
//!
//! The payload carries the four tables the feature vector needs: the TF-IDF
//! vocabulary with its inverse document frequencies, the truncated-SVD basis,
//! and the two target encodings `engineer_features` derives from the training
//! log. Together they are 10.8 MB, of which the basis is 10.0 -- 50,000
//! vocabulary terms by 50 components, `f32` as scikit-learn stored them.
//!
//! It is a blob and not generated Rust because the basis as an array literal
//! is ~120 MB of source, which rustc will not finish. `include_bytes!` puts
//! the same bytes in `.rodata` directly. The cost is that the bytes carry no
//! alignment guarantee, so every scalar is read through `from_le_bytes` rather
//! than transmuted -- a 4-byte copy per component, against the ~2,000
//! component reads a request makes, which is not where the time goes.
//!
//! Section offsets are `const` expressions over the shapes in
//! [`crate::generated`], so a payload that does not match the shapes it was
//! generated with fails the length assertion below at compile time rather than
//! reading the wrong table.

use crate::generated::{N_COMPONENTS, N_DOMAINS, N_DOMAIN_TYPES, N_TERMS};

static BLOB: &[u8] = include_bytes!("model.bin");

const U64: usize = 8;
const U32: usize = 4;
const F32: usize = 4;

/// Section starts, in the order `emit_blob` writes them.
const MAGIC: usize = 0;
const TERM_HASH: usize = MAGIC + 8;
const TERM_INDEX: usize = TERM_HASH + N_TERMS * U64;
const IDF: usize = TERM_INDEX + N_TERMS * U32;
const COMPONENTS: usize = IDF + N_TERMS * F32;
const DOMAIN_HASH: usize = COMPONENTS + N_TERMS * N_COMPONENTS * F32;
const DOMAIN_VALUE: usize = DOMAIN_HASH + N_DOMAINS * U64;
const DOMAIN_TYPE_HASH: usize = DOMAIN_VALUE + N_DOMAINS * F32;
const DOMAIN_TYPE_VALUE: usize = DOMAIN_TYPE_HASH + N_DOMAIN_TYPES * U64;
const END: usize = DOMAIN_TYPE_VALUE + N_DOMAIN_TYPES * F32;

const fn magic_matches() -> bool {
    let want = *b"XGBCLS01";
    let mut i = 0;
    while i < want.len() {
        if BLOB[MAGIC + i] != want[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = assert!(
    magic_matches(),
    "src/model.bin is not a payload scripts/build_model.py wrote"
);
const _: () = assert!(
    BLOB.len() == END,
    "src/model.bin does not match the shapes in src/generated.rs; \
     regenerate both with scripts/build_model.py"
);

fn u64_at(offset: usize) -> u64 {
    u64::from_le_bytes(BLOB[offset..offset + U64].try_into().unwrap())
}

fn u32_at(offset: usize) -> u32 {
    u32::from_le_bytes(BLOB[offset..offset + U32].try_into().unwrap())
}

fn f32_at(offset: usize) -> f32 {
    f32::from_le_bytes(BLOB[offset..offset + F32].try_into().unwrap())
}

/// Index of `hash` in a sorted run of `len` `u64` keys starting at `start`.
fn search(start: usize, len: usize, hash: u64) -> Option<usize> {
    let (mut low, mut high) = (0, len);
    while low < high {
        let mid = low + (high - low) / 2;
        if u64_at(start + mid * U64) < hash {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    match low < len && u64_at(start + low * U64) == hash {
        true => Some(low),
        false => None,
    }
}

/// Vocabulary index of the term hashing to `hash`, or `None` when the shipped
/// TF-IDF did not keep that term.
///
/// The returned index is the term's position in the *vocabulary*, not in the
/// hash-sorted key array, because [`idf`] and [`components`] are stored in
/// vocabulary order -- see `emit_blob` for why that order is worth keeping.
pub fn term(hash: u64) -> Option<usize> {
    search(TERM_HASH, N_TERMS, hash).map(|i| u32_at(TERM_INDEX + i * U32) as usize)
}

/// Inverse document frequency of the term at vocabulary index `term`.
pub fn idf(term: usize) -> f32 {
    f32_at(IDF + term * F32)
}

/// The term's row of the SVD basis: one weight per output component.
pub fn components(term: usize) -> impl Iterator<Item = f32> {
    let start = COMPONENTS + term * N_COMPONENTS * F32;
    BLOB[start..start + N_COMPONENTS * F32]
        .chunks_exact(F32)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
}

/// Median `transfer_bytes` of the tracker domain hashing to `hash`.
pub fn domain_median(hash: u64) -> Option<f64> {
    search(DOMAIN_HASH, N_DOMAINS, hash).map(|i| f64::from(f32_at(DOMAIN_VALUE + i * F32)))
}

/// Median `transfer_bytes` of the (domain, resource type) pair hashing to
/// `hash`.
pub fn domain_type_median(hash: u64) -> Option<f64> {
    search(DOMAIN_TYPE_HASH, N_DOMAIN_TYPES, hash)
        .map(|i| f64::from(f32_at(DOMAIN_TYPE_VALUE + i * F32)))
}
