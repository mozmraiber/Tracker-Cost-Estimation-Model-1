//! The 50 `url_emb_*` features: TF-IDF over URL-path tokens, projected onto
//! the shipped truncated-SVD basis.
//!
//! A port of `src/model/url_embeddings.py` -- `tokenize_url` plus the
//! `TfidfVectorizer` and `TruncatedSVD` it pipelines -- against the fitted
//! vocabulary, idf vector and basis in [`crate::blob`]. It has to be a port
//! rather than a lookup because the features are continuous: there is no
//! finite table of answers to compile in, which is the whole reason this crate
//! is 21 MB where `llm-classifier` is 47 KB.
//!
//! The ensemble downstream is extraordinarily sensitive to these 50 numbers.
//! 370 trees of depth 8 fitted with `tree_method="hist"` put their split
//! thresholds *on* observed feature values, so a one-ulp shift in an embedding
//! is enough to send a request down a different branch: left 1 ulp out, a
//! fifth of requests came back with a different number of bytes and the
//! summed estimate moved 5%. Getting them right therefore meant reproducing
//! scikit-learn's arithmetic exactly rather than accurately -- four separate
//! things, each of which looked like a rounding detail and was not:
//!
//!  * the basis is stored at full `f32` precision, as scikit-learn holds it.
//!    Quantising to 16 bits would halve the payload and moves the median
//!    prediction 9.6%, so there is nothing to trade.
//!  * `f32` throughout, accumulating in ascending vocabulary-index order --
//!    the order scipy's sparse matmul visits a CSR row in. Float addition is
//!    not associative, so the order is part of the answer.
//!  * the L2 normalisation is mixed-precision in the same places
//!    scikit-learn's is; see the comment on it below.
//!  * the projection uses a fused multiply-add, because the C++ `axpy` scipy
//!    compiles does; see the comment on it below.
//!
//! With those four, the features are bit-identical to `engineer_features`' --
//! `tests/test_xgb_classifier_port.py` asserts it over every row of the CSV
//! extract's test half.
//!
//! What is *not* reproduced exactly: Python lowercases and counts characters
//! by Unicode, this lowercases ASCII and treats every non-ASCII byte as a word
//! character. URL paths carrying raw non-ASCII are rare, and the effect is
//! confined to their embeddings.

use crate::blob;
use crate::generated::N_COMPONENTS;
use crate::hash::fnv1a64;

/// Characters `tokenize_url` splits the path on, i.e. its `[/\?&=\.\-_]+`.
/// Splitting on each one rather than on runs of them only produces empty
/// pieces, which contribute no tokens either way.
const DELIMITERS: [char; 7] = ['/', '?', '&', '=', '.', '-', '_'];

/// Embed a URL path, as `URLEmbedder.transform` would.
///
/// `path` is the log's `url_path`, which is what the shipped embedder was
/// fitted and is called on -- not the full URL, and not the query.
pub fn embed(path: &str) -> [f64; N_COMPONENTS] {
    let tokens = tokens(path);

    // Vocabulary index and raw count of every term the path hits. Unigrams
    // then adjacent bigrams, which is `TfidfVectorizer(ngram_range=(1, 2))`.
    // Terms outside the vocabulary are dropped here, exactly as the fitted
    // vectorizer drops them, and so never reach the norm.
    let mut hits: Vec<(usize, f32)> = Vec::new();
    for token in &tokens {
        count(&mut hits, token);
    }
    for pair in tokens.windows(2) {
        count(&mut hits, &format!("{} {}", pair[0], pair[1]));
    }
    hits.sort_unstable_by_key(|&(term, _)| term);

    // `sublinear_tf` then idf then L2, the three steps of
    // `TfidfTransformer.transform`. The term count is an integer, so its log
    // is taken in `f64` and rounded once: `f32::ln` is the platform's `logf`,
    // which is not required to be correctly rounded and need not agree with
    // numpy's to the last bit.
    let mut weights: Vec<f32> = hits
        .iter()
        .map(|&(term, n)| ((f64::from(n).ln() as f32) + 1.0) * blob::idf(term))
        .collect();
    // scikit-learn's `inplace_csr_row_normalize_l2` is mixed-precision, and
    // reproducing it exactly means being mixed in the same places: it squares
    // two `float32`s into a `float32`, accumulates those into a `double`, and
    // divides `float32` data by the `double` root, rounding once on the way
    // back. Doing the squares in `f64` instead -- the obvious "more accurate"
    // reading -- leaves the embeddings 1 ulp out, and 1 ulp is enough to send
    // a request down a different branch of 370 depth-8 trees and move the
    // answer for a popular asset by 18%.
    let norm = weights.iter().map(|w| f64::from(w * w)).sum::<f64>();
    if norm > 0.0 {
        let norm = norm.sqrt();
        for weight in &mut weights {
            *weight = (f64::from(*weight) / norm) as f32;
        }
    }

    // The projection, which is scipy's `csr_matvecs` over one row: an `axpy`
    // per term, in ascending vocabulary-index order, accumulating in `f32`.
    //
    // `mul_add` and not `+= weight * basis`: scipy's `axpy` is
    // `y[i] += a * x[i]` in C++, and every compiler that has a fused
    // multiply-add contracts it into one -- which rounds once where the
    // separate operations round twice. Rust does not contract, so the fusion
    // has to be asked for. It is the last of the four things this function
    // reproduces bit-for-bit, and the one that finally made it exact: without
    // it every embedding is within 1 ulp and a fifth of requests still come
    // out of the ensemble at a different number of bytes.
    //
    // On a target with no FMA, scipy would not fuse either and this becomes
    // the 1-ulp answer instead. `test_xgb_classifier_port` tolerates 2 ulp for
    // that reason rather than asserting equality.
    let mut projected = [0f32; N_COMPONENTS];
    for (&(term, _), &weight) in hits.iter().zip(&weights) {
        for (slot, basis) in projected.iter_mut().zip(blob::components(term)) {
            *slot = weight.mul_add(basis, *slot);
        }
    }
    projected.map(f64::from)
}

/// Add one occurrence of `term`, if the fitted vocabulary kept it.
fn count(hits: &mut Vec<(usize, f32)>, term: &str) {
    let Some(index) = blob::term(fnv1a64(term.as_bytes())) else {
        return;
    };
    // Linear: a URL path yields a few dozen terms, which is below where a hash
    // map starts paying for itself.
    match hits.iter_mut().find(|(seen, _)| *seen == index) {
        Some((_, n)) => *n += 1.0,
        None => hits.push((index, 1.0)),
    }
}

/// The tokens the fitted vectorizer sees for `path`.
///
/// `tokenize_url` splits the path, breaks case and letter/digit boundaries,
/// lowercases and drops anything shorter than two characters; the vectorizer
/// then re-splits what it produced on its default `\b\w\w+\b`. Both steps are
/// here because they do not compose into one: the length filter applies to
/// `tokenize_url`'s token, and the word-run split happens after it, so
/// `a%20b` survives the filter as one token and reaches the vocabulary as
/// `20b`.
fn tokens(path: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for piece in path.split(DELIMITERS) {
        if piece.is_empty() {
            continue;
        }
        // The three `re.sub` passes of `tokenize_url`, in its order. They do
        // not commute: `1a2b` becomes `1 a 2 b` this way round and `1a 2 b`
        // the other.
        let broken = insert_break(piece.as_bytes(), |b| b.is_ascii_lowercase(), |b| {
            b.is_ascii_uppercase()
        });
        let broken = insert_break(&broken, |b| b.is_ascii_alphabetic(), |b| b.is_ascii_digit());
        let broken = insert_break(&broken, |b| b.is_ascii_digit(), |b| b.is_ascii_alphabetic());

        // Valid UTF-8: the passes only ever insert an ASCII space into valid
        // UTF-8, and classify by ASCII, which no continuation byte matches.
        let lowered = String::from_utf8_lossy(&broken).to_ascii_lowercase();
        for token in lowered.split_ascii_whitespace() {
            if token.chars().count() < 2 {
                continue;
            }
            tokens.extend(word_runs(token));
        }
    }
    tokens
}

/// One pass of `re.sub(r'(left)(right)', r'\1 \2', s)`.
///
/// Non-overlapping, like `re.sub`: a pair that matched is consumed whole, so
/// its right-hand byte cannot also be the left-hand byte of the next match.
fn insert_break(bytes: &[u8], left: fn(u8) -> bool, right: fn(u8) -> bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        out.push(bytes[i]);
        if i + 1 < bytes.len() && left(bytes[i]) && right(bytes[i + 1]) {
            out.push(b' ');
            out.push(bytes[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// Maximal runs of at least two word characters, i.e. the vectorizer's
/// `\b\w\w+\b` over one of `tokenize_url`'s tokens.
fn word_runs(token: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut run = String::new();
    for ch in token.chars() {
        if is_word(ch) {
            run.push(ch);
            continue;
        }
        if run.chars().count() >= 2 {
            runs.push(std::mem::take(&mut run));
        } else {
            run.clear();
        }
    }
    if run.chars().count() >= 2 {
        runs.push(run);
    }
    runs
}

/// Python's `\w` under `re.UNICODE`, approximated: ASCII alphanumerics, the
/// underscore, and every non-ASCII character. The underscore never survives
/// `tokenize_url`, which splits on it.
fn is_word(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || !ch.is_ascii()
}
