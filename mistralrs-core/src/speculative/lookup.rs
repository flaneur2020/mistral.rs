//! Prompt-lookup drafting: propose the tokens that followed the longest earlier repeat of a
//! sequence's current suffix.
//!
//! The same mechanism as Strata's suffix drafter (see `.plans/benchmark.md`, W37): the index is
//! built over the sequence's own tokens, so output that quotes its input - a code edit that keeps
//! most lines, a verbatim repeat - can be drafted exactly, and one verify window commits several
//! tokens. It needs no weights and no VRAM, and its proposals are verified like any other draft, so
//! it can only change the speed and never the output.
//!
//! Opt-in through the environment, so nothing changes unless asked:
//!
//! * `MISTRALRS_LOOKUP_DRAFT=<n>`: propose up to `n` tokens (unset or 0: off). The value is
//!   clamped to the draft window the engine was configured with, because the verify batch is sized
//!   for that.
//! * `MISTRALRS_LOOKUP_MIN=<m>`: smallest context match to propose from (default 6). Strata's
//!   default is 3 with a learned cost/benefit policy on top; here the gate is static and stricter,
//!   since a proposal that is not accepted costs a longer verify window.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Acceptance of lookup drafts by match length.
///
/// The split is the one that decides whether the feature pays: a 3-token match is usually the same
/// structure with a different constant, so the proposal is wrong and the longer verify window is
/// wasted (measured at -24% on a repeating pattern prompt, W38), while a long match is usually a
/// verbatim repeat that commits almost every draft (strata measures 95 of 95). Judging them
/// together is what makes a single static gate either lose or gain nothing.
#[derive(Clone, Copy, Default)]
struct Bucket {
    samples: u32,
    /// Drafts this bucket's proposals put in front of the verifier, and how many of them it kept.
    /// The gate is a rate over tokens, not over rounds: a proposal is up to eight drafts long.
    proposed: u32,
    accepted: u32,
}

fn bucket_of(matched: usize) -> usize {
    match matched {
        0..=3 => 0,
        4..=5 => 1,
        6..=7 => 2,
        _ => 3,
    }
}

type Policy = Mutex<[Bucket; 4]>;
fn policy() -> &'static Policy {
    static P: OnceLock<Policy> = OnceLock::new();
    P.get_or_init(|| Mutex::new([Bucket::default(); 4]))
}

/// Proposals taken this step, per sequence: `(seq id, bucket, drafts)`. Replaced on every step, so
/// a step that bypasses speculation cannot leave a stale entry behind.
type Pending = Mutex<Vec<(usize, usize, usize)>>;
fn pending() -> &'static Pending {
    static P: OnceLock<Pending> = OnceLock::new();
    P.get_or_init(|| Mutex::new(Vec::new()))
}

/// How many samples a bucket needs before its rate is trusted; below that a proposal is taken, so
/// the policy can learn, and the cost of learning is a few rounds.
const MIN_SAMPLES: u32 = 4;
/// Below this the match length has not been paying, and its window is refused.
const MIN_RATE: f64 = 0.5;
/// Match length a bucket needs before its own rate is trusted at all.
const COLD_MATCH: usize = 6;

/// Feed the accepted draft count of the round back into the policy.
pub(crate) fn observe_round(rows: &[(usize, usize)]) {
    let mut p = match pending().lock() {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut policy = match policy().lock() {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut seen = 0usize;
    for (seq_id, accepted) in rows {
        let Some(pos) = p.iter().position(|(id, _, _)| id == seq_id) else {
            continue;
        };
        let (_, bucket, drafts) = p.remove(pos);
        let entry = &mut policy[bucket];
        entry.samples += 1;
        entry.proposed += drafts as u32;
        // Drafts are verified in order, so the lookup's own share of the window is its prefix.
        entry.accepted += (*accepted).min(drafts) as u32;
        seen += 1;
    }
    if seen > 0 && std::env::var_os("MISTRALRS_LOOKUP_DEBUG").is_some() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 8 == 0 {
            let rates = policy
                .iter()
                .map(|b| {
                    if b.proposed == 0 {
                        "-".to_string()
                    } else {
                        format!(
                            "{:.2}({})",
                            f64::from(b.accepted) / f64::from(b.proposed),
                            b.samples
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("lookup policy: match 3 / 4-5 / 6-7 / 8+ acceptance = {rates}");
        }
    }
}

use super::proposer::{SpeculativeProposalBatch, SpeculativeTokens};
use crate::sequence::Sequence;

/// End positions kept per trigram, most recent first.
const WAYS: usize = 4;
/// How far a match may extend backwards.
const MAX_MATCH: usize = 32;

pub(crate) struct LookupDrafter {
    max_drafts: usize,
    min_match: usize,
}

impl LookupDrafter {
    pub(crate) fn from_env(max_drafts: usize) -> Option<Self> {
        let want: usize = std::env::var("MISTRALRS_LOOKUP_DRAFT")
            .ok()
            .and_then(|v| v.parse().ok())?;
        if want == 0 || max_drafts == 0 {
            return None;
        }
        let min_match = std::env::var("MISTRALRS_LOOKUP_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        Some(Self {
            max_drafts: want.min(max_drafts),
            min_match,
        })
    }

    pub(crate) fn max_drafts(&self) -> usize {
        self.max_drafts
    }

    /// Propose up to `max_drafts` tokens that followed the longest earlier repeat of `history`'s
    /// suffix, with the length of that match. `None` when nothing is long enough to be worth a
    /// verify window.
    ///
    /// The index is rebuilt from the sequence's tokens on every call: a decode step has a few
    /// thousand tokens of history, so this is microseconds against a round that costs tens of
    /// milliseconds, and it needs no state to invalidate when a draft is rejected.
    pub(crate) fn propose(&self, history: &[u32]) -> Option<(Vec<u32>, usize)> {
        let n = history.len();
        if n < self.min_match + 3 {
            return None;
        }
        let mut index: HashMap<(u32, u32, u32), Vec<usize>> = HashMap::new();
        for (i, w) in history.windows(3).enumerate() {
            let slot = index.entry((w[0], w[1], w[2])).or_default();
            slot.insert(0, i + 2); // i + 2 is the trigram's last token; most recent first
            slot.truncate(WAYS);
        }

        let key = (history[n - 3], history[n - 2], history[n - 1]);
        let mut best: Option<(usize, usize)> = None;
        for &end in index.get(&key)? {
            if end + 1 >= n {
                continue; // the tail's own position
            }
            // Extend the match backwards. Self-overlap is allowed on purpose: a repeating block
            // matches itself, and the continuation of that block is exactly what to propose.
            let mut len = 3;
            // The two occurrences end at `n - 1` and `end`, so the token `len` positions back is
            // `n - 1 - len` on one side and `end - len` on the other.
            while len < MAX_MATCH
                && end >= len
                && n > len
                && history[end - len] == history[n - 1 - len]
            {
                len += 1;
            }
            if best.is_none_or(|(bl, _)| len > bl) {
                best = Some((len, end));
            }
        }
        let (match_len, end) = best?;
        if match_len < self.min_match {
            return None;
        }
        let take = self.max_drafts.min(n - end - 1);
        if take == 0 {
            return None;
        }
        Some((history[end + 1..end + 1 + take].to_vec(), match_len))
    }
}

impl LookupDrafter {
    /// Replace a sequence's MTP drafts with a prompt-lookup proposal when it repeats its own
    /// context and the proposal is at least as long as what the MTP proposed. A source with no
    /// distribution keeps the exact-match rule, so the proposal carries no distribution either.
    pub(crate) fn apply(&self, batch: &mut SpeculativeProposalBatch, sequences: &[&Sequence]) {
        if std::env::var_os("MISTRALRS_LOOKUP_DEBUG").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let i = N.fetch_add(1, Ordering::Relaxed);
            // Count every attempt, not a sample: the useful number is how often a proposal exists.
            {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static TRIES: AtomicUsize = AtomicUsize::new(0);
                static TAKEN: AtomicUsize = AtomicUsize::new(0);
                static MATCH: AtomicUsize = AtomicUsize::new(0);
                let t = TRIES.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some((toks, m)) =
                    self.propose(sequences.first().map(|s| s.get_toks()).unwrap_or(&[]))
                {
                    TAKEN.fetch_add(1, Ordering::Relaxed);
                    MATCH.fetch_add(m, Ordering::Relaxed);
                    let _ = toks;
                }
                if t % 16 == 0 {
                    let taken = TAKEN.load(Ordering::Relaxed);
                    eprintln!(
                        "lookup summary: {t} tries, {taken} proposals exist ({:.0}%), mean match {:.1}",
                        taken as f64 / t as f64 * 100.0,
                        MATCH.load(Ordering::Relaxed) as f64 / taken.max(1) as f64,
                    );
                }
            }
            if i < 400 && (i < 4 || i % 16 == 0) {
                let toks = sequences.first().map(|s| s.get_toks()).unwrap_or(&[]);
                let tail: Vec<String> = toks
                    .iter()
                    .rev()
                    .take(16)
                    .rev()
                    .map(|t| t.to_string())
                    .collect();
                let got = self
                    .propose(toks)
                    .map(|(t, m)| format!("{:?} match {m}", t))
                    .unwrap_or_else(|| "none".to_string());
                let head: Vec<String> = toks.iter().take(8).map(|t| t.to_string()).collect();
                eprintln!(
                    "lookup apply[{i}]: hist={} head={} tail={} propose -> {got}",
                    toks.len(),
                    head.join(","),
                    tail.join(",")
                );
            }
        }
        // Held across the loop: this is the only writer, and the round's feedback is read after the
        // verify, not here.
        let mut taken = pending().lock().ok();
        if let Some(taken) = taken.as_mut() {
            taken.clear();
        }
        for (proposal, seq) in batch.proposals.iter_mut().zip(sequences) {
            let Some((tokens, matched)) = self.propose(seq.get_toks()) else {
                continue;
            };
            if tokens.len() < proposal.tokens.len() {
                continue;
            }
            // Take this match length only if it has been paying (or is still being learned).
            let bucket = bucket_of(matched);
            let (samples, rate) = match policy().lock() {
                Ok(p) => {
                    let b = p[bucket];
                    (
                        b.samples,
                        f64::from(b.accepted) / f64::from(b.proposed.max(1)),
                    )
                }
                Err(_) => (MIN_SAMPLES, 1.0),
            };
            // Cold start: a bucket with no evidence is only trusted for a long match, where the
            // continuation is usually the earlier text itself. Short matches (structure shared, the
            // constant not) are the ones that lose, so they wait for the rate to earn its place.
            if samples < MIN_SAMPLES {
                if matched < COLD_MATCH {
                    continue;
                }
            } else if rate < MIN_RATE {
                continue;
            }
            if let Some(taken) = taken.as_mut() {
                taken.push((*seq.id(), bucket, tokens.len()));
            }
            lookup_probe(tokens.len(), matched);
            proposal.tokens = SpeculativeTokens::Host(tokens);
            proposal.distribution = None;
        }
    }
}

/// Measurement hook (`MISTRALRS_LOOKUP_DEBUG=1`): how many tokens the lookup proposed and how long
/// the match behind them was. Acceptance is the verifier's number, as for MTP drafts.
fn lookup_probe(drafts: usize, matched: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("MISTRALRS_LOOKUP_DEBUG").is_some()) {
        return;
    }
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static DRAFTS: AtomicUsize = AtomicUsize::new(0);
    static MATCH: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    DRAFTS.fetch_add(drafts, Ordering::Relaxed);
    MATCH.fetch_add(matched, Ordering::Relaxed);
    if n % 32 == 0 {
        eprintln!(
            "lookup drafter: {n} proposals taken, {:.1} drafts and {:.1} match tokens on average",
            DRAFTS.load(Ordering::Relaxed) as f64 / n as f64,
            MATCH.load(Ordering::Relaxed) as f64 / n as f64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::LookupDrafter;

    fn drafter(max_drafts: usize, min_match: usize) -> LookupDrafter {
        LookupDrafter {
            max_drafts,
            min_match,
        }
    }

    #[test]
    fn proposes_the_continuation_of_the_longest_earlier_repeat() {
        // The tail repeats the first line, so the tokens after that line are the answer.
        let mut history: Vec<u32> = (100..140).collect();
        history.extend_from_slice(&(100..112).collect::<Vec<u32>>());
        let (tokens, matched) = drafter(4, 6).propose(&history).expect("a match");
        assert_eq!(tokens, vec![112, 113, 114, 115]);
        assert_eq!(matched, 12);
    }

    #[test]
    fn stays_quiet_without_a_long_enough_match() {
        let history: Vec<u32> = (0..64).collect();
        assert!(drafter(4, 6).propose(&history).is_none());
        // Three tokens repeat, which is below the default gate.
        let mut short = vec![7u32, 8, 9, 20, 21, 22, 23, 7, 8, 9];
        assert!(drafter(4, 6).propose(&short).is_none());
        short.extend_from_slice(&[7, 8, 9]);
        assert!(drafter(4, 3).propose(&short).is_some());
    }

    #[test]
    fn never_proposes_more_than_the_window() {
        let mut history: Vec<u32> = (100..200).collect();
        history.extend_from_slice(&(100..130).collect::<Vec<u32>>());
        let (tokens, _) = drafter(2, 6).propose(&history).expect("a match");
        assert_eq!(tokens.len(), 2);
    }
}
