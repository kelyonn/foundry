//! Pattern → NFA (Thompson construction) → DFA (subset construction), over bytes.
//!
//! Three stages, in this order:
//!
//! 1. **Thompson construction** turns a `Pattern` into an NFA with epsilon edges.
//!    Structural and mechanical: each pattern variant becomes a small fragment.
//! 2. **Subset construction** turns that NFA into a DFA. Each DFA state *is* a set
//!    of NFA states; the start state is the epsilon-closure of the NFA start.
//! 3. **Dead-state pruning** removes states from which no accepting state is
//!    reachable. This step is not cosmetic — see `prune` below. Skipping it breaks
//!    the correctness guarantee.

use crate::pattern::{ByteSet, Pattern};
use std::collections::{HashMap, HashSet, VecDeque};

/// Sentinel for "no transition" — the implicit dead state.
///
/// We use a sentinel rather than a real dead state so that `num_states()` counts
/// only live states, and so the vocabulary index never wastes 19 KB caching the
/// answer for a state where the answer is "nothing is legal."
pub const DEAD: u32 = u32::MAX;

pub type StateId = u32;

// ---------------------------------------------------------------------------
// Stage 1: NFA
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct NfaState {
    /// Epsilon edges: traversable without consuming a byte.
    eps: Vec<usize>,
    /// Byte edges: (set of bytes that may be consumed, destination).
    edges: Vec<(ByteSet, usize)>,
}

struct Nfa {
    states: Vec<NfaState>,
}

/// A partially built NFA with one entry and one exit. Thompson's invariant: every
/// fragment has exactly one start and exactly one accept, which is what makes the
/// combinators below composable without special cases.
struct Frag {
    start: usize,
    accept: usize,
}

impl Nfa {
    fn new() -> Self {
        Nfa { states: Vec::new() }
    }

    fn add_state(&mut self) -> usize {
        self.states.push(NfaState::default());
        self.states.len() - 1
    }

    fn eps(&mut self, from: usize, to: usize) {
        self.states[from].eps.push(to);
    }

    fn byte_edge(&mut self, from: usize, set: ByteSet, to: usize) {
        self.states[from].edges.push((set, to));
    }

    /// Build a fragment for `p`. Recursive, mirroring the `Pattern` tree exactly.
    fn build(&mut self, p: &Pattern) -> Frag {
        match p {
            Pattern::Empty => {
                let s = self.add_state();
                let a = self.add_state();
                self.eps(s, a);
                Frag {
                    start: s,
                    accept: a,
                }
            }

            Pattern::Class(set) => {
                let s = self.add_state();
                let a = self.add_state();
                self.byte_edge(s, *set, a);
                Frag {
                    start: s,
                    accept: a,
                }
            }

            // A literal is just a chain of single-byte classes. Written out
            // directly rather than desugared so the common case stays cheap.
            Pattern::Literal(bytes) => {
                let s = self.add_state();
                let mut cur = s;
                for &b in bytes {
                    let next = self.add_state();
                    self.byte_edge(cur, ByteSet::single(b), next);
                    cur = next;
                }
                Frag {
                    start: s,
                    accept: cur,
                }
            }

            // Chain fragments end to end with epsilon edges.
            Pattern::Concat(parts) => {
                if parts.is_empty() {
                    return self.build(&Pattern::Empty);
                }
                let first = self.build(&parts[0]);
                let mut cur_accept = first.accept;
                for part in &parts[1..] {
                    let f = self.build(part);
                    self.eps(cur_accept, f.start);
                    cur_accept = f.accept;
                }
                Frag {
                    start: first.start,
                    accept: cur_accept,
                }
            }

            // One new start that epsilon-branches into every alternative, and one
            // new accept they all epsilon-join into.
            Pattern::Alt(parts) => {
                if parts.is_empty() {
                    // An alternation of nothing matches nothing at all. Build a
                    // fragment with no path from start to accept.
                    let s = self.add_state();
                    let a = self.add_state();
                    return Frag {
                        start: s,
                        accept: a,
                    };
                }
                let s = self.add_state();
                let a = self.add_state();
                for part in parts {
                    let f = self.build(part);
                    self.eps(s, f.start);
                    self.eps(f.accept, a);
                }
                Frag {
                    start: s,
                    accept: a,
                }
            }

            // Bounded repetition is expanded by *copying* the inner fragment
            // `min` times (mandatory), then either a star (unbounded) or
            // `max - min` optional copies. This is where a huge `maxLength`
            // would blow up the state count — a real cost, worth knowing.
            Pattern::Repeat { inner, min, max } => {
                let s = self.add_state();
                let mut cur = s;

                for _ in 0..*min {
                    let f = self.build(inner);
                    self.eps(cur, f.start);
                    cur = f.accept;
                }

                match max {
                    None => {
                        // Kleene star on the tail: a loop state that can either
                        // enter the body again or fall through to accept.
                        let loop_state = self.add_state();
                        let a = self.add_state();
                        self.eps(cur, loop_state);
                        let f = self.build(inner);
                        self.eps(loop_state, f.start);
                        self.eps(f.accept, loop_state);
                        self.eps(loop_state, a);
                        Frag {
                            start: s,
                            accept: a,
                        }
                    }
                    Some(max) => {
                        let a = self.add_state();
                        // Each optional copy may short-circuit straight to accept.
                        self.eps(cur, a);
                        for _ in *min..*max {
                            let f = self.build(inner);
                            self.eps(cur, f.start);
                            cur = f.accept;
                            self.eps(cur, a);
                        }
                        Frag {
                            start: s,
                            accept: a,
                        }
                    }
                }
            }
        }
    }

    /// All NFA states reachable from `states` using only epsilon edges.
    fn eps_closure(&self, states: &[usize], out: &mut Vec<usize>) {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = Vec::new();
        for &s in states {
            if seen.insert(s) {
                stack.push(s);
            }
        }
        while let Some(s) = stack.pop() {
            for &t in &self.states[s].eps {
                if seen.insert(t) {
                    stack.push(t);
                }
            }
        }
        out.clear();
        out.extend(seen);
        // Sorted so the set can be used as a stable HashMap key in subset
        // construction — two identical sets must hash identically.
        out.sort_unstable();
    }
}

// ---------------------------------------------------------------------------
// Stage 2 + 3: DFA
// ---------------------------------------------------------------------------

/// A deterministic finite automaton over bytes.
///
/// Transition table is dense and flat: `trans[state * 256 + byte]`. 1 KB per
/// state at 4 bytes an entry. Dense because the hot path is
/// `step_token`, which does one indexed load per byte of a candidate token, and
/// we do that ~152,000 times when indexing a state.
pub struct Dfa {
    trans: Vec<u32>,
    accepting: Vec<bool>,
    start: StateId,
}

impl Dfa {
    /// Compile a pattern all the way to a pruned DFA.
    pub fn compile(pattern: &Pattern) -> Dfa {
        let mut nfa = Nfa::new();
        let frag = nfa.build(pattern);
        let dfa = subset_construction(&nfa, frag.start, frag.accept);
        dfa.prune()
    }

    #[inline]
    pub fn transition(&self, state: StateId, byte: u8) -> Option<StateId> {
        let next = self.trans[state as usize * 256 + byte as usize];
        if next == DEAD {
            None
        } else {
            Some(next)
        }
    }

    #[inline]
    pub fn is_accepting(&self, state: StateId) -> bool {
        self.accepting[state as usize]
    }

    pub fn start(&self) -> StateId {
        self.start
    }

    pub fn num_states(&self) -> usize {
        self.accepting.len()
    }

    /// Walk a token's bytes through the DFA from `from`.
    ///
    /// `None` means the token is illegal from this state — somewhere in its bytes
    /// it hit the dead state. `Some(s)` is the state we land in, which becomes
    /// the new current state if this token is chosen.
    ///
    /// Note this is *not* byte-at-a-time from the caller's perspective: a token
    /// may cross several DFA states, and the intermediate states are never
    /// observable from outside. Advancing the DFA is always "walk the whole
    /// token." (docs/GRAMMAR.md, "Tokens spanning multiple states")
    #[inline]
    pub fn step_token(&self, from: StateId, token_bytes: &[u8]) -> Option<StateId> {
        let mut s = from;
        for &b in token_bytes {
            s = self.transition(s, b)?;
        }
        Some(s)
    }

    /// Convenience for tests: does the whole byte string match, start to accept?
    pub fn matches(&self, input: &[u8]) -> bool {
        match self.step_token(self.start, input) {
            Some(s) => self.is_accepting(s),
            None => false,
        }
    }

    /// Remove every state from which no accepting state is reachable.
    ///
    /// **This is a correctness step, not an optimization.** Subset construction
    /// happily produces states that are alive (they have outgoing transitions)
    /// but doomed (every path from them eventually dies). If such a state
    /// survives, the masker will happily allow a token leading into it, because
    /// the token "didn't hit the dead state" — and generation then wanders into a
    /// region where nothing can complete the parse. The guarantee we advertise is
    /// "every prefix is extendable to a valid document," which is a statement
    /// about *reachability of an accepting state*, not about surviving one step.
    ///
    /// Implemented as a reverse BFS from the accepting states.
    fn prune(self) -> Dfa {
        let n = self.num_states();

        // Reverse adjacency: for each state, who can reach it in one byte?
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for s in 0..n {
            let mut seen: HashSet<u32> = HashSet::new();
            for b in 0..256 {
                let t = self.trans[s * 256 + b];
                if t != DEAD && seen.insert(t) {
                    preds[t as usize].push(s as u32);
                }
            }
        }

        // BFS backwards from every accepting state.
        let mut live = vec![false; n];
        let mut queue: VecDeque<u32> = VecDeque::new();
        for (s, &accepts) in self.accepting.iter().enumerate() {
            if accepts {
                live[s] = true;
                queue.push_back(s as u32);
            }
        }
        while let Some(s) = queue.pop_front() {
            for &p in &preds[s as usize] {
                if !live[p as usize] {
                    live[p as usize] = true;
                    queue.push_back(p);
                }
            }
        }

        // Renumber the survivors so `num_states()` is honest and the transition
        // table stays dense.
        let mut remap = vec![DEAD; n];
        let mut next_id = 0u32;
        for s in 0..n {
            if live[s] {
                remap[s] = next_id;
                next_id += 1;
            }
        }

        let live_count = next_id as usize;
        let mut trans = vec![DEAD; live_count * 256];
        let mut accepting = vec![false; live_count];
        for s in 0..n {
            if !live[s] {
                continue;
            }
            let ns = remap[s] as usize;
            accepting[ns] = self.accepting[s];
            for b in 0..256 {
                let t = self.trans[s * 256 + b];
                if t != DEAD && live[t as usize] {
                    trans[ns * 256 + b] = remap[t as usize];
                }
            }
        }

        // If the start state itself is dead, the language is empty. Keep a single
        // non-accepting state so callers don't have to handle an empty table.
        if remap[self.start as usize] == DEAD {
            return Dfa {
                trans: vec![DEAD; 256],
                accepting: vec![false],
                start: 0,
            };
        }

        Dfa {
            trans,
            accepting,
            start: remap[self.start as usize],
        }
    }
}

/// Subset construction: each DFA state is a *set* of NFA states.
fn subset_construction(nfa: &Nfa, nfa_start: usize, nfa_accept: usize) -> Dfa {
    let mut scratch = Vec::new();
    nfa.eps_closure(&[nfa_start], &mut scratch);
    let start_set = scratch.clone();

    // Set-of-NFA-states → DFA state id. The dedup here is the whole algorithm:
    // two different paths that reach the same set of NFA states are the same
    // DFA state, which is why the result is finite even though the NFA has loops.
    let mut ids: HashMap<Vec<usize>, u32> = HashMap::new();
    let mut sets: Vec<Vec<usize>> = Vec::new();
    let mut worklist: VecDeque<u32> = VecDeque::new();

    ids.insert(start_set.clone(), 0);
    sets.push(start_set);
    worklist.push_back(0);

    let mut trans: Vec<u32> = vec![DEAD; 256];

    while let Some(dfa_state) = worklist.pop_front() {
        // Clone because we index `sets` mutably below when adding new states.
        let current = sets[dfa_state as usize].clone();

        for b in 0..=255u8 {
            // move(current, b): every NFA state reachable by consuming `b`.
            let mut moved: Vec<usize> = Vec::new();
            for &s in &current {
                for (set, dst) in &nfa.states[s].edges {
                    if set.contains(b) {
                        moved.push(*dst);
                    }
                }
            }
            if moved.is_empty() {
                continue; // stays DEAD
            }

            nfa.eps_closure(&moved, &mut scratch);
            let key = scratch.clone();

            let target = match ids.get(&key) {
                Some(&id) => id,
                None => {
                    let id = sets.len() as u32;
                    ids.insert(key.clone(), id);
                    sets.push(key);
                    trans.extend(std::iter::repeat_n(DEAD, 256));
                    worklist.push_back(id);
                    id
                }
            };
            trans[dfa_state as usize * 256 + b as usize] = target;
        }
    }

    // A DFA state accepts iff its NFA set contains the NFA's accept state.
    let accepting: Vec<bool> = sets
        .iter()
        .map(|set| set.binary_search(&nfa_accept).is_ok())
        .collect();

    Dfa {
        trans,
        accepting,
        start: 0,
    }
}
