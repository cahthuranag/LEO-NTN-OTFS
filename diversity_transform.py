"""
Algebraic Diversity Transform Construction
===========================================
Implements the diversity transform layer from Section IV of:
"Diversity Transforms for Extreme-Reliability LEO NTN OTFS Downlinks"

This module provides:
- GF(2^m) finite field arithmetic (exp/log tables)
- Faithful binary matrix representation via companion matrix (phi map)
- Algorithm 1: Construction of fixed diversity transform G_FIX
- Algorithm 2: Adaptive transform selection with quality metric
- GF(2) linear algebra utilities (rank, multiply, invert)
- MDS property verification (Theorem 1)
- Encoding / decoding interface

Parameters (from paper):
  m = 3, p(x) = x^3 + x + 1, GF(8)
  k_c = 4  (MDS dimension)
  n_s = 6  (number of subchannels)
  rho = k_c * m = 12  (binary input dimension)
  n_out = n_s * m = 18  (binary output dimension)
  d_min = n_s - k_c + 1 = 3
  max_erasures = d_min - 1 = 2

Author: Research implementation
Date: 2026-02-08
"""

import numpy as np
from itertools import combinations


# ============================================================================
#  A. GF(2^m) Finite Field Arithmetic
# ============================================================================

class GF2m:
    """
    Galois field GF(2^m) arithmetic using exp/log tables.

    Default: GF(8) with irreducible polynomial p(x) = x^3 + x + 1 (0b1011).
    Primitive element alpha = 2 (i.e., x).
    """

    def __init__(self, m=3, irred_poly=0b1011):
        self.m = m
        self.q = 1 << m           # 2^m = field size
        self.irred_poly = irred_poly
        self.alpha = 2             # primitive element = x

        # Build exp and log tables
        self.exp_table = [0] * (self.q - 1)  # alpha^i for i=0..q-2
        self.log_table = [0] * self.q         # log_alpha(a) for a=1..q-1

        val = 1
        for i in range(self.q - 1):
            self.exp_table[i] = val
            self.log_table[val] = i
            val = self._mul_no_table(val, self.alpha)

    def _mul_no_table(self, a, b):
        """Polynomial multiplication mod irred_poly (used to build tables)."""
        result = 0
        for i in range(self.m):
            if b & (1 << i):
                result ^= a << i
        # Reduce modulo irred_poly
        for i in range(2 * self.m - 2, self.m - 1, -1):
            if result & (1 << i):
                result ^= self.irred_poly << (i - self.m)
        return result & ((1 << self.m) - 1)

    def add(self, a, b):
        """Addition in GF(2^m): XOR."""
        return a ^ b

    def mul(self, a, b):
        """Multiplication in GF(2^m) via log/exp tables."""
        if a == 0 or b == 0:
            return 0
        log_sum = (self.log_table[a] + self.log_table[b]) % (self.q - 1)
        return self.exp_table[log_sum]

    def inv(self, a):
        """Multiplicative inverse: a^(-1) = alpha^(q-2-log(a))."""
        if a == 0:
            raise ZeroDivisionError("Cannot invert 0 in GF(2^m)")
        log_a = self.log_table[a]
        return self.exp_table[(self.q - 1 - log_a) % (self.q - 1)]

    def pow(self, a, n):
        """Exponentiation: a^n in GF(2^m)."""
        if a == 0:
            return 0 if n > 0 else 1
        log_a = self.log_table[a]
        return self.exp_table[(log_a * n) % (self.q - 1)]

    def elements(self):
        """Return all field elements 0, 1, ..., q-1."""
        return list(range(self.q))

    def nonzero_elements(self):
        """Return all nonzero field elements (GF(2^m)*)."""
        return list(range(1, self.q))


# ============================================================================
#  B. Faithful Matrix Representation (phi map)
# ============================================================================

def companion_matrix(irred_poly, m):
    """
    Companion matrix C of the irreducible polynomial p(x).

    For p(x) = x^m + c_{m-1}x^{m-1} + ... + c_1 x + c_0:
      C[i, m-1] = c_i  (last column = polynomial coefficients)
      C[i+1, i] = 1    (sub-diagonal = 1)

    For p(x) = x^3 + x + 1 (coeff vector [1,1,0] for c_0,c_1,c_2):
      C = [[0, 0, 1],
           [1, 0, 1],
           [0, 1, 0]]

    Returns: (m x m) numpy array over GF(2).
    """
    C = np.zeros((m, m), dtype=int)
    # Sub-diagonal: C[i+1, i] = 1
    for i in range(m - 1):
        C[i + 1, i] = 1
    # Last column: coefficients of p(x) (excluding leading x^m term)
    for i in range(m):
        C[i, m - 1] = (irred_poly >> i) & 1
    return C


def phi(element, gf, C_mat):
    """
    Faithful binary matrix representation:
      phi(0) = zero matrix
      phi(1) = identity matrix
      phi(alpha^i) = C^i

    Maps a GF(2^m) element to its m x m binary matrix representation.

    Args:
        element: integer representing a GF(2^m) element
        gf: GF2m instance
        C_mat: companion matrix (m x m)

    Returns: (m x m) numpy array over GF(2).
    """
    m = gf.m
    if element == 0:
        return np.zeros((m, m), dtype=int)

    log_e = gf.log_table[element]
    # C^log_e over GF(2)
    result = np.eye(m, dtype=int)
    base = C_mat.copy()
    exp = log_e
    while exp > 0:
        if exp & 1:
            result = gf2_matmul(result, base)
        base = gf2_matmul(base, base)
        exp >>= 1
    return result


# ============================================================================
#  D. GF(2) Linear Algebra Utilities
# ============================================================================

def gf2_matmul(A, B):
    """Matrix multiplication over GF(2): (A @ B) mod 2."""
    return np.mod(A @ B, 2).astype(int)


def gf2_rank(matrix):
    """
    Compute rank of a binary matrix over GF(2) via Gaussian elimination.

    Args:
        matrix: 2D numpy array with entries in {0, 1}

    Returns: integer rank
    """
    M = matrix.astype(int).copy()
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        # Find pivot in current column
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap pivot row to current rank position
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate other rows
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


def gf2_inv(matrix):
    """
    Invert a square binary matrix over GF(2) via augmented row reduction.

    Args:
        matrix: (n x n) numpy array with entries in {0, 1}

    Returns: (n x n) inverse matrix over GF(2), or None if singular.
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n), "Matrix must be square"

    # Augment [M | I]
    aug = np.zeros((n, 2 * n), dtype=int)
    aug[:, :n] = matrix.astype(int)
    aug[:, n:] = np.eye(n, dtype=int)

    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            return None  # Singular

        # Swap
        aug[[col, pivot]] = aug[[pivot, col]]

        # Eliminate
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[col]) % 2

    return aug[:, n:]


# ============================================================================
#  C. Algorithm 1: Construction of G_FIX
# ============================================================================

def construct_G_FIX(k_c=4, n_s=6, m=3, irred_poly=0b1011):
    """
    Algorithm 1: Construction of Fixed Diversity Transform Matrix from MDS Codes.

    Steps:
    1. Build RS Vandermonde matrix G_MDS (k_c x n_s) over GF(2^m)
       using n_s distinct nonzero evaluation points from GF(2^m)*
    2. Binary lifting: replace each GF element with its m x m binary
       representation via the phi map
    3. Result: G_FIX in F_2^{(k_c*m) x (n_s*m)}

    Args:
        k_c: MDS code dimension (input blocks), default 4
        n_s: number of subchannels (output blocks), default 6
        m: extension degree, default 3
        irred_poly: irreducible polynomial, default x^3+x+1

    Returns:
        G_FIX: (k_c*m x n_s*m) binary matrix (numpy array)
    """
    gf = GF2m(m, irred_poly)

    # Need n_s distinct nonzero elements from GF(2^m)*
    # GF(2^m)* has 2^m - 1 elements; need n_s <= 2^m - 1
    nonzero = gf.nonzero_elements()
    assert n_s <= len(nonzero), (
        f"Need {n_s} evaluation points but GF(2^{m})* has only {len(nonzero)}")

    # Evaluation points: alpha^0, alpha^1, ..., alpha^{n_s-1}
    eval_points = [gf.exp_table[i] for i in range(n_s)]

    # Step 1: Build Vandermonde G_MDS (k_c x n_s) over GF(2^m)
    # G_MDS[i, j] = eval_points[j]^i
    G_MDS = np.zeros((k_c, n_s), dtype=int)
    for i in range(k_c):
        for j in range(n_s):
            G_MDS[i, j] = gf.pow(eval_points[j], i)

    # Step 2: Binary lifting via phi map
    C_mat = companion_matrix(irred_poly, m)
    rho = k_c * m
    n_out = n_s * m

    G_FIX = np.zeros((rho, n_out), dtype=int)
    for i in range(k_c):
        for j in range(n_s):
            block = phi(G_MDS[i, j], gf, C_mat)
            G_FIX[i * m:(i + 1) * m, j * m:(j + 1) * m] = block

    return G_FIX


# ============================================================================
#  E. MDS Property Verification
# ============================================================================

def verify_mds_property(G_FIX, k_c=4, n_s=6, m=3):
    """
    Verify Theorem 1: For any set E of erased subchannels with
    |E| <= d_min - 1 = n_s - k_c, the submatrix G_FIX restricted
    to surviving blocks has rank rho = k_c * m.

    Tests all C(n_s, k_c) combinations of k_c surviving subchannels.

    Returns:
        (all_pass, results) where results is a list of
        (surviving_indices, rank, passed) tuples.
    """
    rho = k_c * m
    results = []
    all_pass = True

    for surviving in combinations(range(n_s), k_c):
        # Extract columns for surviving subchannels (blocks of m)
        col_indices = []
        for s in surviving:
            col_indices.extend(range(s * m, (s + 1) * m))
        G_sub = G_FIX[:, col_indices]
        r = gf2_rank(G_sub)
        passed = (r == rho)
        if not passed:
            all_pass = False
        results.append((surviving, r, passed))

    return all_pass, results


# ============================================================================
#  F. GL(rho, 2) Candidate Generation
# ============================================================================

def generate_candidate_T(rho, rng):
    """
    Generate a random invertible binary matrix T in GL(rho, F_2).

    Args:
        rho: matrix dimension
        rng: numpy random generator

    Returns: (rho x rho) invertible binary matrix
    """
    max_attempts = 100
    for _ in range(max_attempts):
        T = rng.integers(0, 2, size=(rho, rho))
        if gf2_rank(T) == rho:
            return T
    raise RuntimeError(f"Failed to generate invertible {rho}x{rho} "
                       f"binary matrix after {max_attempts} attempts")


def generate_T_candidates(rho, n_candidates, rng):
    """Generate a pool of n_candidates invertible binary matrices."""
    return [generate_candidate_T(rho, rng) for _ in range(n_candidates)]


# ============================================================================
#  G. Algorithm 2: Adaptive Diversity Transform Selection
# ============================================================================

def quality_metric(G_DIV, reliability, n_s, m):
    """
    Compute quality metric Q(G_DIV, eta) from paper.

    Q = sum_{i=1}^{n_s} eta_i * sum_{j in block_i} w_H(g_j)

    where g_j are columns of G_DIV and block_i = columns [(i-1)*m, i*m).
    w_H is the Hamming weight (number of 1s in a column).

    Higher Q means more coded bits are placed on reliable subchannels.

    Args:
        G_DIV: (rho x n_out) binary matrix
        reliability: (n_s,) reliability vector eta
        n_s: number of subchannels
        m: extension degree

    Returns: scalar quality metric
    """
    Q = 0.0
    for i in range(n_s):
        block_cols = G_DIV[:, i * m:(i + 1) * m]
        # Sum of Hamming weights of columns in this block
        block_weight = np.sum(block_cols)
        Q += reliability[i] * block_weight
    return Q


def select_adaptive_transform(G_FIX, reliability, n_candidates=200,
                               rng=None, m=3, n_s=6):
    """
    Algorithm 2: Adaptive Diversity Transform Selection.

    Searches over random invertible transformations T in GL(rho, F_2)
    to maximize the quality metric Q(T @ G_FIX, eta).

    Args:
        G_FIX: (rho x n_out) fixed binary transform matrix
        reliability: (n_s,) reliability vector eta_i = gamma_i / gamma_avg
        n_candidates: number of candidate transforms to evaluate
        rng: numpy random generator
        m: extension degree
        n_s: number of subchannels

    Returns:
        (T_star, G_DIV_star): optimal transform and resulting diversity matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    rho = G_FIX.shape[0]

    # Quality of identity transform (= fixed transform baseline)
    Q_fixed = quality_metric(G_FIX, reliability, n_s, m)
    T_star = np.eye(rho, dtype=int)
    G_DIV_star = G_FIX.copy()
    Q_star = Q_fixed

    # Search over random candidates
    candidates = generate_T_candidates(rho, n_candidates, rng)

    for T in candidates:
        G_DIV = gf2_matmul(T, G_FIX)
        Q = quality_metric(G_DIV, reliability, n_s, m)
        if Q > Q_star:
            Q_star = Q
            T_star = T
            G_DIV_star = G_DIV

    return T_star, G_DIV_star


# ============================================================================
#  H. Encoding / Decoding Interface
# ============================================================================

def apply_diversity_transform(info_bits, G_DIV):
    """
    Encode information bits using diversity transform.

    coded = info_bits @ G_DIV (mod 2)

    Args:
        info_bits: binary vector of length rho
        G_DIV: (rho x n_out) binary transform matrix

    Returns: coded_bits, binary vector of length n_out
    """
    info_bits = np.asarray(info_bits, dtype=int)
    return np.mod(info_bits @ G_DIV, 2)


def diversity_demapper(received_bits, G_matrix, erased_mask, m=3):
    """
    Recover information bits from received blocks after erasure.

    Extracts columns corresponding to non-erased subchannels and
    solves G_active @ x = received_active over GF(2).

    Args:
        received_bits: binary vector of length n_out (with erased blocks zeroed)
        G_matrix: (rho x n_out) binary transform matrix (G_FIX or G_DIV)
        erased_mask: (n_s,) boolean array, True = erased subchannel
        m: extension degree

    Returns:
        recovered_bits: binary vector of length rho, or None if unrecoverable.
    """
    n_s = len(erased_mask)
    rho = G_matrix.shape[0]

    # Extract surviving subchannel columns
    surviving_cols = []
    surviving_bits = []
    for i in range(n_s):
        if not erased_mask[i]:
            for j in range(m):
                col_idx = i * m + j
                surviving_cols.append(col_idx)
                surviving_bits.append(received_bits[col_idx])

    G_active = G_matrix[:, surviving_cols]
    r_active = np.array(surviving_bits, dtype=int)

    # Check if recovery is possible
    if gf2_rank(G_active) < rho:
        return None

    # Solve via GF(2) Gaussian elimination
    n_cols = len(surviving_cols)
    # Build augmented matrix [G_active^T | r_active^T]^T -> solve G_active^T x = r_active
    # Actually we need to solve: info @ G_active = r_active
    # i.e., G_active^T @ info^T = r_active^T
    G_T = G_active.T.copy()
    r = r_active.copy()

    # Augmented system [G_T | r]
    aug = np.zeros((n_cols, rho + 1), dtype=int)
    aug[:, :rho] = G_T
    aug[:, rho] = r

    # Row reduce
    pivot_cols = []
    row = 0
    for col in range(rho):
        # Find pivot
        pivot = None
        for r_idx in range(row, n_cols):
            if aug[r_idx, col] == 1:
                pivot = r_idx
                break
        if pivot is None:
            continue
        aug[[row, pivot]] = aug[[pivot, row]]
        for r_idx in range(n_cols):
            if r_idx != row and aug[r_idx, col] == 1:
                aug[r_idx] = (aug[r_idx] + aug[row]) % 2
        pivot_cols.append(col)
        row += 1

    if len(pivot_cols) < rho:
        return None

    # Extract solution
    solution = np.zeros(rho, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i, rho]

    return solution


# ============================================================================
#  I. Integration Config Class
# ============================================================================

class DiversityTransformConfig:
    """
    Configuration and interface for diversity transform operations.

    Constructs G_FIX on initialization and provides methods for
    fixed/adaptive transforms and verification.
    """

    def __init__(self, k_c=4, n_s=6, m=3, irred_poly=0b1011):
        self.k_c = k_c
        self.n_s = n_s
        self.m = m
        self.rho = k_c * m
        self.n_out = n_s * m
        self.max_erasures = n_s - k_c
        self.d_min = n_s - k_c + 1
        self.gf = GF2m(m, irred_poly)
        self.G_FIX = construct_G_FIX(k_c, n_s, m, irred_poly)

    def get_fixed_transform(self):
        """Return the fixed diversity transform matrix G_FIX."""
        return self.G_FIX

    def get_adaptive_transform(self, reliability, n_candidates=200, rng=None):
        """
        Compute adaptive diversity transform for given reliability vector.

        Args:
            reliability: (n_s,) reliability vector
            n_candidates: search pool size
            rng: random generator

        Returns: G_DIV (rho x n_out binary matrix)
        """
        T, G_DIV = select_adaptive_transform(
            self.G_FIX, reliability, n_candidates, rng,
            m=self.m, n_s=self.n_s)
        return G_DIV

    def can_recover(self, erased_mask):
        """
        Check if recovery is possible given erased subchannels.

        Uses actual GF(2) rank check on surviving submatrix.

        Args:
            erased_mask: (n_s,) boolean array, True = erased

        Returns: True if rank of surviving submatrix equals rho
        """
        surviving_cols = []
        for i in range(self.n_s):
            if not erased_mask[i]:
                for j in range(self.m):
                    surviving_cols.append(i * self.m + j)
        if len(surviving_cols) < self.rho:
            return False
        G_sub = self.G_FIX[:, surviving_cols]
        return gf2_rank(G_sub) == self.rho

    def verify(self):
        """Verify MDS property for all erasure patterns."""
        return verify_mds_property(self.G_FIX, self.k_c, self.n_s, self.m)


# ============================================================================
#  J. Demo / Verification
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("Algebraic Diversity Transform — Construction & Verification")
    print("=" * 72)

    # --- GF(8) field ---
    gf = GF2m(m=3, irred_poly=0b1011)
    print(f"\nGF(2^3) = GF(8), irreducible polynomial: x^3 + x + 1")
    print(f"Primitive element alpha = 2")
    print(f"Exp table (alpha^i): {gf.exp_table}")
    print(f"Log table: {gf.log_table}")

    # --- Companion matrix ---
    C = companion_matrix(0b1011, 3)
    print(f"\nCompanion matrix C of x^3 + x + 1:")
    print(C)

    # --- phi map verification ---
    print(f"\nphi map (element -> 3x3 binary matrix):")
    for e in range(8):
        P = phi(e, gf, C)
        print(f"  phi({e}) = {P.flatten().tolist()}")

    # --- Construct G_FIX ---
    print("\n" + "-" * 60)
    k_c, n_s, m = 4, 6, 3
    G_FIX = construct_G_FIX(k_c, n_s, m)
    rho = k_c * m
    n_out = n_s * m
    rank = gf2_rank(G_FIX)

    print(f"\nAlgorithm 1: G_FIX construction")
    print(f"  Parameters: k_c={k_c}, n_s={n_s}, m={m}")
    print(f"  G_FIX shape: {G_FIX.shape}  (rho={rho} x n_out={n_out})")
    print(f"  G_FIX rank over GF(2): {rank}")
    print(f"  Full rank: {'YES' if rank == rho else 'NO'}")
    print(f"\nG_FIX matrix:")
    for row in G_FIX:
        print(f"  {''.join(str(x) for x in row)}")

    # --- MDS verification ---
    print("\n" + "-" * 60)
    print(f"\nTheorem 1: MDS property verification")
    print(f"  d_min = {n_s - k_c + 1}, max erasures = {n_s - k_c}")
    print(f"  Testing all C({n_s},{k_c}) = {len(list(combinations(range(n_s), k_c)))} "
          f"surviving subchannel patterns:")

    all_pass, results = verify_mds_property(G_FIX, k_c, n_s, m)
    for surviving, r, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"    Surviving {list(surviving)}: "
              f"rank = {r}/{rho}  [{status}]")

    print(f"\n  Overall: {'ALL PATTERNS PASS' if all_pass else 'SOME PATTERNS FAIL'}")

    # --- Algorithm 2: Adaptive transform ---
    print("\n" + "-" * 60)
    print(f"\nAlgorithm 2: Adaptive transform selection")

    rng = np.random.default_rng(2026)

    # Sample reliability vector (subchannel SNR / avg SNR)
    reliability = np.array([1.2, 0.3, 1.5, 0.8, 1.1, 0.6])
    print(f"  Reliability vector eta: {reliability}")

    Q_fixed = quality_metric(G_FIX, reliability, n_s, m)
    print(f"  Q(G_FIX, eta) = {Q_fixed:.2f}  (fixed baseline)")

    T_star, G_DIV = select_adaptive_transform(
        G_FIX, reliability, n_candidates=500, rng=rng, m=m, n_s=n_s)
    Q_adapt = quality_metric(G_DIV, reliability, n_s, m)
    print(f"  Q(G_DIV, eta) = {Q_adapt:.2f}  (adaptive)")
    print(f"  Gain: Q_adapt/Q_fixed = {Q_adapt / Q_fixed:.4f}")
    print(f"  Adaptive >= Fixed: {'YES' if Q_adapt >= Q_fixed else 'NO'}")

    # --- Encode, erase, decode test ---
    print("\n" + "-" * 60)
    print(f"\nEncode → Erase → Decode test")

    info_bits = rng.integers(0, 2, size=rho)
    print(f"  Info bits ({rho}): {info_bits.tolist()}")

    coded = apply_diversity_transform(info_bits, G_FIX)
    print(f"  Coded bits ({n_out}): {coded.tolist()}")

    # Erase subchannels 1 and 4 (0-indexed)
    erased_mask = np.array([False, True, False, False, True, False])
    n_erased = int(np.sum(erased_mask))
    print(f"  Erased subchannels: {np.where(erased_mask)[0].tolist()} "
          f"({n_erased} erasures, max allowed = {n_s - k_c})")

    # Zero out erased blocks in received
    received = coded.copy()
    for i in range(n_s):
        if erased_mask[i]:
            received[i * m:(i + 1) * m] = 0

    recovered = diversity_demapper(received, G_FIX, erased_mask, m)
    if recovered is not None:
        match = np.array_equal(recovered, info_bits)
        print(f"  Recovered bits: {recovered.tolist()}")
        print(f"  Match: {'YES' if match else 'NO'}")
    else:
        print(f"  Recovery FAILED (rank insufficient)")

    # Test with 3 erasures (should fail for d_min=3)
    print(f"\n  Testing with 3 erasures (exceeds max_erasures={n_s - k_c}):")
    erased_3 = np.array([True, True, False, True, False, False])
    recovered_3 = diversity_demapper(received, G_FIX, erased_3, m)
    if recovered_3 is None:
        print(f"  Recovery correctly FAILED (3 erasures > max_erasures=2)")
    else:
        match_3 = np.array_equal(recovered_3, info_bits)
        print(f"  Recovered (unexpected): match={'YES' if match_3 else 'NO'}")

    print("\n" + "=" * 72)
    print("DiversityTransformConfig integration test")
    print("=" * 72)

    config = DiversityTransformConfig(k_c=4, n_s=6, m=3)
    print(f"  rho={config.rho}, n_out={config.n_out}, "
          f"max_erasures={config.max_erasures}")

    all_pass, _ = config.verify()
    print(f"  MDS verification: {'PASS' if all_pass else 'FAIL'}")

    can = config.can_recover(np.array([False, True, False, False, True, False]))
    print(f"  Can recover with 2 erasures: {can}")

    can3 = config.can_recover(np.array([True, True, False, True, False, False]))
    print(f"  Can recover with 3 erasures: {can3}")

    print("\nDone.")
