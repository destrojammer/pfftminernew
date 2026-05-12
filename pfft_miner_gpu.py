#!/usr/bin/env python3
"""
PFFT Miner Bot — GPU Accelerated (CUDA/pycuda)
Ethereum Mainnet | Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB

=== INSTALL (Vast.ai / Ubuntu + NVIDIA GPU) ===
    pip install pycuda web3 pycryptodome numpy
    (CUDA toolkit biasanya sudah ter-install di Vast.ai GPU instance)

=== RUN ===
    python3 pfft_miner_gpu.py

=== TUNING (opsional) ===
    GPU_BLOCKS=8192  python3 pfft_miner_gpu.py   # RTX 3090 / A100 → lebih besar
    GPU_BLOCKS=16384 python3 pfft_miner_gpu.py   # A100 / H100
    GPU_THREADS=512  python3 pfft_miner_gpu.py   # coba threads per block lebih besar
    ETH_RPC=https://... python3 pfft_miner_gpu.py

=== PERFORMA EKSPEKTASI ===
    CPU Python (original) : ~175k H/s
    CPU multiproc fallback: ~800k–2M H/s (tergantung core)
    RTX 3090 GPU          : ~200–500 MH/s   (~1000–2800× lebih cepat!)
    A100 GPU              : ~600–900 MH/s
"""

import os, sys, json, time, struct, signal, ctypes
from pathlib import Path
import multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CONTRACT    = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB"
CHAIN_ID    = 1
RPC         = os.environ.get("ETH_RPC", "https://ethereum-rpc.publicnode.com")
WALLET_FILE = os.environ.get("PFFT_WALLET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wallet.json"))
GAS_LIMIT            = 200_000
PAUSE_BETWEEN_ROUNDS = 5

# GPU tuning — sesuaikan dengan GPU di Vast.ai
GPU_THREADS = int(os.environ.get("GPU_THREADS", "256"))   # threads per block
GPU_BLOCKS  = int(os.environ.get("GPU_BLOCKS",  "4096"))  # blocks per grid


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Kernel — Keccak-256 PoW solver
# ─────────────────────────────────────────────────────────────────────────────
_CUDA_SRC = r"""
#include <stdint.h>

/* ── Helpers ── */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    /* Byte-swap 64-bit via two 32-bit __byte_perm calls */
    uint32_t lo = __byte_perm((uint32_t)(x),        0u, 0x0123u);
    uint32_t hi = __byte_perm((uint32_t)(x >> 32),  0u, 0x0123u);
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

/* ── Keccak-f[1600] permutation ── */
__device__ void keccakf1600(uint64_t s[25])
{
    /* Round constants */
    const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808AULL, 0x8000000080008000ULL,
        0x000000000000808BULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008AULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000AULL,
        0x000000008000808BULL, 0x800000000000008BULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800AULL, 0x800000008000000AULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };
    /* Rho rotation offsets */
    const int ROTC[24] = {
         1,  3,  6, 10, 15, 21, 28, 36,
        45, 55,  2, 14, 27, 41, 56,  8,
        25, 43, 62, 18, 39, 61, 20, 44
    };
    /* Pi lane permutation indices */
    const int PILN[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,
         8, 21, 24,  4, 15, 23, 19, 13,
        12,  2, 20, 14, 22,  9,  6,  1
    };

    uint64_t t, bc[5];

    #pragma unroll
    for (int r = 0; r < 24; r++) {
        /* Theta */
        bc[0] = s[0]^s[5]^s[10]^s[15]^s[20];
        bc[1] = s[1]^s[6]^s[11]^s[16]^s[21];
        bc[2] = s[2]^s[7]^s[12]^s[17]^s[22];
        bc[3] = s[3]^s[8]^s[13]^s[18]^s[23];
        bc[4] = s[4]^s[9]^s[14]^s[19]^s[24];

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ ROTL64(bc[(i+1)%5], 1);
            s[i]   ^= t; s[i+5] ^= t; s[i+10] ^= t;
            s[i+15]^= t; s[i+20]^= t;
        }

        /* Rho + Pi */
        t = s[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = s[j];
            s[j]  = ROTL64(t, ROTC[i]);
            t     = bc[0];
        }

        /* Chi */
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            bc[0]=s[j]; bc[1]=s[j+1]; bc[2]=s[j+2]; bc[3]=s[j+3]; bc[4]=s[j+4];
            s[j]  ^= (~bc[1]) & bc[2]; s[j+1]^= (~bc[2]) & bc[3];
            s[j+2]^= (~bc[3]) & bc[4]; s[j+3]^= (~bc[4]) & bc[0];
            s[j+4]^= (~bc[0]) & bc[1];
        }

        /* Iota */
        s[0] ^= RC[r];
    }
}

/* ── Main PoW search kernel ── */
__global__ void pow_search(
    uint64_t ch0, uint64_t ch1, uint64_t ch2, uint64_t ch3, /* challenge 32B */
    unsigned long long base_nonce,
    uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3,     /* target 4x BE u64 */
    unsigned long long *out_nonce,
    int *found
)
{
    /* Early exit if another thread already found a nonce */
    if (*found) return;

    uint64_t nonce = (uint64_t)base_nonce
                   + (uint64_t)blockIdx.x * blockDim.x
                   + (uint64_t)threadIdx.x;

    /*
     * Build keccak-256 absorb state for:
     *   input = challenge(32B) || zeros(24B) || nonce_big_endian(8B)
     *
     * Rate = 136 bytes = 17 x uint64 (LE words).
     * Words 0-3  : challenge loaded as LE uint64
     * Words 4-6  : 0 (padding zeros)
     * Word  7    : nonce packed big-endian → bswap64(nonce) in LE
     * Word  8    : 0x01 padding at byte 64
     * Words 9-15 : 0
     * Word  16   : 0x8000000000000000 (end pad at byte 135)
     * Words 17-24: 0 (capacity, untouched)
     */
    uint64_t s[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = 0ULL;

    s[0]  = ch0;
    s[1]  = ch1;
    s[2]  = ch2;
    s[3]  = ch3;
    s[7]  = bswap64(nonce);               /* nonce as big-endian bytes 56-63 */
    s[8]  = 0x0000000000000001ULL;        /* keccak 0x01 pad at byte 64      */
    s[16] = 0x8000000000000000ULL;        /* keccak 0x80 pad at byte 135     */

    keccakf1600(s);

    /* Read output as 256-bit big-endian number (4 x LE word → bswap each) */
    uint64_t h0 = bswap64(s[0]);
    uint64_t h1 = bswap64(s[1]);
    uint64_t h2 = bswap64(s[2]);
    uint64_t h3 = bswap64(s[3]);

    /* Check: hash (as BE 256-bit) <= target */
    int hit = (h0 < t0)
           || (h0 == t0 && h1 < t1)
           || (h0 == t0 && h1 == t1 && h2 < t2)
           || (h0 == t0 && h1 == t1 && h2 == t2 && h3 <= t3);

    if (hit) {
        /* Atomic: only first winner writes result */
        if (atomicCAS(found, 0, 1) == 0) {
            *out_nonce = (unsigned long long)nonce;
        }
    }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# CPU keccak256 (untuk verifikasi on-chain & fallback)
# ─────────────────────────────────────────────────────────────────────────────
from Crypto.Hash import keccak as _keccak_mod

def keccak256(data: bytes) -> bytes:
    return _keccak_mod.new(digest_bits=256, data=data).digest()


# ─────────────────────────────────────────────────────────────────────────────
# GPU Init
# ─────────────────────────────────────────────────────────────────────────────
_gpu_ctx  = None  # pycuda context (kept alive)
_gpu_func = None
_cuda_mod = None   # pycuda.driver module ref

def init_gpu():
    """Compile CUDA kernel. Returns True on success."""
    global _gpu_ctx, _gpu_func, _cuda_mod
    try:
        import pycuda.autoinit               # noqa: F401  — sets up default context
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np  # noqa: F401

        print("  Compiling CUDA kernel...", end=" ", flush=True)
        mod      = SourceModule(_CUDA_SRC, options=["-O3", "--use_fast_math"])
        _gpu_func = mod.get_function("pow_search")
        _cuda_mod = cuda

        dev  = cuda.Device(0)
        name = dev.name()
        vmem = dev.total_memory() // (1024 ** 2)
        sm   = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        print(f"done")
        print(f"  GPU 0 : {name}")
        print(f"          VRAM {vmem} MB | {sm} SMs")
        print(f"          Batch size: {GPU_BLOCKS}×{GPU_THREADS} = {GPU_BLOCKS*GPU_THREADS:,} hashes/call")
        return True
    except Exception as e:
        print(f"⚠️  GPU unavailable: {e}")
        print("    → Falling back to CPU multi-process mining")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GPU PoW Solver
# ─────────────────────────────────────────────────────────────────────────────
def solve_pow_gpu(challenge: bytes, target: int) -> tuple:
    """
    GPU brute-force keccak256 PoW.
    Returns (nonce: int, hash_bytes: None) — hash verified on-chain.
    """
    import numpy as np
    cuda = _cuda_mod

    # Decompose 32-byte challenge into 4 LE uint64 words
    ch0 = int.from_bytes(challenge[0:8],  'little')
    ch1 = int.from_bytes(challenge[8:16], 'little')
    ch2 = int.from_bytes(challenge[16:24],'little')
    ch3 = int.from_bytes(challenge[24:32],'little')

    # Decompose 256-bit target into 4 BE uint64 words
    tb  = target.to_bytes(32, 'big')
    t0  = int.from_bytes(tb[0:8],   'big')
    t1  = int.from_bytes(tb[8:16],  'big')
    t2  = int.from_bytes(tb[16:24], 'big')
    t3  = int.from_bytes(tb[24:32], 'big')

    # Device buffers
    h_zero_i32  = np.zeros(1, dtype=np.int32)
    h_zero_u64  = np.zeros(1, dtype=np.uint64)
    d_found     = cuda.mem_alloc(4)   # int (4 bytes)
    d_result    = cuda.mem_alloc(8)   # unsigned long long (8 bytes)

    batch = GPU_THREADS * GPU_BLOCKS
    base  = 0
    start = time.time()
    last_report = start

    while True:
        # Reset flags
        cuda.memcpy_htod(d_found,  h_zero_i32)
        cuda.memcpy_htod(d_result, h_zero_u64)

        # Launch kernel
        _gpu_func(
            np.uint64(ch0), np.uint64(ch1), np.uint64(ch2), np.uint64(ch3),
            np.uint64(base),
            np.uint64(t0), np.uint64(t1), np.uint64(t2), np.uint64(t3),
            d_result, d_found,
            block=(GPU_THREADS, 1, 1),
            grid=(GPU_BLOCKS, 1)
        )
        cuda.Context.synchronize()

        # Check result
        h_found = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(h_found, d_found)

        if h_found[0]:
            h_nonce = np.empty(1, dtype=np.uint64)
            cuda.memcpy_dtoh(h_nonce, d_result)
            nonce   = int(h_nonce[0])
            elapsed = time.time() - start
            total   = base + batch
            rate    = total / elapsed if elapsed > 0 else 0
            print(f"\n ✅ GPU FOUND  nonce={nonce} | {total:,} hashes | "
                  f"{elapsed:.1f}s | {rate/1e6:.2f} MH/s")
            return nonce, None

        base += batch

        now = time.time()
        if now - last_report >= 5.0:
            elapsed = now - start
            rate    = base / elapsed if elapsed > 0 else 0
            print(f" ⛏️  GPU {base/1e6:.1f}M hashes | {rate/1e6:.2f} MH/s | {elapsed:.0f}s", end='\r')
            last_report = now


# ─────────────────────────────────────────────────────────────────────────────
# CPU Multiprocessing PoW Solver (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _cpu_worker(worker_id: int, num_workers: int,
                challenge: bytes, target: int,
                result_queue: mp.Queue, stop_event):
    """Worker process: tries every num_workers-th nonce starting from worker_id."""
    from Crypto.Hash import keccak as _kmod
    buf   = bytearray(challenge) + bytearray(32)
    nonce = worker_id
    while not stop_event.is_set():
        struct.pack_into('>QQQQ', buf, 32, 0, 0, 0, nonce)
        h     = _kmod.new(digest_bits=256, data=bytes(buf)).digest()
        h_int = int.from_bytes(h, 'big')
        if h_int <= target:
            result_queue.put(nonce)
            return
        nonce += num_workers


def solve_pow_cpu(challenge: bytes, target: int) -> tuple:
    """Multi-process CPU PoW (fallback when GPU unavailable)."""
    num_workers  = mp.cpu_count()
    result_queue = mp.Queue()
    stop_event   = mp.Event()

    procs = [
        mp.Process(target=_cpu_worker,
                   args=(i, num_workers, bytes(challenge), target,
                         result_queue, stop_event),
                   daemon=True)
        for i in range(num_workers)
    ]

    start = time.time()
    print(f" ⛏️  CPU mining on {num_workers} cores...")
    for p in procs: p.start()

    nonce = result_queue.get()   # block until any worker finds it
    stop_event.set()
    for p in procs:
        p.terminate()
        p.join(timeout=2)

    elapsed = time.time() - start
    print(f"\n ✅ CPU FOUND  nonce={nonce} | {elapsed:.1f}s")
    return nonce, None


# ─────────────────────────────────────────────────────────────────────────────
# Generic solver dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def solve_pow(challenge: bytes, target: int, use_gpu: bool) -> tuple:
    if use_gpu:
        return solve_pow_gpu(challenge, target)
    return solve_pow_cpu(challenge, target)


# ─────────────────────────────────────────────────────────────────────────────
# Contract helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
_ABI = [
    {"inputs": [],                                             "name": "currentPowHexZeros", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [],                                             "name": "totalMinted",        "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [],                                             "name": "MAX_SUPPLY",         "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "requested", "type": "uint256"}],    "name": "calculateActualMint","outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "user",      "type": "address"}],    "name": "currentPowChallenge","outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "user",      "type": "address"},
                {"name": "powNonce",  "type": "uint256"}],    "name": "isValidPow",         "outputs": [{"type": "bool"}],   "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "powNonce",  "type": "uint256"}],    "name": "freeMint",            "outputs": [],                   "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "user",      "type": "address"}],    "name": "mintedByAddress",    "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "account",   "type": "address"}],    "name": "balanceOf",          "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
]

def load_contract(w3):
    return w3.eth.contract(
        address=w3.to_checksum_address(CONTRACT),
        abi=_ABI
    )

def get_status(w3, contract, wallet_addr):
    hex_zeros    = contract.functions.currentPowHexZeros().call()
    total_minted = contract.functions.totalMinted().call()
    max_supply   = contract.functions.MAX_SUPPLY().call()
    next_mint    = contract.functions.calculateActualMint(w3.to_wei(1000, 'ether')).call()
    wallet_minted= contract.functions.mintedByAddress(wallet_addr).call()
    wallet_bal   = contract.functions.balanceOf(wallet_addr).call()
    target       = (2**256 - 1) >> (hex_zeros * 4)
    progress     = total_minted * 100 / max_supply if max_supply else 0
    return {
        "hex_zeros":     hex_zeros,
        "difficulty_bits": hex_zeros * 4,
        "total_minted":  total_minted,
        "max_supply":    max_supply,
        "next_mint":     next_mint,
        "wallet_minted": wallet_minted,
        "wallet_bal":    wallet_bal,
        "target":        target,
        "progress":      progress,
    }

def get_challenge(contract, wallet_addr) -> bytes:
    c = contract.functions.currentPowChallenge(wallet_addr).call()
    return c if isinstance(c, bytes) else c.to_bytes(32, 'big')

def submit_mint(w3, wallet, contract, nonce: int) -> bool:
    try:
        fn = contract.functions.freeMint(nonce)
        tx = fn.build_transaction({
            'from':    wallet.address,
            'nonce':   w3.eth.get_transaction_count(wallet.address),
            'chainId': CHAIN_ID,
            'gas':     GAS_LIMIT,
        })
        if 'maxFeePerGas' not in tx and 'maxPriorityFeePerGas' not in tx:
            tx['gasPrice'] = w3.eth.gas_price
        signed   = wallet.sign_transaction(tx)
        tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  📤 TX: https://etherscan.io/tx/0x{tx_hash.hex()}")
        receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status == 1:
            print(f"  ✅ MINT OK  | Block {receipt.blockNumber} | Gas {receipt.gasUsed:,}")
            return True
        print(f"  ❌ REVERTED | Gas {receipt.gasUsed:,}")
        return False
    except Exception as e:
        print(f"  ❌ TX error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Signal handler
# ─────────────────────────────────────────────────────────────────────────────
_running = True
def _handle_signal(sig, frame):
    global _running
    print("\n ⚠️  Stopping miner...")
    _running = False
signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from web3 import Web3
    from eth_account import Account

    print("=" * 65)
    print("  ⛏️  PFFT Miner Bot — GPU Accelerated")
    print(f"  Contract : {CONTRACT}")
    print(f"  RPC      : {RPC}")
    print("=" * 65)

    # ── GPU Init ──────────────────────────────────────────────────
    use_gpu = init_gpu()

    # ── Connect RPC ───────────────────────────────────────────────
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("❌ Cannot connect to RPC")
        sys.exit(1)
    print(f"✅ RPC connected | Block #{w3.eth.block_number:,}")

    # ── Load / create wallet ──────────────────────────────────────
    wallet_path = Path(WALLET_FILE)
    if wallet_path.exists():
        with open(wallet_path) as f:
            wdata = json.load(f)
        pk = wdata.get('private_key_hex') or wdata.get('private_key')
        if not pk.startswith('0x'):
            pk = '0x' + pk
        wallet = Account.from_key(pk)
        print(f"✅ Wallet : {wallet.address}")
    else:
        wallet = Account.create()
        wdata  = {
            "address":         wallet.address,
            "private_key_hex": wallet.key.hex(),
            "created":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note":            "PFFT miner wallet — KEEP SECRET"
        }
        wallet_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wallet_path, 'w') as f:
            json.dump(wdata, f, indent=2)
        os.chmod(wallet_path, 0o600)
        print(f"✅ New wallet : {wallet.address}")
        print(f"   Saved to   : {wallet_path}")

    eth_bal = w3.eth.get_balance(wallet.address) / 1e18
    print(f"💰 ETH balance : {eth_bal:.6f}")
    if eth_bal < 0.00005:
        print("⚠️  Low ETH! Need ~0.00005 ETH for gas.")

    # ── Contract ──────────────────────────────────────────────────
    contract = load_contract(w3)
    s = get_status(w3, contract, wallet.address)
    print(f"\n📊 Contract state:")
    print(f"   Minted      : {s['total_minted']/1e18:,.0f} / {s['max_supply']/1e18:,.0f} PFFT ({s['progress']:.1f}%)")
    print(f"   Next mint   : ~{s['next_mint']/1e18:,.2f} PFFT")
    print(f"   Difficulty  : {s['hex_zeros']} hex zeros ({s['difficulty_bits']}-bit)")
    print(f"   Wallet cap  : {s['wallet_minted']/1e18:,.2f} / 10,000 PFFT")
    print(f"   PFFT balance: {s['wallet_bal']/1e18:,.2f}")

    # ── Mining loop ───────────────────────────────────────────────
    round_num         = 0
    total_mints       = 0
    total_pfft_earned = 0.0
    global_start      = time.time()

    while _running:
        round_num += 1
        print(f"\n{'─'*65}")
        print(f"  Round #{round_num}")
        print(f"{'─'*65}")

        # Refresh status
        try:
            s = get_status(w3, contract, wallet.address)
            print(f"  Supply : {s['total_minted']/1e18:,.0f} ({s['progress']:.1f}%) | "
                  f"Next: ~{s['next_mint']/1e18:,.2f} PFFT | "
                  f"Diff: {s['difficulty_bits']}-bit")
            if s['total_minted'] >= s['max_supply']:
                print("  🏁 Max supply reached — done!")
                break
            if s['wallet_minted'] >= 10_000 * 1e18:
                print("  🏁 Wallet cap (10,000 PFFT) reached — done!")
                break
        except Exception as e:
            print(f"  ⚠️  Status error: {e}, retry in 15s...")
            time.sleep(15)
            continue

        # Get challenge
        challenge = get_challenge(contract, wallet.address)

        # Solve PoW
        print(f"  ⛏️  Mining {s['difficulty_bits']}-bit PoW...")
        t0 = time.time()
        nonce, _ = solve_pow(challenge, s['target'], use_gpu)
        mining_time = time.time() - t0

        # Verify before submitting (catches stale challenge after long mine)
        try:
            is_valid = contract.functions.isValidPow(wallet.address, nonce).call()
            if not is_valid:
                print("  ⚠️  Nonce invalid on-chain (supply changed?) — re-mining...")
                continue
        except Exception as e:
            print(f"  ⚠️  Verify error: {e} — submitting anyway...")

        # Submit
        success = submit_mint(w3, wallet, contract, nonce)
        if success:
            total_mints += 1
            earned        = s['next_mint'] / 1e18
            total_pfft_earned += earned
            print(f"  💰 +{earned:,.2f} PFFT | "
                  f"Session: {total_pfft_earned:,.2f} PFFT from {total_mints} mints")
            try:
                bal = contract.functions.balanceOf(wallet.address).call()
                print(f"  💰 Wallet balance: {bal/1e18:,.2f} PFFT")
            except:
                pass

        # Session summary
        elapsed = time.time() - global_start
        print(f"  📈 {(time.time()-global_start)/60:.1f} min elapsed total")

        if _running:
            print(f"  ⏳ {PAUSE_BETWEEN_ROUNDS}s cooldown...")
            time.sleep(PAUSE_BETWEEN_ROUNDS)

    # Final summary
    print(f"\n{'='*65}")
    print(f"  Session Summary")
    print(f"  Mints      : {total_mints}")
    print(f"  PFFT earned: {total_pfft_earned:,.2f}")
    print(f"  Runtime    : {(time.time()-global_start)/60:.1f} min")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
