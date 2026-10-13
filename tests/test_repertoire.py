import numpy as np
import torch

from senku.long_horizon.repertoire import (
    ADIM, CH, GDIM, ZDIM, CVAERep, MotifRep, chunk_table, load_rep, rep_input, rep_input_cache,
)
from senku.long_horizon.td_head import td_feats


def _cache(lens, si=0):
    """Build test cache: frames sampled every 5 steps."""
    ep, st = [], []
    for e, ln in enumerate(lens):
        n = len(range(0, ln - 1, 5))
        ep.append(np.full(n, si * 100000 + e))
        st.append(np.arange(n) * 5)
    ep, st = np.concatenate(ep), np.concatenate(st)
    return ep, st, np.full(len(ep), si), np.zeros(len(ep), bool)


def test_chunks_never_cross_episode_end_and_history_clamps():
    lens = [56, 201]
    ep, st, src, held = _cache(lens)
    off = np.array([0, 56])
    A = np.arange(257 * 5, dtype=np.float32).reshape(257, 5) / 1000
    tab = chunk_table(ep, st, src, held, [(off, np.array(lens), A)], np.zeros(5, np.float32), np.ones(5, np.float32))
    j = tab["j"]
    # episode 0 (56 steps): starts 0, 5, ..., 30 (t + 25 <= 55)
    e0 = st[j][ep[j] == 0]
    assert e0.tolist() == [0, 5, 10, 15, 20, 25, 30]
    e1 = st[j][ep[j] == 1]
    assert e1.max() == 175                        # t + 25 <= 200
    # chunk content = the episode's own actions
    k = int(np.nonzero((ep[j] == 1) & (st[j] == 10))[0][0])
    np.testing.assert_allclose(tab["chunk"][k].astype(np.float32), A[56 + 10:56 + 35], rtol=1e-3)
    # history clamps to the episode's first frame, never the previous episode
    first1 = int(np.nonzero(ep == 1)[0][0])
    k0 = int(np.nonzero((ep[j] == 1) & (st[j] == 0))[0][0])
    assert tab["j1"][k0] == first1 and tab["j2"][k0] == first1
    k5 = int(np.nonzero((ep[j] == 1) & (st[j] == 5))[0][0])
    assert tab["j1"][k5] == first1 and tab["j2"][k5] == first1
    k10 = int(np.nonzero((ep[j] == 1) & (st[j] == 10))[0][0])
    assert tab["j1"][k10] == first1 + 1 and tab["j2"][k10] == first1


def test_nan_chunks_dropped():
    lens = [56]
    ep, st, src, held = _cache(lens)
    A = np.zeros((56, 5), np.float32)
    A[12] = np.nan
    tab = chunk_table(ep, st, src, held, [(np.array([0]), np.array(lens), A)], np.zeros(5, np.float32),
                      np.ones(5, np.float32))
    assert st[tab["j"]].tolist() == [15, 20, 25, 30]


def test_rep_input_matches_cache_layout():
    torch.manual_seed(0)
    g = torch.randn(4, 3, GDIM)
    s = torch.randn(4, 49, GDIM)
    x = rep_input(g, s)
    assert x.shape == (4, 5 * GDIM + 2 * GDIM)
    # build the same thing from a "cache" of td_feats rows
    X = torch.stack([td_feats(g[:, k], s) for k in range(3)], 1).reshape(12, -1)   # row 3n+k = frame k of item n
    j = torch.arange(4) * 3 + 2
    y = rep_input_cache(X, j, j - 1, j - 2)
    torch.testing.assert_close(x[:, :5 * GDIM], y[:, :5 * GDIM])
    torch.testing.assert_close(x[:, 5 * GDIM:], y[:, 5 * GDIM:])



def _two_mode_data(n=4096, seed=0):
    """Generate synthetic data: state bit picks one chunk type, choice bit picks another."""
    g = torch.Generator().manual_seed(seed)
    b = torch.randint(0, 2, (n,), generator=g)
    m = torch.randint(0, 2, (n,), generator=g)
    z = torch.randn(n, ZDIM, generator=g) * 0.1
    z[:, 0] = b.float() * 2 - 1
    ch = torch.zeros(n, CH, ADIM)
    ch[:, :, 0] = 1 - 2 * b.float()[:, None]
    ch[:, :, 1] = 1 - 2 * m.float()[:, None]
    return z, ch


def _fit(rep, steps=600):
    z, ch = _two_mode_data()
    rep.set_norm(z)
    opt = torch.optim.Adam(rep.parameters(), 1e-3)
    for i in range(steps):
        idx = torch.randint(0, len(z), (256,))
        loss, _ = rep.loss(z[idx], ch[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if rep.kind == "motif" and i % 100 == 99:
            rep.reset_dead(z[:1024], ch[:1024])
    return rep


def _best_of_n_err(rep, n=8):
    z, ch = _two_mode_data(512, seed=1)
    with torch.no_grad():
        _, prop = rep.propose(z, n, torch.Generator().manual_seed(0))
    return ((prop - ch[:, None]) ** 2).mean((2, 3)).min(1).values.mean().item()


def test_cvae_prior_proposals_follow_state():
    """A unimodal Gaussian prior cannot separate the two within-state choices, so some samples decode between them
    (best-of-8 ~0.25 on this toy); what must hold is that proposals follow the state: conditioning on the other
    state's input must be far worse (wrong dim-0 sign costs ~0.8 per chunk)."""
    torch.manual_seed(0)
    rep = _fit(CVAERep(code=4, hidden=128), steps=1500)
    z, ch = _two_mode_data(512, seed=1)
    z_wrong = z.clone()
    z_wrong[:, 0] = -z_wrong[:, 0]
    with torch.no_grad():
        _, right = rep.propose(z, 8, torch.Generator().manual_seed(0))
        _, wrong = rep.propose(z_wrong, 8, torch.Generator().manual_seed(0))
    e_right = ((right - ch[:, None]) ** 2).mean((2, 3)).min(1).values.mean().item()
    e_wrong = ((wrong - ch[:, None]) ** 2).mean((2, 3)).min(1).values.mean().item()
    assert e_right < 0.3 and e_wrong > 3 * e_right


def test_motif_prior_proposals_follow_state_and_codes_used():
    torch.manual_seed(0)
    rep = _fit(MotifRep(k=8, hidden=128))
    assert _best_of_n_err(rep, n=4) < 0.15
    z, ch = _two_mode_data(1024, seed=2)
    with torch.no_grad():
        used = rep.assign(z, ch).unique().numel()
    assert used >= 4                              # the 4 (state, choice) combinations need distinct codes


def test_save_load_roundtrip(tmp_path):
    for rep in (CVAERep(code=4, hidden=32), MotifRep(k=8, hidden=32)):
        rep.set_norm(torch.randn(10, ZDIM))
        p = tmp_path / f"{rep.kind}.pt"
        torch.save(rep.ckpt(), p)
        r2 = load_rep(str(p), "cpu")
        z = torch.randn(3, ZDIM)
        code = torch.zeros(3, 4) if rep.kind == "cvae" else torch.zeros(3, dtype=torch.long)
        torch.testing.assert_close(rep.decode(z, code), r2.decode(z, code))
