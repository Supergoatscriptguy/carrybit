"""Fourier measurements on modular addition networks, after Nanda et al. (2023) and
Swaroop (2026). Every weight vector here is a function of a residue mod p, so its DFT
over that axis says which frequency a neuron uses and with what phase."""

import torch


def dominant_component(w: torch.Tensor):
    """For rows w[i] of length p, return the dominant non-constant frequency, its phase
    such that w[i, x] ~ A cos(2 pi k x / p + phase), its amplitude, and the periodicity
    score of Swaroop (2026): the dominant magnitude over the mean non-DC magnitude."""
    p = w.shape[-1]
    spec = torch.fft.rfft(w.float(), dim=-1)[..., 1 : (p - 1) // 2 + 1]
    mag = spec.abs()
    k = mag.argmax(-1)
    top = spec.gather(-1, k[..., None]).squeeze(-1)
    periodicity = mag.amax(-1) / mag.mean(-1)
    return k + 1, torch.angle(top), 2 * top.abs() / p, periodicity


def wrap(theta):
    return torch.remainder(theta + torch.pi, 2 * torch.pi) - torch.pi


def phase_sum_residual(phase_a, phase_b, phase_out):
    """How far each neuron is from the relation phase_out = phase_a + phase_b."""
    return wrap(phase_out - phase_a - phase_b)


def idealized_mlp(p: int, freq, phase_a, phase_b, phase_out, amp_a, amp_b, amp_out, square=True):
    """Rebuild a two-hot MLP from per-neuron Fourier parameters alone. With square=True the
    input weights are square waves, which Swaroop (2026) found matches trained ReLU MLPs."""
    x = torch.arange(p)
    omega = 2 * torch.pi * freq[:, None] / p
    wave = torch.sign if square else (lambda t: t)
    w_a = amp_a[:, None] * wave(torch.cos(omega * x + phase_a[:, None]))
    w_b = amp_b[:, None] * wave(torch.cos(omega * x + phase_b[:, None]))
    w_out = amp_out[:, None] * torch.cos(omega * x + phase_out[:, None])
    return w_a, w_b, w_out


def mlp_accuracy(p: int, w_a, w_b, w_out) -> float:
    """Accuracy of a bias-free two-hot MLP with the given weights on all p*p pairs."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    hidden = torch.relu(w_a[:, a.flatten()] + w_b[:, b.flatten()])
    logits = w_out.T @ hidden
    return (logits.argmax(0) == ((a + b) % p).flatten()).float().mean().item()


def analyze_neurons(p: int, w_a, w_b, w_out, structured_above: float = 12.0) -> dict:
    """Summary statistics for a set of neurons given their p-periodic input weights for
    each operand and their p-dim map to the logits. Rows are neurons."""
    k_a, ph_a, amp_a, per_a = dominant_component(w_a)
    k_b, ph_b, amp_b, per_b = dominant_component(w_b)
    k_o, ph_o, amp_o, per_o = dominant_component(w_out)
    periodicity = torch.minimum(per_a, per_b)
    structured = periodicity > structured_above
    matched = (k_a == k_b) & (k_a == k_o)
    residual = phase_sum_residual(ph_a, ph_b, ph_o)
    use = structured & matched
    alignment = torch.cos(residual[use]).mean().item() if use.any() else float("nan")
    ideal = idealized_mlp(p, k_a, ph_a, ph_b, ph_o, amp_a, amp_b, amp_o)
    return {
        "periodicity": periodicity.mean().item(),
        "structured_frac": structured.float().mean().item(),
        "matched_frac": matched.float().mean().item(),
        "phase_alignment": alignment,
        "idealized_acc": mlp_accuracy(p, *ideal),
        "n_freqs": len(k_a[structured].unique()),
    }
