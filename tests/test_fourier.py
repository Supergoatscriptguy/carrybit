import torch

from carrybit.fourier import analyze_neurons, dominant_component, idealized_mlp, mlp_accuracy, phase_sum_residual

P = 97


def test_dominant_component_recovers_frequency_and_phase():
    x = torch.arange(P)
    freq = torch.tensor([3, 17, 40])
    phase = torch.tensor([0.4, -2.0, 2.5])
    w = 1.5 * torch.cos(2 * torch.pi * freq[:, None] * x / P + phase[:, None])
    k, ph, amp, periodicity = dominant_component(w)
    assert torch.equal(k, freq)
    assert torch.allclose(phase_sum_residual(ph, torch.zeros(3), phase), torch.zeros(3), atol=1e-4)
    assert torch.allclose(amp, torch.full((3,), 1.5), atol=1e-4)
    assert (periodicity > 40).all()


def test_noise_has_low_periodicity():
    _, _, _, periodicity = dominant_component(torch.randn(64, P))
    assert periodicity.mean() < 6


def test_idealized_mlp_with_consistent_phases_adds_mod_p():
    gen = torch.Generator().manual_seed(0)
    n = 256
    freq = torch.randint(1, (P - 1) // 2 + 1, (n,), generator=gen)
    phase_a = torch.rand(n, generator=gen) * 2 * torch.pi
    phase_b = torch.rand(n, generator=gen) * 2 * torch.pi
    ones = torch.ones(n)
    w = idealized_mlp(P, freq, phase_a, phase_b, phase_a + phase_b, ones, ones, ones)
    assert mlp_accuracy(P, *w) > 0.95
    stats = analyze_neurons(P, *w)
    assert stats["matched_frac"] == 1.0 and stats["phase_alignment"] > 0.99


def test_scrambled_output_phases_break_the_algorithm():
    gen = torch.Generator().manual_seed(1)
    n = 256
    freq = torch.randint(1, (P - 1) // 2 + 1, (n,), generator=gen)
    ph = [torch.rand(n, generator=gen) * 2 * torch.pi for _ in range(3)]
    ones = torch.ones(n)
    w = idealized_mlp(P, freq, ph[0], ph[1], ph[2], ones, ones, ones)
    assert mlp_accuracy(P, *w) < 0.1
