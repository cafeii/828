import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def inputs(dtype=torch.float32, *, requires_grad=False, T=64):
    torch.manual_seed(3407)
    B, H, K, V = 1, 2, 16, 12
    values = [
        torch.randn(B, T, H, K, device="cuda", dtype=dtype),
        torch.randn(B, T, H, K, device="cuda", dtype=dtype),
        torch.randn(B, T, H, V, device="cuda", dtype=dtype),
        -0.3 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        0.05 + 0.9 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        -0.3 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        0.05 + 0.9 * torch.rand(B, T, H, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, device="cuda", dtype=torch.float32),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32),
        torch.randn(B, H, K, V, device="cuda", dtype=torch.float32),
    ]
    if requires_grad:
        values = [x.requires_grad_() for x in values]
    return tuple(values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parallel_forward_and_final_state_match_reference(dtype):
    from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel
    from lit_gpt.mixers.qr_gdn_rule import qr_gdn_reference

    args = inputs(dtype)
    data, state = args[:8], args[8:]
    expected = qr_gdn_reference(*data, initial_state=state)
    actual = qr_gdn_parallel(*data, initial_state=state, output_final_state=True)
    tolerance = 8e-4 if dtype == torch.float32 else 8e-2
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=tolerance, atol=tolerance)
    for value, reference in zip(actual[1], expected[1]):
        torch.testing.assert_close(value.float(), reference.float(), rtol=tolerance, atol=tolerance)


def test_parallel_backward_matches_reference():
    from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel
    from lit_gpt.mixers.qr_gdn_rule import qr_gdn_reference

    args = inputs(requires_grad=True)
    data, state = args[:8], args[8:]
    expected = qr_gdn_reference(*data, initial_state=state)
    weights = [torch.randn_like(expected[0]), *(torch.randn_like(x) for x in expected[1])]
    loss = (expected[0] * weights[0]).sum() + sum((x * w).sum() for x, w in zip(expected[1], weights[1:]))
    expected_grads = torch.autograd.grad(loss, args)

    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    actual = qr_gdn_parallel(*cloned[:8], initial_state=cloned[8:], output_final_state=True)
    loss = (actual[0] * weights[0]).sum() + sum((x * w).sum() for x, w in zip(actual[1], weights[1:]))
    actual_grads = torch.autograd.grad(loss, cloned)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value.float(), reference.float(), rtol=2e-2, atol=2e-2)


def test_zero_qr_read_is_exact_native_gdn_output():
    from lit_gpt.kernels import get_chunk_gated_delta_rule
    from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel

    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = inputs()
    read_logit = torch.zeros_like(read_logit)
    expected, _ = get_chunk_gated_delta_rule()(
        q=q, k=k, v=v, g=g_kv, beta=beta_kv, initial_state=kv,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    actual, _ = qr_gdn_parallel(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit,
        initial_state=(kv, qr), output_final_state=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_chunk_path_matches_naive_and_backpropagates():
    from copy import deepcopy

    from lit_gpt.mixers.qr_gdn import QueryRecallGatedDeltaNet

    torch.manual_seed(2026)
    kwargs = dict(
        hidden_size=64,
        num_heads=4,
        num_groups=4,
        head_dim=16,
        use_short_conv=False,
    )
    chunk = QueryRecallGatedDeltaNet(**kwargs, mode="chunk").cuda().train()
    with torch.no_grad():
        chunk.qr_read_proj.bias.fill_(0.2)
    naive = deepcopy(chunk)
    naive.mode = "naive"

    hidden = torch.randn(1, 64, 64, device="cuda", requires_grad=True)
    expected = naive(hidden.detach().clone().requires_grad_())[0]
    actual = chunk(hidden)[0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-3, atol=3e-3)

    actual.square().mean().backward()
    gradients = [parameter.grad for parameter in chunk.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert chunk.qr_read_proj.weight.grad is not None
    assert torch.count_nonzero(chunk.qr_read_proj.weight.grad) > 0



def test_parallel_split_continuation_matches_single_pass():
    from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel

    args = inputs(T=128)
    data, state = args[:8], args[8:]
    full_output, full_state = qr_gdn_parallel(
        *data, initial_state=state, output_final_state=True
    )
    first_data = tuple(value[:, :64] for value in data)
    second_data = tuple(value[:, 64:] for value in data)
    first_output, middle_state = qr_gdn_parallel(
        *first_data, initial_state=state, output_final_state=True
    )
    second_output, split_state = qr_gdn_parallel(
        *second_data, initial_state=middle_state, output_final_state=True
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=1), full_output, rtol=1e-3, atol=1e-3
    )
    for value, expected in zip(split_state, full_state):
        torch.testing.assert_close(value, expected, rtol=1e-3, atol=1e-3)


def test_parallel_extreme_gates_stay_finite_and_match_reference():
    from lit_gpt.mixers.qr_gdn_parallel import qr_gdn_parallel
    from lit_gpt.mixers.qr_gdn_rule import qr_gdn_reference

    raw = list(inputs(T=64))
    positions = torch.linspace(0, 1, 64, device="cuda")[None, :, None]
    raw[3] = (-(1e-4 + positions * 15.0)).expand_as(raw[3])
    raw[4] = (1e-4 + positions * (0.9998)).expand_as(raw[4])
    raw[5] = (-(15.0 - positions * (15.0 - 1e-4))).expand_as(raw[5])
    raw[6] = (0.9999 - positions * 0.9998).expand_as(raw[6])
    raw[7] = ((positions * 2 - 1) * 5.0).expand_as(raw[7])
    args = tuple(value.detach().clone().requires_grad_() for value in raw)
    data, state = args[:8], args[8:]
    expected = qr_gdn_reference(*data, initial_state=state)
    actual = qr_gdn_parallel(*data, initial_state=state, output_final_state=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=3e-2)
    for value, reference in zip(actual[1], expected[1]):
        torch.testing.assert_close(value, reference, rtol=3e-2, atol=3e-2)
    loss = actual[0].square().mean() + sum(value.square().mean() for value in actual[1])
    gradients = torch.autograd.grad(loss, args)
    assert all(torch.isfinite(value).all() for value in (*actual[1], *gradients))


def test_model_chunk_bf16_autocast_backpropagates():
    from lit_gpt.mixers.qr_gdn import QueryRecallGatedDeltaNet

    torch.manual_seed(314)
    model = QueryRecallGatedDeltaNet(
        hidden_size=64,
        num_heads=4,
        num_groups=4,
        head_dim=16,
        use_short_conv=False,
        mode="chunk",
    ).cuda().train()
    hidden = torch.randn(1, 64, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(hidden)[0]
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
