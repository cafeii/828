import torch
import torch.nn.functional as F

from lit_gpt.mixers.qr_gdn_rule import qr_gdn_affine_reference, qr_gdn_reference


def inputs(requires_grad=False, T=5):
    torch.manual_seed(3407)
    B, H, K, V = 2, 3, 4, 6
    tensors = [
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, K, dtype=torch.float64),
        torch.randn(B, T, H, V, dtype=torch.float64),
        -torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.85 * torch.rand(B, T, H, dtype=torch.float64),
        -torch.rand(B, T, H, dtype=torch.float64),
        0.05 + 0.85 * torch.rand(B, T, H, dtype=torch.float64),
        torch.randn(B, T, H, dtype=torch.float64),
        torch.randn(B, H, K, V, dtype=torch.float64),
        torch.randn(B, H, K, V, dtype=torch.float64),
    ]
    return tuple(x.requires_grad_(requires_grad) for x in tensors)


def test_joint_closed_form_matches_direct_normal_equations():
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = inputs()
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    eye = torch.eye(q.shape[-1], dtype=q.dtype)
    outputs = []
    current_kv, current_qr = kv, qr
    for t in range(q.shape[1]):
        qt, kt, vt = qn[:, t], kn[:, t], v[:, t]
        old_kv, old_qr = current_kv, current_qr
        recalled = torch.einsum("bhk,bhkv->bhv", qt, old_kv)

        base_kv = g_kv[:, t].exp()[..., None, None] * old_kv
        rho_kv = beta_kv[:, t] / (1 - beta_kv[:, t])
        lhs_kv = eye + rho_kv[..., None, None] * kt[..., :, None] * kt[..., None, :]
        rhs_kv = base_kv + rho_kv[..., None, None] * kt[..., None] * vt[..., None, :]
        current_kv = torch.linalg.solve(lhs_kv, rhs_kv)

        qr_read = torch.einsum("bhk,bhkv->bhv", qt, old_qr)
        output = torch.einsum("bhk,bhkv->bhv", qt, current_kv)
        output = output + read_logit[:, t].tanh()[..., None] * qr_read
        outputs.append(q.shape[-1] ** -0.5 * output)

        base_qr = g_qr[:, t].exp()[..., None, None] * old_qr
        rho_qr = beta_qr[:, t] / (1 - beta_qr[:, t])
        lhs_qr = eye + rho_qr[..., None, None] * qt[..., :, None] * qt[..., None, :]
        rhs_qr = base_qr + rho_qr[..., None, None] * qt[..., None] * recalled[..., None, :]
        current_qr = torch.linalg.solve(lhs_qr, rhs_qr)

    actual_outputs, (actual_kv, actual_qr) = qr_gdn_reference(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state=(kv, qr)
    )
    torch.testing.assert_close(actual_outputs, torch.stack(outputs, dim=1), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(actual_kv, current_kv, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(actual_qr, current_qr, rtol=2e-11, atol=2e-11)


def test_block_affine_matches_explicit_outputs_states_and_gradients():
    args = inputs(requires_grad=True)
    data, state = args[:8], args[8:]
    out_a, states_a = qr_gdn_reference(*data, initial_state=state)
    objective_a = out_a.square().sum() + sum(x.square().sum() for x in states_a)
    grads_a = torch.autograd.grad(objective_a, args)

    cloned = tuple(x.detach().clone().requires_grad_() for x in args)
    data_b, state_b = cloned[:8], cloned[8:]
    out_b, states_b = qr_gdn_affine_reference(*data_b, initial_state=state_b)
    objective_b = out_b.square().sum() + sum(x.square().sum() for x in states_b)
    grads_b = torch.autograd.grad(objective_b, cloned)

    torch.testing.assert_close(out_a, out_b, rtol=2e-11, atol=2e-11)
    for actual, expected in zip(states_a, states_b):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    for actual, expected in zip(grads_a, grads_b):
        torch.testing.assert_close(actual, expected, rtol=4e-10, atol=4e-10)


def test_qr_disabled_is_exact_native_gdn():
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, _, kv, qr = inputs()
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    outputs, current = [], kv
    for t in range(q.shape[1]):
        current = g_kv[:, t].exp()[..., None, None] * current
        erased = torch.einsum(
            "bhk,bhkv->bhv", beta_kv[:, t, :, None] * kn[:, t], current
        )
        current = current - kn[:, t, :, :, None] * erased[..., None, :]
        current = current + kn[:, t, :, :, None] * (
            beta_kv[:, t, :, None] * v[:, t]
        )[..., None, :]
        outputs.append(q.shape[-1] ** -0.5 * torch.einsum("bhk,bhkv->bhv", qn[:, t], current))

    actual, (actual_kv, actual_qr) = qr_gdn_reference(
        q,
        k,
        v,
        g_kv,
        beta_kv,
        g_qr,
        torch.zeros_like(beta_qr),
        torch.zeros_like(beta_qr),
        initial_state=(kv, qr),
    )
    torch.testing.assert_close(actual, torch.stack(outputs, dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_kv, current, rtol=1e-12, atol=1e-12)
    expected_qr = qr
    for t in range(q.shape[1]):
        expected_qr = g_qr[:, t].exp()[..., None, None] * expected_qr
    torch.testing.assert_close(actual_qr, expected_qr, rtol=1e-12, atol=1e-12)


def test_qr_delta_write_contracts_query_error():
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = inputs(T=1)
    qn = F.normalize(q, dim=-1)[:, 0]
    recalled = torch.einsum("bhk,bhkv->bhv", qn, kv)
    base_qr = g_qr[:, 0].exp()[..., None, None] * qr
    before = recalled - torch.einsum("bhk,bhkv->bhv", qn, base_qr)
    _, (_, final_qr) = qr_gdn_reference(
        q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, initial_state=(kv, qr)
    )
    after = recalled - torch.einsum("bhk,bhkv->bhv", qn, final_qr)
    torch.testing.assert_close(after, (1 - beta_qr[:, 0])[..., None] * before, rtol=2e-12, atol=2e-12)


def test_current_output_has_no_qr_self_echo_but_future_can_use_write():
    q, k, v, g_kv, beta_kv, g_qr, beta_qr, read_logit, kv, qr = inputs(T=2)
    qr = torch.zeros_like(qr)
    read_logit = read_logit.detach().clone()
    read_logit[:, 0] = 3.0
    read_logit[:, 1] = 3.0
    low = beta_qr.detach().clone()
    high = beta_qr.detach().clone()
    low[:, 0] = 0.0
    high[:, 0] = 0.9
    low[:, 1] = high[:, 1]
    out_low, _ = qr_gdn_reference(
        q, k, v, g_kv, beta_kv, g_qr, low, read_logit, initial_state=(kv, qr)
    )
    out_high, _ = qr_gdn_reference(
        q, k, v, g_kv, beta_kv, g_qr, high, read_logit, initial_state=(kv, qr)
    )
    torch.testing.assert_close(out_low[:, 0], out_high[:, 0], rtol=0, atol=0)
    assert not torch.allclose(out_low[:, 1], out_high[:, 1])


def test_split_sequence_continuation_matches_single_pass():
    args = inputs(T=7)
    data, state = args[:8], args[8:]
    full_output, full_state = qr_gdn_reference(*data, initial_state=state)
    split = 3
    first_data = tuple(x[:, :split] for x in data)
    second_data = tuple(x[:, split:] for x in data)
    first_output, middle_state = qr_gdn_reference(*first_data, initial_state=state)
    second_output, final_state = qr_gdn_reference(*second_data, initial_state=middle_state)
    torch.testing.assert_close(torch.cat((first_output, second_output), dim=1), full_output, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(final_state, full_state):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
