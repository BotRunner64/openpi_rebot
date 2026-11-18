import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip
import openpi.models.pi0_config as _pi0_config
from openpi.shared import nnx_utils


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_rtc_loss_and_prefix_sampling(monkeypatch):
    # Keep the real Gemma, vision, and flow paths small enough to run on CPU.
    decode_variant = siglip.decode_variant
    monkeypatch.setattr(siglip, "decode_variant", lambda _: decode_variant("mu/56"))
    config = _pi0_config.Pi0Config(
        pi05=True,
        rtc_max_delay=3,
        action_horizon=4,
        action_dim=4,
        max_token_len=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    key = jax.random.key(42)
    model = config.create(key)
    obs, actions = config.fake_obs(4), config.fake_act(4)
    delay = jax.random.randint(jax.random.fold_in(key, 3), (4,), 0, 4)
    postfix = jnp.arange(4) >= delay[:, None]
    assert 0 < postfix.sum() < postfix.size
    graph, state = nnx.split(model)

    def objective(bias, state, obs, actions):
        net = nnx.merge(graph, state)
        net.action_out_proj.bias.value = bias
        losses = net.compute_loss(key, obs, actions, train=True)
        return losses.mean(), losses

    (_, loss), grad = jax.jit(jax.value_and_grad(objective, has_aux=True))(
        model.action_out_proj.bias.value, state, obs, actions
    )
    assert jnp.isfinite(grad).all()
    assert jnp.linalg.norm(grad) > 0
    np.testing.assert_array_equal(loss > 0, postfix)
    assert jnp.isfinite(loss).all()

    sample = nnx_utils.module_jit(model.sample_actions)
    prefix = actions[:, :3]
    chunk = sample(key, obs, num_steps=2, action_prefix=prefix)
    np.testing.assert_array_equal(chunk[:, :3], prefix)
    assert jnp.isfinite(chunk[:, 3:]).all()
    np.testing.assert_allclose(
        sample(key, obs, num_steps=2), sample(key, obs, num_steps=2, action_prefix=prefix[:, :0])
    )
