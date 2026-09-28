from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader


def test_qwen3_qk_norm_paths_are_mapped_to_megatron_attention_norms():
    mapping = MegatronDynamicLoader.LAYER_NORM_SUFFIX_TO_MEGATRON

    assert mapping["q_norm.weight"][0] == "self_attention.q_layernorm.weight"
    assert mapping["k_norm.weight"][0] == "self_attention.k_layernorm.weight"
