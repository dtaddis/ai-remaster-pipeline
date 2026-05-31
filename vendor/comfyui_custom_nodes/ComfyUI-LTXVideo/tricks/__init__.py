from .nodes.attn_bank_nodes import (
    LTXAttentionBankNode,
    LTXAttentioOverrideNode,
    LTXPrepareAttnInjectionsNode,
)
from .nodes.attn_override_node import LTXAttnOverrideNode
from .nodes.latent_guide_node import AddLatentGuideNode
from .nodes.ltx_feta_enhance_node import LTXFetaEnhanceNode
from .nodes.ltx_flowedit_nodes import LTXFlowEditCFGGuiderNode, LTXFlowEditSamplerNode
from .nodes.ltx_inverse_model_pred_nodes import (
    LTXForwardModelSamplingPredNode,
    LTXReverseModelSamplingPredNode,
)
from .nodes.ltx_pag_node import LTXPerturbedAttentionNode
from .nodes.modify_ltx_model_node import ModifyLTXModelNode
from .nodes.rectified_sampler_nodes import (
    LTXRFForwardODESamplerNode,
    LTXRFReverseODESamplerNode,
)

NODE_CLASS_MAPPINGS = {
    "ModifyLTXModel": ModifyLTXModelNode,
    "AddLatentGuide": AddLatentGuideNode,
    "LTXForwardModelSamplingPred": LTXForwardModelSamplingPredNode,
    "LTXReverseModelSamplingPred": LTXReverseModelSamplingPredNode,
    "LTXRFForwardODESampler": LTXRFForwardODESamplerNode,
    "LTXRFReverseODESampler": LTXRFReverseODESamplerNode,
    "LTXAttentionBank": LTXAttentionBankNode,
    "LTXPrepareAttnInjections": LTXPrepareAttnInjectionsNode,
    "LTXAttentioOverride": LTXAttentioOverrideNode,
    "LTXPerturbedAttention": LTXPerturbedAttentionNode,
    "LTXAttnOverride": LTXAttnOverrideNode,
    "LTXFlowEditCFGGuider": LTXFlowEditCFGGuiderNode,
    "LTXFlowEditSampler": LTXFlowEditSamplerNode,
    "LTXFetaEnhance": LTXFetaEnhanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModifyLTXModel": "Modify LTX Model",
    "AddLatentGuide": "Add LTX Latent Guide",
    "LTXAddImageGuide": "Add LTX Image Guide",
    "LTXForwardModelSamplingPred": "LTX Forward Model Pred",
    "LTXReverseModelSamplingPred": "LTX Reverse Model Pred",
    "LTXRFForwardODESampler": "LTX Rf-Inv Forward Sampler",
    "LTXRFReverseODESampler": "LTX Rf-Inv Reverse Sampler",
    "LTXAttentionBank": "LTX Attention Bank",
    "LTXPrepareAttnInjections": "LTX Prepare Attn Injection",
    "LTXAttentioOverride": "LTX Attn Block Override",
    "LTXPerturbedAttention": "LTX Apply Perturbed Attention",
    "LTXAttnOverride": "LTX Attention Override",
    "LTXFlowEditCFGGuider": "LTX Flow Edit CFG Guider",
    "LTXFlowEditSampler": "LTX Flow Edit Sampler",
    "LTXFetaEnhance": "LTX Feta Enhance",
}
