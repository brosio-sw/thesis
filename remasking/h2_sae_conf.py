"""
H2 SAE reconstructor for computing residual norms from public SAEs.

Loads public Top-K SAE checkpoints from AwesomeInterpretability/llada-mask-topk-sae
and provides methods to compute reconstruction residuals.
"""

from __future__ import annotations

from typing import Dict
import torch
from huggingface_hub import hf_hub_download


class LayerwiseTopKSAEReconstructor:
    """Load and use public Top-K SAEs to compute residual norms per layer."""

    def __init__(
        self,
        repo_id: str = "AwesomeInterpretability/llada-mask-topk-sae",
        layer_ids: list[int] | None = None,
        trainer_idx: int = 0,
        device: str = "cpu",
    ):
        """
        Initialize SAE reconstructor.

        Args:
            repo_id: HuggingFace repo ID containing SAE checkpoints
            layer_ids: List of layer IDs to load SAEs for
            trainer_idx: SAE trainer index (for models with multiple trainers)
            device: Device to load SAEs onto
        """
        self.repo_id = repo_id
        self.layer_ids = list(layer_ids or [])
        self.trainer_idx = trainer_idx
        self.device = device
        self.saes: Dict[int, Dict] = {}  # layer -> SAE state/params
        self._load_saes()

    def _load_saes(self) -> None:
        """Load SAE checkpoints from HuggingFace hub."""
        for layer_id in self.layer_ids:
            try:
                # SAE files are organized as resid_post_layer_<id>/trainer_<idx>/ae.pt
                filename = f"resid_post_layer_{layer_id}/trainer_{self.trainer_idx}/ae.pt"
                
                # Download from HuggingFace hub
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    repo_type="model",
                )
                
                # Load the checkpoint
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                self.saes[layer_id] = checkpoint
                
            except Exception as e:
                print(f"[sae] Warning: Failed to load SAE for layer {layer_id}: {e}")

    @torch.no_grad()
    def residual_norm(self, layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute residual norm ||h - h_recon|| for given hidden states.

        Args:
            layer_id: Layer ID to get SAE for
            hidden_states: Hidden states tensor [B, L, D]

        Returns:
            Residual norms [B, L] (one norm per token per batch)
        """
        if layer_id not in self.saes:
            # If SAE not available, return zeros (all on-manifold)
            B, L, D = hidden_states.shape
            return torch.zeros(B, L, device=hidden_states.device, dtype=hidden_states.dtype)

        sae = self.saes[layer_id]
        
        try:
            # Extract encoder/decoder weights from checkpoint
            encoder_weight = sae.get("encoder.weight")
            encoder_bias = sae.get("encoder.bias")
            decoder_weight = sae.get("decoder.weight")
            b_dec = sae.get("b_dec")
            
            if encoder_weight is None or decoder_weight is None:
                B, L, D = hidden_states.shape
                return torch.zeros(B, L, device=hidden_states.device, dtype=hidden_states.dtype)
            
            # Keep SAE math on CPU to avoid GPU OOM from large SAE matrices.
            work_dtype = torch.float32
            cpu_device = torch.device("cpu")
            encoder_weight = encoder_weight.to(cpu_device, dtype=work_dtype)
            decoder_weight = decoder_weight.to(cpu_device, dtype=work_dtype)
            if encoder_bias is not None:
                encoder_bias = encoder_bias.to(cpu_device, dtype=work_dtype)
            if b_dec is not None:
                b_dec = b_dec.to(cpu_device, dtype=work_dtype)
            
            # Encode: h -> z
            # z = h @ encoder_weight.T + encoder_bias, then Top-K activation
            original_shape = hidden_states.shape
            h_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(cpu_device, dtype=work_dtype)  # [B*L, D]
            
            z = torch.matmul(h_flat, encoder_weight.t())  # [B*L, dict_size]
            if encoder_bias is not None:
                z = z + encoder_bias
            
            # Top-K activation (sparsity)
            z = torch.relu(z)
            k = sae.get("k")
            if k is not None and isinstance(k, torch.Tensor):
                k = k.item()
            if k is None:
                k = 50  # default
            
            # Apply Top-K: keep only top k activations
            if k > 0 and z.shape[-1] > k:
                topk_vals, topk_indices = torch.topk(z, k, dim=-1)
                z_topk = torch.zeros_like(z)
                z_topk.scatter_(-1, topk_indices, topk_vals)
                z = z_topk
            
            # Decode: z -> h_recon
            # h_recon = z @ decoder_weight.T + b_dec
            h_recon = torch.matmul(z, decoder_weight.t())  # [B*L, D]
            if b_dec is not None:
                h_recon = h_recon + b_dec
            
            # Compute residual norm
            residual = h_flat - h_recon
            norms = torch.norm(residual, p=2, dim=-1)  # [B*L]
            norms = norms.reshape(original_shape[:-1])  # [B, L]
            
            return norms.to(hidden_states.device, dtype=hidden_states.dtype)
                
        except Exception as e:
            print(f"[sae] Warning: Failed to compute residual for layer {layer_id}: {e}")
            import traceback
            traceback.print_exc()
            B, L, D = hidden_states.shape
            return torch.zeros(B, L, device=hidden_states.device, dtype=hidden_states.dtype)
