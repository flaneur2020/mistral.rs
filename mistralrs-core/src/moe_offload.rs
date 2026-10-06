//! External routed-experts providers.
//!
//! A [`RoutedExpertsProvider`] installed before a model is loaded can take over the routed
//! experts of supported MoE blocks. The model keeps its router and shared expert; the provider
//! decides where the routed experts live and how they run (for example in host memory, or split
//! between a device-resident cache and the CPU). Blocks it declines load as usual.
//!
//! Supported models: Qwen4-Exp (decoder layers and the MTP head).
//!
//! Notes for providers:
//! - Taken-over experts are never loaded by the model, so automatic device mapping still counts
//!   their bytes; pass an explicit device-layer map when the remaining weights fit on device.
//! - Models with externally owned experts disable CUDA decode graph capture.

use std::sync::{Arc, OnceLock};

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{Comm, QuantizedConfig, ShardedVarBuilder};

use crate::layers::Activation;
use crate::moe::{MoEExperts, MoEExpertsConfig};

/// Which MoE block is being loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSite {
    /// Decoder layer index; for the MTP head, the layer index the model assigns to it.
    pub layer: usize,
    /// Whether the block belongs to the MTP draft head rather than a decoder layer.
    pub is_mtp: bool,
}

/// Everything the model would pass to [`MoEExperts::new`] for one block.
pub struct ExpertLoadContext<'a> {
    pub site: ExpertSite,
    pub config: &'a MoEExpertsConfig,
    /// The block's builder; the experts live under `vb.pp("experts")`.
    pub vb: ShardedVarBuilder,
    /// The device holding the rest of the block (router, shared expert, activations).
    pub layer_device: &'a Device,
    pub comm: &'a Arc<Comm>,
    pub loading_isq: bool,
    pub quantization_config: &'a Option<QuantizedConfig>,
    pub act: Activation,
}

impl ExpertLoadContext<'_> {
    /// Build the model's own experts layer for this block on `device`.
    pub fn load_builtin(&self, device: Device) -> Result<MoEExperts> {
        MoEExperts::new(
            self.config,
            self.vb.clone(),
            device,
            self.comm,
            self.loading_isq,
            self.quantization_config,
            self.act,
        )
    }
}

/// Routed experts of one MoE block.
pub trait RoutedExperts: Send + Sync {
    /// `xs: [batch, seq, hidden]` on the layer device; `topk_weights` / `topk_ids`:
    /// `[tokens, top_k]` from the model's router. Returns `[tokens, hidden]` on the layer device.
    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor>;
}

impl RoutedExperts for MoEExperts {
    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        MoEExperts::forward(self, xs, topk_weights, topk_ids)
    }
}

/// Supplies routed experts for MoE blocks at load time.
pub trait RoutedExpertsProvider: Send + Sync {
    /// Return `Some` to own this block's routed experts, or `None` to let the model load them.
    fn build(&self, ctx: ExpertLoadContext<'_>) -> Result<Option<Arc<dyn RoutedExperts>>>;
}

static PROVIDER: OnceLock<Arc<dyn RoutedExpertsProvider>> = OnceLock::new();

/// Install the process-wide provider. Must run before the model loads; at most once.
pub fn install_routed_experts_provider(provider: Arc<dyn RoutedExpertsProvider>) -> Result<()> {
    PROVIDER
        .set(provider)
        .map_err(|_| candle_core::Error::msg("a routed-experts provider is already installed"))
}

pub(crate) fn routed_experts_provider() -> Option<&'static Arc<dyn RoutedExpertsProvider>> {
    PROVIDER.get()
}
