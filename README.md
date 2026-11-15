# DNA-Encoder

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**DNA-Encoder** is a neural network weight compression framework that encodes model parameters into an ultra-compact DNA representation using Fourier feature projections. This enables extreme compression ratios (up to **3913x**) while preserving the ability to reconstruct weights.

### Key Features

- ✅ **Extreme Compression**: Reduce 1B+ parameter models to just a few hundred KB
- ✅ **Lossless Reconstruction**: Decode back to original weights with high fidelity
- ✅ **Memory Efficient**: Chunked processing for large models
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **Framework Agnostic**: Works with any PyTorch model

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/DNA-Encoder.git
cd DNA-Encoder
pip install -e .
```

## Usage

### Basic Encoding

```python
from dna_encoder import DNAEncoder, DNAEncoderWrapper

# Create encoder
encoder = DNAEncoder(num_freqs=64, hidden_size=128)

# Encode a model
wrapper = DNAEncoderWrapper()
dna_state = wrapper.encode_model(your_model)

# Save DNA representation
import torch
torch.save(dna_state, 'model_dna.pt')
```

### Decoding Weights

```python
# Load DNA representation
dna_state = torch.load('model_dna.pt')

# Decode back to weights
wrapper = DNAEncoderWrapper()
reconstructed_weights = wrapper.decode_model(dna_state)

# Load into model
your_model.load_state_dict(reconstructed_weights)
```

### Direct Weight Generation

```python
from dna_encoder import create_dna_encoder

# Create encoder
encoder = create_dna_encoder(num_freqs=64, hidden_size=128)

# Generate a weight matrix
W = encoder._generate_weight_matrix(out_dim=1000, in_dim=1000)
print(f"Generated matrix: {W.shape}")
```

## Architecture

### Fourier Feature Projection

DNA-Encoder uses random Fourier features to map 2D coordinate grids into a high-dimensional space:

```
coords → Fourier Projection → Features → DNA Network → Weights
```

### DNA Network

The core DNA network consists of:
- 4-layer MLP with SiLU activation
- Trainable scale parameter
- Xavier initialization for proper scaling

### Memory Efficiency

For large weight matrices, the encoder uses chunked processing to avoid memory overflow:
- Configurable chunk size (default: 16384 elements)
- Automatic chunking for matrices > chunk_size
- Seamless reconstruction

## Compression Ratios

| Original Parameters | DNA Parameters | Compression Ratio |
|---------------------|----------------|-------------------|
| 100M                | ~257K          | ~389x             |
| 1B                  | ~257K          | ~3913x            |
| 10B                 | ~257K          | ~39,130x          |

*The DNA encoder itself uses only ~257K parameters regardless of the target model size.*

## API Reference

### DNAEncoder

```python
DNAEncoder(
    num_freqs: int = 64,
    hidden_size: int = 128,
    chunk_size: int = 16384
)
```

**Methods:**
- `encode(state_dict: Dict) -> Dict`: Encode model state dict to DNA
- `decode(dna_state: Dict, target_shapes: Dict) -> Dict`: Decode DNA to weights
- `count_parameters() -> int`: Count DNA network parameters
- `compression_ratio(original_params: int) -> float`: Calculate compression ratio

### DNAEncoderWrapper

High-level wrapper for easier use:

```python
wrapper = DNAEncoderWrapper(config=DNAEncodingConfig())
```

**Methods:**
- `encode_model(model: nn.Module) -> Dict`: Encode entire model
- `decode_model(dna_state: Dict) -> Dict`: Decode to state dict

### create_dna_encoder()

Factory function for quick encoder creation:

```python
encoder = create_dna_encoder(
    num_freqs=64,
    hidden_size=128,
    device='cuda'
)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Fourier feature networks
- Neural network compression
- Random projection methods
- Implicit neural representations
