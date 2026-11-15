# DNA-Encoder

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Encoder/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Encode a PyTorch state dict through Fourier-feature generator networks.

## What it does

DNAEncoder stores, per weight matrix, a small Fourier-feature network
instead of the matrix itself, and decodes back to a state dict. The
stored size is the generator parameters, not the original weights.

## Installation

    pip install torch numpy pytest

## Usage

    from dna_encoder import DNAEncoderWrapper

    wrapper = DNAEncoderWrapper()
    dna_state = wrapper.encode_model(your_model)
    reconstructed = wrapper.decode_model(dna_state)
    your_model.load_state_dict(reconstructed)

## Architecture

- Random Fourier features over a 2D coordinate grid
- 4-layer MLP with SiLU and a trainable scale, per weight matrix
- Chunked generation (default chunk 16384 elements) for large matrices

## API

- DNAEncoder(num_freqs=64, hidden_size=128, chunk_size=16384): encode,
  decode, count_parameters, compression_ratio
- DNAEncoderWrapper: encode_model / decode_model
- create_dna_encoder(...) factory

## Honest results and limitations

- Reconstruction is NOT lossless. The decoder emits a smooth
  low-frequency approximation of each matrix; trained weights are not
  smooth, so expect substantial relative error unless each matrix is
  fitted (trained) individually.
- The ratio table often quoted (3913x for 1B params) is parameter
  arithmetic only: 1e9 / ~257K generator params. No reconstruction-
  quality or task-accuracy measurement backs it.
- Encoding cost: one small network per weight matrix must be optimized
  to be useful at all; without that fitting step the decode is just a
  random smooth field.
- GPU support exists but all published numbers here are CPU.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generated-weight reconstructions
  of trained models were not usable without per-layer training, and that
  noise robustness of the related carrier-mode scheme failed about half
  the trials. The robust design kept in the toolkit is the Reed-Solomon
  slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
