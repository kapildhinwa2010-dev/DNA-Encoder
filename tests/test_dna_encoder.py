"""
Tests for DNA Encoder module
"""

import pytest
import torch
import torch.nn as nn
from dna_encoder import (
    DNAEncoder,
    DNAEncoderWrapper,
    DNAEncodingConfig,
    create_dna_encoder
)


class TestDNAEncoderInitialization:
    """Test DNAEncoder initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        encoder = DNAEncoder()
        assert encoder.num_freqs == 64
        assert encoder.hidden_size == 128
        assert encoder.chunk_size == 16384
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        encoder = DNAEncoder(num_freqs=32, hidden_size=64, chunk_size=8192)
        assert encoder.num_freqs == 32
        assert encoder.hidden_size == 64
        assert encoder.chunk_size == 8192
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        encoder = DNAEncoder()
        assert hasattr(encoder, 'B')
        assert isinstance(encoder.B, torch.Tensor)
        assert encoder.B.shape == (64, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        encoder = DNAEncoder(num_freqs=32, hidden_size=64)
        count = encoder.count_parameters()
        # 32*2 input + 4 layers (64, 64, 64, 1) + scale
        expected_params = (
            32 * 2 * 64 + 64 +  # Layer 1
            64 * 64 + 64 +      # Layer 2
            64 * 64 + 64 +      # Layer 3
            64 * 1 + 1 +        # Layer 4
            1                   # Scale
        )
        assert count == expected_params


class TestDNAEncoderWeightGeneration:
    """Test weight generation"""
    
    def test_generate_small_matrix(self):
        """Test generating small weight matrix"""
        encoder = DNAEncoder()
        W = encoder._generate_weight_matrix(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_large_matrix(self):
        """Test generating large weight matrix"""
        encoder = DNAEncoder(chunk_size=1000)
        W = encoder._generate_weight_matrix(100, 100)
        assert W.shape == (100, 100)
    
    def test_generate_matrix_device(self):
        """Test weight generation on different devices"""
        encoder = DNAEncoder()
        W_cpu = encoder._generate_weight_matrix(10, 10, device=torch.device('cpu'))
        assert W_cpu.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        encoder = DNAEncoder()
        W = encoder._generate_weight_matrix(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestDNAEncoderChunkedProcessing:
    """Test chunked processing for large matrices"""
    
    def test_chunked_vs_unchunked(self):
        """Test that chunked and unchunked produce same results"""
        encoder = DNAEncoder(chunk_size=50)
        
        # Small matrix (no chunking needed)
        W_small = encoder._generate_weight_matrix(10, 10)
        
        # Force chunking by setting small chunk size
        encoder_small_chunk = DNAEncoder(chunk_size=5)
        encoder_small_chunk.B.data = encoder.B.data.clone()
        encoder_small_chunk.net.load_state_dict(encoder.net.state_dict())
        encoder_small_chunk.scale.data = encoder.scale.data.clone()
        
        W_chunked = encoder_small_chunk._generate_weight_matrix(10, 10)
        
        # Should be close (may differ slightly due to floating point)
        assert torch.allclose(W_small, W_chunked, rtol=1e-5, atol=1e-5)
    
    def test_large_matrix_chunking(self):
        """Test that large matrices are processed correctly"""
        encoder = DNAEncoder(chunk_size=100)
        W = encoder._generate_weight_matrix(500, 500)
        assert W.shape == (500, 500)
        # Check consistency
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestDNAEncoderEncodeDecode:
    """Test encoding and decoding"""
    
    def test_encode_decode_simple(self):
        """Test basic encode/decode cycle"""
        encoder = DNAEncoder(num_freqs=32, hidden_size=64)
        
        # Create a simple state dict
        state_dict = {
            'weight1': torch.randn(10, 10),
            'bias1': torch.randn(10),
            'weight2': torch.randn(20, 10),
        }
        
        # Encode
        dna_state = encoder.encode(state_dict)
        assert 'B' in dna_state
        assert 'net' in dna_state
        assert 'scale' in dna_state
        assert 'config' in dna_state
    
    def test_decode_shapes(self):
        """Test decoding with shape information"""
        encoder = DNAEncoder(num_freqs=32, hidden_size=64)
        
        target_shapes = {
            'weight1': (10, 10),
            'bias1': (10,),
            'weight2': (20, 10),
        }
        
        # Create dummy DNA state
        dna_state = encoder.encode({})
        
        # Decode
        reconstructed = encoder.decode(dna_state, target_shapes)
        
        assert reconstructed['weight1'].shape == (10, 10)
        assert reconstructed['bias1'].shape == (10,)
        assert reconstructed['weight2'].shape == (20, 10)


class TestDNAEncoderWrapper:
    """Test DNAEncoderWrapper"""
    
    def test_encode_model(self):
        """Test encoding a model"""
        wrapper = DNAEncoderWrapper()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        dna_state = wrapper.encode_model(model)
        assert 'B' in dna_state
        assert 'net' in dna_state
        assert 'shapes' in dna_state
        assert 'model_class' in dna_state
    
    def test_decode_model(self):
        """Test decoding a model"""
        wrapper = DNAEncoderWrapper()
        
        # Create and encode a model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        dna_state = wrapper.encode_model(model)
        reconstructed = wrapper.decode_model(dna_state)
        
        # Check that we get weights back
        assert 'weight' in reconstructed or len(reconstructed) > 0


class TestCreateDNAEncoder:
    """Test create_dna_encoder factory function"""
    
    def test_default_creation(self):
        """Test default encoder creation"""
        encoder = create_dna_encoder()
        assert isinstance(encoder, DNAEncoder)
        assert encoder.num_freqs == 64
        assert encoder.hidden_size == 128
    
    def test_custom_creation(self):
        """Test custom encoder creation"""
        encoder = create_dna_encoder(
            num_freqs=32,
            hidden_size=64,
            chunk_size=8192
        )
        assert encoder.num_freqs == 32
        assert encoder.hidden_size == 64
        assert encoder.chunk_size == 8192


class TestDNAEncodingConfig:
    """Test DNAEncodingConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = DNAEncodingConfig()
        assert config.num_freqs == 64
        assert config.hidden_size == 128
        assert config.chunk_size == 16384
        assert config.dtype == torch.float32
        assert config.device is None
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DNAEncodingConfig(
            num_freqs=32,
            hidden_size=64,
            device='cpu'
        )
        assert config.num_freqs == 32
        assert config.hidden_size == 64
        assert config.device == 'cpu'


class TestCompressionRatio:
    """Test compression ratio calculations"""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        encoder = DNAEncoder(num_freqs=64, hidden_size=128)
        
        # DNA encoder has: 64*2*128 + 128 + 128*128 + 128 + 128*128 + 128 + 128*1 + 1 + 1
        dna_params = encoder.count_parameters()
        
        # For a 1B parameter model
        original_params = 1_000_000_000
        ratio = encoder.compression_ratio(original_params)
        
        expected_ratio = original_params / dna_params
        assert abs(ratio - expected_ratio) < 0.01


class TestGradientFlow:
    """Test gradient flow through encoder"""
    
    def test_gradients_exist(self):
        """Test that gradients flow through encoder"""
        encoder = DNAEncoder()
        
        # Generate weights with gradient tracking
        W = encoder._generate_weight_matrix(10, 10)
        loss = W.sum()
        loss.backward()
        
        # Check that parameters have gradients
        assert encoder.scale.grad is not None
        for param in encoder.net.parameters():
            assert param.grad is not None


class TestDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        encoder = DNAEncoder()
        W = encoder._generate_weight_matrix(10, 10)
        assert W.device == encoder.B.device
        assert W.device == encoder.scale.device
        for param in encoder.net.parameters():
            assert param.device == encoder.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        encoder = DNAEncoder()
        W = encoder._generate_weight_matrix(10, 10)
        assert W.dtype == encoder.B.dtype
        assert W.dtype == encoder.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
