import torch
import torch.nn as nn
from src.utils import PositionalEncoding
from config.settings import Config

class NiftyPredictor(nn.Module):
    """
    Transformer Encoder-Decoder model for Nifty index prediction.
    """
    def __init__(self):
        super().__init__()
        
        self.d_model = Config.D_MODEL
        self.num_heads = Config.NUM_HEADS
        self.num_encoder_layers = Config.NUM_ENCODER_LAYERS
        self.num_decoder_layers = Config.NUM_DECODER_LAYERS
        self.dim_feedforward = Config.DIM_FEEDFORWARD
        self.dropout = Config.DROPOUT
        self.input_features_dim = Config.INPUT_FEATURES_DIM
        self.output_features_dim = Config.OUTPUT_FEATURES_DIM
        self.decoder_sequence_length = Config.DECODER_SEQUENCE_LENGTH

        # 1. Initial Linear Projection Layer (Input Embedding for continuous data)
        # Converts the raw input features (e.g., 100) to d_model (512)
        self.input_linear_projection = nn.Linear(self.input_features_dim, self.d_model)

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=max(Config.ENCODER_SEQUENCE_LENGTH, Config.DECODER_SEQUENCE_LENGTH))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True # Input/output tensors are (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True # Input/output tensors are (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)

        # 5. Final Linear Layer (Output Head)
        # Projects the d_model output from the decoder to the desired output features (OHLCV)
        # The output sequence length is multiplied because we predict all features for all future steps
        self.output_linear_projection = nn.Linear(self.d_model, self.output_features_dim)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Nifty Predictor.

        Args:
            src (torch.Tensor): Input sequence to the encoder.
                                Shape: (batch_size, encoder_sequence_length, input_features_dim)
            tgt (torch.Tensor): Input sequence to the decoder (shifted target for training).
                                Shape: (batch_size, decoder_sequence_length, output_features_dim)

        Returns:
            torch.Tensor: Predicted output sequence.
                          Shape: (batch_size, decoder_sequence_length, output_features_dim)
        """
        # 1. Apply initial linear projection to encoder input
        # (batch_size, enc_seq_len, input_features_dim) -> (batch_size, enc_seq_len, d_model)
        src = self.input_linear_projection(src)
        
        # 2. Add positional encoding to encoder input
        src = self.positional_encoding(src)

        # 3. Encoder Forward Pass
        # Encoder output: (batch_size, enc_seq_len, d_model)
        encoder_output = self.transformer_encoder(src)

        # 4. Prepare Decoder Input
        # The decoder input (tgt) needs to be projected to d_model
        # (batch_size, dec_seq_len, output_features_dim) -> (batch_size, dec_seq_len, d_model)
        # Note: In a true sequence-to-sequence setup, the decoder input might be a special
        # <SOS> token followed by previous predicted tokens during inference,
        # or the shifted ground truth during training (teacher forcing).
        # Here, we project the target OHLCV directly to d_model for simplicity in training.
        # For inference, you'd start with an <SOS> equivalent and feed back predictions.
        
        # Create a dummy decoder input for training (e.g., zeros or a learned start token)
        # For simplicity in this training example, we'll use a tensor of zeros for decoder input
        # and let the cross-attention guide the generation.
        # In a more advanced setup, you might have a learned <SOS> token or use the last encoder output.
        
        # A common practice is to use the actual target sequence, but mask future values.
        # For simplicity, let's assume the decoder input is just a sequence of zeros
        # that gets its positional encoding and then relies on cross-attention.
        # This is a simplification for training the structure.
        
        # For a proper sequence-to-sequence decoder, `tgt` would be the shifted actual target,
        # and we'd apply masking. However, PyTorch's TransformerDecoderLayer handles
        # the masked self-attention internally when `tgt_mask` is provided.
        # For this example, we'll use a simple approach for `tgt` input for training.
        
        # The `tgt` tensor passed to forward is the ground truth target sequence for training.
        # We need to project it to d_model first.
        # (batch_size, dec_seq_len, output_features_dim) -> (batch_size, dec_seq_len, d_model)
        
        # For training, the decoder input typically comes from the actual target sequence,
        # shifted by one, and then masked.
        # Let's assume `tgt` already represents the shifted target (OHLCV).
        
        # Project target to d_model for decoder input
        # This is a placeholder for a more sophisticated decoder input.
        # For training, `tgt` is the ground truth OHLCV sequence.
        # We need to convert it to d_model.
        # A common way is to use a linear layer or simply assume it's already in d_model if it's
        # some kind of learned start token.
        
        # For our current setup, the `tgt` in `forward(src, tgt)` is the actual target OHLCV.
        # The decoder needs an input that is also `d_model` dimensional.
        # A simple approach is to use a learned start token or just a linear projection of the target.
        # Let's create a dummy decoder input `decoder_input_tensor` for the decoder's self-attention.
        # This will be `(batch_size, dec_seq_len, d_model)`.
        # For training, a common approach is to use the ground truth target, shifted and masked.
        # PyTorch's TransformerDecoderLayer expects `tgt` to be `d_model` dimensional.
        
        # Simplification: For training, we can pass a tensor of zeros or a learned start token
        # as the actual input to the decoder, and let the cross-attention from the encoder_output
        # guide the prediction. The `tgt` argument in `forward` is actually the *target* for loss calculation.
        
        # Let's define a learned start token for the decoder input.
        # This token will be repeated for the decoder sequence length.
        # This is a common practice for generative decoders.
        
        # For the `nn.TransformerDecoder`, the `tgt` argument is the input to the decoder's self-attention.
        # This `tgt` input needs to be `d_model` dimensional.
        # It's typically a sequence of `<SOS>` tokens followed by the previously generated tokens.
        # During training, it's the `<SOS>` followed by the ground truth target sequence (teacher forcing).
        # We need to create this `decoder_input_tensor` from scratch or a learned token.
        
        # Let's use a simple approach: a learned start token repeated for the sequence length.
        # This `start_token` will be learned.
        # It's a single vector of `d_model` dimensions.
        self.start_token = nn.Parameter(torch.randn(self.d_model))
        
        # Repeat the start token for the decoder sequence length and batch size.
        # (batch_size, dec_seq_len, d_model)
        decoder_input_tensor = self.start_token.unsqueeze(0).unsqueeze(0).repeat(src.size(0), self.decoder_sequence_length, 1)
        
        # Add positional encoding to decoder input
        decoder_input_tensor = self.positional_encoding(decoder_input_tensor)

        # Generate a causal mask for the decoder's self-attention
        # This ensures that predictions for position i can only depend on known outputs at positions less than i.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.decoder_sequence_length).to(src.device)

        # 5. Decoder Forward Pass
        # Decoder output: (batch_size, dec_seq_len, d_model)
        decoder_output = self.transformer_decoder(
            tgt=decoder_input_tensor,
            memory=encoder_output, # Encoder output serves as memory for cross-attention
            tgt_mask=tgt_mask # Apply causal mask to decoder self-attention
        )

        # 6. Final Linear Projection
        # (batch_size, dec_seq_len, d_model) -> (batch_size, dec_seq_len, output_features_dim)
        predictions = self.output_linear_projection(decoder_output)
        
        return predictions
