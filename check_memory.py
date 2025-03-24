import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusion.model.builder import get_vae, get_tokenizer_and_text_encoder

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2  # Convert to MB

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VAE
    print("Loading VAE...")
    vae = get_vae("AutoencoderDC", "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", device)
    vae_size = get_model_size(vae)
    print(f"VAE size: {vae_size:.2f} MB")
    
    # Load Text Encoder
    print("\nLoading Text Encoder...")
    tokenizer, text_encoder = get_tokenizer_and_text_encoder("gemma-2-2b-it", device)
    text_encoder_size = get_model_size(text_encoder)
    print(f"Text Encoder size: {text_encoder_size:.2f} MB")
    
    # Total size
    total_size = vae_size + text_encoder_size
    print(f"\nTotal size: {total_size:.2f} MB")

if __name__ == "__main__":
    main() 