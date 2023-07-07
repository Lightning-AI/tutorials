import torch
import torchaudio

def synthesize_speech(text):
    """Synthesizes speech from the given text."""

    
    tacotron2_model = torch.load("tacotron2_model.pt")

    
    mel_spectrogram = tacotron2_model(text)

    
    vocoder = torchaudio.models.WaveGlow(
        n_mels=80,
        sample_rate=22050,
    )

 
    audio = vocoder.infer(mel_spectrogram)

    torchaudio.save(audio, "speech.wav")

if __name__ == "__main__":
    text = "Hello, world!"

    synthesize_speech(text)