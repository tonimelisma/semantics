from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

def transcribe_finnish(audio_file):
    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="finnish", task="transcribe")

    # Load audio file
    speech, sampling_rate = sf.read(audio_file)

    # Process the audio file
    input_features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

if __name__ == "__main__":
    audio_file = 'suomi.wav'  # Path to your audio file
    result = transcribe_finnish(audio_file)
    print("Transcription:", result)
