
import whisper
import time

start_time = time.time()


# "base", "small", "medium", "large" options
model = whisper.load_model("large")

path = "path/to/your/audio/file.wav"  # path to audio file

result = model.transcribe(path, fp16=False)


with open("transcription.txt", "w", encoding="utf-8") as file:
    file.write(result["text"])

print("Text is written in 'transcription.txt'.")

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Processing time: {elapsed_time:.2f} seconds")
