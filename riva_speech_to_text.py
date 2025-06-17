import argparse
import grpc
import time
try:
    import riva_api.riva_audio_pb2 as ra # RIVA 2.0.0 and above
except:
    import riva_api.audio_pb2 as ra
import riva_api.riva_asr_pb2 as rasr
import riva_api.riva_asr_pb2_grpc as rasr_srv
import wave

audio_file = "<add path to .wav file>"
server = "localhost:50051"

wf = wave.open(audio_file, 'rb')
with open(audio_file, 'rb') as fh:
    data = fh.read()

channel = grpc.insecure_channel(server)
client = rasr_srv.RivaSpeechRecognitionStub(channel)
config = rasr.RecognitionConfig(
    encoding=ra.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=wf.getframerate(),
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=False,
    audio_channel_count=1
)

request = rasr.RecognizeRequest(config=config, audio=data)

response = client.Recognize(request)
print(response)