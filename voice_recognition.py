# Class for voice recognition functionality
class VoiceRecognition:
    @staticmethod
    def record_audio(duration=5, fs=44100, device_index=1):
        try:
            st.info("Recording... Speak now!")
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64', device=device_index)
            sd.wait()
            st.success("Recording complete!")
            return recording.flatten()
        except Exception as e:
            st.error(f"Recording failed: {e}")
            return None

    @staticmethod
    def audio_to_text(audio_data, fs=44100):
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sd.write(temp_audio.name, audio_data, fs)
            with sr.AudioFile(temp_audio.name) as source:
                audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I could not understand the audio."
            except sr.RequestError:
                return "Error with the Speech Recognition service."

    @staticmethod
    def text_to_speech(text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            tts.save(temp_audio.name)
            st.audio(temp_audio.name, format="audio/mp3")