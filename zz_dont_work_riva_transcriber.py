from flask import Flask, request, jsonify
import os
import re
import requests
import time
import yt_dlp

app = Flask(__name__)

class RivaTranscriber:
    def __init__(self, riva_url, api_key, output_dir='/txt'):
        self.riva_url = riva_url
        self.api_key = api_key
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_mp4_360p_link(self, youtube_url):
        """
        Extracts the MP4 360p or closest possible link using yt-dlp.
        """
        try:
            ydl_opts = {
                'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]',  # Tries to get 360p or best available below 360p
                'quiet': True,
                'noplaylist': True,
                'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s')
            }
            
            ydld= yt_dlp.YoutubeDL(ydl_opts)
            with ydld  as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                video_url = info_dict.get("url", None)
                if not video_url:
                    raise Exception("No valid MP4 link found.")
                return video_url
        except Exception as e:
            raise Exception(f"Failed to extract MP4 link: {str(e)}")

    def post_links(self, links):
        """ Posts links to Riva for transcription and returns job ID. """
        url = f"{self.riva_url}/transcribe"
        payload = {
            "links": links,
            "prompt": "Describe the video based on title, audio, tags, characters, and activities."
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_transcript(self, job_id):
        """ Retrieves transcript by job ID, polling until completion. """
        url = f"{self.riva_url}/transcribe/{job_id}"
        while True:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            if result["status"] == "completed":
                return result["transcript"]
            elif result["status"] == "failed":
                raise Exception(f"Transcription failed: {result.get('error', 'Unknown error')}")
            time.sleep(5)

    def transcribe_links(self, youtube_links):
        """ Extracts MP4 links, posts for transcription, and saves to file. """
        mp4_links = []
        for link in youtube_links:
            try:
                mp4_link = self.extract_mp4_360p_link(link)
                mp4_links.append(mp4_link)
            except Exception as e:
                print(f"Failed to extract MP4 360p link for {link}: {str(e)}")
        if not mp4_links:
            raise Exception("No valid MP4 360p links extracted")
        
        job_info = self.post_links(mp4_links)
        job_id = job_info["job_id"]
        transcript = self.get_transcript(job_id)

        filename = os.path.join(self.output_dir, f"transcript_{job_id}.txt")
        with open(filename, 'w') as file:
            file.write(transcript)
        
        return filename

# Flask routes
@app.route('/download/<resolution>', methods=['POST'])
def download_by_resolution(resolution):
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "Missing 'url' parameter in the request body."}), 400
    if not is_valid_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL."}), 400
    
    transcriber = RivaTranscriber(riva_url="http://localhost:8000", api_key="your_api_key_here")
    result = transcriber.download_video(url, resolution)
    if "error" in result:
        return jsonify(result), 500
    else:
        return jsonify(result), 200

@app.route('/video_info', methods=['POST'])
def video_info():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "Missing 'url' parameter in the request body."}), 400
    if not is_valid_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL."}), 400
    
    transcriber = RivaTranscriber(riva_url="http://localhost:8000", api_key="your_api_key_here")
    result = transcriber.get_video_info(url)
    return jsonify(result), 200

def is_valid_youtube_url(url):
    pattern = r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+(&\S*)?$"
    return re.match(pattern, url) is not None

if __name__ == "__main__":
    # Main functionality to run transcriber on a list of YouTube links directly
    riva_url = "http://localhost:8000/"
    api_key = "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"
    youtube_links = [
        "https://www.youtube.com/watch?v=GaiWyMP4N24",
        "https://www.youtube.com/watch?v=EPWrVyyd3U4",
        "https://www.youtube.com/watch?v=2hSVDLPWeKA",
        "https://www.youtube.com/watch?v=vluo9tQcTDU",
        "https://www.youtube.com/watch?v=91eoGiLSCgY",
        "https://www.youtube.com/watch?v=DCYNq2dyVj0",
        "https://www.youtube.com/watch?v=NBDAna_qxY8",
        "https://www.youtube.com/watch?v=f5Iph5tGRz0",
        "https://www.youtube.com/watch?v=VOT3jZ-Uc6Q"
    ]
    output_dir = "document_repo/sports/soccer/txt"
    transcriber = RivaTranscriber(riva_url, api_key, output_dir)
    filename = transcriber.transcribe_links(youtube_links)
    print(f"Transcript saved to {filename}")

    # Start the Flask app in debug mode
    app.run(debug=True)