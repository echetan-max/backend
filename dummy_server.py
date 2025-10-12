# dummy_server.py
from flask import Flask
from threading import Thread
import worker  # import your worker module

app = Flask(__name__)

@app.route("/")
def home():
    return "Quantum Worker is running 🚀"

if __name__ == "__main__":
    # Start worker loop in background
    Thread(target=worker.main, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
