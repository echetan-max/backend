# dummy_server.py
from flask import Flask
from threading import Thread
from worker import main as run_worker  # import your job loop

app = Flask(__name__)

@app.route("/")
def home():
    return "Quantum Worker is running 🚀"

if __name__ == "__main__":
    # Start worker in background
    Thread(target=run_worker, daemon=True).start()
    # Start Flask app to satisfy Render port scan
    app.run(host="0.0.0.0", port=10000)
