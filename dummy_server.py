if __name__ == "__main__":
    from worker import main as run_worker
    from threading import Thread

    # Start worker in background
    Thread(target=run_worker, daemon=True).start()

    # Start Flask to expose a port
    app.run(host="0.0.0.0", port=10000)
