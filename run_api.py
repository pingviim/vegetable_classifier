import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vegetable Classification API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )