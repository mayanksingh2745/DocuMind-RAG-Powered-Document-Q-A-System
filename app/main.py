from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="RAG-Powered Document Q&A System API",
        version="1.0.0",
    )

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # We will include routers here once they are created
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    from fastapi.responses import HTMLResponse
    import os

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        static_file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if os.path.exists(static_file_path):
            with open(static_file_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        return HTMLResponse(content="<h1>DocuMind Frontend Not Found</h1>", status_code=404)

    return app

app = create_app()
