from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.database import Base, engine, SessionLocal
from app.api import sites, ml, auth
from app.db.seed import seed_dummy_users
import app.models.models  # Ensure models are imported
import logging
import time
import asyncio
from datetime import datetime

# Configure logging FIRST - ensure output goes to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,  # Force reconfiguration
    handlers=[
        logging.StreamHandler()  # Explicitly use StreamHandler for console output
    ]
)
logger = logging.getLogger(__name__)
# Ensure logger outputs to console
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    try:
        # Create tables (for demo; consider Alembic for prod)
        Base.metadata.create_all(bind=engine)
        
        # Seed demo users if none exist
        with SessionLocal() as db:
            seed_dummy_users(db)
        logger.info("✅ Database initialized successfully")
        print("✅ Database initialized successfully")
        logger.info("🚀 Backend server ready! Waiting for requests...")
        print("🚀 Backend server ready! Waiting for requests...")
        print("📝 All requests will be logged below:")
        print("")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        print(f"❌ Warning: Database initialization failed: {e}")
        print("  Make sure PostgreSQL is running and POSTGRES_PASSWORD is set correctly")
    
    yield  # Application runs here
    
    # Shutdown - handle cancellation gracefully
    try:
        logger.info("🛑 Shutting down gracefully...")
        print("🛑 Shutting down gracefully...")
    except asyncio.CancelledError:
        # This is normal during shutdown, just ignore it
        pass
    except Exception as e:
        # Log any other shutdown errors but don't raise
        logger.warning(f"Shutdown warning: {e}")

app = FastAPI(
    title="Solar AI Backend",
    version="0.1.0",
    lifespan=lifespan
)

# CORS: Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    start_time = time.time()
    
    # Log request - use both logger and print to ensure visibility
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"🌐 [{timestamp}] {request.method} {request.url.path}"
    logger.info(log_msg)
    print(log_msg)  # Also print to ensure visibility
    if request.query_params:
        query_log = f"   Query params: {dict(request.query_params)}"
        logger.info(query_log)
        print(query_log)
    if request.client:
        client_log = f"   Client: {request.client.host}:{request.client.port}"
        logger.info(client_log)
        print(client_log)
    
    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response - use both logger and print
        response_log = f"✅ [{timestamp}] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s"
        logger.info(response_log)
        print(response_log)  # Also print to ensure visibility
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        error_log = f"❌ [{timestamp}] {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s"
        logger.error(error_log)
        print(error_log)  # Also print to ensure visibility
        raise

app.include_router(sites.router, prefix="/api")
app.include_router(ml.router, prefix="/api")
app.include_router(auth.router, prefix="/api")

@app.get("/")
def root():
    """Health check endpoint."""
    logger.info("🏥 Health check requested")
    return {"status": "ok", "service": "Solar AI Backend"}