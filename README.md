# 🛰️🌍SolarCast.AI - Full Stack AI/ML Application⚡☀️

### 📌 **Key Idea**

Solar panels produce less energy when clouds block sunlight. If we can accurately **forecast cloud movement and density**, solar energy systems can be adjusted **in real-time** or **planned more efficiently** for example, by storing more energy in advance or adjusting grid supply strategies.

## 🌟 Features

### Core Capabilities
- **Cloud Detection**: Real-time satellite-based cloud coverage analysis
- **Cloud Forecasting**: Advanced weather monitoring and cloud forecast predictions
- **Solar Energy Prediction**: Deep-learning models for solar irradiance and energy output prediction
- **Advanced Analytics**: Live model health monitoring and performance tracking
- **System Engagement Tracking**: Real-time usage analytics using OpenWeatherMap API

### Technical Features

The application features JWT-based authentication and authorization, interactive dashboards with real-time charts, a modern UI built with React, TypeScript, and Tailwind CSS, a FastAPI backend with async/await support, multiple ML models with confidence scoring, model performance monitoring, Google Earth Engine integration, and a PostgreSQL database with SQLAlchemy ORM.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI 0.115.0
- **Server**: Uvicorn with ASGI
- **Database**: PostgreSQL with SQLAlchemy 2.0
- **Authentication**: JWT (python-jose)
- **ML Libraries**: 
  - PyTorch
  - XGBoost
  - scikit-learn
  - NumPy, Pandas
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Google Earth Engine**: earthengine-api, geemap

### Frontend
- **Framework**: React 18.3
- **Language**: TypeScript 5.8
- **Build Tool**: Vite 5.4
- **UI Library**: shadcn/ui (Radix UI)
- **Styling**: Tailwind CSS 3.4
- **Charts**: Chart.js, Recharts
- **State Management**: React Query (TanStack Query)
- **Routing**: React Router DOM 6.30

### Infrastructure
- **Database**: PostgreSQL
- **API**: RESTful API with FastAPI
- **Deployment**: Docker-ready (nginx configuration available)

---

## 📁 Project Structure


```text
SolarCastAI/
|
|-- .gitignore
|-- requirements.txt
|-- README.md
|
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |
|   |-- api/
|   |   |-- __init__.py
|   |   |-- auth.py
|   |   |-- ml.py
|   |   `-- sites.py
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- deps.py
|   |   `-- security.py
|   |
|   |-- db/
|   |   |-- __init__.py
|   |   |-- database.py
|   |   `-- seed.py
|   |
|   |-- ml/
|   |   |-- __init__.py
|   |   |-- best_model.pth
|   |   |-- cloud_detection.py
|   |   |-- cloud_forecasting.py
|   |   |-- gee_config.py
|   |   |-- gee_config_simple.py
|   |   |-- requirements.txt
|   |   |-- setup_gee.py
|   |   |-- solar_energy_output_prediction.py
|   |   `-- solar_irradiance_prediction.py
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   `-- models.py
|   |
|   |-- schemas/
|   |   |-- __init__.py
|   |   `-- schemas.py
|   |
|   `-- services/
|       |-- __init__.py
|       |-- crud.py
|       |-- ml_service.py
|       `-- model_monitor.py
|
|-- Frontend/
|   |-- Main/
|   |   |-- src/
|   |   |   |-- components/
|   |   |   |   |-- ui/
|   |   |   |   |-- ApiStatus.tsx
|   |   |   |   |-- CTASection.tsx
|   |   |   |   |-- DataSourcesSection.tsx
|   |   |   |   |-- FeaturesSection.tsx
|   |   |   |   |-- Footer.tsx
|   |   |   |   |-- Header.tsx
|   |   |   |   |-- HeroSection.tsx
|   |   |   |   |-- MLPredictor.tsx
|   |   |   |   |-- ProtectedRoute.tsx
|   |   |   |   |-- SitesManager.tsx
|   |   |   |   `-- TechnologySection.tsx
|   |   |   |
|   |   |   |-- contexts/
|   |   |   |   |-- AuthContext.tsx
|   |   |   |   `-- ThemeContext.tsx
|   |   |   |
|   |   |   |-- hooks/
|   |   |   |   |-- use-api.ts
|   |   |   |   |-- use-mobile.tsx
|   |   |   |   `-- use-toast.ts
|   |   |   |
|   |   |   |-- lib/
|   |   |   |   |-- api.ts
|   |   |   |   `-- utils.ts
|   |   |   |
|   |   |   |-- pages/
|   |   |   |   |-- Dashboard.tsx
|   |   |   |   |-- Satellite.tsx
|   |   |   |   |-- Weather.tsx
|   |   |   |   |-- Solar.tsx
|   |   |   |   `-- Advanced.tsx
|   |   |   |
|   |   |   |-- assets/
|   |   |   |   |-- earth-hero.jpg
|   |   |   |   `-- hero-earth-satellite.jpg
|   |   |   |
|   |   |   |-- App.css
|   |   |   |-- App.tsx
|   |   |   |-- index.css
|   |   |   |-- main.tsx
|   |   |   `-- vite-env.d.ts
|   |   |
|   |   |-- public/
|   |   |   `-- robots.txt
|   |   |
|   |   |-- components.json
|   |   |-- env.example
|   |   |-- eslint.config.js
|   |   |-- index.html
|   |   |-- package.json
|   |   |-- package-lock.json
|   |   |-- postcss.config.js
|   |   |-- README.md
|   |   |-- tailwind.config.ts
|   |   |-- tsconfig.app.json
|   |   |-- tsconfig.json
|   |   |-- tsconfig.node.json
|   |   `-- vite.config.ts
|   |
|   `-- package-lock.json
|
`-- scripts/
    |-- check_users.py
    `-- force_seed_users.py
```


---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python** 3.10+ (recommended: 3.11 or 3.12)
- **Node.js** 18+ (or Bun)
- **PostgreSQL** 12+
- **Google Earth Engine Account** (for satellite imagery features)
- **OpenWeatherMap API Key** (for weather data)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SolarCastAI
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Install Dependencies

```bash
# Install main backend dependencies
pip install -r requirements.txt

# Install ML-specific dependencies
cd app/ml
pip install -r requirements.txt
cd ../..
```

#### Setup Google Earth Engine

```bash
cd app/ml
python setup_gee.py
# Follow the authentication prompts
cd ../..
```

### 3. Frontend Setup

```bash
cd Frontend/Main

# Install dependencies (using npm)
npm install

# OR using bun
bun install
```

### 4. Database Setup

#### Create PostgreSQL Database

```sql
CREATE DATABASE sola_ai;
```

#### Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=sola_ai

# JWT Configuration
SECRET_KEY=your-secret-key-here  # REQUIRED: Generate a strong random key for production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Google Earth Engine
GEE_CREDENTIALS_PATH=path/to/credentials.json
GEE_PROJECT_ID=your-gcp-project-id  # Optional: Your Google Cloud Project ID for GEE

# OpenWeatherMap API
OPENWEATHERMAP_API_KEY=your-api-key-here
```

#### Initialize Database

```bash
# The database tables will be created automatically on first run
# Or manually seed users:
python scripts/check_users.py
```

---

## 🏃 Running the Application

### Backend Server

```bash
# From root directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Frontend Development Server

```bash
cd Frontend/Main
npm run dev
# or
bun run dev
```

The frontend will be available at:
- **Frontend**: http://localhost:5173 (Vite default port)

### Production Build

#### Frontend Build

```bash
cd Frontend/Main
npm run build
# Output will be in dist/ directory
```

#### Backend Production

```bash
# Use a production ASGI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📡 API Endpoints

### Authentication

- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `GET /api/auth/me` - Get current user

### ML Predictions

- `POST /api/ml/cloud-detection/run` - Run cloud detection analysis
- `POST /api/ml/cloud-forecasting/run` - Run cloud forecasting
- `POST /api/ml/solar-energy-prediction/run` - Run solar energy prediction
- `POST /api/ml/irradiance` - Predict solar irradiance
- `POST /api/ml/energy-output` - Predict energy output

### Model Monitoring

- `GET /api/ml/models/status` - Get model health status
- `GET /api/ml/models/health` - Detailed model health metrics

### Site Management

- `GET /api/sites` - List all sites
- `POST /api/sites` - Create a new site
- `GET /api/sites/{id}` - Get site details
- `PUT /api/sites/{id}` - Update site
- `DELETE /api/sites/{id}` - Delete site

### Health Check

- `GET /` - Health check endpoint

For detailed API documentation, visit http://localhost:8000/docs after starting the server.

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_HOST` | PostgreSQL host | Yes |
| `POSTGRES_PORT` | PostgreSQL port | Yes |
| `POSTGRES_USER` | PostgreSQL username | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | Database name | Yes |
| `SECRET_KEY` | JWT secret key | Yes |
| `ALGORITHM` | JWT algorithm | No (default: HS256) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry | No (default: 30) |
| `GEE_CREDENTIALS_PATH` | GEE credentials path | Yes (for ML features) |
| `GEE_PROJECT_ID` | Google Cloud Project ID for GEE | No (has default) |
| `OPENWEATHERMAP_API_KEY` | OpenWeatherMap API key | Yes (for weather) |

### ML Model Configuration

ML models are located in `app/ml/`:
- `best_model.pth` - Pre-trained model weights
- Model configuration is handled in respective Python files

---

## 🧪 Testing

### Backend Tests

```bash
# Run tests (if test files exist)
pytest
```

### Frontend Tests

```bash
cd Frontend/Main
npm run test
```

### Database Utilities

```bash
# Check database users
python scripts/check_users.py

# Force seed users (resets database)
python scripts/force_seed_users.py
```

---

## 📊 Project Statistics

- **Backend Files**: ~25 Python source files
- **Frontend Files**: ~100 TypeScript/React files
- **Total Production Files**: ~120-150 files
- **ML Models**: 4 active models
  - Cloud Detection Model
  - Cloud Forecasting Model
  - Solar Irradiance Prediction Model
  - Solar Energy Output Prediction Model

---

## 🚢 Deployment

### Docker Deployment (Recommended)

1. Build Docker images
2. Configure environment variables
3. Run with docker-compose

### Manual Deployment

1. **Backend**: Deploy using Gunicorn or Uvicorn with reverse proxy (nginx)
2. **Frontend**: Build static files and serve with nginx or CDN
3. **Database**: Use managed PostgreSQL service (AWS RDS, Azure, etc.)

### Environment Setup

- **Development**: Use local PostgreSQL and development servers
- **Staging**: Use staging database and test environment
- **Production**: Use production database with SSL and proper security

---

## 🔒 Security Considerations

- JWT tokens with expiration
- Password hashing using bcrypt
- CORS configuration for API
- Environment variable protection
- SQL injection prevention (SQLAlchemy ORM)
- Input validation (Pydantic schemas)

---

## 📝 Development Guidelines

### Code Style

- **Backend**: Follow PEP 8 Python style guide
- **Frontend**: Use ESLint and Prettier configurations
- **TypeScript**: Strict type checking enabled

### Git Workflow

- Use feature branches
- Commit messages should be descriptive
- Review code before merging

### Adding New Features

1. Create feature branch
2. Implement backend API endpoint
3. Create frontend component/page
4. Add tests
5. Update documentation
6. Submit pull request

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is proprietary. All rights reserved.

---

## 🆘 Support

For issues and questions:
- Check the documentation in `CLEAN_FOLDER_STRUCTURE.md`
- Review API documentation at `/docs` endpoint
- Contact the development team

---

## 🎯 Roadmap

- [ ] Additional ML model integrations
- [ ] Real-time data streaming
- [ ] Mobile app support
- [ ] Advanced analytics dashboard
- [ ] Multi-tenant support
- [ ] API rate limiting
- [ ] Caching layer implementation

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Google Earth Engine](https://earthengine.google.com/)
- [OpenWeatherMap API](https://openweathermap.org/api)

---

**Last Updated**: 2025
**Version**: 1.0.0
**Status**: Production Ready
