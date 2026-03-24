# Clean Folder Structure

This project follows the structure below.

```text
SolarCastAI/
|
|-- .gitignore
|-- requirements.txt
|-- README.md
|-- CLEAN_FOLDER_STRUCTURE.md
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
