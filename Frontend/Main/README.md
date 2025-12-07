# SolarCast.AI - Authentication & Navigation System

## Overview
This application now includes a dynamic header navigation system that changes based on user authentication status, with dedicated pages for each dashboard section.

## Authentication Features

### Login State
- **Demo Mode**: Any email/password combination will work for testing
- **Persistent**: Login state is saved in localStorage and persists across page refreshes
- **Context-based**: Uses React Context for global state management

### Navigation Items

#### When NOT Logged In:
- Features
- Technology  
- About
- Login/Register buttons

#### When Logged In:
- **Dashboard** - Real-time monitoring of solar energy systems and weather conditions, Land Availability & Suitability
- **Satellite** - Real-time satellite data and cloud cover analysis
- **Weather** - Time-based weather forecast and solar production predictions
- **Solar** - Detailed solar energy production analysis and optimization
- **Advanced** - Performance insights and predictive analytics
- Logout button

## Dashboard Pages

### 1. Dashboard (`/dashboard`)
- **Land Availability & Suitability**: Available sites, solar potential, development status
- **Real-time Monitoring**: Solar energy systems performance, weather conditions
- **Quick Actions**: View reports, site analysis, solar forecast, weather alerts

### 2. Satellite (`/satellite`)
- **Satellite Status**: Active satellites, coverage area, update frequency, data quality
- **Cloud Cover Analysis**: Regional cloud coverage, solar irradiance impact
- **Data Metrics**: Visibility index, last update, data trends
- **Quick Actions**: Live feed, coverage map, cloud forecast, analytics

### 3. Weather (`/weather`)
- **Current Weather**: Temperature, humidity, wind speed, cloud cover
- **24-Hour Forecast**: Hourly weather predictions with solar production impact
- **Solar Production Predictions**: Daily production forecast, weather impact analysis
- **7-Day Forecast**: Weekly weather and efficiency predictions
- **Quick Actions**: Hourly view, trends, production, alerts

### 4. Solar (`/solar`)
- **Production Overview**: Current output, daily production, efficiency, capacity factor
- **Production Analysis**: Hourly production patterns, performance metrics
- **Optimization Tools**: Performance targets, system monitoring, efficiency trends
- **Quick Actions**: Live data, analytics, configure, set goals

### 5. Advanced (`/advanced`)
- **AI Insights**: AI confidence, prediction rate, optimization score, system health
- **Predictive Analytics**: Energy production forecast, weather impact predictions
- **Performance Insights**: Efficiency trends, time analysis, risk assessment
- **AI Recommendations**: Optimization suggestions, predictive maintenance
- **Quick Actions**: AI insights, analytics, predictions, reports

## How to Use

1. **Login**: Navigate to `/login` and enter any email/password
2. **Register**: Navigate to `/register` and fill out the form
3. **Navigation**: After login, the header will automatically show dashboard navigation items
4. **Page Navigation**: Click on any dashboard item to navigate to the respective page
5. **Logout**: Click the logout button to return to the public navigation
6. **Mobile**: On mobile devices, logged-in users see a hamburger menu for dashboard navigation

## Technical Implementation

- **AuthContext**: Manages authentication state globally
- **localStorage**: Persists login status across sessions
- **React Router**: Handles navigation between dashboard pages
- **Responsive Design**: Mobile-friendly navigation with collapsible menu
- **Route Protection**: Different navigation based on authentication status

## File Structure

```
src/
├── contexts/
│   └── AuthContext.tsx      # Authentication context
├── components/
│   └── Header.tsx           # Dynamic header component
├── pages/
│   ├── Login.tsx            # Login page with auth integration
│   ├── Register.tsx         # Registration page with auth integration
│   ├── Dashboard.tsx        # Dashboard overview page
│   ├── Satellite.tsx        # Satellite data page
│   ├── Weather.tsx          # Weather forecast page
│   ├── Solar.tsx            # Solar production page
│   └── Advanced.tsx         # Advanced analytics page
└── App.tsx                  # Main app with AuthProvider and routes
```

## Development

```bash
npm install
npm run dev
```

The application will run on `http://localhost:5173` by default.

## Features

- **Responsive Design**: All pages are mobile-friendly with responsive layouts
- **Real-time Data**: Simulated real-time data for demonstration purposes
- **Interactive Elements**: Progress bars, badges, and interactive cards
- **Consistent UI**: Unified design language across all dashboard pages
- **Quick Actions**: Actionable shortcuts for common tasks on each page
