import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, Globe, Satellite, Cloud, Database } from "lucide-react";

export function DataSourcesSection() {
  return (
    <section className="py-16 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-800 dark:text-slate-100 mb-4">
            Data Sources
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Our platform integrates with leading global data providers to deliver accurate and comprehensive solar energy insights.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Google Earth Engine */}
          <Card className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
                  <Globe className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-slate-800 dark:text-slate-100">
                    Google Earth Engine
                  </CardTitle>
                  <Badge variant="secondary" className="mt-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                    Satellite Data
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">
                High-resolution satellite imagery and geospatial data for environmental monitoring and analysis.
              </p>
              <a
                href="https://earthengine.google.com"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300 transition-colors"
              >
                Visit Earth Engine
                <ExternalLink className="h-3 w-3" />
              </a>
            </CardContent>
          </Card>

          {/* OpenWeatherMap */}
          <Card className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                  <Cloud className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-slate-800 dark:text-slate-100">
                    OpenWeatherMap
                  </CardTitle>
                  <Badge variant="secondary" className="mt-1 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    Weather Data
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">
                Real-time weather data, forecasts, and historical weather information for global locations.
              </p>
              <a
                href="https://openweathermap.org"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
              >
                Visit OpenWeatherMap
                <ExternalLink className="h-3 w-3" />
              </a>
            </CardContent>
          </Card>

          {/* OpenStreetMap */}
          <Card className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                  <Database className="h-6 w-6 text-orange-600 dark:text-orange-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-slate-800 dark:text-slate-100">
                    OpenStreetMap
                  </CardTitle>
                  <Badge variant="secondary" className="mt-1 bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200">
                    Map Data
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">
                Free, editable world map data created and maintained by a community of contributors.
              </p>
              <a
                href="https://www.openstreetmap.org"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-orange-600 dark:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 transition-colors"
              >
                Visit OpenStreetMap
                <ExternalLink className="h-3 w-3" />
              </a>
            </CardContent>
          </Card>

          {/* NASA/ESA Copernicus */}
          <Card className="bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl transition-all duration-300">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                  <Satellite className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-slate-800 dark:text-slate-100">
                    NASA / ESA Copernicus
                  </CardTitle>
                  <Badge variant="secondary" className="mt-1 bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                    Satellite Imagery
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">
                High-quality satellite imagery and Earth observation data from space agencies.
              </p>
              <a
                href="https://www.esa.int/Applications/Observing_the_Earth/Copernicus"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
              >
                Visit Copernicus
                <ExternalLink className="h-3 w-3" />
              </a>
            </CardContent>
          </Card>
        </div>

        {/* Attribution Footer */}
        <div className="mt-12 text-center">
          <div className="inline-block bg-white dark:bg-slate-800 rounded-lg px-6 py-4 shadow-md border border-slate-200 dark:border-slate-700">
            <p className="text-sm text-slate-600 dark:text-slate-300">
              <span className="font-medium">Data Attribution:</span> Map data © OpenStreetMap contributors | 
              Satellite imagery © NASA / ESA Copernicus | 
              Weather data © OpenWeatherMap | 
              Earth Engine data © Google
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
