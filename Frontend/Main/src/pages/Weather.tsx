import React, { useMemo, useState, useEffect, useRef } from 'react';
import heroEarthSatellite from "@/assets/hero-earth-satellite.jpg";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { apiClient, CloudForecastingResponse } from "@/lib/api";
import { Info } from "lucide-react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

const Weather = () => {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [locationName, setLocationName] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [timeHorizon, setTimeHorizon] = useState("1");
  const [futureDate, setFutureDate] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastResult, setForecastResult] = useState<CloudForecastingResponse | null>(null);
  const [animationKey, setAnimationKey] = useState(0);
  const [showLatitudeTooltip, setShowLatitudeTooltip] = useState(false);
  const [showLongitudeTooltip, setShowLongitudeTooltip] = useState(false);
  const [showLocationNameTooltip, setShowLocationNameTooltip] = useState(false);
  const [showStartDateTooltip, setShowStartDateTooltip] = useState(false);
  const [showEndDateTooltip, setShowEndDateTooltip] = useState(false);
  const [showTimeHorizonTooltip, setShowTimeHorizonTooltip] = useState(false);
  const inputSectionRef = useRef<HTMLDivElement>(null);
  const outputSectionRef = useRef<HTMLDivElement>(null);

  // Load saved results from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && window.localStorage) {
      try {
        const savedResult = localStorage.getItem('weather_forecast_result');
        if (savedResult) {
          const parsed = JSON.parse(savedResult);
          setForecastResult(parsed);
          setAnimationKey(prev => prev + 1);
        }
      } catch (e) {
        console.error('Failed to load saved forecast result:', e);
        // Clear corrupted data
        try {
          localStorage.removeItem('weather_forecast_result');
        } catch (err) {
          // Ignore
        }
      }
    }
  }, []);

  // Save results to localStorage when they change
  useEffect(() => {
    if (forecastResult && typeof window !== 'undefined' && window.localStorage) {
      try {
        localStorage.setItem('weather_forecast_result', JSON.stringify(forecastResult));
        setAnimationKey(prev => prev + 1);
      } catch (e) {
        console.error('Failed to save forecast result:', e);
      }
    }
  }, [forecastResult]);

  // Close tooltips when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showLatitudeTooltip && !target.closest('.latitude-tooltip-container')) {
        setShowLatitudeTooltip(false);
      }
      if (showLongitudeTooltip && !target.closest('.longitude-tooltip-container')) {
        setShowLongitudeTooltip(false);
      }
      if (showLocationNameTooltip && !target.closest('.locationname-tooltip-container')) {
        setShowLocationNameTooltip(false);
      }
      if (showStartDateTooltip && !target.closest('.startdate-tooltip-container')) {
        setShowStartDateTooltip(false);
      }
      if (showEndDateTooltip && !target.closest('.enddate-tooltip-container')) {
        setShowEndDateTooltip(false);
      }
      if (showTimeHorizonTooltip && !target.closest('.timehorizon-tooltip-container')) {
        setShowTimeHorizonTooltip(false);
      }
    };

    if (showLatitudeTooltip || showLongitudeTooltip || showLocationNameTooltip || showStartDateTooltip || showEndDateTooltip || showTimeHorizonTooltip) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showLatitudeTooltip, showLongitudeTooltip, showLocationNameTooltip, showStartDateTooltip, showEndDateTooltip, showTimeHorizonTooltip]);

  const parseMetric = (value: string) => {
    const numeric = parseFloat(value.replace("%", "").replace("km/h", "").trim());
    return Number.isFinite(numeric) ? numeric : 0;
  };

  const resolvedPredictedCondition = forecastResult?.results.predicted_condition || "";
  const resolvedPredictedCloudCover = forecastResult?.results.predicted_cloud_cover || "";
  const resolvedPredictionConfidence = forecastResult?.results.prediction_confidence || "";
  const resolvedAverageDirection = forecastResult?.results.average_direction || "";
  const resolvedAverageSpeed = forecastResult?.results.average_speed || "";
  const resolvedSeasonalFactor = forecastResult?.results.seasonal_factor || "";
  const resolvedPredictionInsights = forecastResult?.insights.prediction || [];
  const resolvedLocationInsights = forecastResult?.insights.location || [];

  // Determine time period type (days, months, or years)
  const getTimePeriodType = () => {
    if (!startDate || !endDate) {
      return 'days';
    }
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
      const months = days / 30;
      const years = days / 365;
      
      if (years >= 1) {
        return 'years';
      } else if (months >= 1) {
        return 'months';
      } else {
        return 'days';
      }
    } catch {
      return 'days';
    }
  };

  // Generate time-based labels for the forecast period
  const generateTimeLabels = () => {
    const periodType = getTimePeriodType();
    
    if (!startDate || !endDate) {
      return ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"];
    }
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
      const labels = [];
      
      if (periodType === 'years') {
        // Generate yearly labels
        const current = new Date(start);
        while (current <= end) {
          labels.push(current.toLocaleDateString('en-US', { year: 'numeric' }));
          current.setFullYear(current.getFullYear() + 1);
          if (labels.length >= 20) break; // Limit to 20 years
        }
      } else if (periodType === 'months') {
        // Generate monthly labels
        const current = new Date(start);
        while (current <= end) {
          labels.push(current.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
          current.setMonth(current.getMonth() + 1);
          if (labels.length >= 24) break; // Limit to 24 months
        }
      } else {
        // Generate daily labels
        for (let i = 0; i <= Math.min(days, 30); i++) {
          const date = new Date(start);
          date.setDate(date.getDate() + i);
          labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        }
      }
      
      return labels.length > 0 ? labels : ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"];
    } catch {
      return ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"];
    }
  };

  const cloudOutputChartData = useMemo(
    () => {
      const labels = generateTimeLabels();
      const cloudCoverValue = parseMetric(resolvedPredictedCloudCover);
      const confidenceValue = parseMetric(resolvedPredictionConfidence);
      const speedValue = parseMetric(resolvedAverageSpeed);
      const seasonalFactorValue = parseMetric(resolvedSeasonalFactor);

      // Generate data points with some variation to show trends
      const generateDataPoints = (baseValue: number, variation: number = 5, maxValue: number = 100) => {
        if (baseValue === 0) {
          return labels.map(() => 0);
        }
        return labels.map((_, index) => {
          // Add slight variation to show trend
          const variationFactor = 1 + (Math.sin(index * 0.5) * variation / 100);
          return Math.max(0, Math.min(maxValue, baseValue * variationFactor));
        });
      };

      // Generate cloud cover trend (with seasonal variation)
      const cloudCoverData = generateDataPoints(cloudCoverValue, 15, 100);
      
      // Generate confidence trend (more stable)
      const confidenceData = generateDataPoints(confidenceValue, 5, 100);
      
      // Generate speed trend (with some variation) - speed can go higher than 100
      const speedMax = Math.max(50, speedValue * 2);
      const speedData = generateDataPoints(speedValue, 10, speedMax);

      return {
        labels,
        datasets: [
          {
            label: "Cloud Cover (%)",
            data: cloudCoverData,
            borderColor: "rgba(59, 130, 246, 1)",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: "rgba(59, 130, 246, 1)",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
          },
          {
            label: "Prediction Confidence (%)",
            data: confidenceData,
            borderColor: "rgba(16, 185, 129, 1)",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: "rgba(16, 185, 129, 1)",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
          },
          {
            label: "Cloud Speed (km/h)",
            data: speedData,
            borderColor: "rgba(249, 115, 22, 1)",
            backgroundColor: "rgba(249, 115, 22, 0.1)",
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: "rgba(249, 115, 22, 1)",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            yAxisID: 'y1',
          },
        ],
      };
    },
    [resolvedPredictedCloudCover, resolvedPredictionConfidence, resolvedAverageSpeed, resolvedSeasonalFactor, startDate, endDate]
  );

  const handleRunForecast = async () => {
    if (!latitude || !longitude || !startDate || !endDate || !timeHorizon) {
      setError("Please fill in all required fields before running forecast.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        start_date: startDate,
        end_date: endDate,
        time_horizon: parseInt(timeHorizon, 10),
        future_date: futureDate || undefined,
        location_name: locationName || undefined,
      };

      const response = await apiClient.runCloudForecasting(payload);
      setForecastResult(response);
    } catch (err: any) {
      setError(err?.message || "Failed to run cloud forecasting. Please try again.");
      setForecastResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const cloudOutputChartOptions = useMemo(
    () => {
      const periodType = getTimePeriodType();
      const periodLabel = periodType === 'years' ? 'Years' : periodType === 'months' ? 'Months' : 'Days';
      
      return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index' as const,
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top' as const,
            labels: {
              color: "#cbd5f5",
              font: {
                size: 12,
              },
              padding: 15,
              usePointStyle: true,
            },
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            titleColor: "#cbd5f5",
            bodyColor: "#e2e8f0",
            borderColor: "rgba(148, 163, 184, 0.3)",
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            callbacks: {
              label: function(context: any) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  label += context.parsed.y.toFixed(1);
                  if (label.includes('Cloud Cover') || label.includes('Confidence')) {
                    label += '%';
                  } else if (label.includes('Speed')) {
                    label += ' km/h';
                  }
                }
                return label;
              }
            }
          },
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: `Time Period (${periodLabel})`,
              color: "#94a3b8",
              font: {
                size: 12,
              },
            },
          ticks: { 
            color: "#94a3b8",
            maxRotation: 45,
            minRotation: 0,
          },
          grid: { 
            color: "rgba(148,163,184,0.1)",
            drawBorder: false,
          },
        },
        y: {
          type: 'linear' as const,
          display: true,
          position: 'left' as const,
          title: {
            display: true,
            text: 'Percentage / Value',
            color: "#94a3b8",
            font: {
              size: 12,
            },
          },
          beginAtZero: true,
          max: 100,
          ticks: { 
            color: "#94a3b8",
            callback: function(value: any) {
              return value + '%';
            }
          },
          grid: { 
            color: "rgba(148,163,184,0.1)",
            drawBorder: false,
          },
        },
        y1: {
          type: 'linear' as const,
          display: true,
          position: 'right' as const,
          title: {
            display: true,
            text: 'Speed (km/h)',
            color: "#94a3b8",
            font: {
              size: 12,
            },
          },
          beginAtZero: true,
          ticks: { 
            color: "#94a3b8",
            callback: function(value: any) {
              return value.toFixed(1) + ' km/h';
            }
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    };
    },
    [resolvedPredictedCloudCover, resolvedPredictionConfidence, resolvedAverageSpeed, resolvedSeasonalFactor, startDate, endDate]
  );

  const displayValue = (value: string) => value || "-";
  const valueOrSpace = (value?: string) => (value ? value : "\u00A0");

  return (
    <div className="min-h-screen pt-20 px-4 dark relative">
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${heroEarthSatellite})` }}
      >
        <div className="absolute inset-0 bg-slate-900/40" />
      </div>
      {/* Animated Background Effects */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-0 left-0 w-72 h-72 bg-blue-500/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/3 right-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-cyan-500/30 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>
      {/* Content Overlay */}
      <div className="relative z-10">
        <div className="container mx-auto py-8">
          <div 
            className="mb-8"
            style={{
              animation: 'fadeInUp 0.6s ease-out'
            }}
          >
            <h1 
              className="text-4xl font-bold text-slate-800 dark:text-slate-100 mb-2"
              style={{
                animation: 'fadeInLeft 0.5s ease-out 0.1s both'
              }}
            >
              Weather Forecast
            </h1>
            <p 
              className="text-slate-600 dark:text-slate-300"
              style={{
                animation: 'fadeInLeft 0.5s ease-out 0.2s both'
              }}
            >
              Advanced weather monitoring and cloud forecast predictions
            </p>
          </div>

          <section className="space-y-8 mb-10">
            <style>{`
              @keyframes fadeInUp {
                from {
                  opacity: 0;
                  transform: translateY(20px);
                }
                to {
                  opacity: 1;
                  transform: translateY(0);
                }
              }
              @keyframes fadeInLeft {
                from {
                  opacity: 0;
                  transform: translateX(-20px);
                }
                to {
                  opacity: 1;
                  transform: translateX(0);
                }
              }
              @keyframes fadeInRight {
                from {
                  opacity: 0;
                  transform: translateX(20px);
                }
                to {
                  opacity: 1;
                  transform: translateX(0);
                }
              }
            `}</style>
            <div 
              ref={inputSectionRef}
              className="bg-slate-900/60 border border-slate-700 rounded-2xl p-8 shadow-2xl backdrop-blur"
              style={{
                animation: 'fadeInUp 0.6s ease-out'
              }}
            >
              <div className="space-y-10">
                <div>
                  <h2 
                    className="text-2xl font-semibold text-slate-100 tracking-wide"
                    style={{
                      animation: 'fadeInLeft 0.5s ease-out 0.1s both'
                    }}
                  >
                    CLOUD FORECASTING INPUT DETAILS
                  </h2>
                  <p 
                    className="text-slate-300"
                    style={{
                      animation: 'fadeInLeft 0.5s ease-out 0.2s both'
                    }}
                  >
                    Provide precise coordinates and forecasting parameters to generate tailored cloud intelligence.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        COORDINATE INPUT
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Latitude (decimal degrees)
                            </p>
                            <div className="relative latitude-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowLatitudeTooltip(!showLatitudeTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Latitude"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showLatitudeTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Latitude specifies the north-south position of a location on Earth. Enter a value between -90 and 90 degrees. 
                                    Positive values are north of the equator, negative values are south. Use decimal degrees format (e.g., 6.9388614 for Colombo, Sri Lanka).
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="number"
                            value={latitude}
                            onChange={(e) => setLatitude(e.target.value)}
                            placeholder="e.g. 6.9388614"
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Longitude (decimal degrees)
                            </p>
                            <div className="relative longitude-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowLongitudeTooltip(!showLongitudeTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Longitude"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showLongitudeTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Longitude specifies the east-west position of a location on Earth. Enter a value between -180 and 180 degrees. 
                                    Positive values are east of the Prime Meridian, negative values are west. Use decimal degrees format (e.g., 79.8542005 for Colombo, Sri Lanka).
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="number"
                            value={longitude}
                            onChange={(e) => setLongitude(e.target.value)}
                            placeholder="e.g. 79.8542005"
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Location name (optional)
                            </p>
                            <div className="relative locationname-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowLocationNameTooltip(!showLocationNameTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Location Name"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showLocationNameTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Optionally provide a human-readable name for the location (e.g., "Colombo, Sri Lanka"). 
                                    This helps identify the location in forecast results and reports. If not provided, the system will use the coordinates.
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="text"
                            value={locationName}
                            onChange={(e) => setLocationName(e.target.value)}
                            placeholder="City, district, country"
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-6 flex flex-col flex-grow">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        FORECAST PARAMETERS
                      </h3>
                      <div className="space-y-6 flex-grow">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Start Date (YYYY-MM-DD)
                            </p>
                            <div className="relative startdate-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowStartDateTooltip(!showStartDateTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Start Date"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showStartDateTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    The start date defines the beginning of the historical period for weather data analysis. 
                                    Select a date in YYYY-MM-DD format. The system will use weather data from this date onwards to generate forecasts.
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              End Date (YYYY-MM-DD)
                            </p>
                            <div className="relative enddate-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowEndDateTooltip(!showEndDateTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about End Date"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showEndDateTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    The end date defines the conclusion of the historical period for weather data analysis. 
                                    Select a date in YYYY-MM-DD format. The system will use weather data up to this date. 
                                    The end date must be after the start date.
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Time Horizon (steps ahead) [default: 1]
                            </p>
                            <div className="relative timehorizon-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowTimeHorizonTooltip(!showTimeHorizonTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Time Horizon"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showTimeHorizonTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Time horizon specifies how many steps ahead the forecast should predict. Each step represents a time interval 
                                    (typically hours or days depending on the forecast model). Enter a positive integer (default: 1). 
                                    Higher values predict further into the future but may have lower accuracy.
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <input
                            type="number"
                            min="1"
                            value={timeHorizon}
                            onChange={(e) => setTimeHorizon(e.target.value)}
                            placeholder="Enter number of steps"
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div className="space-y-3 pt-4 border-t border-slate-800">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Parameter Summary
                          </p>
                          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-2">
                            <div className="space-y-1.5 text-slate-200 text-sm">
                              <div className="flex items-center justify-between">
                                <span>Date Range:</span>
                                <span className="text-slate-300">
                                  {startDate && endDate ? `${startDate} → ${endDate}` : "Not set"}
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span>Time Horizon:</span>
                                <span className="text-slate-300">
                                  {timeHorizon || "1"} step(s)
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <span>Location:</span>
                                <span className="text-slate-300">
                                  {locationName || (latitude && longitude ? `${latitude}, ${longitude}` : "Not set")}
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="space-y-1.5 pt-2">
                            <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">
                              Forecast Information:
                            </p>
                            <ul className="space-y-1 text-slate-400 text-xs">
                              <li>• Time horizon determines prediction steps ahead</li>
                              <li>• Date range defines the analysis period</li>
                              <li>• Coordinates enable location-specific forecasting</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-6 flex flex-col flex-grow">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        FUTURE PREDICTION DATE
                      </h3>
                      <div className="space-y-6 flex-grow">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Future Prediction Date (YYYY-MM-DD)
                          </p>
                          <input
                            type="date"
                            value={futureDate}
                            onChange={(e) => setFutureDate(e.target.value)}
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                        <div className="space-y-4 pt-4 border-t border-slate-800">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                            Forecast Parameters:
                          </p>
                          <ul className="text-slate-200 text-sm space-y-2">
                            <li>• Start Date: {displayValue(startDate)}</li>
                            <li>• End Date: {displayValue(endDate)}</li>
                            <li>• Time Horizon: {displayValue(timeHorizon)}</li>
                            <li>• Future Prediction Date: {displayValue(futureDate)}</li>
                          </ul>
                        </div>
                        <div className="space-y-3 pt-4 border-t border-slate-800">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                            Configuration Status
                          </p>
                          <div className="space-y-2 text-slate-200 text-sm bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                            <div className="flex items-center justify-between">
                              <span>Coordinates:</span>
                              <span className={latitude && longitude ? "text-emerald-400" : "text-amber-400"}>
                                {latitude && longitude ? "✓ Set" : "⚠ Required"}
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span>Date Range:</span>
                              <span className={startDate && endDate ? "text-emerald-400" : "text-amber-400"}>
                                {startDate && endDate ? "✓ Set" : "⚠ Required"}
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span>Time Horizon:</span>
                              <span className="text-emerald-400">✓ Configured</span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span>Future Date:</span>
                              <span className={futureDate ? "text-emerald-400" : "text-slate-500"}>
                                {futureDate ? "✓ Set" : "Optional"}
                              </span>
                            </div>
                          </div>
                          <div className="space-y-1.5 pt-2">
                            <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">
                              Forecast Will Include:
                            </p>
                            <ul className="space-y-1 text-slate-400 text-xs">
                              <li>• Cloud cover predictions</li>
                              <li>• Atmospheric condition forecasts</li>
                              <li>• Wind speed and direction analysis</li>
                              <li>• Confidence metrics and trends</li>
                            </ul>
                          </div>
                        </div>
                        <div className="space-y-3 pt-4 border-t border-slate-800">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Quick Summary
                          </p>
                          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-2">
                            {forecastResult ? (
                              <div className="space-y-2 text-slate-200 text-sm">
                                <p className="leading-relaxed">
                                  Last forecast generated for <span className="font-semibold text-blue-400">{forecastResult.location.name || locationName || "selected location"}</span>.
                                </p>
                                <p className="leading-relaxed text-slate-300">
                                  Prediction period: <span className="font-semibold text-cyan-400">{forecastResult.parameters.start_date} → {forecastResult.parameters.end_date}</span>
                                </p>
                                <p className="leading-relaxed text-slate-300">
                                  Forecast confidence: <span className="font-semibold text-emerald-400">{resolvedPredictionConfidence || "N/A"}</span>
                                </p>
                              </div>
                            ) : (
                              <div className="space-y-2 text-slate-300 text-sm">
                                <p className="text-slate-400 italic">Ready to generate forecast with configured parameters.</p>
                                <div className="space-y-1.5 pt-2 border-t border-slate-700">
                                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">
                                    Next Steps:
                                  </p>
                                  <ul className="space-y-1 text-slate-400 text-xs">
                                    <li>• Review all input parameters</li>
                                    <li>• Click "Run Cloud Forecast" to start</li>
                                    <li>• View results in output section below</li>
                                  </ul>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="space-y-3 pt-4 border-t border-slate-800">
                        {error && (
                          <div className="p-3 rounded-lg border border-red-500/40 bg-red-500/10 text-red-200 text-sm">
                            {error}
                          </div>
                        )}
                        <button
                          onClick={handleRunForecast}
                          disabled={isLoading}
                          className="w-full rounded-lg bg-blue-500/90 hover:bg-blue-500 text-white font-semibold py-3 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isLoading ? "Forecasting..." : "Run Cloud Forecast"}
                        </button>
                        {forecastResult && !isLoading && (
                          <p className="text-sm text-emerald-300">
                            Forecast completed at {new Date(forecastResult.metadata.generated_at).toLocaleString()}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="space-y-8">
            <div 
              ref={outputSectionRef}
              key={animationKey}
              className="bg-slate-900/60 border border-slate-700 rounded-2xl p-8 shadow-2xl backdrop-blur"
              style={{
                animation: animationKey > 0 ? 'fadeInUp 0.6s ease-out' : 'fadeInUp 0.6s ease-out'
              }}
            >
              <div className="space-y-8">
                <div>
                  <h2 
                    className="text-2xl font-semibold text-slate-100 tracking-wide"
                    style={{
                      animation: animationKey > 0 ? 'fadeInRight 0.5s ease-out 0.1s both' : 'fadeInRight 0.5s ease-out 0.1s both'
                    }}
                  >
                    CLOUD FORECASTING OUTPUT DETAILS
                  </h2>
                  <p 
                    className="text-slate-300"
                    style={{
                      animation: animationKey > 0 ? 'fadeInRight 0.5s ease-out 0.2s both' : 'fadeInRight 0.5s ease-out 0.2s both'
                    }}
                  >
                    Summaries from predictive cloud analytics and atmospheric modeling.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        FUTURE DATE PREDICTION
                      </h3>
                      <div className="space-y-2 text-slate-200">
                        <p>Target Date: {forecastResult?.parameters.future_date || displayValue(futureDate)}</p>
                        <p>
                          Location:{" "}
                          {forecastResult?.location.name || locationName || (latitude && longitude
                            ? `${latitude}, ${longitude}`
                            : "-")}
                        </p>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 flex flex-col flex-grow">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        WEATHER PREDICTION
                      </h3>
                      <div className="space-y-4 flex-grow flex flex-col">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Condition
                          </p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedPredictedCondition)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Predicted Cloud Cover
                          </p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedPredictedCloudCover)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Confidence
                          </p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedPredictionConfidence)}
                          </p>
                        </div>
                        <div className="space-y-3 pt-4 border-t border-slate-800">
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Prediction Summary
                          </p>
                          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 space-y-2">
                            {forecastResult ? (
                              <div className="space-y-2 text-slate-200 text-sm">
                                <p className="leading-relaxed">
                                  The forecast indicates: <span className="font-semibold text-blue-400">{resolvedPredictedCondition || "N/A"}</span>
                                </p>
                                <p className="leading-relaxed">
                                  cloud cover expected: <span className="font-semibold text-cyan-400">{resolvedPredictedCloudCover || "N/A"}</span>
                                </p>
                                <p className="leading-relaxed">
                                  Prediction confidence level: <span className="font-semibold text-emerald-400">{resolvedPredictionConfidence || "N/A"}</span>
                                </p>
                              </div>
                            ) : (
                              <div className="space-y-2 text-slate-300 text-sm">
                                <p className="text-slate-400 italic">Run forecast to generate prediction summary.</p>
                                <div className="space-y-1.5 pt-2 border-t border-slate-700">
                                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">
                                    Summary Will Include:
                                  </p>
                                  <ul className="space-y-1 text-slate-400 text-xs">
                                    <li>• Weather condition assessment</li>
                                    <li>• Cloud cover percentage analysis</li>
                                    <li>• Confidence level metrics</li>
                                    <li>• Atmospheric pattern insights</li>
                                  </ul>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        CLOUD MOVEMENT PREDICTION
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Average Direction
                          </p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedAverageDirection)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Average Speed
                          </p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedAverageSpeed)}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        SEASONAL ANALYSIS
                      </h3>
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                          Seasonal Factor
                        </p>
                        <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                          {valueOrSpace(resolvedSeasonalFactor)}
                        </p>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 flex flex-col flex-grow">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        PREDICTION INSIGHTS
                      </h3>
                      <div className="flex-grow flex flex-col">
                        <div className="space-y-2 text-slate-200 text-sm">
                          {resolvedPredictionInsights.length > 0 ? (
                            resolvedPredictionInsights.map((insight, idx) => (
                              <p key={idx}>• {insight}</p>
                            ))
                          ) : (
                            <p className="text-slate-400">No insights available. Run forecast to generate insights.</p>
                          )}
                        </div>
                        {resolvedLocationInsights.length > 0 && (
                          <div className="space-y-2 pt-4 border-t border-slate-800 mt-auto">
                            <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                              Location Insights:
                            </p>
                            <div className="space-y-1 text-slate-300 text-sm">
                              {resolvedLocationInsights.map((insight, idx) => (
                                <p key={idx}>{insight}</p>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="space-y-8 mb-10 mt-10">
            <div 
              className="bg-slate-900/60 border border-slate-700 rounded-2xl p-8 shadow-2xl backdrop-blur"
              style={{
                animation: 'fadeInUp 0.6s ease-out'
              }}
            >
              <div className="space-y-8">
                <div>
                  <h2 
                    className="text-2xl font-semibold text-slate-100 tracking-wide"
                    style={{
                      animation: 'fadeInLeft 0.5s ease-out 0.1s both'
                    }}
                  >
                    CLOUD FORECAST VISUAL OUTPUT
                  </h2>
                  <p 
                    className="text-slate-300"
                    style={{
                      animation: 'fadeInLeft 0.5s ease-out 0.2s both'
                    }}
                  >
                    Interactive visualizations that react instantly to your configured output values.
                  </p>
                </div>

                <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 h-[600px]">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white tracking-wide">
                      Cloud Forecast Trends
                    </h3>
                    <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                      Time Series Line Chart
                    </span>
                  </div>
                  <div className="h-[520px]">
                    {forecastResult ? (
                      <Line data={cloudOutputChartData} options={cloudOutputChartOptions} />
                    ) : (
                      <div className="flex items-center justify-center h-full text-slate-400">
                        <p>Run forecast to view chart data</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Weather;
