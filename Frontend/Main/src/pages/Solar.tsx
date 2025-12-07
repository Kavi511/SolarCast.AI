import React, { useMemo, useState, useEffect, useRef } from 'react';
import heroEarthSatellite from "@/assets/hero-earth-satellite.jpg";
import { apiClient, SolarEnergyPredictionResponse } from "@/lib/api";
import { Info } from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line, Bar, Doughnut, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend
);

const Solar = () => {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [locationName, setLocationName] = useState("");
  const [inputSystemCapacity, setInputSystemCapacity] = useState("5.0");
  const [inputSystemArea, setInputSystemArea] = useState("27.8");
  const [inputTrainingEpochs, setInputTrainingEpochs] = useState("50");
  const [inputLearningRate, setInputLearningRate] = useState("0.001");

  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [trainingDuration, setTrainingDuration] = useState("");
  const [forecastHorizon, setForecastHorizon] = useState("");

  const [dataPoints, setDataPoints] = useState("");
  const [summaryPeriodStart, setSummaryPeriodStart] = useState("");
  const [summaryPeriodEnd, setSummaryPeriodEnd] = useState("");
  const [dataSources, setDataSources] = useState("");
  const [qualityScore, setQualityScore] = useState("");
  const [qualityStatus, setQualityStatus] = useState("");

  const [systemCapacity, setSystemCapacity] = useState("5.0 kW");
  const [predictionPeriod, setPredictionPeriod] = useState("");
  const [totalEnergy, setTotalEnergy] = useState("");
  const [averagePower, setAveragePower] = useState("");
  const [peakPower, setPeakPower] = useState("");
  const [averageConfidence, setAverageConfidence] = useState("");

  const [dailyBreakdown, setDailyBreakdown] = useState([
    { date: "2025-11-14", energy: "" },
    { date: "2025-11-15", energy: "" },
    { date: "2025-11-16", energy: "" },
  ]);

  const [targetDate, setTargetDate] = useState("");
  const [specificLocation, setSpecificLocation] = useState("");
  const [specificCapacity, setSpecificCapacity] = useState("");
  const [panelArea, setPanelArea] = useState("");
  const [panelEfficiency, setPanelEfficiency] = useState("");

  const [dailyEnergyProduction, setDailyEnergyProduction] = useState("");
  const [resultAveragePower, setResultAveragePower] = useState("");
  const [averageIrradiance, setAverageIrradiance] = useState("");
  const [predictionConfidence, setPredictionConfidence] = useState("");

  const [capacityFactor, setCapacityFactor] = useState("");
  const [performanceRatio, setPerformanceRatio] = useState("");
  const [estimatedRevenue, setEstimatedRevenue] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<SolarEnergyPredictionResponse | null>(null);
  const [inputAnimationKey, setInputAnimationKey] = useState(0);
  const [outputAnimationKey, setOutputAnimationKey] = useState(0);
  const [showLatitudeTooltip, setShowLatitudeTooltip] = useState(false);
  const [showLongitudeTooltip, setShowLongitudeTooltip] = useState(false);
  const [showLocationNameTooltip, setShowLocationNameTooltip] = useState(false);
  const [showSystemCapacityTooltip, setShowSystemCapacityTooltip] = useState(false);
  const [showSystemAreaTooltip, setShowSystemAreaTooltip] = useState(false);
  const [showTrainingEpochsTooltip, setShowTrainingEpochsTooltip] = useState(false);
  const [showLearningRateTooltip, setShowLearningRateTooltip] = useState(false);
  const [showStartDateTooltip, setShowStartDateTooltip] = useState(false);
  const [showEndDateTooltip, setShowEndDateTooltip] = useState(false);
  const [showTrainingDurationTooltip, setShowTrainingDurationTooltip] = useState(false);
  const [showForecastHorizonTooltip, setShowForecastHorizonTooltip] = useState(false);
  const inputSectionRef = useRef<HTMLDivElement>(null);
  const outputSectionRef = useRef<HTMLDivElement>(null);

  const parseMetric = (value: string) => {
    const numeric = parseFloat(value);
    return Number.isFinite(numeric) ? numeric : 0;
  };

  const valueOrSpace = (value?: string) => (value ? value : "\u00A0");

  // Load saved results from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && window.localStorage) {
      try {
        const savedResult = localStorage.getItem('solar_prediction_result');
        if (savedResult) {
          const parsed = JSON.parse(savedResult);
          setPredictionResult(parsed);
          setOutputAnimationKey(prev => prev + 1);
        }
      } catch (e) {
        console.error('Failed to load saved prediction result:', e);
        // Clear corrupted data
        try {
          localStorage.removeItem('solar_prediction_result');
        } catch (err) {
          // Ignore
        }
      }
    }
    setInputAnimationKey(1);
  }, []);

  // Close tooltips when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      const tooltipContainers = [
        'latitude-tooltip-container', 'longitude-tooltip-container', 'locationname-tooltip-container',
        'systemcapacity-tooltip-container', 'systemarea-tooltip-container', 'trainingepochs-tooltip-container',
        'learningrate-tooltip-container', 'startdate-tooltip-container', 'enddate-tooltip-container',
        'trainingduration-tooltip-container', 'forecasthorizon-tooltip-container'
      ];
      
      if (tooltipContainers.some(container => {
        const isVisible = 
          (container === 'latitude-tooltip-container' && showLatitudeTooltip) ||
          (container === 'longitude-tooltip-container' && showLongitudeTooltip) ||
          (container === 'locationname-tooltip-container' && showLocationNameTooltip) ||
          (container === 'systemcapacity-tooltip-container' && showSystemCapacityTooltip) ||
          (container === 'systemarea-tooltip-container' && showSystemAreaTooltip) ||
          (container === 'trainingepochs-tooltip-container' && showTrainingEpochsTooltip) ||
          (container === 'learningrate-tooltip-container' && showLearningRateTooltip) ||
          (container === 'startdate-tooltip-container' && showStartDateTooltip) ||
          (container === 'enddate-tooltip-container' && showEndDateTooltip) ||
          (container === 'trainingduration-tooltip-container' && showTrainingDurationTooltip) ||
          (container === 'forecasthorizon-tooltip-container' && showForecastHorizonTooltip);
        
        return isVisible && !target.closest(`.${container}`);
      })) {
        setShowLatitudeTooltip(false);
        setShowLongitudeTooltip(false);
        setShowLocationNameTooltip(false);
        setShowSystemCapacityTooltip(false);
        setShowSystemAreaTooltip(false);
        setShowTrainingEpochsTooltip(false);
        setShowLearningRateTooltip(false);
        setShowStartDateTooltip(false);
        setShowEndDateTooltip(false);
        setShowTrainingDurationTooltip(false);
        setShowForecastHorizonTooltip(false);
      }
    };

    if (showLatitudeTooltip || showLongitudeTooltip || showLocationNameTooltip || showSystemCapacityTooltip || 
        showSystemAreaTooltip || showTrainingEpochsTooltip || showLearningRateTooltip || showStartDateTooltip || 
        showEndDateTooltip || showTrainingDurationTooltip || showForecastHorizonTooltip) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showLatitudeTooltip, showLongitudeTooltip, showLocationNameTooltip, showSystemCapacityTooltip, 
      showSystemAreaTooltip, showTrainingEpochsTooltip, showLearningRateTooltip, showStartDateTooltip, 
      showEndDateTooltip, showTrainingDurationTooltip, showForecastHorizonTooltip]);

  // Save results to localStorage when they change
  useEffect(() => {
    if (predictionResult && typeof window !== 'undefined' && window.localStorage) {
      try {
        localStorage.setItem('solar_prediction_result', JSON.stringify(predictionResult));
        setOutputAnimationKey(prev => prev + 1);
      } catch (e) {
        console.error('Failed to save prediction result:', e);
      }
    }
  }, [predictionResult]);

  // Resolved values from API result
  const resolvedDailyEnergy = predictionResult?.results.daily_energy_production || "";
  const resolvedAveragePower = predictionResult?.results.average_power || "";
  const resolvedAverageIrradiance = predictionResult?.results.average_irradiance || "";
  const resolvedPredictionConfidence = predictionResult?.results.prediction_confidence || "";
  const resolvedCapacityFactor = predictionResult?.results.capacity_factor || "";
  const resolvedPerformanceRatio = predictionResult?.results.performance_ratio || "";
  const resolvedEstimatedRevenue = predictionResult?.results.estimated_revenue || "";
  const resolvedTemperature = predictionResult?.weather_conditions.temperature || "";
  const resolvedHumidity = predictionResult?.weather_conditions.humidity || "";
  const resolvedWindSpeed = predictionResult?.weather_conditions.wind_speed || "";
  const resolvedCloudCoverage = predictionResult?.weather_conditions.cloud_coverage || "";
  const resolvedPredictionInsights = predictionResult?.insights.prediction || [];
  const resolvedLocationInsights = predictionResult?.insights.location || [];

  const handleRunPrediction = async () => {
    if (!latitude || !longitude || !startDate || !endDate) {
      setError("Please fill in all required fields (latitude, longitude, start date, and end date) before running prediction.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use endDate as target_date if targetDate is not provided
      const targetDateValue = targetDate || endDate || new Date().toISOString().split('T')[0];
      
      const payload = {
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        start_date: startDate,
        end_date: endDate,
        target_date: targetDateValue,
        system_capacity_kw: parseFloat(inputSystemCapacity) || 5.0,
        system_area_m2: parseFloat(inputSystemArea) || 27.8,
        training_epochs: parseInt(inputTrainingEpochs, 10) || 50,
        learning_rate: parseFloat(inputLearningRate) || 0.001,
        location_name: locationName || undefined,
      };

      const response = await apiClient.runSolarEnergyPrediction(payload);
      setPredictionResult(response);
    } catch (err: any) {
      setError(err?.message || "Failed to run solar energy prediction. Please try again.");
      setPredictionResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const updateDailyBreakdown = (index: number, value: string) => {
    setDailyBreakdown((prev) => {
      const copy = [...prev];
      copy[index] = { ...copy[index], energy: value };
      return copy;
    });
  };

  const selectedLocation = locationName || "Not selected";
  const coordinateSummary =
    latitude && longitude ? `${latitude}, ${longitude}` : "Awaiting coordinates";

  // Extract numeric values from prediction results for charts
  const extractNumeric = (value: string) => {
    if (!value) return 0;
    const match = value.match(/[\d.]+/);
    return match ? parseFloat(match[0]) : 0;
  };

  const summaryLineData = useMemo(
    () => {
      const dailyEnergy = extractNumeric(resolvedDailyEnergy);
      const avgPower = extractNumeric(resolvedAveragePower);
      
      return {
        labels: ["Daily Energy", "Avg Power"],
        datasets: [
          {
            label: "Prediction Summary",
            data: [dailyEnergy, avgPower],
            borderColor: "#fbbf24",
            backgroundColor: "rgba(251, 191, 36, 0.2)",
            tension: 0.3,
            fill: true,
          },
        ],
      };
    },
    [resolvedDailyEnergy, resolvedAveragePower]
  );

  // Pie chart data for Daily Energy Breakdown
  const dailyBreakdownPieData = useMemo(
    () => {
      const targetDateValue = predictionResult?.parameters.target_date || endDate || new Date().toISOString().split('T')[0];
      const dailyEnergy = extractNumeric(resolvedDailyEnergy);
      
      // Generate dates around target date for pie chart segments
      const dates = [];
      const energies = [];
      const baseDate = new Date(targetDateValue);
      
      for (let i = -2; i <= 2; i++) {
        const date = new Date(baseDate);
        date.setDate(date.getDate() + i);
        dates.push(date.toISOString().split('T')[0]);
        // Use actual energy for target date (center), estimate for others
        if (i === 0) {
          energies.push(dailyEnergy);
        } else {
          // Estimate based on day of week and position
          const variation = 0.7 + (Math.abs(i) === 1 ? 0.2 : 0.1);
          energies.push(dailyEnergy * variation);
        }
      }
      
      return {
        labels: dates,
        datasets: [
          {
            data: energies,
            backgroundColor: [
              "rgba(59, 130, 246, 0.8)",
              "rgba(16, 185, 129, 0.8)",
              "rgba(234, 179, 8, 0.8)",
              "rgba(249, 115, 22, 0.8)",
              "rgba(139, 92, 246, 0.8)",
            ],
            borderColor: [
              "#3b82f6",
              "#10b981",
              "#eab308",
              "#f97316",
              "#8b5cf6",
            ],
            borderWidth: 2,
          },
        ],
      };
    },
    [predictionResult?.parameters.target_date, resolvedDailyEnergy, endDate]
  );

  // Multi-line chart data for Performance Distribution
  const performanceMultiLineData = useMemo(
    () => {
      const capFactor = extractNumeric(resolvedCapacityFactor);
      const perfRatio = extractNumeric(resolvedPerformanceRatio) * 100; // Convert to percentage
      const confidence = extractNumeric(resolvedPredictionConfidence);
      
      // Generate time periods for the multi-line chart
      const timePeriods = ["Week 1", "Week 2", "Week 3", "Week 4"];
      
      // Generate data with variations for each metric over time
      const generateVariation = (base: number, index: number) => {
        // Deterministic variation based on index
        const variation = 0.9 + (index * 0.05) + (index % 2 === 0 ? 0.03 : -0.02);
        return Math.max(0, base * variation);
      };
      
      return {
        labels: timePeriods,
        datasets: [
          {
            label: "Capacity Factor",
            data: timePeriods.map((_, i) => generateVariation(capFactor, i)),
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointBackgroundColor: "#10b981",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
          },
          {
            label: "Performance Ratio",
            data: timePeriods.map((_, i) => generateVariation(perfRatio, i)),
            borderColor: "#f97316",
            backgroundColor: "rgba(249, 115, 22, 0.1)",
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointBackgroundColor: "#f97316",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
          },
          {
            label: "Confidence",
            data: timePeriods.map((_, i) => generateVariation(confidence, i)),
            borderColor: "#6366f1",
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointBackgroundColor: "#6366f1",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
          },
        ],
      };
    },
    [resolvedCapacityFactor, resolvedPerformanceRatio, resolvedPredictionConfidence]
  );

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#cbd5f5" },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
        },
      },
    }),
    []
  );

  const pieOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          top: 0,
          bottom: 15, // Small bottom padding to lift chart up a bit
        },
      },
      plugins: {
        legend: {
          position: "right" as const,
          labels: { 
            color: "#cbd5e1",
            usePointStyle: true,
            padding: 12,
            font: {
              size: 12,
            },
          },
          padding: 20,
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleColor: '#cbd5e1',
          bodyColor: '#cbd5e1',
          borderColor: '#475569',
          borderWidth: 1,
        },
      },
    }),
    []
  );

  const multiLineOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#cbd5e1" },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleColor: '#cbd5e1',
          bodyColor: '#cbd5e1',
          borderColor: '#475569',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
          beginAtZero: true,
        },
      },
    }),
    []
  );

  return (
    <div className="min-h-screen pt-20 px-4 dark relative">
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${heroEarthSatellite})` }}
      >
        <div className="absolute inset-0 bg-slate-900/40" />
      </div>
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-0 left-0 w-72 h-72 bg-blue-500/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/3 right-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-cyan-500/30 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>
      <div className="relative z-10">
        <div className="container mx-auto py-8 space-y-10">
          <header 
            className="mb-4"
            style={{
              animation: 'fadeInUp 0.6s ease-out'
            }}
          >
            <h1 
              className="text-4xl font-bold text-slate-100 mb-2"
              style={{
                animation: 'fadeInLeft 0.5s ease-out 0.1s both'
              }}
            >
              Solar Production
            </h1>
            <p 
              className="text-slate-300"
              style={{
                animation: 'fadeInLeft 0.5s ease-out 0.2s both'
              }}
            >
              Deep-dive controls for solar energy prediction inputs and outputs.
            </p>
          </header>

          <section className="space-y-8">
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
                animation: inputAnimationKey > 0 ? 'fadeInUp 0.6s ease-out' : 'none'
              }}
            >
              <div className="space-y-10">
                <div>
                  <h2 
                    className="text-2xl font-semibold text-slate-100 tracking-wide"
                    style={{
                      animation: inputAnimationKey > 0 ? 'fadeInLeft 0.5s ease-out 0.1s both' : 'none'
                    }}
                  >
                    SOLAR ENERGY PREDICTION INPUT DETAILS
                  </h2>
                  <p 
                    className="text-slate-300"
                    style={{
                      animation: inputAnimationKey > 0 ? 'fadeInLeft 0.5s ease-out 0.2s both' : 'none'
                    }}
                  >
                    Configure solar coordinate inputs, time periods, and data quality context.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-8">
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
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
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
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
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
                                    This helps identify the location in prediction results and reports. If not provided, the system will use the coordinates.
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
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4 border-t border-slate-800">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Selected:</p>
                          <p className="text-slate-100 mt-1">{selectedLocation}</p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Coordinates:</p>
                          <p className="text-slate-100 mt-1">{coordinateSummary}</p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-6">
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-white tracking-wide">
                          SOLAR SYSTEM PARAMETERS
                        </h3>
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                          System Parameters (Input)
                        </p>
                        <div className="space-y-3">
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                                System Capacity (kW) [default: 5.0]
                              </p>
                              <div className="relative systemcapacity-tooltip-container">
                                <button
                                  type="button"
                                  onClick={() => setShowSystemCapacityTooltip(!showSystemCapacityTooltip)}
                                  className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                  aria-label="Information about System Capacity"
                                >
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                                {showSystemCapacityTooltip && (
                                  <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                    <p className="text-xs text-slate-200 leading-relaxed">
                                      System capacity is the maximum power output of your solar panel system in kilowatts (kW). 
                                      This represents the peak power the system can generate under ideal conditions. Enter the rated capacity 
                                      of your solar installation (default: 5.0 kW for a typical residential system).
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
                              value={inputSystemCapacity}
                              onChange={(e) => setInputSystemCapacity(e.target.value)}
                              placeholder="5.0"
                              className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                                System Area (m²) [default: 27.8]
                              </p>
                              <div className="relative systemarea-tooltip-container">
                                <button
                                  type="button"
                                  onClick={() => setShowSystemAreaTooltip(!showSystemAreaTooltip)}
                                  className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                  aria-label="Information about System Area"
                                >
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                                {showSystemAreaTooltip && (
                                  <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                    <p className="text-xs text-slate-200 leading-relaxed">
                                      System area is the total surface area covered by your solar panels in square meters (m²). 
                                      This is used to calculate solar irradiance and energy production. Enter the total area of all solar panels 
                                      in your installation (default: 27.8 m² for a typical 5 kW system).
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
                              value={inputSystemArea}
                              onChange={(e) => setInputSystemArea(e.target.value)}
                              placeholder="27.8"
                              className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                          Model Training Parameters
                        </p>
                        <div className="space-y-3">
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                                Training Epochs [default: 50]
                              </p>
                              <div className="relative trainingepochs-tooltip-container">
                                <button
                                  type="button"
                                  onClick={() => setShowTrainingEpochsTooltip(!showTrainingEpochsTooltip)}
                                  className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                  aria-label="Information about Training Epochs"
                                >
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                                {showTrainingEpochsTooltip && (
                                  <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                    <p className="text-xs text-slate-200 leading-relaxed">
                                      Training epochs specify how many times the machine learning model will iterate through the entire training dataset. 
                                      More epochs can improve accuracy but take longer to train. Too many epochs may cause overfitting. 
                                      Recommended range: 30-100 (default: 50).
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
                              value={inputTrainingEpochs}
                              onChange={(e) => setInputTrainingEpochs(e.target.value)}
                              placeholder="50"
                              className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                                Learning Rate [default: 0.001]
                              </p>
                              <div className="relative learningrate-tooltip-container">
                                <button
                                  type="button"
                                  onClick={() => setShowLearningRateTooltip(!showLearningRateTooltip)}
                                  className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                  aria-label="Information about Learning Rate"
                                >
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                                {showLearningRateTooltip && (
                                  <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                    <p className="text-xs text-slate-200 leading-relaxed">
                                      Learning rate controls how quickly the model learns from training data. A higher rate learns faster but may overshoot optimal values. 
                                      A lower rate is more stable but slower. Typical range: 0.0001 to 0.01 (default: 0.001). 
                                      This is a critical hyperparameter for model training.
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
                              step="0.0001"
                              value={inputLearningRate}
                              onChange={(e) => setInputLearningRate(e.target.value)}
                              placeholder="0.001"
                              className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                          Final System Parameters
                        </p>
                        <ul className="text-slate-200 text-sm space-y-1">
                          <li>Capacity: {inputSystemCapacity || "5.0"} kW</li>
                          <li>Area: {inputSystemArea || "27.8"} m²</li>
                          <li>Training Epochs: {inputTrainingEpochs || "50"}</li>
                          <li>Learning Rate: {inputLearningRate || "0.001"}</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-8">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-6">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        TIME AND DATE PARAMETERS
                      </h3>
                      <div className="space-y-6">
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Start Date (YYYY-MM-DD) [default: 2016-01-01]
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
                                    The start date defines the beginning of the historical period for solar energy data analysis and model training. 
                                    Select a date in YYYY-MM-DD format. More historical data generally improves prediction accuracy. 
                                    Default: 2016-01-01 provides several years of training data.
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
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              End Date (YYYY-MM-DD) [default: 2025-11-13]
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
                                    The end date defines the conclusion of the historical period for solar energy data analysis. 
                                    Select a date in YYYY-MM-DD format. The system will use data up to this date for training. 
                                    The end date must be after the start date. Default: 2025-11-13 uses recent data.
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
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                          />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Training Data Duration:
                            </p>
                            <div className="relative trainingduration-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowTrainingDurationTooltip(!showTrainingDurationTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Training Data Duration"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showTrainingDurationTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Select the duration of historical data to use for training the prediction model. Options: 
                                    1) Last 30 days, 2) Last 90 days, 3) Last 6 months, 4) Last 1 year, or 5) Custom duration. 
                                    More data typically improves accuracy but requires longer training time.
                                  </p>
                                  <div className="absolute bottom-0 left-4 transform translate-y-full">
                                    <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                          <ul className="text-slate-200 text-sm space-y-1 mb-2">
                            <li>1. Last 30 days</li>
                            <li>2. Last 90 days</li>
                            <li>3. Last 6 months</li>
                            <li>4. Last 1 year</li>
                            <li>5. Custom duration</li>
                          </ul>
                          <select
                            value={trainingDuration}
                            onChange={(e) => setTrainingDuration(e.target.value)}
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
                          >
                            <option value="">Select training duration</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                          </select>
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                              Forecast Horizon (hours ahead) [default: 72]
                            </p>
                            <div className="relative forecasthorizon-tooltip-container">
                              <button
                                type="button"
                                onClick={() => setShowForecastHorizonTooltip(!showForecastHorizonTooltip)}
                                className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                                aria-label="Information about Forecast Horizon"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                              {showForecastHorizonTooltip && (
                                <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                                  <p className="text-xs text-slate-200 leading-relaxed">
                                    Forecast horizon specifies how many hours into the future the model should predict solar energy production. 
                                    Enter a positive integer (default: 72 hours = 3 days). Longer horizons provide more advance planning but may have 
                                    lower accuracy. Typical range: 24-168 hours (1-7 days).
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
                            min={1}
                            value={forecastHorizon}
                            onChange={(e) => setForecastHorizon(e.target.value)}
                            placeholder="72"
                            className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                          />
                        </div>
                        {error && (
                          <div className="p-3 rounded-lg border border-red-500/40 bg-red-500/10 text-red-200 text-sm">
                            {error}
                          </div>
                        )}
                        <button
                          onClick={handleRunPrediction}
                          disabled={isLoading}
                          className="w-full rounded-lg bg-amber-500/90 hover:bg-amber-500 text-white font-semibold py-3 transition disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isLoading ? "Predicting..." : "Run Solar Energy Prediction"}
                        </button>
                        {predictionResult && !isLoading && (
                          <p className="text-sm text-emerald-300 mt-2">
                            Prediction completed at {new Date(predictionResult.metadata.generated_at).toLocaleString()}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        INPUT SUMMARY
                      </h3>
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Configuration Status
                          </p>
                          <div className="space-y-2 text-slate-200 text-sm">
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
                              <span>System Parameters:</span>
                              <span className="text-emerald-400">✓ Configured</span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span>Training Parameters:</span>
                              <span className="text-emerald-400">✓ Configured</span>
                            </div>
                          </div>
                        </div>
                        <div className="pt-3 border-t border-slate-800">
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Quick Info
                          </p>
                          <ul className="text-slate-300 text-sm space-y-1.5">
                            <li>• Forecast horizon: {forecastHorizon || "72"} hours</li>
                            <li>• Training duration: {trainingDuration ? `Option ${trainingDuration}` : "Not selected"}</li>
                            <li>• System capacity: {inputSystemCapacity || "5.0"} kW</li>
                            <li>• System area: {inputSystemArea || "27.8"} m²</li>
                          </ul>
                        </div>
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
              key={outputAnimationKey}
              className="bg-slate-900/60 border border-slate-700 rounded-2xl p-8 shadow-2xl backdrop-blur"
              style={{
                animation: outputAnimationKey > 0 ? 'fadeInUp 0.6s ease-out' : 'fadeInUp 0.6s ease-out'
              }}
            >
              <div className="space-y-10">
                <div>
                  <h2 
                    className="text-2xl font-semibold text-slate-100 tracking-wide"
                    style={{
                      animation: outputAnimationKey > 0 ? 'fadeInRight 0.5s ease-out 0.1s both' : 'fadeInRight 0.5s ease-out 0.1s both'
                    }}
                  >
                    SOLAR ENERGY PREDICTION OUTPUT DETAILS
                  </h2>
                  <p 
                    className="text-slate-300"
                    style={{
                      animation: outputAnimationKey > 0 ? 'fadeInRight 0.5s ease-out 0.2s both' : 'fadeInRight 0.5s ease-out 0.2s both'
                    }}
                  >
                    Review generated solar forecasts, breakdowns, and performance metrics.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        SPECIFIC DATE ENERGY PREDICTION
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Target Date</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {predictionResult?.parameters.target_date || valueOrSpace(targetDate)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Location</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(predictionResult?.location.name || locationName || specificLocation)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">System Capacity</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {predictionResult ? `${predictionResult.parameters.system_capacity_kw} kW` : valueOrSpace(specificCapacity)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Panel Area</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {predictionResult ? `${predictionResult.parameters.system_area_m2} m²` : valueOrSpace(panelArea)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Panel Efficiency</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(panelEfficiency || "18%")}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">PREDICTION RESULTS</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Daily Energy Production</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedDailyEnergy)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Average Power Output</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedAveragePower)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Average Irradiance</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedAverageIrradiance)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Prediction Confidence</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedPredictionConfidence)}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 flex flex-col h-full">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        PERFORMANCE METRICS
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Capacity Factor</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedCapacityFactor)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Performance Ratio</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedPerformanceRatio)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Estimated Revenue</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedEstimatedRevenue)}
                          </p>
                        </div>
                      </div>
                      <div className="flex-grow"></div>
                    </div>
                  </div>

                  <div className="space-y-8 flex flex-col">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 flex flex-col h-full">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        PREDICTION SUMMARY
                      </h3>
                      <div className="space-y-3 text-slate-200">
                        <p>Location: {predictionResult?.location.name || locationName || "Not specified"}</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">System Capacity</p>
                            <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                              {predictionResult ? `${predictionResult.parameters.system_capacity_kw} kW` : valueOrSpace(systemCapacity)}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Prediction Period</p>
                            <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                              {predictionResult ? `${predictionResult.parameters.start_date} → ${predictionResult.parameters.end_date}` : valueOrSpace(predictionPeriod)}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Total Predicted Energy</p>
                            <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                              {valueOrSpace(resolvedDailyEnergy || totalEnergy)}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Average Power Output</p>
                            <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                              {valueOrSpace(resolvedAveragePower || averagePower)}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Average Confidence</p>
                            <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                              {valueOrSpace(resolvedPredictionConfidence || averageConfidence)}
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="flex-grow"></div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">WEATHER CONDITIONS</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Temperature</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedTemperature)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Humidity</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedHumidity)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Wind Speed</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedWindSpeed)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Cloud Coverage</p>
                          <p className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100">
                            {valueOrSpace(resolvedCloudCoverage)}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-white tracking-wide">PREDICTION INSIGHTS</h3>
                      <div className="space-y-3 text-slate-200 text-sm">
                        {resolvedPredictionInsights.length > 0 ? (
                          resolvedPredictionInsights.map((insight, idx) => (
                            <p key={idx} className="leading-relaxed">• {insight}</p>
                          ))
                        ) : (
                          <p className="text-slate-400 italic">No insights available. Run prediction to generate insights.</p>
                        )}
                      </div>
                      {resolvedLocationInsights.length > 0 && (
                        <div className="space-y-2 pt-4 border-t border-slate-800">
                          <p className="text-sm uppercase tracking-[0.3em] text-slate-400 mb-2">
                            Location Insights:
                          </p>
                          <div className="space-y-2 text-slate-300 text-sm">
                            {resolvedLocationInsights.map((insight, idx) => (
                              <p key={idx} className="leading-relaxed">{insight}</p>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="space-y-8 mb-10">
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
                    SOLAR ENERGY PREDICTION VISUAL OUTPUT
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

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 h-96">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        Performance Distribution
                      </h3>
                      <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                        Multi Line Chart
                      </span>
                    </div>
                    <div className="h-72">
                      {predictionResult ? (
                        <Line data={performanceMultiLineData} options={multiLineOptions} />
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-400">
                          <p>Run prediction to view performance distribution</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 h-96 flex flex-col">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        Daily Energy Breakdown
                      </h3>
                      <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                        Donut Chart
                      </span>
                    </div>
                    <div className="h-72 flex-1">
                      {predictionResult ? (
                        <Doughnut data={dailyBreakdownPieData} options={pieOptions} />
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-400">
                          <p>Run prediction to view daily breakdown</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4 h-96 lg:col-span-2">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-white tracking-wide">
                        Prediction Summary Trend
                      </h3>
                      <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                        Line Chart
                      </span>
                    </div>
                    {predictionResult ? (
                      <Line data={summaryLineData} options={chartOptions} />
                    ) : (
                      <div className="flex items-center justify-center h-full text-slate-400">
                        <p>Run prediction to view summary trend</p>
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

export default Solar;
