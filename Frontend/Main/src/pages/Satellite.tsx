import React, { useState, useEffect, useRef } from 'react';
import heroEarthSatellite from "@/assets/hero-earth-satellite.jpg";
import { apiClient, CloudDetectionResponse } from "@/lib/api";
import { Info } from "lucide-react";

const Satellite = () => {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [locationName, setLocationName] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [timePreference, setTimePreference] = useState("");
  const [cloudThreshold, setCloudThreshold] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<CloudDetectionResponse | null>(null);
  const [animationKey, setAnimationKey] = useState(0);
  const [showThresholdTooltip, setShowThresholdTooltip] = useState(false);
  const [showLatitudeTooltip, setShowLatitudeTooltip] = useState(false);
  const [showLongitudeTooltip, setShowLongitudeTooltip] = useState(false);
  const [showLocationNameTooltip, setShowLocationNameTooltip] = useState(false);
  const [showStartDateTooltip, setShowStartDateTooltip] = useState(false);
  const [showEndDateTooltip, setShowEndDateTooltip] = useState(false);
  const [showTimePreferenceTooltip, setShowTimePreferenceTooltip] = useState(false);
  const inputSectionRef = useRef<HTMLDivElement>(null);
  const outputSectionRef = useRef<HTMLDivElement>(null);

  // One-time cleanup of old oversized data on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && window.localStorage) {
      try {
        const savedResult = localStorage.getItem('satellite_detection_result');
        if (savedResult) {
          try {
            const parsed = JSON.parse(savedResult);
            // Check if this is old data with visualizations (oversized)
            if (parsed.visualizations && (
              parsed.visualizations.composite || 
              parsed.visualizations.satellite_image ||
              parsed.visualizations.cloud_overlay ||
              parsed.visualizations.mask
            )) {
              console.warn('⚠️ Found old oversized data with images. Cleaning up...');
              // Remove visualizations and save lightweight version
              const lightweightResult = {
                location: parsed.location,
                parameters: parsed.parameters,
                results: parsed.results,
                insights: parsed.insights,
                metadata: parsed.metadata,
              };
              try {
                localStorage.setItem('satellite_detection_result', JSON.stringify(lightweightResult));
                console.log('✅ Cleaned up old oversized data - removed base64 images');
              } catch (cleanupError: any) {
                if (cleanupError.name === 'QuotaExceededError') {
                  // Still too large, try removing everything and starting fresh
                  localStorage.removeItem('satellite_detection_result');
                  console.log('🧹 Removed all data due to quota issues');
                } else {
                  console.error('❌ Failed to clean up old data:', cleanupError);
                }
              }
            }
          } catch (e) {
            // Ignore parse errors in cleanup
          }
        }
      } catch (e) {
        // Ignore errors in cleanup
      }
    }
  }, []); // Run once on mount

  // Load saved results from localStorage on mount and when page becomes visible
  useEffect(() => {
    const loadSavedResult = () => {
      if (typeof window !== 'undefined' && window.localStorage) {
        try {
          const savedResult = localStorage.getItem('satellite_detection_result');
          if (savedResult) {
            try {
              const parsed = JSON.parse(savedResult);
              
              // Check if this is old data with visualizations (oversized) - clean it up
              if (parsed.visualizations && (
                parsed.visualizations.composite || 
                parsed.visualizations.satellite_image ||
                parsed.visualizations.cloud_overlay ||
                parsed.visualizations.mask
              )) {
                console.warn('⚠️ Found old oversized data with images. Cleaning up...');
                // Remove visualizations and save lightweight version
                const lightweightResult = {
                  location: parsed.location,
                  parameters: parsed.parameters,
                  results: parsed.results,
                  insights: parsed.insights,
                  metadata: parsed.metadata,
                };
                try {
                  localStorage.setItem('satellite_detection_result', JSON.stringify(lightweightResult));
                  console.log('✅ Cleaned up old oversized data - removed base64 images');
                  parsed.visualizations = undefined; // Remove from parsed object
                } catch (cleanupError: any) {
                  if (cleanupError.name === 'QuotaExceededError') {
                    // Still too large, remove everything
                    localStorage.removeItem('satellite_detection_result');
                    console.log('🧹 Removed all data due to quota issues');
                    return; // Don't set detection result
                  } else {
                    console.error('❌ Failed to clean up old data:', cleanupError);
                  }
                }
              }
              
              // Validate and set result
              if (parsed && parsed.results) {
                // Note: Loaded data won't have visualizations (they're excluded to save space)
                // Visualizations are only available when fresh from API
                setDetectionResult(parsed);
                setAnimationKey(prev => prev + 1);
                console.log('✅ Satellite detection result loaded from localStorage (summary only, no images)');
              }
            } catch (parseError) {
              console.error('Failed to parse saved detection result:', parseError);
              // Only clear if truly corrupted, don't clear on minor issues
            }
          }
        } catch (e) {
          console.error('Failed to load saved detection result:', e);
          // Don't clear localStorage on error - keep data
        }
      }
    };

    // Load on mount
    loadSavedResult();

    // Reload when page becomes visible (user returns to tab)
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadSavedResult();
      }
    };

    // Reload when window gains focus
    const handleFocus = () => {
      loadSavedResult();
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleFocus);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleFocus);
    };
  }, []);

  // Save results to localStorage when they change - ALWAYS save, never remove
  // IMPORTANT: Exclude visualizations (base64 images) to avoid localStorage quota issues
  useEffect(() => {
    if (detectionResult && typeof window !== 'undefined' && window.localStorage) {
      try {
        // Create a lightweight version without base64 images for localStorage
        const lightweightResult = {
          location: detectionResult.location,
          parameters: detectionResult.parameters,
          results: detectionResult.results,
          insights: detectionResult.insights,
          metadata: detectionResult.metadata,
          // Exclude visualizations to save space
          // visualizations are kept in component state only
        };
        
        localStorage.setItem('satellite_detection_result', JSON.stringify(lightweightResult));
        setAnimationKey(prev => prev + 1);
        console.log('✅ Satellite detection result saved to localStorage (without images)');
      } catch (e: any) {
        if (e.name === 'QuotaExceededError') {
          console.error('❌ localStorage quota exceeded. Clearing old data and retrying...');
          // Try to clear and save again
          try {
            // Clear all satellite-related data
            localStorage.removeItem('satellite_detection_result');
            // Save lightweight version
            const lightweightResult = {
              location: detectionResult.location,
              parameters: detectionResult.parameters,
              results: detectionResult.results,
              insights: detectionResult.insights,
              metadata: detectionResult.metadata,
            };
            localStorage.setItem('satellite_detection_result', JSON.stringify(lightweightResult));
            console.log('✅ Retry successful after clearing old data');
          } catch (retryError) {
            console.error('❌ Failed to save even after clearing:', retryError);
          }
        } else {
          console.error('❌ Failed to save detection result:', e);
        }
      }
    }
    // Note: We intentionally don't clear localStorage when detectionResult is null
    // This ensures previous results persist even if state is reset
  }, [detectionResult]);

  // Close tooltips when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showThresholdTooltip && !target.closest('.threshold-tooltip-container')) {
        setShowThresholdTooltip(false);
      }
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
      if (showTimePreferenceTooltip && !target.closest('.timepreference-tooltip-container')) {
        setShowTimePreferenceTooltip(false);
      }
    };

    if (showThresholdTooltip || showLatitudeTooltip || showLongitudeTooltip || showLocationNameTooltip || showStartDateTooltip || showEndDateTooltip || showTimePreferenceTooltip) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showThresholdTooltip, showLatitudeTooltip, showLongitudeTooltip, showLocationNameTooltip, showStartDateTooltip, showEndDateTooltip, showTimePreferenceTooltip]);

  // Animate input section on mount
  useEffect(() => {
    setAnimationKey(1);
  }, []);

  const timePreferenceLabels: Record<string, string> = {
    "1": "Morning (06:00-12:00)",
    "2": "Afternoon (12:00-18:00)",
    "3": "Evening (18:00-24:00)",
    "4": "Any time",
  };

  const valueOrSpace = (value?: string) => (value ? value : "\u00A0");
  const formatPercentage = (value?: number) =>
    typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "\u00A0";
  const formatDecimal = (value?: number) =>
    typeof value === "number" ? value.toFixed(3) : "\u00A0";

  const resolveTimePreferenceLabel = (value?: string | number) => {
    if (!value) return "-";
    const key = value.toString();
    return timePreferenceLabels[key] || "-";
  };

  const resolvedLocationName = detectionResult?.location.name || locationName;
  const resolvedStartDate = detectionResult?.parameters.start_date || startDate;
  const resolvedEndDate = detectionResult?.parameters.end_date || endDate;
  const resolvedTimeChoice =
    detectionResult?.parameters.time_choice ?? (timePreference ? parseInt(timePreference, 10) : undefined);
  const resolvedTimeLabel = resolveTimePreferenceLabel(resolvedTimeChoice);
  const resolvedThreshold =
    detectionResult?.parameters.cloud_threshold ?? (cloudThreshold ? parseInt(cloudThreshold, 10) : undefined);

  const handleRunDetection = async () => {
    if (!latitude || !longitude || !startDate || !endDate || !timePreference || !cloudThreshold) {
      setError("Please fill in all required fields before running detection.");
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
        time_choice: parseInt(timePreference, 10),
        cloud_threshold: parseInt(cloudThreshold, 10),
        location_name: locationName || undefined,
      };

      const response = await apiClient.runCloudDetection(payload);
      setDetectionResult(response);
      // Only update localStorage on successful detection
      // Previous result stays in localStorage until new one succeeds
    } catch (err: any) {
      setError(err?.message || "Failed to run cloud detection. Please try again.");
      // Don't clear previous result - keep it in localStorage
      // setDetectionResult(null); // Removed - keep previous result
    } finally {
      setIsLoading(false);
    }
  };

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
        <div className="absolute top-0 left-0 w-72 h-72 bg-green-500/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/3 right-0 w-96 h-96 bg-green-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-green-500/30 rounded-full blur-3xl animate-pulse delay-2000"></div>
      </div>
      {/* Content Overlay */}
      <div className="relative z-10">
      <div className="container mx-auto py-8">
        {/* Header */}
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
            Satellite Imagery
          </h1>
          <p 
            className="text-slate-600 dark:text-slate-300"
            style={{
              animation: 'fadeInLeft 0.5s ease-out 0.2s both'
            }}
          >
            Real-time satellite data and cloud cover analysis
          </p>
        </div>

        {/* AI-Powered Cloud Detection & Analysis */}
        <div className="mb-8">
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div 
              ref={inputSectionRef}
              className="bg-slate-900/80 border border-slate-700 rounded-xl p-6 shadow-lg space-y-6"
              style={{
                animation: 'fadeInUp 0.6s ease-out'
              }}
            >
              <div>
                <h2 
                  className="text-2xl font-semibold text-slate-100 tracking-wide mb-1"
                  style={{
                    animation: 'fadeInLeft 0.5s ease-out 0.1s both'
                  }}
                >
                  CLOUD DETECTION INPUT DETAILS
                </h2>
                <p 
                  className="text-slate-300"
                  style={{
                    animation: 'fadeInLeft 0.5s ease-out 0.2s both'
                  }}
                >
                  Configure location coordinates, time parameters, and cloud detection thresholds for analysis.
                </p>
              </div>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
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
                  className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <div className="flex items-center gap-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
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
                  className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <div className="flex items-center gap-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
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
                          This helps identify the location in results and reports. If not provided, the system will use the coordinates.
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
                  className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>

              <div className="space-y-4">
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                  Time and Date Parameters:
                </p>
                <div className="space-y-2 text-slate-100">
                  <div className="flex items-center gap-2">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
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
                            The start date defines the beginning of the time period for satellite image analysis. 
                            Select a date in YYYY-MM-DD format. The system will search for satellite images from this date onwards.
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
                    className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <div className="flex items-center gap-2">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
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
                            The end date defines the conclusion of the time period for satellite image analysis. 
                            Select a date in YYYY-MM-DD format. The system will search for satellite images up to this date. 
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
                    className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                        Time of Day Preference
                      </p>
                      <div className="relative timepreference-tooltip-container">
                        <button
                          type="button"
                          onClick={() => setShowTimePreferenceTooltip(!showTimePreferenceTooltip)}
                          className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                          aria-label="Information about Time of Day Preference"
                        >
                          <Info className="w-3.5 h-3.5" />
                        </button>
                        {showTimePreferenceTooltip && (
                          <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                            <p className="text-xs text-slate-200 leading-relaxed">
                              Select your preferred time of day for satellite image capture. Options: 1) Morning (06:00-12:00), 
                              2) Afternoon (12:00-18:00), 3) Evening (18:00-24:00), or 4) Any time. This helps filter satellite 
                              images to match your analysis needs.
                            </p>
                            <div className="absolute bottom-0 left-4 transform translate-y-full">
                              <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-600"></div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    <ul className="space-y-1 text-slate-100 text-sm mb-3">
                      <li>1. Morning (06:00-12:00)</li>
                      <li>2. Afternoon (12:00-18:00)</li>
                      <li>3. Evening (18:00-24:00)</li>
                      <li>4. Any time</li>
                    </ul>
                    <select
                      value={timePreference}
                      onChange={(e) => setTimePreference(e.target.value)}
                      className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="">Select preference</option>
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="3">3</option>
                      <option value="4">4</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                      Cloud Coverage Threshold %
                    </p>
                    <div className="relative threshold-tooltip-container">
                      <button
                        type="button"
                        onClick={() => setShowThresholdTooltip(!showThresholdTooltip)}
                        className="text-slate-400 hover:text-slate-300 transition-colors focus:outline-none"
                        aria-label="Information about Cloud Coverage Threshold"
                      >
                        <Info className="w-3.5 h-3.5" />
                      </button>
                      {showThresholdTooltip && (
                        <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-lg z-50">
                          <p className="text-xs text-slate-200 leading-relaxed">
                            The cloud coverage threshold determines the maximum percentage of cloud coverage 
                            allowed in satellite images. Images with cloud coverage above this threshold will 
                            be filtered out. Enter a value between 0-100% (e.g., 50 means images with more 
                            than 50% cloud coverage will be excluded).
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
                    min="0"
                    max="100"
                    value={cloudThreshold}
                    onChange={(e) => setCloudThreshold(e.target.value)}
                    placeholder="e.g. 50"
                    className="w-full rounded-lg border border-slate-700 bg-slate-800/70 px-4 py-2 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>
              </div>

              <div className="space-y-2 text-slate-100">
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                  Time Parameters:
                </p>
                <ul className="space-y-1">
                  <li>• Start Date: {startDate || "-"}</li>
                  <li>• End Date: {endDate || "-"}</li>
                  <li>
                    • Time Preference:{" "}
                    {timePreference ? timePreferenceLabels[timePreference] : "-"}
                  </li>
                  <li>• Cloud Threshold: {cloudThreshold ? `${cloudThreshold}%` : "-"}</li>
                </ul>
              </div>
              {error && (
                <div className="p-3 rounded-lg border border-red-500/40 bg-red-500/10 text-red-200 text-sm">
                  {error}
                </div>
              )}
              <button
                onClick={handleRunDetection}
                disabled={isLoading}
                className="w-full rounded-lg bg-green-500/90 hover:bg-green-500 text-white font-semibold py-3 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? "Analyzing..." : "Run Cloud Detection"}
              </button>
              {detectionResult && !isLoading && (
                <p className="text-sm text-emerald-300">
                  Analysis completed at {new Date(detectionResult.metadata.generated_at).toLocaleString()}
                </p>
              )}
            </div>

            <div 
              ref={outputSectionRef}
              key={animationKey}
              className="bg-slate-900/80 border border-slate-700 rounded-xl p-6 shadow-lg space-y-6"
              style={{
                animation: animationKey > 0 ? 'fadeInUp 0.6s ease-out' : 'fadeInUp 0.6s ease-out'
              }}
            >
              <div className="mb-6">
                <h2 
                  className="text-2xl font-semibold text-slate-100 tracking-wide mb-1"
                  style={{
                    animation: animationKey > 0 ? 'fadeInRight 0.5s ease-out 0.1s both' : 'fadeInRight 0.5s ease-out 0.1s both'
                  }}
                >
                  CLOUD DETECTION OUTPUT DETAILS
                </h2>
                <p 
                  className="text-slate-300"
                  style={{
                    animation: animationKey > 0 ? 'fadeInRight 0.5s ease-out 0.2s both' : 'fadeInRight 0.5s ease-out 0.2s both'
                  }}
                >
                  Summaries from cloud detection analysis and atmospheric modeling.
                </p>
              </div>
              <div className="space-y-8 text-slate-100">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">1. Location:</p>
                  <p className="min-h-6 text-lg font-semibold">
                    {valueOrSpace(resolvedLocationName)}
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">2. Analysis Period:</p>
                  <p className="min-h-6 text-lg font-semibold">
                    {resolvedStartDate && resolvedEndDate ? `${resolvedStartDate} → ${resolvedEndDate}` : "\u00A0"}
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">3. Time Preference:</p>
                  <p className="min-h-6 text-lg font-semibold">
                    {valueOrSpace(resolvedTimeLabel)}
                  </p>
                </div>
                <div className="space-y-3">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">4. Cloud Coverage</p>
                  <div className="space-y-2 text-lg">
                    <p className="min-h-6 text-lg font-semibold">
                      {resolvedThreshold !== undefined ? `${resolvedThreshold}%` : "\u00A0"}
                    </p>
                    <p className="min-h-6 text-base">
                      • Cloud Type: {valueOrSpace(detectionResult?.results.cloud_type || "")}
                    </p>
                    <p className="min-h-6 text-base">
                      • Cloud Coverage: {formatPercentage(detectionResult?.results.cloud_coverage)}
                    </p>
                    <p className="min-h-6 text-base">
                      • Cloud Density: {formatDecimal(detectionResult?.results.cloud_density)}
                    </p>
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">5. Detailed Analysis:</p>
                  <div className="min-h-6 space-y-3 text-sm bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                    {detectionResult?.insights.analysis?.length ? (
                      <div className="space-y-2">
                        {detectionResult.insights.analysis.map((item, idx) => (
                          <p key={idx} className="text-slate-200 leading-relaxed">• {item}</p>
                        ))}
                      </div>
                    ) : detectionResult ? (
                      <div className="space-y-2 text-slate-300">
                        <p className="text-slate-400 italic">Analysis summary will appear here after detection completes.</p>
                        <div className="space-y-1.5 pt-2 border-t border-slate-700">
                          <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">Expected Information:</p>
                          <ul className="space-y-1 text-slate-400 text-xs">
                            <li>• Cloud pattern identification</li>
                            <li>• Coverage percentage breakdown</li>
                            <li>• Atmospheric conditions assessment</li>
                            <li>• Temporal analysis insights</li>
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2 text-slate-300">
                        <p className="text-slate-400 italic">Run cloud detection to generate detailed analysis.</p>
                        <div className="space-y-1.5 pt-2 border-t border-slate-700">
                          <p className="text-xs uppercase tracking-[0.2em] text-slate-500 mb-1">Analysis Will Include:</p>
                          <ul className="space-y-1 text-slate-400 text-xs">
                            <li>• Cloud pattern identification and classification</li>
                            <li>• Coverage percentage breakdown by time period</li>
                            <li>• Atmospheric conditions assessment</li>
                            <li>• Temporal analysis and trend insights</li>
                            <li>• Location-specific weather patterns</li>
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                {(detectionResult?.insights.location?.length ?? 0) > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-400">6. Location Insights:</p>
                    <div className="space-y-1 text-sm text-slate-300">
                      {detectionResult.insights.location.map((line, idx) => (
                        <p key={idx}>{line}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

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
                  SATELLITE IMAGERY VISUAL OUTPUT
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

              <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white tracking-wide">
                    Cloud Detection Visualization
                  </h3>
                  <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                    Composite Image
                  </span>
                </div>
                <div className="w-full">
                  {detectionResult?.visualizations?.composite ? (
                    <img
                      src={`data:image/png;base64,${detectionResult.visualizations.composite}`}
                      alt="Cloud detection visualization"
                      className="w-full rounded-lg border border-slate-700"
                      style={{
                        imageRendering: 'auto',
                        filter: 'contrast(1.15) brightness(1.08) saturate(1.1)',
                        WebkitBackfaceVisibility: 'hidden',
                        backfaceVisibility: 'hidden',
                      }}
                      loading="eager"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-96 text-slate-400 border border-slate-700 rounded-lg">
                      <p>Run detection to view visualization</p>
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

export default Satellite;
