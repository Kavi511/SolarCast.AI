// API Base Configuration
const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').trim();

// Types for API responses
export interface Site {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  created_at: string;
}

export interface SiteCreate {
  name: string;
  latitude: number;
  longitude: number;
}

export interface Observation {
  id: number;
  site_id: number;
  timestamp: string;
  temperature: number;
  humidity: number;
  wind_speed: number;
  solar_irradiance: number;
}

export interface ObservationCreate {
  site_id: number;
  temperature: number;
  humidity: number;
  wind_speed: number;
  solar_irradiance: number;
}

export interface MLResponse {
  cloud_cover_pct?: number;
  irradiance_wm2?: number;
  energy_output_kw?: number;
}

export interface CloudDetectionPayload {
  latitude: number;
  longitude: number;
  start_date: string;
  end_date: string;
  time_choice?: number;
  cloud_threshold?: number;
  location_name?: string;
}

export interface CloudDetectionResponse {
  location: {
    name: string;
    latitude: number;
    longitude: number;
    climate: string;
  };
  parameters: {
    start_date: string;
    end_date: string;
    time_choice: number;
    cloud_threshold: number;
  };
  results: {
    cloud_type: string;
    cloud_coverage: number;
    cloud_density: number;
    confidence: number;
  };
  visualizations: {
    composite?: string;
    satellite_image?: string;
    cloud_overlay?: string;
    mask?: string;
  };
  insights: {
    location: string[];
    analysis: string[];
  };
  metadata: {
    generated_at: string;
    cloud_threshold: number;
    time_choice: number;
  };
}

export interface CloudForecastingPayload {
  latitude: number;
  longitude: number;
  start_date: string;
  end_date: string;
  time_horizon?: number;
  future_date?: string;
  location_name?: string;
}

export interface CloudForecastingResponse {
  location: {
    name: string;
    latitude: number;
    longitude: number;
    climate: string;
  };
  parameters: {
    start_date: string;
    end_date: string;
    time_horizon: number;
    future_date?: string;
  };
  results: {
    predicted_condition: string;
    predicted_cloud_cover: string;
    prediction_confidence: string;
    average_direction: string;
    average_speed: string;
    seasonal_factor: string;
  };
  insights: {
    prediction: string[];
    location: string[];
  };
  metadata: {
    generated_at: string;
    time_horizon: number;
    num_forecasts: number;
  };
}

export interface SolarEnergyPredictionPayload {
  latitude: number;
  longitude: number;
  start_date: string;
  end_date: string;
  target_date: string;
  system_capacity_kw?: number;
  system_area_m2?: number;
  training_epochs?: number;
  learning_rate?: number;
  location_name?: string;
}

export interface SolarEnergyPredictionResponse {
  location: {
    name: string;
    latitude: number;
    longitude: number;
    climate: string;
  };
  parameters: {
    start_date: string;
    end_date: string;
    target_date: string;
    system_capacity_kw: number;
    system_area_m2: number;
    training_epochs: number;
    learning_rate: number;
  };
  results: {
    daily_energy_production: string;
    average_power: string;
    average_irradiance: string;
    prediction_confidence: string;
    capacity_factor: string;
    performance_ratio: string;
    estimated_revenue: string;
  };
  weather_conditions: {
    temperature: string;
    humidity: string;
    wind_speed: string;
    cloud_coverage: string;
  };
  insights: {
    prediction: string[];
    location: string[];
  };
  metadata: {
    generated_at: string;
    data_points: number;
    model_trained: boolean;
  };
}

export interface User {
  id: number;
  email: string;
  company_name: string | null;
  account_type: string | null;
  created_at: string;
}

export interface UserRegister {
  email: string;
  password: string;
  company_name?: string;
  account_type?: string;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// API Client class
class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const base = this.baseURL.trim().replace(/\/$/, ''); // Remove trailing slash if present
    const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${base}${path}`;
    const token = this.getAuthToken();
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      // CRITICAL: Handle non-OK responses - MUST reject for 401/403 errors
      if (!response.ok) {
        const statusCode = response.status;
        console.error(`HTTP ERROR: Status ${statusCode}`);
        console.error(`Response Status: ${statusCode} ${response.statusText}`);
        
        let errorMessage = `HTTP error! status: ${statusCode}`;
        let errorDetail = null;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
          errorDetail = errorData;
          console.error("Error Details:", errorDetail);
        } catch {
          // If JSON parsing fails, use status text
          errorMessage = response.statusText || errorMessage;
          console.error("Could not parse error JSON, using status text");
        }
        
        // Log specific error codes
        if (statusCode === 401) {
          console.error("401 UNAUTHORIZED: Invalid credentials");
        } else if (statusCode === 400) {
          console.error("400 BAD REQUEST: Invalid input");
        } else if (statusCode === 500) {
          console.error("500 INTERNAL SERVER ERROR: Server error");
        }
        
        // Create a proper error that will be caught
        const error = new Error(errorMessage);
        (error as any).status = statusCode;
        (error as any).response = response;
        (error as any).detail = errorDetail;
        
        console.error(`THROWING ERROR: ${errorMessage}`);
        throw error; // CRITICAL: This MUST throw to prevent success
      }
      
      // Handle empty responses
      const text = await response.text();
      if (!text) {
        return null as T;
      }
      
      try {
        const data = JSON.parse(text) as T;
        return data;
      } catch (parseError) {
        throw new Error('Invalid JSON response from server');
      }
    } catch (error: any) {
      console.error('API request failed:', error);
      
      // Handle network errors specifically
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        const url = new URL(this.baseURL);
        throw new Error(`Cannot connect to backend server at ${this.baseURL}. Please make sure the backend is running on port ${url.port || '8001'}.`);
      }
      
      // Re-throw with better message if it's already an Error
      if (error instanceof Error) {
        throw error;
      }
      
      // Fallback for unknown errors
      throw new Error(`Request failed: ${error?.message || 'Unknown error'}`);
    }
  }

  // Auth API
  async register(userData: UserRegister): Promise<User> {
    return this.request<User>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async login(credentials: UserLogin): Promise<TokenResponse> {
    // CRITICAL: Clear any existing tokens before attempting login
    this.logout();
    
    console.log("API CLIENT: Starting login request");
    console.log("Request URL:", `${this.baseURL}/auth/login`);
    console.log("Request Method: POST");
    console.log("Request Body:", { email: credentials.email, password: "***" });
    
    try {
      const response = await this.request<TokenResponse>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(credentials),
      });
      
      console.log("API CLIENT: Request successful");
      console.log("Response received:", {
        hasToken: !!response?.access_token,
        hasUser: !!response?.user,
        tokenType: response?.token_type
      });
      
      // CRITICAL: Only store token if login was successful and we have a valid response
      if (response && response.access_token && response.user) {
        console.log("API CLIENT: Storing token and user data");
        localStorage.setItem('auth_token', response.access_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        console.log("API CLIENT: Token and user data stored successfully");
        return response;
      } else {
        // Invalid response - clear tokens and throw error
        console.error("API CLIENT: Invalid response structure");
        this.logout();
        throw new Error('Invalid response from server - authentication failed');
      }
    } catch (error: any) {
      console.error("API CLIENT: Request failed");
      console.error("Error Status:", error?.status || "Unknown");
      console.error("Error Message:", error?.message || "Unknown error");
      
      // CRITICAL: Ensure tokens are cleared on ANY error (401, network error, etc.)
      this.logout();
      
      // Re-throw error so Login component can display it
      throw error;
    }
  }

  async forgotPassword(email: string): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  }

  async getCurrentUser(): Promise<User> {
    return this.request<User>('/api/auth/me');
  }

  async getUserProfile(userId: number): Promise<User> {
    return this.request<User>(`/api/auth/profile/${userId}`);
  }

  logout(): void {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  }

  // Sites API
  async getSites(): Promise<Site[]> {
    return this.request<Site[]>('/api/sites');
  }

  async createSite(site: SiteCreate): Promise<Site> {
    return this.request<Site>('/api/sites', {
      method: 'POST',
      body: JSON.stringify(site),
    });
  }

  async getSite(id: number): Promise<Site> {
    return this.request<Site>(`/api/sites/${id}`);
  }

  async deleteSite(id: number): Promise<{ status: string }> {
    return this.request<{ status: string }>(`/api/sites/${id}`, {
      method: 'DELETE',
    });
  }

  // Observations API
  async getObservations(siteId: number): Promise<Observation[]> {
    return this.request<Observation[]>(`/api/sites/${siteId}/observations`);
  }

  async createObservation(siteId: number, observation: Omit<ObservationCreate, 'site_id'>): Promise<Observation> {
    const payload = { ...observation, site_id: siteId };
    return this.request<Observation>(`/api/sites/${siteId}/observations`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // ML API
  async detectCloudCover(image: File): Promise<{ cloud_cover_pct: number }> {
    const formData = new FormData();
    formData.append('image', image);

    const response = await fetch(`${this.baseURL}/api/ml/cloud-detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async runCloudDetection(payload: CloudDetectionPayload): Promise<CloudDetectionResponse> {
    return this.request<CloudDetectionResponse>('/api/ml/cloud-detection/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async runCloudForecasting(payload: CloudForecastingPayload): Promise<CloudForecastingResponse> {
    return this.request<CloudForecastingResponse>('/api/ml/cloud-forecasting/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async runSolarEnergyPrediction(payload: SolarEnergyPredictionPayload): Promise<SolarEnergyPredictionResponse> {
    return this.request<SolarEnergyPredictionResponse>('/api/ml/solar-energy-prediction/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictIrradiance(cloudCoverPct: number): Promise<{ irradiance_wm2: number }> {
    const formData = new FormData();
    formData.append('cloud_cover_pct', cloudCoverPct.toString());

    const response = await fetch(`${this.baseURL}/api/ml/irradiance`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async predictEnergyOutput(
    irradianceWm2: number,
    panelAreaM2: number = 1.6,
    panelEfficiency: number = 0.20
  ): Promise<{ energy_output_kw: number }> {
    const formData = new FormData();
    formData.append('irradiance_wm2', irradianceWm2.toString());
    formData.append('panel_area_m2', panelAreaM2.toString());
    formData.append('panel_efficiency', panelEfficiency.toString());

    const response = await fetch(`${this.baseURL}/api/ml/energy-output`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<{ status: string; service: string }> {
    return this.request<{ status: string; service: string }>('/');
  }

  // Weather API status check
  async weatherStatus(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>('/api/ml/weather-status');
  }

  async geeStatus(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>('/api/ml/gee-status');
  }

  // Models status check
  async modelsStatus(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>('/api/ml/models-status');
  }

  // Models health metrics
  async modelsHealth(): Promise<{ models: any[]; timestamp: number }> {
    return this.request<{ models: any[]; timestamp: number }>('/api/ml/models-health');
  }
}

// Export singleton instance
export const apiClient = new ApiClient(API_BASE_URL);
