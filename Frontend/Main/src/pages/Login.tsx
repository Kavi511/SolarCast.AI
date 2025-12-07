import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import earthHero from "@/assets/earth-hero.jpg";
import { Eye, EyeOff } from "lucide-react";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const { setIsLoggedIn, setUser } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    
    // Validate empty fields
    const emailTrimmed = email.trim();
    const passwordTrimmed = password.trim();
    
    if (!emailTrimmed && !passwordTrimmed) {
      setError("Please enter your email and password.");
      console.log("Validation: Both email and password are empty");
      return;
    }
    
    // Validate email format
    if (emailTrimmed) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(emailTrimmed)) {
        setError("Invalid email format.");
        console.log("Validation: Invalid email format");
        return;
      }
    }
    
    setLoading(true);

    // ALERT: Starting login process
    console.log("LOGIN PROCESS STARTED");
    console.log("Sending request to backend:", { email, password: "***" });

    // CRITICAL: Clear any existing tokens and auth state BEFORE login attempt
    apiClient.logout();
    setIsLoggedIn(false);
    setUser(null);

    try {
      // ALERT: Making API call
      console.log("API CALL: POST /api/auth/login");
      
      // CRITICAL: Direct backend call - backend MUST validate credentials
      const response = await apiClient.login({ email, password });
      
      // ALERT: Received response
      console.log("RESPONSE RECEIVED:", response);
      console.log("Response Status: 200 OK");
      
      // CRITICAL: Verify response is complete and valid
      if (!response) {
        console.error("ERROR: No response from server");
        throw new Error("Invalid response from server - authentication failed");
      }
      
      if (!response.access_token) {
        console.error("ERROR: No access token in response");
        throw new Error("Invalid response from server - no access token");
      }
      
      if (!response.user) {
        console.error("ERROR: No user data in response");
        throw new Error("Invalid response from server - no user data");
      }
      
      // ALERT: Response validation passed
      console.log("RESPONSE VALIDATION: All checks passed");
      console.log("Token received:", response.access_token.substring(0, 20) + "...");
      console.log("User data:", response.user);
      
      // CRITICAL: Verify token was stored in localStorage
      const storedToken = localStorage.getItem('auth_token');
      if (!storedToken || storedToken !== response.access_token) {
        console.error("ERROR: Token storage verification failed");
        apiClient.logout();
        setIsLoggedIn(false);
        setUser(null);
        throw new Error("Failed to store authentication token");
      }
      
      // ALERT: Token stored successfully
      console.log("TOKEN STORED: Successfully saved to localStorage");
      
      // CRITICAL: Only update auth state if ALL checks passed
      console.log("AUTH STATE: Setting isLoggedIn = true");
      setIsLoggedIn(true);
      setUser(response.user);
      
      // Show success message
      setSuccess("Successfully logging");
      console.log("LOGIN SUCCESSFUL!");
      
      // Navigate to home ONLY after successful authentication
      // Small delay to show success message
      setTimeout(() => {
        navigate("/");
      }, 500);
    } catch (err: any) {
      // ALERT: Error caught
      console.error("ERROR CAUGHT:", err);
      
      // Clear success message on error
      setSuccess("");
      
      // Determine response code
      const statusCode = err?.status || err?.response?.status || "Unknown";
      console.log("Response Status Code:", statusCode);
      
      // CRITICAL: Ensure we're logged out on ANY error
      console.log("CLEARING AUTH STATE: Logging out due to error");
      apiClient.logout();
      setIsLoggedIn(false);
      setUser(null);
      
      // CRITICAL: Verify tokens are cleared
      const tokenAfterError = localStorage.getItem('auth_token');
      if (tokenAfterError) {
        console.error("SECURITY ISSUE: Token still exists after error - clearing now!");
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
      }
      
      // Extract error message from backend response
      let errorMessage = "Invalid email or password. Please check your credentials.";
      
      // Properly extract error message
      if (err) {
        if (typeof err === 'string') {
          errorMessage = err;
        } else if (err.message && typeof err.message === 'string') {
          errorMessage = err.message;
        } else if (err.detail && typeof err.detail === 'string') {
          errorMessage = err.detail;
        } else if (typeof err === 'object') {
          const msg = err.message || err.detail || err.error || err.msg;
          if (msg && typeof msg === 'string') {
            errorMessage = msg;
          }
        }
      }
      
      // Handle 401 errors specifically
      if (statusCode === 401) {
        if (errorMessage.includes("not registered")) {
          // Keep the "Email not registered..." message
        } else if (errorMessage.includes("Incorrect password")) {
          // Keep the "Incorrect password..." message
        } else {
          errorMessage = "Invalid email or password. Please check your credentials.";
        }
      }
      
      // Ensure errorMessage is always a string
      if (typeof errorMessage !== 'string') {
        errorMessage = "Invalid email or password. Please check your credentials.";
      }
      
      // Show error message in UI
      setError(errorMessage);
      console.error("ERROR MESSAGE DISPLAYED:", errorMessage);
      console.error("FULL ERROR OBJECT:", JSON.stringify(err, null, 2));
      
      // DO NOT navigate on error - stay on login page
    } finally {
      setLoading(false);
      console.log("LOGIN PROCESS COMPLETED");
    }
  };


    return (
    <div
      className="min-h-screen bg-background bg-cover bg-center bg-no-repeat relative dark"
      style={{ backgroundImage: `url(${earthHero})` }}
    >
      <div className="absolute inset-0 bg-background/40"></div>
      <div className="relative z-10">
        {/* Navigation handled by main App.tsx */}
        <div className="flex items-center justify-center min-h-screen py-16 px-4">
        <div className="w-full max-w-md">
          <Card className="bg-white/10 backdrop-blur-md border-white/20">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl font-bold text-white drop-shadow-lg">
                Welcome Back
              </CardTitle>
              <CardDescription className="text-white/90 text-sm md:text-base">
                Sign in to your SolarCast.AI account to access advanced solar forecasting
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {error && (
                  <div className="bg-red-500/20 border border-red-500/50 text-red-200 p-3 rounded text-sm">
                    {typeof error === 'string' ? error : "Invalid email or password. Please check your credentials."}
                  </div>
                )}
                
                {success && (
                  <div className="bg-green-500/20 border border-green-500/50 text-green-200 p-3 rounded text-sm">
                    {success}
                  </div>
                )}
                
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="text"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-white/70 hover:text-white transition-colors"
                    >
                      {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                  </div>
                </div>
                
                <Button type="submit" variant="hero" size="lg" className="w-full" disabled={loading}>
                  {loading ? "Signing In..." : "Sign In"}
                </Button>
                
                <div className="text-center space-y-2">
                  <Link to="/forgot-password" className="text-sm text-primary hover:underline">
                    Forgot your password?
                  </Link>
                  
                  <p className="text-sm text-muted-foreground">
                    Don't have an account?{" "}
                    <Link to="/register" className="text-primary hover:underline">
                      Sign up
                    </Link>
                  </p>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
        </div>
      </div>
    </div>
  );
};

export default Login;