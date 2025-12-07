import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api";
import earthHero from "@/assets/earth-hero.jpg";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    // Validate email
    const emailTrimmed = email.trim();
    
    if (!emailTrimmed) {
      setError("Please enter your email address.");
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailTrimmed)) {
      setError("Invalid email format.");
      return;
    }

    setLoading(true);

    try {
      const response = await apiClient.forgotPassword(emailTrimmed);
      setSuccess(response.message || "Password reset instructions have been sent to your email address. Please check your inbox.");
      setEmail("");
    } catch (err: any) {
      const errorMessage = err?.message || err?.detail || "Failed to send password reset email. Please try again.";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-background bg-cover bg-center bg-no-repeat relative dark"
      style={{ backgroundImage: `url(${earthHero})` }}
    >
      <div className="absolute inset-0 bg-background/40"></div>
      <div className="relative z-10">
        <div className="flex items-center justify-center min-h-screen py-16 px-4">
          <div className="w-full max-w-md">
            <Card className="bg-white/10 backdrop-blur-md border-white/20">
              <CardHeader className="text-center">
                <CardTitle className="text-3xl font-bold text-white drop-shadow-lg">
                  Forgot Password
                </CardTitle>
                <CardDescription className="text-white/90 text-sm md:text-base">
                  Enter your email address and we'll send you instructions to reset your password
                </CardDescription>
              </CardHeader>
              
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  {error && (
                    <div className="bg-red-500/20 border border-red-500/50 text-red-200 p-3 rounded text-sm">
                      {error}
                    </div>
                  )}
                  
                  {success && (
                    <div className="bg-green-500/20 border border-green-500/50 text-green-200 p-3 rounded text-sm">
                      {success}
                    </div>
                  )}
                  
                  <div className="space-y-2">
                    <Label htmlFor="email">Email Address</Label>
                    <Input
                      id="email"
                      type="text"
                      placeholder="Enter your email address"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70"
                    />
                  </div>
                  
                  <Button type="submit" variant="hero" size="lg" className="w-full" disabled={loading}>
                    {loading ? "Sending..." : "Send Reset Instructions"}
                  </Button>
                  
                  <div className="text-center space-y-2">
                    <Link to="/login" className="text-sm text-primary hover:underline block">
                      Back to Login
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

export default ForgotPassword;

