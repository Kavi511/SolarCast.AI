import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAuth } from "@/contexts/AuthContext";
import earthHero from "@/assets/earth-hero.jpg";
import { Eye, EyeOff } from "lucide-react";

const Register = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    companyName: "",
    accountType: "",
    agreeToTerms: false,
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const navigate = useNavigate();
  const { register } = useAuth();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (error) setError("");
  };

  const handleSelectChange = (value: string) => {
    setFormData(prev => ({ ...prev, accountType: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    // Check if all fields are empty
    const emailTrimmed = formData.email.trim();
    const passwordTrimmed = formData.password.trim();
    const confirmPasswordTrimmed = formData.confirmPassword.trim();
    const companyNameTrimmed = formData.companyName.trim();
    
    if (!emailTrimmed && !passwordTrimmed && !confirmPasswordTrimmed && !companyNameTrimmed && !formData.accountType && !formData.agreeToTerms) {
      setError("All fields are required");
      return;
    }

    // Check if company/organization name is missing
    if (!companyNameTrimmed) {
      setError("Please enter company/organization name");
      return;
    }

    // Check if account type is missing
    if (!formData.accountType) {
      setError("Please select account type");
      return;
    }

    if (!formData.email || !formData.password || !formData.agreeToTerms) {
      setError("Please fill in all required fields and agree to terms");
      return;
    }

    if (!formData.email.endsWith("@gmail.com")) {
      setError("Only Gmail addresses are allowed for registration");
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 8) {
      setError("Password must be at least 8 characters long");
      return;
    }

    setLoading(true);
    try {
      await register(
        formData.email,
        formData.password,
        formData.companyName || undefined,
        formData.accountType || undefined
      );
      
      // Show success message
      setSuccess("Account created successfully");
      
      // Navigate to profile page after showing success message
      setTimeout(() => {
        navigate("/profile");
      }, 1000);
    } catch (err: any) {
      setSuccess("");
      setError(err.message || "Registration failed. Please try again.");
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
        <div className="flex items-center justify-center min-h-screen pt-24 pb-16 px-4">
        <div className="w-full max-w-md">
          <Card className="bg-white/10 backdrop-blur-md border-white/20">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl font-bold text-white drop-shadow-lg">
                Join SolarCast.AI
              </CardTitle>
              <CardDescription className="text-white/90 text-sm md:text-base">
                Create your account and start leveraging satellite intelligence for solar forecasting
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
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
                  <Label htmlFor="companyName">Company/Organization Name</Label>
                  <Input
                    id="companyName"
                    name="companyName"
                    placeholder="Your company or organization (optional)"
                    value={formData.companyName}
                    onChange={handleInputChange}
                    className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="accountType">Account Type</Label>
                  <Select onValueChange={handleSelectChange}>
                    <SelectTrigger className="bg-white/10 backdrop-blur-md border-white/20 text-white">
                      <SelectValue placeholder="Select account type (optional)" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Individual">Individual</SelectItem>
                      <SelectItem value="Business">Business</SelectItem>
                      <SelectItem value="Enterprise">Enterprise</SelectItem>
                      <SelectItem value="Research">Research Institution</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="email">Email (Gmail only)</Label>
                  <Input
                    id="email"
                    name="email"
                    type="text"
                    placeholder="yourname@gmail.com"
                    value={formData.email}
                    onChange={handleInputChange}
                    className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      name="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Create a strong password (min 8 characters)"
                      value={formData.password}
                      onChange={handleInputChange}
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
                
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <div className="relative">
                    <Input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showConfirmPassword ? "text" : "password"}
                      placeholder="Confirm your password"
                      value={formData.confirmPassword}
                      onChange={handleInputChange}
                      className="bg-white/10 backdrop-blur-md border-white/20 text-white placeholder:text-white/70 pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-white/70 hover:text-white transition-colors"
                    >
                      {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="terms"
                    checked={formData.agreeToTerms}
                    onCheckedChange={(checked) => 
                      setFormData(prev => ({ ...prev, agreeToTerms: checked as boolean }))
                    }
                  />
                  <Label htmlFor="terms" className="text-sm">
                    I agree to the{" "}
                    <a href="#" className="text-primary hover:underline">
                      Terms of Service
                    </a>{" "}
                    and{" "}
                    <a href="#" className="text-primary hover:underline">
                      Privacy Policy
                    </a>
                  </Label>
                </div>
                
                <Button 
                  type="submit" 
                  variant="hero" 
                  size="lg" 
                  className="w-full"
                  disabled={loading || !formData.agreeToTerms || !formData.email.trim() || !formData.password.trim() || !formData.confirmPassword.trim() || !formData.companyName.trim() || !formData.accountType}
                >
                  {loading ? "Creating Account..." : "Create Account"}
                </Button>
                
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">
                    Already have an account?{" "}
                    <Link to="/login" className="text-primary hover:underline">
                      Sign in
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

export default Register;
