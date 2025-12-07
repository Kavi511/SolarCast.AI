import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { ProtectedRoute } from "./components/ProtectedRoute";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Register from "./pages/Register";
import ForgotPassword from "./pages/ForgotPassword";
import NotFound from "./pages/NotFound";
import Dashboard from "./pages/Dashboard";
import Satellite from "./pages/Satellite";
import Weather from "./pages/Weather";
import Solar from "./pages/Solar";
import Advanced from "./pages/Advanced";
import Profile from "./pages/Profile";
import Header from "./components/Header";
import Footer from "./components/Footer";
import HeroSection from "./components/HeroSection";
import FeaturesSection from "./components/FeaturesSection";
import TechnologySection from "./components/TechnologySection";
import CTASection from "./components/CTASection";


const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <TooltipProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <Header />
              <main>
                <Routes>
                  <Route path="/" element={
                    <div className="bg-black dark" data-main-page="true">
                      <HeroSection />
                      <FeaturesSection />
                      <TechnologySection />
                      <CTASection />
                      <Index />
                      <Footer />
                    </div>
                  } />
                  <Route path="/login" element={<Login />} />
                  <Route path="/register" element={<Register />} />
                  <Route path="/forgot-password" element={<ForgotPassword />} />
                  <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
                  <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
                  <Route path="/satellite" element={<ProtectedRoute><Satellite /></ProtectedRoute>} />
                  <Route path="/weather" element={<ProtectedRoute><Weather /></ProtectedRoute>} />
                  <Route path="/solar" element={<ProtectedRoute><Solar /></ProtectedRoute>} />
                  <Route path="/advanced" element={<ProtectedRoute><Advanced /></ProtectedRoute>} />
                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </main>
            </BrowserRouter>
          </TooltipProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
