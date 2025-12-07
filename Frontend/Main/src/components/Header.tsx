import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { User, LogIn, LogOut, Menu, X, RefreshCw, Clock, Sun, Moon } from "lucide-react";

const Header = () => {
  const { isLoggedIn, logout } = useAuth();
  const { isDarkMode, toggleDarkMode } = useTheme();
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());


  const isAuthPage = location.pathname === '/login' || location.pathname === '/register';

  // Auto-refresh timer effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (autoRefresh && isLoggedIn) {
      interval = setInterval(() => {
        setCurrentTime(new Date());
        // Trigger data refresh here
        console.log('Auto-refreshing data...');
      }, 30000); // Refresh every 30 seconds
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, isLoggedIn]);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);



  // Toggle auto-refresh
  const toggleAutoRefresh = () => {
    setAutoRefresh(!autoRefresh);
  };

  // Format time, date, and day
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: true, 
      hour: 'numeric', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const formatDay = (date: Date) => {
    return date.toLocaleDateString('en-US', { weekday: 'long' });
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-gray-900/95 backdrop-blur supports-[backdrop-filter]:bg-gray-900/60 border-b border-gray-800 dark-mode-transition">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-4">
            <Link to="/" className={`text-2xl font-bold ${isAuthPage ? 'text-white' : 'text-foreground'}`}>
              SolarCast.AI
            </Link>
          </div>

          {/* Desktop Navigation */}
          {!isAuthPage && (
            <nav className="hidden md:flex items-center space-x-8">
              {isLoggedIn ? (
                // Logged in navigation items
                <>
                  <Link 
                    to="/dashboard" 
                    className={`transition-all duration-150 ease-in-out ${
                      location.pathname === '/dashboard' 
                        ? 'text-foreground font-semibold border-b-2 border-blue-500 pb-1' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Dashboard
                  </Link>
                  <Link 
                    to="/satellite" 
                    className={`transition-all duration-150 ease-in-out ${
                      location.pathname === '/satellite' 
                        ? 'text-foreground font-semibold border-b-2 border-blue-500 pb-1' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Satellite
                  </Link>
                  <Link 
                    to="/weather" 
                    className={`transition-all duration-150 ease-in-out ${
                      location.pathname === '/weather' 
                        ? 'text-foreground font-semibold border-b-2 border-blue-500 pb-1' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Weather
                  </Link>
                  <Link 
                    to="/solar" 
                    className={`transition-all duration-150 ease-in-out ${
                      location.pathname === '/solar' 
                        ? 'text-foreground font-semibold border-b-2 border-blue-500 pb-1' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Solar
                  </Link>
                  <Link 
                    to="/advanced" 
                    className={`transition-all duration-150 ease-in-out ${
                      location.pathname === '/advanced' 
                        ? 'text-foreground font-semibold border-b-2 border-blue-500 pb-1' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Advanced
                  </Link>
                </>
              ) : (
                // Non-logged in navigation items
                <>
                  <a href="#features" className="text-muted-foreground hover:text-foreground transition-smooth">
                    Features
                  </a>
                  <a href="#technology" className="text-muted-foreground hover:text-foreground transition-smooth">
                    Technology
                  </a>
                  <a href="#about" className="text-muted-foreground hover:text-foreground transition-smooth">
                    About
                  </a>
                </>
              )}
            </nav>
                           )}

                 {/* Right side - Time, Location, Controls, Auth */}
          <div className="flex items-center space-x-4">
                        {/* Time Display (only for logged-in users) */}
            {isLoggedIn && !isAuthPage && (
              <div className="hidden lg:flex items-center space-x-4 text-sm">
                {/* Time */}
                <div className="flex items-center space-x-1 text-muted-foreground">
                  <Clock className="w-4 h-4" />
                  <span>{formatTime(currentTime)}</span>
                </div>

                {/* Date */}
                <div className="text-muted-foreground">
                  {formatDate(currentTime)}
                </div>
              </div>
            )}

            {/* Control Buttons (only for logged-in users) */}
            {/* Dark mode toggle removed - dashboard pages are forced to dark mode */}

            {/* Auth Buttons */}
            <div className="flex items-center space-x-2">
              {isLoggedIn ? (
                <>
                  <Link to="/profile">
                    <Button
                      variant="outline"
                      className="flex items-center space-x-2"
                    >
                      <User className="w-4 h-4" />
                      <span className="hidden sm:inline">Profile</span>
                    </Button>
                  </Link>
                  <Button
                    variant="outline"
                    onClick={logout}
                    className="flex items-center space-x-2"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="hidden sm:inline">Logout</span>
                  </Button>
                </>
              ) : (
                <>
                  <Link to="/login">
                    <Button variant="outline" className="flex items-center space-x-2">
                      <LogIn className="w-4 h-4" />
                      <span className="hidden sm:inline">Login</span>
                    </Button>
                  </Link>
                  <Link to="/register">
                    <Button className="flex items-center space-x-2">
                      <User className="w-4 h-4" />
                      <span className="hidden sm:inline">Register</span>
                    </Button>
                  </Link>
                </>
              )}
            </div>

            {/* Mobile Menu Button */}
            {!isAuthPage && (
              <Button
                variant="ghost"
                size="sm"
                className="md:hidden"
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              >
                {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            )}
          </div>
        </div>

        {/* Mobile Menu */}
        {!isAuthPage && isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-border/50">
            {/* Mobile Navigation */}
            <nav className="flex flex-col space-y-4 mb-4">
              {isLoggedIn ? (
                // Logged in mobile navigation
                <>
                  <Link
                    to="/dashboard"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/dashboard' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Dashboard
                  </Link>
                  <Link
                    to="/satellite"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/satellite' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Satellite
                  </Link>
                  <Link
                    to="/weather"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/weather' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Weather
                  </Link>
                  <Link
                    to="/solar"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/solar' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Solar
                  </Link>
                  <Link
                    to="/advanced"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/advanced' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Advanced
                  </Link>
                  <Link
                    to="/profile"
                    className={`transition-all duration-150 ease-in-out px-2 py-1 ${
                      location.pathname === '/profile' 
                        ? 'text-foreground font-semibold bg-slate-800/50 rounded border-l-4 border-blue-500' 
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    Profile
                  </Link>
                </>
              ) : (
                // Non-logged in mobile navigation
                <>
                  <a href="#features" className="text-muted-foreground hover:text-foreground transition-smooth px-2 py-1">
                    Features
                  </a>
                  <a href="#technology" className="text-muted-foreground hover:text-foreground transition-smooth px-2 py-1">
                    Technology
                  </a>
                  <a href="#about" className="text-muted-foreground hover:text-foreground transition-smooth px-2 py-1">
                    About
                  </a>
                </>
              )}
                               </nav>

                   {/* Mobile Controls for Logged-in Users */}
            {isLoggedIn && (
              <div className="border-t border-border/50 pt-4">
                {/* Mobile Time */}
                <div className="space-y-2 mb-4 text-sm">
                  <div className="flex items-center space-x-2 text-muted-foreground">
                    <Clock className="w-4 h-4" />
                    <span>{formatTime(currentTime)}</span>
                  </div>
                  <div className="text-muted-foreground">
                    {formatDate(currentTime)}
                  </div>
                </div>

                {/* Mobile Control Buttons - Dark mode toggle removed */}
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;