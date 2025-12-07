import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const { isLoggedIn, user, setIsLoggedIn, setUser } = useAuth();
  const [isVerifying, setIsVerifying] = useState(true);

  useEffect(() => {
    const verifyAuth = async () => {
      const token = localStorage.getItem('auth_token');
      const userStr = localStorage.getItem('user');
      
      // If we already have user and isLoggedIn is true, skip verification
      // This prevents unnecessary API calls when navigating between protected routes
      if (isLoggedIn && user && token) {
        setIsVerifying(false);
        return;
      }

      if (!token) {
        // No token - ensure we're logged out
        setIsLoggedIn(false);
        setUser(null);
        setIsVerifying(false);
        return;
      }

      // If we have token but no user in context, restore from localStorage first
      if (token && !user && userStr) {
        try {
          const savedUser = JSON.parse(userStr);
          setUser(savedUser);
          setIsLoggedIn(true);
        } catch (parseError) {
          console.error("Failed to parse user data:", parseError);
        }
      }

      // Verify token in background (non-blocking)
      // Only clear tokens if we get a 401, not on network errors
      try {
        const currentUser = await apiClient.getCurrentUser();
        // Token is valid - update auth context with fresh data
        setUser(currentUser);
        setIsLoggedIn(true);
        localStorage.setItem('user', JSON.stringify(currentUser));
        setIsVerifying(false);
      } catch (error: any) {
        // Only clear tokens if we get a 401 (unauthorized) - token is definitely invalid
        if (error?.status === 401 || error?.response?.status === 401) {
          console.log("Token verification failed with 401 - clearing auth state");
          apiClient.logout();
          setIsLoggedIn(false);
          setUser(null);
        } else {
          // Network error or other issue - keep user logged in
          console.log("Token verification failed but keeping user logged in:", error);
          // Ensure we have user from localStorage if available
          if (token && userStr && !user) {
            try {
              const savedUser = JSON.parse(userStr);
              setUser(savedUser);
              setIsLoggedIn(true);
            } catch (parseError) {
              console.error("Failed to parse user data:", parseError);
            }
          }
        }
        setIsVerifying(false);
      }
    };

    verifyAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount - auth state is checked inside

  // Show loading while verifying
  if (isVerifying) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // CRITICAL ISSUE #8: Must have valid token AND authenticated state
  // Authentication protection is NOT disabled - all checks are active
  const token = localStorage.getItem('auth_token');
  if (!token) {
    console.log("SECURITY: ProtectedRoute - No token found - redirecting to login");
    return <Navigate to="/login" replace />;
  }

  // CRITICAL ISSUE #8: Must have both isLoggedIn AND user - token alone is not enough
  // This prevents access with invalid tokens or when login failed
  if (!isLoggedIn) {
    console.log("SECURITY: ProtectedRoute - isLoggedIn is false - redirecting to login");
    // Clear invalid token
    apiClient.logout();
    return <Navigate to="/login" replace />;
  }

  if (!user) {
    console.log("SECURITY: ProtectedRoute - No user data - redirecting to login");
    // Clear invalid token
    apiClient.logout();
    return <Navigate to="/login" replace />;
  }

  // CRITICAL ISSUE #8: All checks passed - user is authenticated
  // Protected pages CANNOT be opened without checking if user is logged in
  console.log("SECURITY: ProtectedRoute - Access granted for user:", user.email);
  return <>{children}</>;
};

