import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { apiClient, User } from '@/lib/api';

interface AuthContextType {
  isLoggedIn: boolean;
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  register: (email: string, password: string, companyName?: string, accountType?: string) => Promise<void>;
  setIsLoggedIn: (value: boolean) => void;
  setUser: (user: User | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false); // Start as false, verify token on mount
  const [user, setUser] = useState<User | null>(null);

  const login = async (email: string, password: string) => {
    // Clear any existing tokens before attempting login
    apiClient.logout();
    setIsLoggedIn(false);
    setUser(null);
    
    try {
      const response = await apiClient.login({ email, password });
      setIsLoggedIn(true);
      setUser(response.user);
    } catch (error: any) {
      // Ensure we're logged out on failure
      apiClient.logout();
      setIsLoggedIn(false);
      setUser(null);
      throw new Error(error.message || 'Login failed');
    }
  };

  const register = async (email: string, password: string, companyName?: string, accountType?: string) => {
    try {
      const userData = await apiClient.register({ email, password, company_name: companyName, account_type: accountType });
      // Auto-login after registration
      const loginResponse = await apiClient.login({ email, password });
      setIsLoggedIn(true);
      setUser(loginResponse.user);
    } catch (error: any) {
      throw new Error(error.message || 'Registration failed');
    }
  };

  const logout = () => {
    apiClient.logout();
    setIsLoggedIn(false);
    setUser(null);
  };

  // Restore auth state from localStorage on mount, then verify token in background
  useEffect(() => {
    const restoreAuthState = async () => {
      const token = localStorage.getItem('auth_token');
      const userStr = localStorage.getItem('user');
      
      if (token && userStr) {
        try {
          // First, restore user from localStorage immediately (optimistic restore)
          const savedUser = JSON.parse(userStr);
          setUser(savedUser);
          setIsLoggedIn(true);
          
          // Then verify token in background (non-blocking)
          try {
            const currentUser = await apiClient.getCurrentUser();
            // Update with fresh user data if verification succeeds
            setUser(currentUser);
            // Update localStorage with fresh user data
            localStorage.setItem('user', JSON.stringify(currentUser));
          } catch (verifyError: any) {
            // Only clear tokens if we get a 401 (unauthorized) - token is definitely invalid
            // For network errors or other issues, keep the user logged in
            if (verifyError?.status === 401 || verifyError?.response?.status === 401) {
              console.log("Token verification failed with 401 - clearing auth state");
              apiClient.logout();
              setIsLoggedIn(false);
              setUser(null);
            } else {
              // Network error or other issue - keep user logged in with saved data
              console.log("Token verification failed but keeping user logged in:", verifyError);
            }
          }
        } catch (parseError) {
          // Failed to parse user data - clear everything
          console.error("Failed to parse user data from localStorage:", parseError);
          apiClient.logout();
          setIsLoggedIn(false);
          setUser(null);
        }
      } else {
        // No token or user data, ensure we're logged out
        setIsLoggedIn(false);
        setUser(null);
      }
    };
    restoreAuthState();
  }, []);

  const value = {
    isLoggedIn,
    user,
    login,
    logout,
    register,
    setIsLoggedIn,
    setUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
