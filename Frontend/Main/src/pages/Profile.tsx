import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient, User } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import earthHero from "@/assets/earth-hero.jpg";

const Profile = () => {
  const { user: contextUser, logout } = useAuth();
  const [user, setUser] = useState<User | null>(contextUser);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        if (contextUser?.id) {
          const userData = await apiClient.getCurrentUser();
          setUser(userData);
        } else {
          // Try to get from context
          setUser(contextUser);
        }
      } catch (error) {
        console.error("Failed to fetch user profile:", error);
        setUser(contextUser);
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [contextUser]);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <div
        className="min-h-screen bg-background bg-cover bg-center bg-no-repeat relative dark flex items-center justify-center"
        style={{ backgroundImage: `url(${earthHero})` }}
      >
        <div className="absolute inset-0 bg-background/40"></div>
        <div className="relative z-10 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
          <p className="mt-4 text-white">Loading profile...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div
        className="min-h-screen bg-background bg-cover bg-center bg-no-repeat relative dark flex items-center justify-center px-4"
        style={{ backgroundImage: `url(${earthHero})` }}
      >
        <div className="absolute inset-0 bg-background/40"></div>
        <div className="relative z-10 w-full max-w-md">
          <Card className="bg-white/10 backdrop-blur-md border-white/20">
            <CardHeader>
              <CardTitle className="text-white">No User Data</CardTitle>
              <CardDescription className="text-white/90">Unable to load user profile</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => navigate("/login")} className="w-full">
                Go to Login
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen bg-background bg-cover bg-center bg-no-repeat bg-fixed relative dark overflow-auto"
      style={{ backgroundImage: `url(${earthHero})` }}
    >
      <div className="absolute inset-0 bg-background/40"></div>
      <div className="relative z-10 pt-24 pb-12 px-4 min-h-screen">
        <div className="max-w-4xl mx-auto w-full">
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-2 text-white drop-shadow-lg">User Profile</h1>
            <p className="text-white/90">View and manage your account information</p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 w-full">
            <Card className="bg-white/10 backdrop-blur-md border-white/20">
              <CardHeader>
                <CardTitle className="text-white">Account Information</CardTitle>
                <CardDescription className="text-white/90">Your account details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-white/80">Email Address</label>
                  <p className="text-lg font-semibold mt-1 text-white">{user.email}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-white/80">User ID</label>
                  <p className="text-lg font-semibold mt-1 text-white">#{user.id}</p>
                </div>

                <div>
                  <label className="text-sm font-medium text-white/80">Account Created</label>
                  <p className="text-lg font-semibold mt-1 text-white">{formatDate(user.created_at)}</p>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/10 backdrop-blur-md border-white/20">
              <CardHeader>
                <CardTitle className="text-white">Company Details</CardTitle>
                <CardDescription className="text-white/90">Your organization information</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-white/80">Company/Organization Name</label>
                  <p className="text-2xl font-bold mt-1 text-white drop-shadow-lg">
                    {user.company_name || <span className="text-white/60 italic">Not provided</span>}
                  </p>
                </div>

                <div>
                  <label className="text-sm font-medium text-white/80">Account Type</label>
                  <p className="text-lg font-semibold mt-1 text-white">
                    {user.account_type || <span className="text-white/60 italic">Not selected</span>}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="mt-6 bg-white/10 backdrop-blur-md border-white/20">
            <CardHeader>
              <CardTitle className="text-white">Account Actions</CardTitle>
              <CardDescription className="text-white/90">Manage your account</CardDescription>
            </CardHeader>
            <CardContent className="flex gap-4">
              <Button onClick={() => navigate("/dashboard")} variant="outline" className="bg-white/10 border-white/20 text-white hover:bg-white/20">
                Go to Dashboard
              </Button>
              <Button onClick={handleLogout} variant="destructive">
                Logout
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Profile;

