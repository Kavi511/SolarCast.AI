import { Button } from "@/components/ui/button";
import { ArrowRight, Zap, Satellite, Brain } from "lucide-react";
import heroImage from "@/assets/hero-earth-satellite.jpg";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden dark">
      {/* Background Image with Overlay */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${heroImage})` }}
      >
        <div className="absolute inset-0 bg-black/60" />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto space-y-8">

          {/* Main Headline with Typing Animation */}
          <h1 className="text-5xl md:text-7xl font-bold leading-tight cursor-default">
            <span className="block text-white drop-shadow-lg typewriter-line mx-auto" style={{
              animation: 'typewriter 4s ease-out 0.5s forwards', 
              overflow: 'hidden', 
              whiteSpace: 'nowrap', 
              width: '0'
            }}>
              SolarCast.AI
            </span>
            <span className="block text-white drop-shadow-lg" style={{animation: 'spaceFloat 5s ease-in-out infinite', animationDelay: '1s'}}>
              Intelligence from Space
            </span>
            <span className="block text-white drop-shadow-lg" style={{animation: 'spaceFloat 6s ease-in-out infinite', animationDelay: '2s'}}>
              to Power the Future
            </span>
          </h1>

          {/* Subheadline */}
          <div className="opacity-100">
            <p className="text-xl md:text-2xl text-slate-200 max-w-2xl mx-auto leading-relaxed">
              Harness multi-temporal satellite imagery and advanced AI algorithms to generate 
              precise solar energy production forecasts with unprecedented accuracy.
            </p>
          </div>

          {/* Key Stats */}
          <div className="opacity-100">
            <div className="flex flex-col md:flex-row items-center justify-center gap-8 py-8">
              <div className="flex items-center gap-3 transform hover:scale-110 transition-transform duration-300">
                <Satellite className="w-6 h-6 text-blue-400" />
                <span className="text-lg text-slate-200">
                  <span className="font-semibold text-blue-400">99.2%</span> Accuracy
                </span>
              </div>
              <div className="flex items-center gap-3 transform hover:scale-110 transition-transform duration-300">
                <Zap className="w-6 h-6 text-amber-400" />
                <span className="text-lg text-slate-200">
                  <span className="font-semibold text-amber-400">24/7</span> Monitoring
                </span>
              </div>
              <div className="flex items-center gap-3 transform hover:scale-110 transition-transform duration-300">
                <Brain className="w-6 h-6 text-emerald-400" />
                <span className="text-lg text-slate-200">
                  <span className="font-semibold text-emerald-400">Real-time</span> Predictions
                </span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default HeroSection;
