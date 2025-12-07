import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Satellite, Brain, BarChart3, Cloud, Globe, Zap } from "lucide-react";
import aiPredictionImage from "@/assets/h9Ec3oA2XWnXSQVuah-xg.jpg";

export default function FeaturesSection() {
  const features = [
    {
      icon: Globe,
      title: "Global Coverage",
      description: "Worldwide satellite network provides comprehensive coverage for solar installations anywhere on Earth with consistent data quality.",
      color: "primary"
    },
    {
      icon: BarChart3,
      title: "Dashboard Overview",
      description: "Real-time monitoring of solar energy systems and weather conditions",
      color: "accent"
    },
    {
      icon: Satellite,
      title: "Satellite Imagery",
      description: "Real-time satellite data and cloud cover analysis",
      color: "success"
    },
    {
      icon: Cloud,
      title: "Weather Forecast",
      description: "7-day weather forecast and solar production predictions",
      color: "primary"
    },
    {
      icon: Zap,
      title: "Solar Production",
      description: "Detailed solar energy production analysis and optimization",
      color: "accent"
    },
    {
      icon: Brain,
      title: "Advanced Analytics",
      description: "Performance insights and predictive analytics",
      color: "success"
    }
  ];

  return (
    <section id="features" className="relative py-20 bg-black dark">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-6 text-slate-100">
          Powerful Features for
        </h2>
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-6">
          <span className="text-gradient-primary">Solar Energy Forecasting</span>
        </h2>

        <p className="text-xl text-slate-300 text-center max-w-3xl mx-auto mb-16">
          Our platform combines cutting-edge satellite technology with advanced AI to deliver 
          unparalleled solar energy forecasting capabilities.
        </p>

        {/* Main Visual */}
        <div className="mb-16">
          <Card className="shadow-card border-slate-700 bg-slate-800/50 backdrop-blur-sm overflow-hidden">
            <CardContent className="p-0">
              <div className="grid lg:grid-cols-2 gap-0">
                <div className="p-8 lg:p-12 flex flex-col justify-center space-y-6">
                  <div className="space-y-4">
                    <div className="inline-flex items-center gap-2 bg-blue-900/20 rounded-full px-4 py-2 text-sm">
                      <Brain className="w-4 h-4 text-blue-400" />
                      <span className="text-blue-400 font-medium">AI Technology</span>
                    </div>
                    <h3 className="text-3xl font-bold text-slate-100">
                      Next-Generation Solar Forecasting
                    </h3>
                    <p className="text-slate-300 text-lg">
                      Our proprietary AI algorithms process terabytes of satellite data 
                      to predict solar energy production with unprecedented precision, 
                      helping optimize renewable energy investments worldwide.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-2xl font-bold text-blue-400">99.2%</div>
                      <div className="text-sm text-slate-400">Forecast Accuracy</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-amber-400">&lt;1min</div>
                      <div className="text-sm text-slate-400">Processing Time</div>
                    </div>
                  </div>
                </div>
                <div className="relative">
                  <img 
                    src={aiPredictionImage} 
                    alt="AI prediction visualization"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-l from-transparent to-slate-800/20" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card 
                key={index}
                className="shadow-card border-slate-700 bg-slate-800/50 backdrop-blur-sm hover-glow transition-spring group h-64 flex flex-col"
              >
                <CardHeader className="space-y-4 flex-shrink-0">
                  <div className={`w-12 h-12 rounded-lg ${
                    feature.color === 'primary' ? 'bg-blue-900/20' :
                    feature.color === 'accent' ? 'bg-amber-900/20' : 
                    'bg-emerald-900/20'
                  } flex items-center justify-center group-hover:bg-opacity-30 transition-smooth`}>
                    <Icon className={`w-6 h-6 ${
                      feature.color === 'primary' ? 'text-blue-400' :
                      feature.color === 'accent' ? 'text-amber-400' : 
                      'text-emerald-400'
                    }`} />
                  </div>
                  <CardTitle className="text-xl font-semibold text-slate-100">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow flex items-start">
                  <p className="text-slate-300 leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
}