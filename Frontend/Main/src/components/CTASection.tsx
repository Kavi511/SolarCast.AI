import { Card, CardContent } from "@/components/ui/card";
import { Eye, Target } from "lucide-react";

const CTASection = () => {
  return (
    <section id="about" className="relative py-20 bg-black dark">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-6 text-slate-100">
          Vision & Mission
        </h2>

        <p className="text-xl text-slate-300 text-center max-w-3xl mx-auto mb-16">
          Our commitment to transforming the future of renewable energy through innovation and intelligence
        </p>

        {/* Vision Section */}
        <div className="mb-16">
          <Card className="shadow-card border-slate-700 bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-8">
              <div className="flex items-start gap-6">
                <div className="w-16 h-16 rounded-full bg-blue-900/20 flex items-center justify-center flex-shrink-0">
                  <Eye className="w-8 h-8 text-blue-400" />
                </div>
                <div className="space-y-4">
                  <h3 className="text-3xl font-bold text-slate-100">Vision</h3>
                  <p className="text-lg text-slate-300 leading-relaxed">
                    "To empower a sustainable future by unlocking the full potential of solar energy through intelligent, data-driven forecasting, enabling communities, businesses, and governments to make smarter energy decisions".
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Mission Section */}
        <div className="mb-16">
          <Card className="shadow-card border-slate-700 bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-8">
              <div className="flex items-start gap-6">
                <div className="w-16 h-16 rounded-full bg-amber-900/20 flex items-center justify-center flex-shrink-0">
                  <Target className="w-8 h-8 text-amber-400" />
                </div>
                <div className="space-y-4">
                  <h3 className="text-3xl font-bold text-slate-100">Mission</h3>
                  <p className="text-lg text-slate-300 leading-relaxed">
                    "Our mission is to harness the power of artificial intelligence and satellite imagery to deliver accurate, real-time solar energy forecasts. By integrating advanced cloud detection, movement prediction, and irradiance modeling into a user-friendly web platform, we aim to provide actionable insights that optimize solar energy production, reduce uncertainty in renewable power planning, and accelerate the global transition toward clean energy".
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default CTASection;