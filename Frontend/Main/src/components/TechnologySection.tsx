import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, Cpu, Database } from "lucide-react";
import satelliteTemporalImage from "@/assets/satellite-temporal.jpg";

export default function TechnologySection() {

  return (
    <section id="technology" className="relative py-20 bg-black dark">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-6 text-slate-100">
          Advanced Technology
        </h2>

        <p className="text-xl text-slate-300 text-center max-w-3xl mx-auto mb-16">
          Our multi-temporal satellite analysis combines decades of space technology expertise 
          with cutting-edge artificial intelligence.
        </p>

        <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
          {/* Technology Description */}
          <div className="space-y-8">
            <div className="space-y-6">
              <h3 className="text-3xl font-bold text-slate-100">
                Multi-Temporal Satellite Analysis
              </h3>
              <p className="text-lg text-slate-300 leading-relaxed">
                By analyzing satellite imagery across multiple time periods, our AI can 
                identify subtle changes in solar panel installations, weather patterns, 
                and environmental factors that directly impact energy production.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-blue-900/20 flex items-center justify-center mt-1">
                    <Layers className="w-4 h-4 text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-slate-100">Temporal Change Detection</h4>
                    <p className="text-slate-300">
                      Track solar panel installations, degradation, and maintenance 
                      activities over time to improve prediction accuracy.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-amber-900/20 flex items-center justify-center mt-1">
                    <Cpu className="w-4 h-4 text-amber-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-slate-100">AI Pattern Recognition</h4>
                    <p className="text-slate-300">
                      Advanced neural networks identify complex patterns in satellite 
                      data that correlate with energy production variations.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-emerald-900/20 flex items-center justify-center mt-1">
                    <Database className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-slate-100">Massive Dataset Training</h4>
                    <p className="text-slate-300">
                      Models trained on petabytes of historical satellite data 
                      and energy production records for maximum accuracy.
                    </p>
                  </div>
                </div>
              </div>


            </div>
          </div>

          {/* Satellite Image */}
          <div className="relative">
            <Card className="shadow-card border-slate-700 bg-slate-800/50 backdrop-blur-sm overflow-hidden">
              <CardContent className="p-0">
                <img 
                  src={satelliteTemporalImage} 
                  alt="Multi-temporal satellite analysis"
                  className="w-full h-[400px] object-cover brightness-125 contrast-110"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/70 via-transparent to-transparent" />
                
                {/* Before and After Labels */}
                <div className="absolute top-4 left-4">
                  <div className="bg-slate-900/80 text-white px-4 py-2 rounded-lg font-bold text-lg drop-shadow-lg border border-slate-600">
                    BEFORE
                  </div>
                </div>
                <div className="absolute top-4 right-4">
                  <div className="bg-slate-900/80 text-white px-4 py-2 rounded-lg font-bold text-lg drop-shadow-lg border border-slate-600">
                    AFTER
                  </div>
                </div>
                
                <div className="absolute bottom-4 left-4 right-4">
                  <Badge variant="secondary" className="mb-2 bg-blue-900/90 text-blue-100 border-blue-700">
                    Before & After Analysis
                  </Badge>
                  <p className="text-sm text-slate-100 font-medium drop-shadow-lg">
                    Multi-temporal satellite comparison showing solar panel installation changes over time
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>


      </div>
    </section>
  );
}