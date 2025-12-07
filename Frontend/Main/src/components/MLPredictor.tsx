import { useState } from 'react';
import { useCloudDetection, useIrradiancePrediction, useEnergyOutputPrediction } from '@/hooks/use-api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Upload, Sun, Zap, Cloud } from 'lucide-react';

export function MLPredictor() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [cloudCoverPct, setCloudCoverPct] = useState<number | null>(null);
  const [irradianceWm2, setIrradianceWm2] = useState<number | null>(null);
  const [energyOutputKw, setEnergyOutputKw] = useState<number | null>(null);

  const cloudDetection = useCloudDetection();
  const irradiancePrediction = useIrradiancePrediction();
  const energyOutputPrediction = useEnergyOutputPrediction();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setCloudCoverPct(null);
      setIrradianceWm2(null);
      setEnergyOutputKw(null);
    }
  };

  const handleCloudDetection = () => {
    if (selectedFile) {
      cloudDetection.mutate(selectedFile, {
        onSuccess: (data) => {
          setCloudCoverPct(data.cloud_cover_pct);
          // Automatically predict irradiance after cloud detection
          handleIrradiancePrediction(data.cloud_cover_pct);
        },
      });
    }
  };

  const handleIrradiancePrediction = (cloudCover: number) => {
    irradiancePrediction.mutate(cloudCover, {
      onSuccess: (data) => {
        setIrradianceWm2(data.irradiance_wm2);
        // Automatically predict energy output after irradiance prediction
        handleEnergyOutputPrediction(data.irradiance_wm2);
      },
    });
  };

  const handleEnergyOutputPrediction = (irradiance: number) => {
    energyOutputPrediction.mutate({
      irradianceWm2: irradiance,
      panelAreaM2: 1.6,
      panelEfficiency: 0.20,
    }, {
      onSuccess: (data) => {
        setEnergyOutputKw(data.energy_output_kw);
      },
    });
  };

  const getCloudCoverStatus = (pct: number) => {
    if (pct < 20) return { label: 'Clear', color: 'bg-green-500' };
    if (pct < 50) return { label: 'Partly Cloudy', color: 'bg-yellow-500' };
    if (pct < 80) return { label: 'Cloudy', color: 'bg-orange-500' };
    return { label: 'Overcast', color: 'bg-gray-500' };
  };

  const getIrradianceStatus = (wm2: number) => {
    if (wm2 > 800) return { label: 'Excellent', color: 'bg-green-500' };
    if (wm2 > 600) return { label: 'Good', color: 'bg-blue-500' };
    if (wm2 > 400) return { label: 'Moderate', color: 'bg-yellow-500' };
    return { label: 'Low', color: 'bg-red-500' };
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Cloud Detection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="image-upload">Upload Satellite Image</Label>
            <Input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="mt-2"
            />
          </div>
          {selectedFile && (
            <div className="flex items-center gap-2">
              <Badge variant="outline">{selectedFile.name}</Badge>
              <Button
                onClick={handleCloudDetection}
                disabled={cloudDetection.isPending}
                size="sm"
              >
                {cloudDetection.isPending ? 'Analyzing...' : 'Detect Clouds'}
              </Button>
            </div>
          )}
          {cloudCoverPct !== null && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Cloud Cover</span>
                <span className="text-sm text-muted-foreground">{cloudCoverPct.toFixed(1)}%</span>
              </div>
              <Progress value={cloudCoverPct} className="h-2" />
              <Badge variant="outline" className={getCloudCoverStatus(cloudCoverPct).color}>
                {getCloudCoverStatus(cloudCoverPct).label}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sun className="h-5 w-5" />
            Solar Irradiance Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="cloud-cover-input">Cloud Cover (%)</Label>
              <Input
                id="cloud-cover-input"
                type="number"
                value={cloudCoverPct || ''}
                onChange={(e) => setCloudCoverPct(Number(e.target.value))}
                placeholder="0-100"
                min="0"
                max="100"
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={() => cloudCoverPct !== null && handleIrradiancePrediction(cloudCoverPct)}
                disabled={cloudCoverPct === null || irradiancePrediction.isPending}
                className="w-full"
              >
                {irradiancePrediction.isPending ? 'Predicting...' : 'Predict Irradiance'}
              </Button>
            </div>
          </div>
          {irradianceWm2 !== null && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Solar Irradiance</span>
                <span className="text-sm text-muted-foreground">{irradianceWm2.toFixed(1)} W/m²</span>
              </div>
              <Progress value={(irradianceWm2 / 1000) * 100} className="h-2" />
              <Badge variant="outline" className={getIrradianceStatus(irradianceWm2).color}>
                {getIrradianceStatus(irradianceWm2).label}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Energy Output Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="irradiance-input">Irradiance (W/m²)</Label>
              <Input
                id="irradiance-input"
                type="number"
                value={irradianceWm2 || ''}
                onChange={(e) => setIrradianceWm2(Number(e.target.value))}
                placeholder="0-1000"
                min="0"
              />
            </div>
            <div>
              <Label htmlFor="panel-area">Panel Area (m²)</Label>
              <Input
                id="panel-area"
                type="number"
                defaultValue="1.6"
                placeholder="1.6"
                min="0"
              />
            </div>
            <div>
              <Label htmlFor="efficiency">Efficiency (%)</Label>
              <Input
                id="efficiency"
                type="number"
                defaultValue="20"
                placeholder="20"
                min="0"
                max="100"
              />
            </div>
          </div>
          <Button
            onClick={() => irradianceWm2 !== null && handleEnergyOutputPrediction(irradianceWm2)}
            disabled={irradianceWm2 === null || energyOutputPrediction.isPending}
            className="w-full"
          >
            {energyOutputPrediction.isPending ? 'Calculating...' : 'Calculate Energy Output'}
          </Button>
          {energyOutputKw !== null && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Energy Output</span>
                <span className="text-lg font-bold text-green-600">
                  {energyOutputKw.toFixed(3)} kW
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                Based on {irradianceWm2?.toFixed(1)} W/m² irradiance
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Summary Card */}
      {energyOutputKw !== null && (
        <Card className="border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
          <CardHeader>
            <CardTitle className="text-green-800 dark:text-green-200">
              Prediction Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {cloudCoverPct?.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Cloud Cover</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {irradianceWm2?.toFixed(0)} W/m²
                </div>
                <div className="text-xs text-muted-foreground">Irradiance</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {energyOutputKw.toFixed(3)} kW
                </div>
                <div className="text-xs text-muted-foreground">Energy Output</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
