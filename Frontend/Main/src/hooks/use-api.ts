import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, type Site, type SiteCreate, type Observation, type ObservationCreate } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

// Sites hooks
export const useSites = () => {
  return useQuery({
    queryKey: ['sites'],
    queryFn: () => apiClient.getSites(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useSite = (id: number) => {
  return useQuery({
    queryKey: ['sites', id],
    queryFn: () => apiClient.getSite(id),
    enabled: !!id,
  });
};

export const useCreateSite = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: (site: SiteCreate) => apiClient.createSite(site),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sites'] });
      toast({
        title: 'Success',
        description: 'Site created successfully',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to create site',
        variant: 'destructive',
      });
    },
  });
};

export const useDeleteSite = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteSite(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sites'] });
      toast({
        title: 'Success',
        description: 'Site deleted successfully',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to delete site',
        variant: 'destructive',
      });
    },
  });
};

// Observations hooks
export const useObservations = (siteId: number) => {
  return useQuery({
    queryKey: ['observations', siteId],
    queryFn: () => apiClient.getObservations(siteId),
    enabled: !!siteId,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useCreateObservation = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ siteId, observation }: { siteId: number; observation: Omit<ObservationCreate, 'site_id'> }) =>
      apiClient.createObservation(siteId, observation),
    onSuccess: (_, { siteId }) => {
      queryClient.invalidateQueries({ queryKey: ['observations', siteId] });
      toast({
        title: 'Success',
        description: 'Observation added successfully',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to add observation',
        variant: 'destructive',
      });
    },
  });
};

// ML hooks
export const useCloudDetection = () => {
  const { toast } = useToast();

  return useMutation({
    mutationFn: (image: File) => apiClient.detectCloudCover(image),
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to detect cloud cover',
        variant: 'destructive',
      });
    },
  });
};

export const useIrradiancePrediction = () => {
  const { toast } = useToast();

  return useMutation({
    mutationFn: (cloudCoverPct: number) => apiClient.predictIrradiance(cloudCoverPct),
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to predict irradiance',
        variant: 'destructive',
      });
    },
  });
};

export const useEnergyOutputPrediction = () => {
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ irradianceWm2, panelAreaM2, panelEfficiency }: {
      irradianceWm2: number;
      panelAreaM2?: number;
      panelEfficiency?: number;
    }) => apiClient.predictEnergyOutput(irradianceWm2, panelAreaM2, panelEfficiency),
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to predict energy output',
        variant: 'destructive',
      });
    },
  });
};

// Health check hook
export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 30 * 1000, // Check every 30 seconds
    retry: 3,
  });
};

// Weather API status hook
export const useWeatherStatus = () => {
  return useQuery({
    queryKey: ['weather-status'],
    queryFn: () => apiClient.weatherStatus(),
    refetchInterval: 60 * 1000, // Check every minute
    retry: 3,
  });
};

// Models status hook
export const useModelsStatus = () => {
  return useQuery({
    queryKey: ['models-status'],
    queryFn: () => apiClient.modelsStatus(),
    refetchInterval: 60 * 1000, // Check every minute
    retry: 3,
  });
};

// Google Earth Engine status hook
export const useGEEStatus = () => {
  return useQuery({
    queryKey: ['gee-status'],
    queryFn: () => apiClient.geeStatus(),
    refetchInterval: 60 * 1000, // Check every minute
    retry: 3,
  });
};