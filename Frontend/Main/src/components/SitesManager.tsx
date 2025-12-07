import { useState } from 'react';
import { useSites, useCreateSite, useDeleteSite } from '@/hooks/use-api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Trash2, Plus, MapPin } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const siteSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
});

type SiteFormData = z.infer<typeof siteSchema>;

export function SitesManager() {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const { data: sites, isLoading, error } = useSites();
  const createSite = useCreateSite();
  const deleteSite = useDeleteSite();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<SiteFormData>({
    resolver: zodResolver(siteSchema),
  });

  const onSubmit = (data: SiteFormData) => {
    createSite.mutate(data, {
      onSuccess: () => {
        setIsDialogOpen(false);
        reset();
      },
    });
  };

  const handleDelete = (id: number) => {
    if (confirm('Are you sure you want to delete this site?')) {
      deleteSite.mutate(id);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center p-6">
          <div className="text-muted-foreground">Loading sites...</div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-destructive">Error loading sites: {error.message}</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Solar Sites</h2>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Site
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Solar Site</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div>
                <Label htmlFor="name">Site Name</Label>
                <Input
                  id="name"
                  {...register('name')}
                  placeholder="Enter site name"
                />
                {errors.name && (
                  <p className="text-sm text-destructive mt-1">{errors.name.message}</p>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="latitude">Latitude</Label>
                  <Input
                    id="latitude"
                    type="number"
                    step="any"
                    {...register('latitude', { valueAsNumber: true })}
                    placeholder="0.0"
                  />
                  {errors.latitude && (
                    <p className="text-sm text-destructive mt-1">{errors.latitude.message}</p>
                  )}
                </div>
                <div>
                  <Label htmlFor="longitude">Longitude</Label>
                  <Input
                    id="longitude"
                    type="number"
                    step="any"
                    {...register('longitude', { valueAsNumber: true })}
                    placeholder="0.0"
                  />
                  {errors.longitude && (
                    <p className="text-sm text-destructive mt-1">{errors.longitude.message}</p>
                  )}
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={createSite.isPending}>
                  {createSite.isPending ? 'Creating...' : 'Create Site'}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sites?.map((site) => (
          <Card key={site.id}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{site.name}</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(site.id)}
                  disabled={deleteSite.isPending}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">
                    {site.latitude.toFixed(4)}, {site.longitude.toFixed(4)}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Created: {new Date(site.created_at).toLocaleDateString()}
                </div>
                <Badge variant="secondary">Site ID: {site.id}</Badge>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {sites?.length === 0 && (
        <Card>
          <CardContent className="flex items-center justify-center p-6">
            <div className="text-center text-muted-foreground">
              <MapPin className="h-8 w-8 mx-auto mb-2" />
              <p>No sites found</p>
              <p className="text-sm">Create your first solar site to get started</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
