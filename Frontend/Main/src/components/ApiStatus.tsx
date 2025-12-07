import { useHealthCheck } from '@/hooks/use-api';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

export function ApiStatus() {
  const { data, isLoading, error } = useHealthCheck();

  const getStatusIcon = () => {
    if (isLoading) return <Loader2 className="h-4 w-4 animate-spin" />;
    if (error) return <AlertCircle className="h-4 w-4 text-destructive" />;
    return <CheckCircle2 className="h-4 w-4 text-green-500" />;
  };

  const getStatusText = () => {
    if (isLoading) return 'Checking...';
    if (error) return 'Disconnected';
    return 'Connected';
  };

  const getStatusColor = () => {
    if (isLoading) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    if (error) return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
    return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  };

  return (
    <Card className="w-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          {getStatusIcon()}
          API Status
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6 flex-1 flex flex-col justify-between">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Backend:</span>
            <Badge variant="outline" className={getStatusColor()}>
              {getStatusText()}
            </Badge>
          </div>
          {data && (
            <div className="text-xs text-muted-foreground">
              Service: {data.service}
            </div>
          )}
          {error && (
            <div className="text-xs text-destructive">
              Error: {error.message}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
