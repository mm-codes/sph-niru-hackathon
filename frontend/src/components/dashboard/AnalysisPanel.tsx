import { useState } from "react";
import { Upload, FileAudio, FileVideo, FileText, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import AnalysisResults from "./AnalysisResults";

const AnalysisPanel = () => {
  const [analyzing, setAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisType, setAnalysisType] = useState<"audio" | "video" | "text">("audio");

  const handleAnalyze = async () => {
    setAnalyzing(true);
    
    // Simulate analysis
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setAnalyzing(false);
    setShowResults(true);
    toast.success("Analysis complete");
  };

  if (showResults) {
    return <AnalysisResults onBack={() => setShowResults(false)} type={analysisType} />;
  }

  return (
    <Card className="p-6 bg-card border-border">
      <h2 className="text-lg font-semibold text-foreground mb-6">Multi-Modal Content Analysis</h2>

      <Tabs defaultValue="audio" onValueChange={(v) => setAnalysisType(v as any)}>
        <TabsList className="grid w-full grid-cols-3 bg-secondary">
          <TabsTrigger value="audio">
            <FileAudio className="w-4 h-4 mr-2" />
            Audio
          </TabsTrigger>
          <TabsTrigger value="video">
            <FileVideo className="w-4 h-4 mr-2" />
            Video
          </TabsTrigger>
          <TabsTrigger value="text">
            <FileText className="w-4 h-4 mr-2" />
            Text
          </TabsTrigger>
        </TabsList>

        <TabsContent value="audio" className="space-y-4 mt-6">
          <div className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-primary/50 transition-colors cursor-pointer">
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-sm text-foreground mb-2">
              Upload audio file for voice clone detection
            </p>
            <p className="text-xs text-muted-foreground">
              Supports MP3, WAV, M4A • Max 50MB
            </p>
          </div>
        </TabsContent>

        <TabsContent value="video" className="space-y-4 mt-6">
          <div className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-primary/50 transition-colors cursor-pointer">
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-sm text-foreground mb-2">
              Upload video file for deepfake detection
            </p>
            <p className="text-xs text-muted-foreground">
              Supports MP4, MOV, AVI • Max 100MB
            </p>
          </div>
        </TabsContent>

        <TabsContent value="text" className="space-y-4 mt-6">
          <Textarea
            placeholder="Paste text content for misinformation analysis (Swahili-English supported)..."
            className="min-h-[200px] bg-background border-border text-foreground"
          />
        </TabsContent>
      </Tabs>

      <div className="mt-6 flex gap-3">
        <Button 
          onClick={handleAnalyze}
          disabled={analyzing}
          className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
        >
          {analyzing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            "Run Analysis"
          )}
        </Button>
      </div>
    </Card>
  );
};

export default AnalysisPanel;
