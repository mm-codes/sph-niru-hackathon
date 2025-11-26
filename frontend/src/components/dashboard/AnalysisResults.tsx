import { ArrowLeft, AlertTriangle, CheckCircle2, Send } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";

interface AnalysisResultsProps {
  onBack: () => void;
  type: "audio" | "video" | "text";
}

const AnalysisResults = ({ onBack, type }: AnalysisResultsProps) => {
  const mockResults = {
    audio: {
      overall: 87,
      risk: "high",
      modules: {
        spectral: 89,
        pitch: 92,
        phoneme: 81
      },
      findings: [
        "Detected synthetic voice characteristics",
        "Pitch anomalies at 2.3s and 5.7s timestamps",
        "Kenyan accent patterns inconsistent with speaker profile"
      ]
    },
    video: {
      overall: 76,
      risk: "medium",
      modules: {
        lipSync: 78,
        temporal: 81,
        branding: 69
      },
      findings: [
        "Minor lip-sync discrepancies detected",
        "Frame inconsistencies at transitions",
        "Media branding appears authentic"
      ]
    },
    text: {
      overall: 91,
      risk: "high",
      modules: {
        propaganda: 94,
        misinformation: 89,
        sentiment: 90
      },
      findings: [
        "High propaganda score - incitement language detected",
        "False claims about election integrity",
        "Swahili-English code-switching patterns match PolitiKweli dataset"
      ]
    }
  };

  const results = mockResults[type];

  const handleSendForReview = () => {
    toast.success("Content sent for expert review");
  };

  return (
    <div className="space-y-6">
      <Button variant="ghost" onClick={onBack} className="mb-4">
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to Analysis
      </Button>

      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-foreground">Analysis Results</h2>
          <Badge 
            variant="outline" 
            className={results.risk === "high" ? "bg-destructive/20 text-destructive border-destructive/50" : "bg-warning/20 text-warning border-warning/50"}
          >
            {results.risk} risk
          </Badge>
        </div>

        {/* Overall Score */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-medium text-foreground">Localized Deepfake Confidence Score</h3>
            <span className="text-3xl font-bold text-destructive">{results.overall}%</span>
          </div>
          <Progress value={results.overall} className="h-3" />
          <p className="text-xs text-muted-foreground mt-2">
            {results.overall > 80 ? "High likelihood of AI-generated or manipulated content" : "Moderate risk detected"}
          </p>
        </div>

        {/* Module Scores */}
        <div className="space-y-4 mb-8">
          <h3 className="text-sm font-semibold text-foreground mb-3">Module Breakdown</h3>
          {Object.entries(results.modules).map(([module, score]) => (
            <div key={module} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground capitalize">{module}</span>
                <span className="font-semibold text-foreground">{score}%</span>
              </div>
              <Progress value={score} className="h-2" />
            </div>
          ))}
        </div>

        {/* Findings */}
        <div className="space-y-4 mb-8">
          <h3 className="text-sm font-semibold text-foreground mb-3">Key Findings</h3>
          <div className="space-y-3">
            {results.findings.map((finding, index) => (
              <div key={index} className="flex gap-3 p-3 bg-secondary rounded-lg">
                {results.risk === "high" ? (
                  <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                )}
                <p className="text-sm text-foreground">{finding}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <Button 
            onClick={handleSendForReview}
            className="flex-1 bg-primary hover:bg-primary/90"
          >
            <Send className="w-4 h-4 mr-2" />
            Send for Expert Review
          </Button>
          <Button variant="outline" className="flex-1">
            Export Report
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default AnalysisResults;
