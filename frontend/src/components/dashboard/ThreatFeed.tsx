import { useState } from "react";
import { AlertTriangle, Twitter, Video, CheckCircle2, XCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ThreatItem {
  id: string;
  source: "twitter" | "tiktok";
  type: "audio" | "video" | "text";
  content: string;
  confidence: number;
  risk: "high" | "medium" | "low";
  timestamp: string;
  details: {
    audioScore?: number;
    visualScore?: number;
    textScore?: number;
  };
}

const mockThreats: ThreatItem[] = [
  {
    id: "1",
    source: "twitter",
    type: "video",
    content: "Video claiming President making false statement about national security",
    confidence: 89,
    risk: "high",
    timestamp: "2 min ago",
    details: { audioScore: 92, visualScore: 87, textScore: 88 }
  },
  {
    id: "2",
    source: "tiktok",
    type: "audio",
    content: "Audio clip attributed to Cabinet Secretary with suspicious voice patterns",
    confidence: 76,
    risk: "medium",
    timestamp: "5 min ago",
    details: { audioScore: 82, textScore: 70 }
  },
  {
    id: "3",
    source: "twitter",
    type: "text",
    content: "Viral post containing propaganda narratives about election process",
    confidence: 84,
    risk: "high",
    timestamp: "8 min ago",
    details: { textScore: 84 }
  }
];

const ThreatFeed = () => {
  const [threats] = useState<ThreatItem[]>(mockThreats);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high": return "bg-destructive/20 text-destructive border-destructive/50";
      case "medium": return "bg-warning/20 text-warning border-warning/50";
      case "low": return "bg-success/20 text-success border-success/50";
      default: return "bg-muted";
    }
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Live Threat Feed</h2>
        </div>
        <Badge variant="outline" className="bg-primary/10 text-primary border-primary/50">
          {threats.length} Active
        </Badge>
      </div>

      <ScrollArea className="h-[600px] pr-4">
        <div className="space-y-4">
          {threats.map((threat) => (
            <Card key={threat.id} className="p-4 bg-secondary border-border hover:border-primary/50 transition-colors">
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    {threat.source === "twitter" ? (
                      <Twitter className="w-4 h-4 text-primary" />
                    ) : (
                      <Video className="w-4 h-4 text-primary" />
                    )}
                    <span className="text-xs text-muted-foreground uppercase">{threat.source}</span>
                    <span className="text-xs text-muted-foreground">•</span>
                    <span className="text-xs text-muted-foreground">{threat.timestamp}</span>
                  </div>
                  <Badge className={getRiskColor(threat.risk)}>
                    {threat.risk} risk
                  </Badge>
                </div>

                <p className="text-sm text-foreground">{threat.content}</p>

                <div className="flex items-center gap-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Confidence Score</span>
                      <span className="font-semibold text-foreground">{threat.confidence}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-destructive to-destructive/60 transition-all"
                        style={{ width: `${threat.confidence}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs">
                  {threat.details.audioScore && (
                    <div className="text-center p-2 bg-background rounded">
                      <p className="text-muted-foreground mb-1">Audio</p>
                      <p className="font-semibold text-foreground">{threat.details.audioScore}%</p>
                    </div>
                  )}
                  {threat.details.visualScore && (
                    <div className="text-center p-2 bg-background rounded">
                      <p className="text-muted-foreground mb-1">Visual</p>
                      <p className="font-semibold text-foreground">{threat.details.visualScore}%</p>
                    </div>
                  )}
                  {threat.details.textScore && (
                    <div className="text-center p-2 bg-background rounded">
                      <p className="text-muted-foreground mb-1">Text</p>
                      <p className="font-semibold text-foreground">{threat.details.textScore}%</p>
                    </div>
                  )}
                </div>

                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline" className="flex-1">
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                    Send for Review
                  </Button>
                  <Button size="sm" variant="outline" className="flex-1">
                    <XCircle className="w-4 h-4 mr-2" />
                    Mark Safe
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </Card>
  );
};

export default ThreatFeed;
