import { useState } from "react";
import { ArrowLeft, Download, Share2, FileText, Calendar, Search, Filter, TrendingUp, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DashboardHeader from "@/components/dashboard/DashboardHeader";

interface AnalysisReport {
  id: string;
  date: string;
  contentType: "audio" | "video" | "text";
  title: string;
  score: number;
  risk: "high" | "medium" | "low";
  status: "completed" | "flagged" | "reviewed";
  summary: string;
  findings: string[];
}

const mockReports: AnalysisReport[] = [
  {
    id: "KS-2025-00147",
    date: "Nov 25, 2025",
    contentType: "audio",
    title: "Voice Recording - Political Statement",
    score: 87,
    risk: "high",
    status: "flagged",
    summary: "Detected synthetic voice characteristics with high confidence",
    findings: [
      "Detected synthetic voice characteristics",
      "Pitch anomalies at 2.3s and 5.7s timestamps",
      "Kenyan accent patterns inconsistent"
    ]
  },
  {
    id: "KS-2025-00146",
    date: "Nov 24, 2025",
    contentType: "video",
    title: "Social Media Video - Public Figure",
    score: 76,
    risk: "medium",
    status: "completed",
    summary: "Minor discrepancies detected, content likely authentic",
    findings: [
      "Minor lip-sync discrepancies detected",
      "Frame inconsistencies at transitions",
      "Media branding appears authentic"
    ]
  },
  {
    id: "KS-2025-00145",
    date: "Nov 23, 2025",
    contentType: "text",
    title: "News Article - Misinformation Check",
    score: 91,
    risk: "high",
    status: "flagged",
    summary: "High propaganda score with incitement language detected",
    findings: [
      "High propaganda score - incitement language detected",
      "False claims about election integrity",
      "Swahili-English code-switching patterns match PolitiKweli dataset"
    ]
  },
  {
    id: "KS-2025-00144",
    date: "Nov 22, 2025",
    contentType: "audio",
    title: "News Broadcast - Election Coverage",
    score: 23,
    risk: "low",
    status: "reviewed",
    summary: "Content appears authentic with no major anomalies",
    findings: [
      "Spectral analysis shows natural voice patterns",
      "No significant pitch inconsistencies detected",
      "Audio quality consistent throughout recording"
    ]
  },
  {
    id: "KS-2025-00143",
    date: "Nov 21, 2025",
    contentType: "text",
    title: "Social Media Post - Health Misinformation",
    score: 85,
    risk: "high",
    status: "flagged",
    summary: "Potential health misinformation with high confidence",
    findings: [
      "Medical claims contradict WHO guidelines",
      "Language patterns match known misinformation vectors",
      "High similarity to flagged content from previous reports"
    ]
  },
  {
    id: "KS-2025-00142",
    date: "Nov 20, 2025",
    contentType: "video",
    title: "TikTok Video - Entertainment Content",
    score: 34,
    risk: "low",
    status: "completed",
    summary: "Authentic content with no detected manipulations",
    findings: [
      "Visual consistency maintained throughout",
      "No deepfake markers detected",
      "Audio-visual synchronization normal"
    ]
  }
];

const Reports = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedReport, setSelectedReport] = useState<AnalysisReport | null>(null);
  const [filterRisk, setFilterRisk] = useState<"all" | "high" | "medium" | "low">("all");
  const [filterType, setFilterType] = useState<"all" | "audio" | "video" | "text">("all");

  const filteredReports = mockReports.filter(report => {
    const matchesSearch = report.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         report.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = filterRisk === "all" || report.risk === filterRisk;
    const matchesType = filterType === "all" || report.contentType === filterType;
    return matchesSearch && matchesRisk && matchesType;
  });

  const stats = {
    total: mockReports.length,
    flagged: mockReports.filter(r => r.status === "flagged").length,
    reviewed: mockReports.filter(r => r.status === "reviewed").length,
    avgScore: Math.round(mockReports.reduce((sum, r) => sum + r.score, 0) / mockReports.length)
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high":
        return "bg-destructive/20 text-destructive border-destructive/50";
      case "medium":
        return "bg-warning/20 text-warning border-warning/50";
      case "low":
        return "bg-success/20 text-success border-success/50";
      default:
        return "bg-muted/20 text-muted-foreground border-muted/50";
    }
  };

  const getContentTypeIcon = (type: string) => {
    switch (type) {
      case "audio":
        return "🎵";
      case "video":
        return "🎬";
      case "text":
        return "📝";
      default:
        return "📄";
    }
  };

  const handleDownloadReport = (reportId: string) => {
    alert(`Downloading report ${reportId}...`);
  };

  const handleShareReport = (reportId: string) => {
    alert(`Sharing report ${reportId}...`);
  };

  if (selectedReport) {
    return (
      <div className="min-h-screen bg-background">
        <DashboardHeader />
        
        <main className="container mx-auto px-4 py-6">
          <Button 
            variant="ghost" 
            onClick={() => setSelectedReport(null)}
            className="mb-6"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Reports
          </Button>

          <div className="space-y-6">
            {/* Report Header */}
            <Card className="p-8 bg-card border-border">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-4xl">{getContentTypeIcon(selectedReport.contentType)}</span>
                    <div>
                      <h1 className="text-3xl font-bold text-foreground">{selectedReport.title}</h1>
                      <p className="text-sm text-muted-foreground">{selectedReport.id} • {selectedReport.date}</p>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="outline" className={getRiskColor(selectedReport.risk)}>
                    {selectedReport.risk.toUpperCase()} RISK
                  </Badge>
                </div>
              </div>

              {/* Overall Score */}
              <div className="mb-8 p-6 bg-background rounded-lg">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-lg font-semibold text-foreground">Deepfake Confidence Score</h3>
                  <span className="text-5xl font-bold text-destructive">{selectedReport.score}%</span>
                </div>
                <Progress value={selectedReport.score} className="h-3 mb-3" />
                <p className="text-sm text-muted-foreground">
                  {selectedReport.score > 80 ? "⚠️ High likelihood of AI-generated or manipulated content" : 
                   selectedReport.score > 50 ? "⚡ Moderate risk detected - requires review" :
                   "✓ Content appears authentic with low risk"}
                </p>
              </div>

              {/* Summary */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-foreground mb-2">Summary</h3>
                <p className="text-foreground">{selectedReport.summary}</p>
              </div>

              {/* Key Findings */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-foreground mb-3">Key Findings</h3>
                <ul className="space-y-2">
                  {selectedReport.findings.map((finding, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-sm text-foreground">
                      <span className="text-primary mt-1">•</span>
                      <span>{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Actions */}
              <div className="flex gap-3 pt-6 border-t border-border">
                <Button 
                  variant="outline"
                  onClick={() => handleDownloadReport(selectedReport.id)}
                  className="flex-1"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Report
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => handleShareReport(selectedReport.id)}
                  className="flex-1"
                >
                  <Share2 className="w-4 h-4 mr-2" />
                  Share Report
                </Button>
                <Button className="flex-1 bg-primary hover:bg-primary/90">
                  Flag for Expert Review
                </Button>
              </div>
            </Card>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      
      <main className="container mx-auto px-4 py-6">
        <div className="space-y-6">
          {/* Page Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Analysis Reports</h1>
              <p className="text-muted-foreground mt-1">View all content analysis results and reports</p>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-4 bg-card border-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Total Analyses</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total}</p>
                </div>
                <FileText className="w-8 h-8 text-primary/30" />
              </div>
            </Card>

            <Card className="p-4 bg-card border-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Flagged</p>
                  <p className="text-2xl font-bold text-destructive">{stats.flagged}</p>
                </div>
                <AlertTriangle className="w-8 h-8 text-destructive/30" />
              </div>
            </Card>

            <Card className="p-4 bg-card border-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Expert Reviewed</p>
                  <p className="text-2xl font-bold text-success">{stats.reviewed}</p>
                </div>
                <CheckCircle2 className="w-8 h-8 text-success/30" />
              </div>
            </Card>

            <Card className="p-4 bg-card border-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Average Score</p>
                  <p className="text-2xl font-bold text-foreground">{stats.avgScore}%</p>
                </div>
                <TrendingUp className="w-8 h-8 text-primary/30" />
              </div>
            </Card>
          </div>

          {/* Filters and Search */}
          <Card className="p-4 bg-card border-border">
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search reports by ID or title..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-background border-border text-foreground"
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Risk:</span>
                </div>
                {["all", "high", "medium", "low"].map(risk => (
                  <Button
                    key={risk}
                    variant={filterRisk === risk ? "default" : "outline"}
                    size="sm"
                    onClick={() => setFilterRisk(risk as any)}
                    className={filterRisk === risk ? "bg-primary text-primary-foreground" : ""}
                  >
                    {risk.charAt(0).toUpperCase() + risk.slice(1)}
                  </Button>
                ))}

                <div className="ml-auto flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Type:</span>
                </div>
                {["all", "audio", "video", "text"].map(type => (
                  <Button
                    key={type}
                    variant={filterType === type ? "default" : "outline"}
                    size="sm"
                    onClick={() => setFilterType(type as any)}
                    className={filterType === type ? "bg-primary text-primary-foreground" : ""}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </Button>
                ))}
              </div>
            </div>
          </Card>

          {/* Reports List */}
          <div className="space-y-3">
            {filteredReports.length > 0 ? (
              filteredReports.map((report) => (
                <Card 
                  key={report.id}
                  className="p-4 bg-card border-border hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => setSelectedReport(report)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="text-3xl">{getContentTypeIcon(report.contentType)}</div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-foreground">{report.title}</h3>
                        <div className="flex items-center gap-3 mt-1">
                          <span className="text-xs text-muted-foreground">{report.id}</span>
                          <span className="text-xs text-muted-foreground">•</span>
                          <span className="text-xs text-muted-foreground flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {report.date}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 ml-4">
                      <div className="text-right">
                        <p className="text-lg font-bold text-foreground">{report.score}%</p>
                        <Progress value={report.score} className="w-24 h-2 mt-1" />
                      </div>
                      <Badge variant="outline" className={`${getRiskColor(report.risk)} min-w-fit`}>
                        {report.risk.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                </Card>
              ))
            ) : (
              <Card className="p-8 text-center bg-card border-border">
                <FileText className="w-12 h-12 mx-auto text-muted-foreground/50 mb-3" />
                <p className="text-muted-foreground">No reports found matching your filters</p>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Reports;
