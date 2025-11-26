import { useState } from "react";
import { Shield, Activity, AlertTriangle, CheckCircle2, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import ThreatFeed from "@/components/dashboard/ThreatFeed";
import AnalysisPanel from "@/components/dashboard/AnalysisPanel";
import StatsOverview from "@/components/dashboard/StatsOverview";

const Index = () => {
  const [activeTab, setActiveTab] = useState("monitor");

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      
      <main className="container mx-auto px-4 py-6 space-y-6">
        <StatsOverview />
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2 bg-card">
            <TabsTrigger value="monitor">Real-Time Monitor</TabsTrigger>
            <TabsTrigger value="analyze">Analyze Content</TabsTrigger>
          </TabsList>
          
          <TabsContent value="monitor" className="space-y-6 mt-6">
            <ThreatFeed />
          </TabsContent>
          
          <TabsContent value="analyze" className="space-y-6 mt-6">
            <AnalysisPanel />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
