import { Activity, AlertTriangle, CheckCircle2, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";

const stats = [
  {
    label: "Active Scans",
    value: "1,247",
    change: "+12.5%",
    icon: Activity,
    color: "text-primary",
    bgColor: "bg-primary/10"
  },
  {
    label: "Threats Detected",
    value: "89",
    change: "+5.2%",
    icon: AlertTriangle,
    color: "text-destructive",
    bgColor: "bg-destructive/10"
  },
  {
    label: "Verified Content",
    value: "3,421",
    change: "+18.3%",
    icon: CheckCircle2,
    color: "text-success",
    bgColor: "bg-success/10"
  },
  {
    label: "Avg Response Time",
    value: "1.2s",
    change: "-0.3s",
    icon: Clock,
    color: "text-accent",
    bgColor: "bg-accent/10"
  }
];

const StatsOverview = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="p-6 bg-card border-border hover:shadow-lg transition-shadow">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-muted-foreground mb-1">{stat.label}</p>
              <p className="text-3xl font-bold text-foreground mb-2">{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.change} from last hour</p>
            </div>
            <div className={`p-3 rounded-lg ${stat.bgColor}`}>
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};

export default StatsOverview;
