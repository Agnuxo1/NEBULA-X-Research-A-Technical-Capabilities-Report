export interface Project {
  name: string;
  description: string;
  url: string;
  keyFeatures: string[];
}

export interface Publication {
  title: string;
  summary: string;
  publicationDate: string;
}

export interface MarketAnalysis {
  summary: string;
  sources: {
    title: string;
    uri: string;
  }[];
}

export interface PerformanceValidation {
  competition: string;
  rank: number | string;
  totalTeams: number;
  category: string;
}

export interface Team {
  name: string;
  title: string;
  bio: string;
}

export interface ReportData {
  title: string;
  executiveSummary: string;
  visionStatement: string;
  technologyPortfolio: Project[];
  researchTraction: Publication[];
  marketAnalysis: MarketAnalysis;
  performanceValidation: PerformanceValidation[];
  team: Team;
}
