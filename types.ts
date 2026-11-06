/* NEBULA-X Research: AI Investment Proposal Generator
 * Copyright (c) 2025 Francisco Angulo de Lafuente
 * Author: Francisco Angulo de Lafuente
 * ORCID: https://orcid.org/0009-0001-1634-7063
 * Attribution record date: 2026-11-06
 */

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
