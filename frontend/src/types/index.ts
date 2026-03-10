// ─── Enums ────────────────────────────────────────────────────────────────────

export enum EmploymentStatus {
  Employed = "employed",
  SelfEmployed = "self_employed",
  Unemployed = "unemployed",
  Retired = "retired",
  Student = "student",
  PartTime = "part_time",
}

export enum LoanType {
  Personal = "personal",
  Mortgage = "mortgage",
  Auto = "auto",
  Business = "business",
  Student = "student",
  HomeEquity = "home_equity",
  CreditCard = "credit_card",
}

export enum LoanStatus {
  Pending = "pending",
  UnderReview = "under_review",
  Approved = "approved",
  Rejected = "rejected",
  Disbursed = "disbursed",
  Closed = "closed",
  Defaulted = "defaulted",
}

export enum CollateralType {
  None = "none",
  RealEstate = "real_estate",
  Vehicle = "vehicle",
  Equipment = "equipment",
  FinancialAsset = "financial_asset",
}

export enum DocumentType {
  PayStub = "pay_stub",
  TaxReturn = "tax_return",
  BankStatement = "bank_statement",
  GovernmentId = "government_id",
  PropertyDeed = "property_deed",
  Other = "other",
}

export enum RiskGrade {
  A = "A",
  B = "B",
  C = "C",
  D = "D",
  E = "E",
  F = "F",
}

export enum ThinkingMode {
  On = "on",
  Off = "off",
  Auto = "auto",
}

export enum ThinkingVisibility {
  Hidden = "hidden",
  Collapsed = "collapsed",
  Streaming = "streaming",
  Full = "full",
}

export enum ThinkingBudgetPreset {
  None = "none",
  Minimal = "minimal",
  Short = "short",
  Standard = "standard",
  Extended = "extended",
  Deep = "deep",
  Unlimited = "unlimited",
}

// ─── Customer Types ───────────────────────────────────────────────────────────

export interface Customer {
  id: number;
  first_name: string;
  last_name: string;
  full_name: string;
  email: string;
  phone?: string;
  date_of_birth?: string;
  ssn_last4?: string;
  address?: string;
  city?: string;
  state?: string;
  zip_code?: string;

  // Employment
  employment_status: EmploymentStatus;
  employer_name?: string;
  annual_income: number;
  monthly_income: number;
  years_employed?: number;
  job_title?: string;

  // Credit
  credit_score: number;
  credit_history_years: number;
  num_credit_accounts: number;
  num_open_accounts: number;
  num_closed_accounts: number;
  total_credit_limit: number;
  total_credit_used: number;
  credit_utilization_ratio: number;
  credit_grade: string;

  // Debt
  total_debt: number;
  monthly_debt_payments: number;
  mortgage_balance: number;
  auto_loan_balance: number;
  student_loan_balance: number;
  credit_card_balance: number;
  other_debt: number;
  dti_ratio: number;

  // Payment History
  on_time_payments: number;
  late_payments_30d: number;
  late_payments_60d: number;
  late_payments_90d: number;
  collections: number;
  bankruptcies: number;
  foreclosures: number;
  charge_offs: number;

  // Inquiries
  hard_inquiries_6m: number;
  hard_inquiries_12m: number;
  soft_inquiries_12m: number;

  // Assets
  checking_balance: number;
  savings_balance: number;
  investment_balance: number;
  property_value: number;
  vehicle_value: number;

  is_active: boolean;
  created_at: string;
  updated_at: string;
  notes?: string;
}

export interface CustomerListResponse {
  customers: Customer[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ─── Loan Types ───────────────────────────────────────────────────────────────

export interface LoanApplication {
  id: number;
  customer_id: number;
  loan_type: LoanType;
  requested_amount: number;
  approved_amount?: number;
  interest_rate?: number;
  term_months: number;
  monthly_payment?: number;
  status: LoanStatus;
  application_date: string;
  decision_date?: string;
  disbursement_date?: string;
  collateral_type: CollateralType;
  collateral_value: number;
  loan_purpose?: string;
  rejection_reason?: string;
  credit_score_at_application?: number;
  dti_at_application?: number;
  risk_grade_at_application?: string;
}

export interface CustomerDocument {
  id: number;
  customer_id: number;
  document_type: DocumentType;
  file_name: string;
  file_size_bytes?: number;
  mime_type?: string;
  is_parsed: boolean;
  is_verified: boolean;
  uploaded_at: string;
}

// ─── Credit Score Types ───────────────────────────────────────────────────────

export interface CreditScoreResponse {
  customer_id: number;
  score: number;
  grade: string;
  grade_label: string;
  payment_history_score: number;
  credit_utilization_score: number;
  credit_age_score: number;
  credit_mix_score: number;
  new_credit_score: number;
  payment_history_weight: number;
  credit_utilization_weight: number;
  credit_age_weight: number;
  credit_mix_weight: number;
  new_credit_weight: number;
  factors: Record<string, string>;
  breakdown: Record<string, unknown>;
  recommendations: string[];
}

export interface LoanAdjustedScore {
  customer_id: number;
  base_score: number;
  adjusted_score: number;
  final_grade: string;
  final_grade_label: string;
  dti_adjustment: number;
  collateral_adjustment: number;
  term_risk_adjustment: number;
  purpose_adjustment: number;
  total_adjustment: number;
  adjustments: Record<string, string>;
  loan_recommendation: string;
}

export interface DTIRatio {
  customer_id: number;
  monthly_income: number;
  monthly_debt_payments: number;
  additional_payment: number;
  front_end_ratio: number;
  back_end_ratio: number;
  combined_dti: number;
  risk_classification: string;
  risk_label: string;
  impact_description: string;
  thresholds: Record<string, number>;
}

export interface CollateralEval {
  collateral_type: string;
  collateral_value: number;
  loan_amount: number;
  ltv_ratio: number;
  coverage_ratio: number;
  haircut_pct: number;
  adjusted_value: number;
  risk_assessment: string;
  risk_description: string;
  score_impact: number;
}

export interface RiskWeightedScore {
  customer_id: number;
  base_score: number;
  risk_score: number;
  risk_grade: string;
  risk_label: string;
  recommendation: string;
  breakdown: Record<string, unknown>;
  key_risk_factors: string[];
  positive_factors: string[];
}

// ─── Agent / Chat Types ───────────────────────────────────────────────────────

export interface ToolCall {
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_output: unknown;
  duration_ms: number;
  success: boolean;
  error?: string;
}

export interface ExecutionStep extends ToolCall {
  step: number;
}

export interface AgentResponse {
  session_id: string;
  answer: string;
  execution_trace: ExecutionStep[];
  moe_traces?: Record<string, unknown>;
  tokens_used: Record<string, number>;
  thinking?: ThinkingData;
  total_duration_ms: number;
  model: string;
}

export interface ThinkingData {
  content: string;
  tokens: number;
  step: number;
  budget_preset: string;
  duration_ms?: number;
  was_enforced?: boolean;
  utilization_pct?: number;
}

// ─── MoE Types ────────────────────────────────────────────────────────────────

export interface MoELayerTrace {
  layer_id: string;
  layer_index: number;
  experts_activated: number[];
  gating_weights: number[];
  expert_load: number[];
  entropy: number;
  top_k: number;
  num_experts: number;
}

export interface MoERequestTrace {
  request_id: string;
  layers: MoELayerTrace[];
  thinking_phase_experts: Record<string, unknown>;
  response_phase_experts: Record<string, unknown>;
  total_expert_activations: number;
  duration_ms?: number;
}

export interface HeatmapCell {
  count: number;
  frequency: number;
  avg_weight: number;
}

export interface ExpertHeatmap {
  layers: string[];
  experts: number[];
  data: HeatmapCell[][];
  total_requests: number;
}

export interface LayerActivity {
  layer_id: string;
  layer_index: number;
  top_experts: Array<{ expert_id: number; count: number; frequency: number; avg_weight: number }>;
  entropy: number;
  load_distribution: number[];
  gating_weight_stats: Record<string, number>;
  activation_count: number;
}

export interface EntropyDataPoint {
  timestamp: number;
  layer: string;
  entropy: number;
  request_id?: string;
}

// ─── Thinking Types ───────────────────────────────────────────────────────────

export interface CoTConfig {
  mode: ThinkingMode;
  budget: string;
  visibility: ThinkingVisibility;
  enable_thinking: boolean;
  auto_classify?: boolean;
}

export interface CoTPreset {
  name: string;
  description: string;
  mode: string;
  budget: string;
  visibility: string;
  budget_tokens: number;
  latency_impact: string;
}

export interface ThinkingTrace {
  request_id: string;
  session_id: string;
  mode: string;
  budget_preset: string;
  budget_tokens: number;
  thinking_content: string;
  thinking_tokens_used: number;
  thinking_duration_ms?: number;
  response_content: string;
  response_tokens: number;
  was_budget_enforced: boolean;
  utilization_pct: number;
  created_at: string;
}

export interface ThinkingStats {
  total_requests: number;
  thinking_on_count: number;
  thinking_off_count: number;
  avg_thinking_tokens: number;
  avg_budget_utilization_pct: number;
  budget_enforced_count: number;
  avg_thinking_duration_ms: number;
  preset_distribution: Record<string, number>;
  mode_distribution: Record<string, number>;
}

// ─── Chat / WebSocket Types ───────────────────────────────────────────────────

export type WebSocketEventType =
  | "thinking_start"
  | "thinking_delta"
  | "thinking_end"
  | "response_start"
  | "response_delta"
  | "response_end"
  | "tool_call"
  | "tool_result"
  | "session_start"
  | "done"
  | "error"
  | "pong";

export interface WebSocketEvent {
  type: WebSocketEventType;
  content?: string;
  session_id?: string;
  tool_calls?: unknown[];
  tokens_used?: number;
  duration_ms?: number;
  full_thinking_content?: string;
  full_response?: string;
  thinking_tokens?: number;
  response_tokens?: number;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  thinking?: string;
  thinkingTokens?: number;
  thinkingDurationMs?: number;
  wasThinkingEnforced?: boolean;
  toolCalls?: ExecutionStep[];
  moeTrace?: MoERequestTrace;
  timestamp: Date;
  isStreaming?: boolean;
  error?: string;
}
