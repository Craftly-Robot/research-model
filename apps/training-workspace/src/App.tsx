import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Ban,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Cpu,
  Database,
  FileArchive,
  Gauge,
  KeyRound,
  ListRestart,
  LogOut,
  Play,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Square,
} from "lucide-react";

type Profile = {
  profile_id: string;
  version: number;
  description: string;
  readiness: string;
  run_type: string;
  model_profile: string;
  curriculum_mode: string;
  scheduler: string;
  distributed_strategy: string;
  steps: number;
  sequence_length: number;
  resources: { nodes: number; gpus_per_node: number; gpu_memory_gib: number };
};

type Job = {
  job_id: string;
  status: string;
  version: number;
  created_at: string;
  updated_at: string;
  failure_detail?: string;
  spec: {
    profile_id: string;
    model_profile: string;
    scheduler: string;
    distributed_strategy: string;
    steps: number;
    spec_hash: string;
    resources: { nodes: number; gpus_per_node: number; gpu_memory_gib: number };
  };
};

type TrainingEvent = {
  event_id: string;
  sequence: number;
  timestamp: string;
  stage: string;
  status: string;
  kind: string;
  rank: number;
  world_size: number;
  step?: number;
  max_steps?: number;
  loss?: number;
  validation_loss?: number;
  tokens_per_second?: number;
  gpu_memory_bytes?: number;
  message?: string;
};

type Artifact = { artifact_id: string; kind: string; uri: string; size_bytes: number; promoted: boolean; created_at: string };
type AuditEvent = { audit_id: string; action: string; outcome: string; actor_id: string; created_at: string };
type Checkpoint = { checkpoint_id: string; step: number; promoted: boolean; reload_verified: boolean; created_at: string };
type Evaluation = { evaluation_id: string; decision: string; status: string; created_at: string };
type Session = { access_token: string; refresh_token: string; session_id: string; expires_in: number };

const terminalStates = new Set(["succeeded", "failed", "blocked", "cancelled"]);
const apiBase = (import.meta.env.VITE_CRAFTLY_API_URL || "").replace(/\/$/, "");
const logoPath = "/WhatsApp%20Image%202026-06-10%20at%208.29.10%20PM.jpeg";

function formatNumber(value?: number, digits = 2) {
  if (value === undefined || value === null) return "-";
  return Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
}

function formatBytes(value?: number) {
  if (!value) return "-";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let current = value;
  let unit = 0;
  while (current >= 1024 && unit < units.length - 1) {
    current /= 1024;
    unit += 1;
  }
  return `${formatNumber(current, 1)} ${units[unit]}`;
}

function Sparkline({ values, tone = "green" }: { values: number[]; tone?: "green" | "amber" }) {
  if (values.length < 2) return <div className="spark-empty">Awaiting metrics</div>;
  const width = 520;
  const height = 110;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((value, index) => `${(index / (values.length - 1)) * width},${height - ((value - min) / range) * (height - 14) - 7}`).join(" ");
  return (
    <svg className={`sparkline ${tone}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Metric trend">
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Status({ value }: { value: string }) {
  const Icon = value === "succeeded" ? CheckCircle2 : value === "failed" || value === "blocked" ? CircleAlert : value === "running" ? Activity : Clock3;
  return <span className={`status status-${value}`}><Icon size={14} />{value}</span>;
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [bootstrapToken, setBootstrapToken] = useState("");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<Job | null>(null);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [createProfile, setCreateProfile] = useState("");
  const [datasetUri, setDatasetUri] = useState("");
  const [datasetHash, setDatasetHash] = useState("");
  const [tokenizerUri, setTokenizerUri] = useState("");
  const [tokenizerHash, setTokenizerHash] = useState("");
  const streamAbort = useRef<AbortController | null>(null);
  const sessionRef = useRef<Session | null>(null);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  const request = useCallback(async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    if (!session) throw new Error("Workspace session is not authenticated");
    const execute = (accessToken: string) => fetch(`${apiBase}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}`, ...(init.headers || {}) },
    });
    let response = await execute(session.access_token);
    if (response.status === 401) {
      const refreshed = await fetch(`${apiBase}/v1/training/token/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id, refresh_token: session.refresh_token }),
      });
      if (!refreshed.ok) throw new Error("Workspace session expired");
      const token = await refreshed.json();
      setSession((current) => current ? { ...current, access_token: token.access_token, expires_in: token.expires_in } : current);
      response = await execute(token.access_token);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `${response.status} ${response.statusText}`);
    }
    return response.json() as Promise<T>;
  }, [session]);

  const login = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/v1/training/token/exchange`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bootstrap_token: bootstrapToken }),
      });
      if (!response.ok) throw new Error("Workspace credential was rejected");
      setSession(await response.json());
      setBootstrapToken("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  const loadWorkspace = useCallback(async () => {
    try {
      const [profilePayload, jobPayload] = await Promise.all([
        request<{ profiles: Profile[] }>("/v1/training/profiles"),
        request<{ jobs: Job[] }>("/v1/training/jobs?limit=200"),
      ]);
      setProfiles(profilePayload.profiles);
      setJobs(jobPayload.jobs);
      setCreateProfile((current) => current || profilePayload.profiles[0]?.profile_id || "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Workspace refresh failed");
    }
  }, [request]);

  useEffect(() => {
    if (session) void loadWorkspace();
  }, [session, loadWorkspace]);

  const selectJob = useCallback(async (job: Job) => {
    streamAbort.current?.abort();
    setSelected(job);
    setEvents([]);
    setError("");
    try {
      const [artifactPayload, auditPayload, checkpointPayload, evaluationPayload] = await Promise.all([
        request<{ artifacts: Artifact[] }>(`/v1/training/jobs/${job.job_id}/artifacts`),
        request<{ audit_events: AuditEvent[] }>(`/v1/training/jobs/${job.job_id}/audit`),
        request<{ checkpoints: Checkpoint[] }>(`/v1/training/jobs/${job.job_id}/checkpoints`),
        request<{ evaluations: Evaluation[] }>(`/v1/training/jobs/${job.job_id}/evaluations`),
      ]);
      setArtifacts(artifactPayload.artifacts);
      setAudit(auditPayload.audit_events);
      setCheckpoints(checkpointPayload.checkpoints);
      setEvaluations(evaluationPayload.evaluations);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Job details failed");
    }
    const abort = new AbortController();
    streamAbort.current = abort;
    let cursor = 0;
    let backoff = 1000;
    while (!abort.signal.aborted) {
      try {
        let activeSession = sessionRef.current;
        if (!activeSession) return;
        let response = await fetch(`${apiBase}/v1/training/jobs/${job.job_id}/events?after_sequence=${cursor}`, {
          headers: { Authorization: `Bearer ${activeSession.access_token}`, Accept: "text/event-stream", "Last-Event-ID": String(cursor) },
          signal: abort.signal,
        });
        if (response.status === 401) {
          const refreshed = await fetch(`${apiBase}/v1/training/token/refresh`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: activeSession.session_id, refresh_token: activeSession.refresh_token }),
            signal: abort.signal,
          });
          if (!refreshed.ok) throw new Error("Workspace session expired");
          const token = await refreshed.json();
          activeSession = { ...activeSession, access_token: token.access_token, expires_in: token.expires_in };
          sessionRef.current = activeSession;
          setSession(activeSession);
          response = await fetch(`${apiBase}/v1/training/jobs/${job.job_id}/events?after_sequence=${cursor}`, {
            headers: { Authorization: `Bearer ${activeSession.access_token}`, Accept: "text/event-stream", "Last-Event-ID": String(cursor) },
            signal: abort.signal,
          });
        }
        if (!response.ok || !response.body) throw new Error("Live event stream unavailable");
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += value;
          const frames = buffer.split("\n\n");
          buffer = frames.pop() || "";
          for (const frame of frames) {
            if (frame.startsWith(":")) continue;
            const idLine = frame.split("\n").find((line) => line.startsWith("id:"));
            const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
            if (!dataLine) continue;
            cursor = Math.max(cursor, Number(idLine?.slice(3).trim() || 0));
            const event = JSON.parse(dataLine.slice(5).trim()) as TrainingEvent;
            setEvents((current) => [...current.filter((item) => item.sequence !== event.sequence), event].sort((a, b) => a.sequence - b.sequence).slice(-5000));
          }
        }
        const currentJob = await request<Job>(`/v1/training/jobs/${job.job_id}`);
        setSelected(currentJob);
        if (terminalStates.has(currentJob.status)) return;
        backoff = 1000;
      } catch (reason) {
        if (abort.signal.aborted) return;
        setError(reason instanceof Error ? `${reason.message}; reconnecting after event ${cursor}` : "Live stream reconnecting");
        await new Promise((resolve) => window.setTimeout(resolve, backoff));
        backoff = Math.min(backoff * 2, 15000);
      }
    }
  }, [request, session]);

  useEffect(() => () => streamAbort.current?.abort(), []);

  const createJob = async () => {
    setBusy(true);
    setError("");
    try {
      const created = await request<Job>("/v1/training/jobs", {
        method: "POST",
        body: JSON.stringify({
          profile_id: createProfile,
          idempotency_key: `web-${crypto.randomUUID()}`,
          git_commit: "0000000",
          container_digest: `sha256:${"0".repeat(64)}`,
          metadata: { client: "training-workspace-web" },
          dataset_manifest_uri: datasetUri || null,
          dataset_manifest_sha256: datasetHash || null,
          tokenizer_uri: tokenizerUri || null,
          tokenizer_sha256: tokenizerHash || null,
        }),
      });
      await loadWorkspace();
      void selectJob(created);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Job submission failed");
    } finally {
      setBusy(false);
    }
  };

  const transition = async (action: "cancel" | "resume") => {
    if (!selected) return;
    setBusy(true);
    try {
      const updated = await request<Job>(`/v1/training/jobs/${selected.job_id}/${action}`, { method: "POST" });
      setSelected(updated);
      await loadWorkspace();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `Job ${action} failed`);
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    streamAbort.current?.abort();
    if (session) {
      await fetch(`${apiBase}/v1/training/token/revoke`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${session.access_token}` },
        body: JSON.stringify({ session_id: session.session_id, refresh_token: session.refresh_token }),
      }).catch(() => undefined);
    }
    setSession(null);
  };

  const filteredJobs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return jobs.filter((job) => !needle || `${job.job_id} ${job.status} ${job.spec.profile_id}`.toLowerCase().includes(needle));
  }, [jobs, query]);
  const selectedProfile = profiles.find((profile) => profile.profile_id === createProfile);
  const immutableInputsReady = selectedProfile?.run_type !== "pretrain" || (
    datasetUri.length > 0 && tokenizerUri.length > 0 && /^[0-9a-f]{64}$/.test(datasetHash) && /^[0-9a-f]{64}$/.test(tokenizerHash)
  );
  const losses = events.flatMap((event) => event.loss === undefined ? [] : [event.loss]);
  const validationLosses = events.flatMap((event) => event.validation_loss === undefined ? [] : [event.validation_loss]);
  const latest = events.at(-1);

  if (!session) {
    return (
      <main className="auth-shell">
        <section className="auth-panel">
          <img src={logoPath} alt="Craftly" className="brand-logo" onError={(event) => { event.currentTarget.style.display = "none"; }} />
          <div><p className="eyebrow">TRAINING CONTROL PLANE</p><h1>Craftly Workspace</h1></div>
          <label>Bootstrap credential<input type="password" autoComplete="off" value={bootstrapToken} onChange={(event) => setBootstrapToken(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void login(); }} /></label>
          {error && <p className="error"><CircleAlert size={16} />{error}</p>}
          <button className="primary" disabled={busy || bootstrapToken.length < 32} onClick={() => void login()}><KeyRound size={17} />Authenticate</button>
        </section>
      </main>
    );
  }

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand"><img src={logoPath} alt="" onError={(event) => { event.currentTarget.style.display = "none"; }} /><div><strong>Craftly</strong><span>Training Workspace</span></div></div>
        <div className="top-actions"><span className="connection"><span />CONTROL PLANE</span><button className="icon-button" title="Refresh workspace" onClick={() => void loadWorkspace()}><RefreshCw size={17} /></button><button className="icon-button" title="End session" onClick={() => void logout()}><LogOut size={17} /></button></div>
      </header>

      <section className="command-band">
        <div><p className="eyebrow">IMMUTABLE PROFILE</p><select value={createProfile} onChange={(event) => setCreateProfile(event.target.value)}>{profiles.map((profile) => <option key={`${profile.profile_id}-${profile.version}`} value={profile.profile_id}>{profile.profile_id} | v{profile.version}</option>)}</select></div>
        <button className="primary" disabled={busy || !createProfile || !immutableInputsReady} onClick={() => void createJob()}><Play size={17} />Submit job</button>
        <div className="profile-summary">{selectedProfile?.description}</div>
        {selectedProfile?.run_type === "pretrain" && <div className="immutable-inputs"><label>Dataset manifest<input value={datasetUri} onChange={(event) => setDatasetUri(event.target.value)} placeholder="s3://bucket/dataset/manifest.json" /></label><label>Dataset SHA-256<input value={datasetHash} maxLength={64} onChange={(event) => setDatasetHash(event.target.value.toLowerCase())} /></label><label>Tokenizer<input value={tokenizerUri} onChange={(event) => setTokenizerUri(event.target.value)} placeholder="s3://bucket/tokenizer.json" /></label><label>Tokenizer SHA-256<input value={tokenizerHash} maxLength={64} onChange={(event) => setTokenizerHash(event.target.value.toLowerCase())} /></label></div>}
      </section>

      {error && <div className="error-banner"><CircleAlert size={17} /><span>{error}</span><button className="icon-button" title="Dismiss" onClick={() => setError("")}><Ban size={15} /></button></div>}

      <div className="workspace-grid">
        <aside className="job-pane">
          <div className="pane-heading"><div><p className="eyebrow">JOBS</p><h2>{jobs.length} runs</h2></div><label className="search"><Search size={15} /><input aria-label="Filter jobs" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter" /></label></div>
          <div className="job-list">{filteredJobs.map((job) => <button key={job.job_id} className={`job-row ${selected?.job_id === job.job_id ? "selected" : ""}`} onClick={() => void selectJob(job)}><div><strong>{job.spec.profile_id}</strong><small>{job.job_id.slice(0, 8)} | {new Date(job.created_at).toLocaleString()}</small></div><Status value={job.status} /><ChevronRight size={16} /></button>)}</div>
        </aside>

        <section className="detail-pane">
          {!selected ? <div className="empty-state"><Server size={32} /><h2>Select a training job</h2></div> : <>
            <div className="run-heading"><div><p className="eyebrow">{selected.job_id}</p><h2>{selected.spec.profile_id}</h2><Status value={selected.status} /></div><div className="run-actions">{!terminalStates.has(selected.status) && <button className="danger" disabled={busy} onClick={() => void transition("cancel")}><Square size={15} />Cancel</button>}{["failed", "cancelled"].includes(selected.status) && <button className="secondary" disabled={busy} onClick={() => void transition("resume")}><ListRestart size={16} />Resume</button>}</div></div>

            <div className="metric-strip">
              <div><Gauge size={17} /><span>Step</span><strong>{formatNumber(latest?.step, 0)} / {formatNumber(latest?.max_steps || selected.spec.steps, 0)}</strong></div>
              <div><Activity size={17} /><span>Loss</span><strong>{formatNumber(latest?.loss, 4)}</strong></div>
              <div><Cpu size={17} /><span>Tokens/sec</span><strong>{formatNumber(latest?.tokens_per_second, 0)}</strong></div>
              <div><Database size={17} /><span>GPU memory</span><strong>{formatBytes(latest?.gpu_memory_bytes)}</strong></div>
            </div>

            <div className="chart-grid">
              <section><div className="section-heading"><div><p className="eyebrow">TRAINING</p><h3>Loss</h3></div><strong>{formatNumber(losses.at(-1), 4)}</strong></div><Sparkline values={losses} /></section>
              <section><div className="section-heading"><div><p className="eyebrow">VALIDATION</p><h3>Validation loss</h3></div><strong>{formatNumber(validationLosses.at(-1), 4)}</strong></div><Sparkline values={validationLosses} tone="amber" /></section>
            </div>

            <section className="runtime-facts"><div><Server size={16} /><span>Scheduler</span><strong>{selected.spec.scheduler}</strong></div><div><Cpu size={16} /><span>Topology</span><strong>{selected.spec.resources.nodes} x {selected.spec.resources.gpus_per_node} GPU</strong></div><div><ShieldCheck size={16} /><span>Strategy</span><strong>{selected.spec.distributed_strategy}</strong></div><div><Clock3 size={16} /><span>Updated</span><strong>{new Date(selected.updated_at).toLocaleTimeString()}</strong></div></section>

            <div className="lower-grid">
              <section className="log-section"><div className="section-heading"><div><p className="eyebrow">LIVE EVENTS</p><h3>Worker stream</h3></div><span>{events.length}</span></div><div className="log-view">{events.slice(-200).map((event) => <div key={event.event_id}><time>{new Date(event.timestamp).toLocaleTimeString()}</time><b>{event.stage}</b><span>{event.message || `${event.status}${event.step !== undefined ? ` | step ${event.step}` : ""}`}</span></div>)}</div></section>
              <div className="side-stack"><section><div className="section-heading"><div><p className="eyebrow">CHECKPOINTS</p><h3>Version lifecycle</h3></div><FileArchive size={18} /></div><div className="compact-list">{checkpoints.map((item) => <div key={item.checkpoint_id}><span>step {item.step}</span><strong>{item.promoted ? "promoted" : item.reload_verified ? "verified" : "pending"}</strong></div>)}{evaluations.map((item) => <div key={item.evaluation_id}><span>evaluation</span><strong>{item.decision}</strong></div>)}</div></section><section><div className="section-heading"><div><p className="eyebrow">ARTIFACTS</p><h3>Durable outputs</h3></div><Database size={18} /></div><div className="compact-list">{artifacts.map((item) => <div key={item.artifact_id}><span>{item.kind}</span><strong>{formatBytes(item.size_bytes)}</strong></div>)}</div></section><section><div className="section-heading"><div><p className="eyebrow">AUDIT</p><h3>Job history</h3></div><ShieldCheck size={18} /></div><div className="compact-list">{audit.map((item) => <div key={item.audit_id}><span>{item.action}</span><strong>{item.outcome}</strong></div>)}</div></section></div>
            </div>
          </>}
        </section>
      </div>
    </main>
  );
}
