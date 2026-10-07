const { useState, useEffect, useRef, useMemo } = React;
const api = async (path, opts) => {
  const r = await fetch("/api" + path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const COLOR = { low: "var(--low)", steady: "var(--steady)", high: "var(--high)" };
const BAND = { low: "Low risk", steady: "Steady", high: "High risk" };

function Segmented({ field, value, onChange }) {
  return (
    <div className="field">
      <label>{field.label}</label>
      <div className="seg">
        {field.options.map(o => (
          <button key={o.value} className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}>
            {o.label}{o.hint && <span>{o.hint}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

function Gauge({ pct, band }) {
  const len = Math.PI * 100;
  return (
    <div className="gauge">
      <svg viewBox="0 0 240 130">
        <path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke="#ece9e0" strokeWidth="14" strokeLinecap="round" />
        <path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke={COLOR[band]} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${(len * pct) / 100} ${len}`} style={{ transition: "stroke-dasharray .5s ease, stroke .3s" }} />
      </svg>
      <div className="num">{pct.toFixed(1)}<sup>%</sup></div>
    </div>
  );
}

function PredictionResult({ res, schema, form }) {
  const high = res.prediction === 1;
  const title = high ? "High-risk profile" : res.risk_band === "steady" ? "Steady profile" : "Low-risk profile";
  const chips = schema.map(f => [f.label, f.options.find(o => o.value === form[f.key]).label]);
  return (
    <div className="predbox" style={{ "--c": COLOR[res.risk_band] }}>
      <p className="eyebrow">Prediction</p>
      <div className="predtitle">{title}</div>
      <p>{high ? "The model expects this customer to leave quietly." : "The model expects this customer to stay with the network."}
        {" "}Retention probability: <b>{res.retention_probability}%</b>.</p>
      <div className="chips">{chips.map(([k, v]) => <span key={k}><em>{k}</em> {v}</span>)}</div>
    </div>
  );
}

function Drivers({ items }) {
  const max = Math.max(10, ...items.map(d => Math.abs(d.impact)));
  return items.map(d => {
    const w = (Math.abs(d.impact) / max) * 50;
    const up = d.impact > 0;
    return (
      <div className="drv" key={d.key}>
        <div>{d.label}<small>{d.value}</small></div>
        <div className="track"><i style={{ width: w + "%", left: up ? "50%" : 50 - w + "%", background: up ? COLOR.high : COLOR.low }} /></div>
        <b style={{ color: up ? COLOR.high : COLOR.low }}>{up ? "+" : ""}{d.impact}</b>
      </div>
    );
  });
}

const DEFAULTS = { tenure_group: 0, charge_level: 0, multiple_lines: 0, tech_support: 0, internet_service: 0, online_security: 0, contract: 0 };

function Predict({ schema }) {
  const [form, setForm] = useState(DEFAULTS);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => {
    setBusy(true);
    const t = setTimeout(() => post("/predict", form).then(r => { setRes(r); setErr(""); }).catch(e => setErr(e.message)).finally(() => setBusy(false)), 220);
    return () => clearTimeout(t);
  }, [form]);
  const groups = useMemo(() => [...new Set(schema.map(f => f.group))], [schema]);
  return (
    <div className="grid">
      <div className="card">
        {groups.map(g => (
          <div className="group" key={g}>
            <p className="eyebrow">{g}</p>
            {schema.filter(f => f.group === g).map(f => <Segmented key={f.key} field={f} value={form[f.key]} onChange={v => setForm({ ...form, [f.key]: v })} />)}
          </div>
        ))}
      </div>
      <div className="sticky">
        {err && <p className="err">{err}</p>}
        {res && (
          <div className={busy ? "fade" : ""}>
            <div className="card">
              <p className="eyebrow">Churn probability</p>
              <Gauge pct={res.churn_probability} band={res.risk_band} />
              <PredictionResult res={res} schema={schema} form={form} />
            </div>
            <div className="card sec">
              <p className="eyebrow">What moves the score <span style={{ textTransform: "none", letterSpacing: 0 }}>· points vs. an average customer</span></p>
              <Drivers items={res.drivers} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Insights() {
  const [d, setD] = useState(null);
  useEffect(() => { api("/insights").then(setD); }, []);
  if (!d) return <p>Loading…</p>;
  return (
    <>
      <div className="tiles">
        <div className="tile"><div className="v">{d.customers.toLocaleString()}</div><div className="l">Customers analysed</div></div>
        <div className="tile"><div className="v">{d.churn_rate}%</div><div className="l">Overall churn rate</div></div>
      </div>
      <div className="grid" style={{ alignItems: "start" }}>
        {d.features.map(f => (
          <div className="card feat" key={f.key}>
            <p className="eyebrow">{f.label}</p>
            {f.segments.map(s => (
              <div className="bar" key={s.label}>
                <span>{s.label}</span>
                <div className="t"><i style={{ width: Math.min(100, s.churn_rate * 1.6) + "%" }} /><u style={{ left: Math.min(100, d.churn_rate * 1.6) + "%" }} /></div>
                <span>{s.churn_rate}% · {s.customers.toLocaleString()}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <p style={{ color: "var(--mute)", fontSize: 13 }}>The dark marker on each bar is the overall churn rate.</p>
    </>
  );
}

function Batch() {
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const run = async file => {
    setBusy(true); setErr("");
    const fd = new FormData(); fd.append("file", file);
    try { setOut(await api("/predict/batch", { method: "POST", body: fd })); } catch (e) { setErr(e.message); setOut(null); }
    setBusy(false);
  };
  const download = () => {
    const cols = Object.keys(out.rows[0]);
    const csv = [cols.join(","), ...out.rows.map(r => cols.map(c => r[c]).join(","))].join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "churn_scores.csv"; a.click();
  };
  const sorted = out ? [...out.rows].sort((a, b) => b.probability - a.probability) : [];
  return (
    <>
      <div className="drop">
        <p style={{ margin: "0 0 14px" }}>Upload a CSV with <b>Tenure, Monthly Charges, Multiple Lines, Tech Support, Internet Service, Online Security, Contract</b> (raw values) or the engineered feature columns.</p>
        <input type="file" accept=".csv" id="f" hidden onChange={e => e.target.files[0] && run(e.target.files[0])} />
        <button className="btn" onClick={() => document.getElementById("f").click()}>{busy ? "Scoring…" : "Choose CSV"}</button>
        {err && <p className="err">{err}</p>}
      </div>
      {out && (
        <div className="sec">
          <div className="tiles">
            <div className="tile"><div className="v">{out.count}</div><div className="l">Customers scored</div></div>
            <div className="tile"><div className="v" style={{ color: COLOR.high }}>{out.summary.high}</div><div className="l">High risk</div></div>
            <div className="tile"><div className="v" style={{ color: COLOR.steady }}>{out.summary.steady}</div><div className="l">Steady</div></div>
            <div className="tile"><div className="v" style={{ color: COLOR.low }}>{out.summary.low}</div><div className="l">Low risk</div></div>
          </div>
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <p className="eyebrow" style={{ margin: 0 }}>Ranked by churn probability</p>
              <button className="btn ghost" onClick={download}>Download scores</button>
            </div>
            <div className="scroll"><table className="tbl">
              <thead><tr><th>Row</th><th>Risk</th><th>Probability</th><th>Tenure Group</th><th>Charge Level</th><th>Multiple Lines</th><th>Tech Support</th><th>Internet Service</th><th>Online Security</th><th>Contract</th></tr></thead>
              <tbody>{sorted.map(r => (
                <tr key={r.row}><td>{r.row}</td><td><span className={"tag " + r.risk_band}>{BAND[r.risk_band]}</span></td><td>{r.probability}%</td>
                  <td>{r.tenure_group}</td><td>{r.charge_level}</td><td>{r.multiple_lines}</td><td>{r.tech_support}</td><td>{r.internet_service}</td><td>{r.online_security}</td><td>{r.contract}</td></tr>))}</tbody>
            </table></div>
          </div>
        </div>
      )}
    </>
  );
}

function Model() {
  const [m, setM] = useState(null);
  useEffect(() => { api("/model").then(setM); }, []);
  if (!m) return <p>Loading…</p>;
  const mt = m.metrics;
  return (
    <div className="grid">
      <div className="card">
        <p className="eyebrow">Test performance · {m.split} split</p>
        <div className="tiles" style={{ marginBottom: 0 }}>
          {[["AUC", mt.auc], ["Recall", mt.recall], ["Precision", mt.precision], ["F1", mt.f1]].map(([l, v]) => (
            <div className="tile" key={l}><div className="v">{v.toFixed(2)}</div><div className="l">{l}</div></div>))}
        </div>
        <p style={{ color: "var(--mute)", marginBottom: 0 }}>Tuned for recall: the model catches roughly 3 in 4 churners, accepting some false alarms — the right trade when a retention call is cheap.</p>
      </div>
      <div className="card">
        <p className="eyebrow">{m.name}</p>
        {m.members.map(x => <div className="kv" key={x}><span>Member</span><span>{x}</span></div>)}
        <div className="kv"><span>Voting</span><span>Soft</span></div>
        <div className="kv"><span>Class balancing</span><span>{m.balancing}</span></div>
        <div className="kv"><span>Decision threshold</span><span>{m.threshold}</span></div>
        <div className="kv"><span>Training data</span><span>{m.training_rows.toLocaleString()} customers</span></div>
        <div className="kv"><span>Loaded from</span><span>{m.source}</span></div>
      </div>
    </div>
  );
}

const TABS = { predict: ["Predict", "Score a customer", "Describe a customer and see their quiet-exit risk update live."],
  insights: ["Portfolio", "Where churn concentrates", "Historic churn rate across every customer segment."],
  batch: ["Batch", "Score a customer list", "Rank a whole book of customers by churn risk."],
  model: ["Model", "How the model performs", "The tuned hybrid ensemble behind every score."] };

function App() {
  const [tab, setTab] = useState("predict");
  const [schema, setSchema] = useState(null);
  const [ok, setOk] = useState(null);
  const [build, setBuild] = useState("");
  useEffect(() => { api("/schema").then(r => setSchema(r.features)); api("/health").then(h => { setOk(true); setBuild(h.build); }).catch(() => setOk(false)); }, []);
  return (
    <div className="shell">
      <header>
        <div className="brand">
          <svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="#0f5c5a" strokeWidth="2.2" strokeLinecap="round"><path d="M5 20a14 14 0 0 1 20 0M9 16a9 9 0 0 1 12 0"/><circle cx="15" cy="23" r="1.6" fill="#0f5c5a"/></svg>
          <div>Silent Churn Detector</div>
        </div>
        <nav>{Object.entries(TABS).map(([k, v]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{v[0]}</button>)}</nav>
        <div className="status"><span className={"dot" + (ok === false ? " off" : "")} />{ok === false ? "API offline" : "Model online · " + build}</div>
      </header>
      <div className="lead"><h1>{TABS[tab][1]}</h1><p>{TABS[tab][2]}</p></div>
      {tab === "predict" && (schema ? <Predict schema={schema} /> : <p>Loading…</p>)}
      {tab === "insights" && <Insights />}
      {tab === "batch" && <Batch />}
      {tab === "model" && <Model />}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
