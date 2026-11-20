"""
app.py - Modern Web UI for AeroVantage X9 Drone Manual Assistant.

Run with:
    python app.py
or:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from rag import answer

app = FastAPI(title="AeroVantage X9 Drone Assistant", version="1.0.0")

ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "title": "LED Diagnostic Code",
        "question": "What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?"
    },
    {
        "id": "Q2",
        "title": "Underwater Sonar Mapping (Unsupported)",
        "question": "Can the AeroVantage X9 be used for underwater sonar mapping at a depth of 50 metres?"
    },
    {
        "id": "Q3",
        "title": "Emergency RTH Cutoff Voltage",
        "question": "At what minimum per-cell voltage, in volts, does the X9 trigger a forced emergency Return-to-Home (RTH) landing?"
    },
    {
        "id": "Q4",
        "title": "Certified Wind Speed Limit",
        "question": "What is the maximum sustained wind speed at which the X9 is certified to operate safely?"
    },
    {
        "id": "Q5",
        "title": "DJI CEO Query (Out of Corpus)",
        "question": "Who is the current Chief Executive Officer of DJI Innovations?"
    }
]


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
def api_query(req: QueryRequest) -> Dict[str, Any]:
    start_time = time.time()
    res = answer(req.question)
    elapsed = round(time.time() - start_time, 3)
    return {
        "question": req.question,
        "answer": res.get("answer", "[not found]"),
        "sources": res.get("sources", []),
        "latency_sec": elapsed
    }


@app.get("/api/benchmark")
def api_benchmark() -> List[Dict[str, Any]]:
    results = []
    for item in ASSESSMENT_QUESTIONS:
        start_time = time.time()
        res = answer(item["question"])
        elapsed = round(time.time() - start_time, 3)
        results.append({
            "id": item["id"],
            "title": item["title"],
            "question": item["question"],
            "answer": res.get("answer", "[not found]"),
            "sources": res.get("sources", []),
            "latency_sec": elapsed
        })
    return results


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroVantage X9 Technical Manual Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-input: #0f172a;
            --border-color: #334155;
            --accent-primary: #38bdf8;
            --accent-hover: #0284c7;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 900px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        header {
            text-align: center;
            padding-bottom: 0.5rem;
        }

        .badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-primary);
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-bottom: 0.75rem;
        }

        h1 {
            font-size: 1.85rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }

        .section-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .preset-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
        }

        .btn-preset {
            background: #273549;
            color: #e2e8f0;
            border: 1px solid #3b4f6b;
            padding: 0.45rem 0.85rem;
            border-radius: 8px;
            font-size: 0.825rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-preset:hover {
            background: var(--accent-hover);
            color: #ffffff;
            border-color: var(--accent-primary);
            transform: translateY(-1px);
        }

        .input-group {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }

        textarea {
            flex: 1;
            min-height: 85px;
            padding: 0.85rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-input);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }

        .btn-primary {
            background: var(--accent-primary);
            color: #0f172a;
            font-weight: 600;
            font-size: 0.95rem;
            border: none;
            border-radius: 8px;
            padding: 0 1.5rem;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 140px;
        }

        .btn-primary:hover {
            background: #7dd3fc;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            box-shadow: none;
        }

        .btn-secondary {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            cursor: pointer;
        }

        .btn-secondary:hover {
            color: var(--text-primary);
            border-color: var(--text-secondary);
        }

        #result-box {
            display: none;
            margin-top: 1.25rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1.25rem;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .answer-content {
            background: var(--bg-input);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            font-size: 1rem;
            line-height: 1.6;
            border-left: 4px solid var(--accent-primary);
            margin-bottom: 1rem;
            font-family: 'Inter', sans-serif;
        }

        .answer-content.not-found {
            border-left-color: var(--accent-danger);
            color: #fca5a5;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        .meta-badges {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .source-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            background: #1e3a5f;
            color: #7dd3fc;
            border: 1px solid #0284c7;
        }

        .source-pill.empty {
            background: #3f1d24;
            color: #fca5a5;
            border-color: #991b1b;
        }

        .latency-pill {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Benchmark Table */
        #benchmark-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 1rem;
        }

        #benchmark-table th, #benchmark-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        #benchmark-table th {
            color: var(--text-secondary);
            font-weight: 600;
            background: #172033;
        }

        #benchmark-table tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(15, 23, 42, 0.3);
            border-radius: 50%;
            border-top-color: #0f172a;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge">AeroVantage X9 Industrial Hexacopter</span>
            <h1>Technical Maintenance Manual Assistant</h1>
            <p class="subtitle">Strict Zero-Hallucination Retrieval-Augmented Generation (50-Page Flight Manual)</p>
        </header>

        <div class="card">
            <div class="section-title">
                <span>Preset Assessment Questions</span>
                <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: normal;">Click any question to populate</span>
            </div>
            <div class="preset-buttons">
                <button class="btn-preset" onclick="setQuestion(1)">Q1: LED Blink Code</button>
                <button class="btn-preset" onclick="setQuestion(2)">Q2: Underwater Sonar (Unsupported)</button>
                <button class="btn-preset" onclick="setQuestion(3)">Q3: RTH Cutoff Voltage</button>
                <button class="btn-preset" onclick="setQuestion(4)">Q4: Certified Wind Limit</button>
                <button class="btn-preset" onclick="setQuestion(5)">Q5: DJI CEO (Out-of-Corpus)</button>
            </div>

            <div class="input-group">
                <textarea id="question-input" placeholder="Enter any technical maintenance query... (e.g., What does the LED blink code indicate?)"></textarea>
                <button id="ask-btn" class="btn-primary" onclick="submitQuery()">Ask Manual</button>
            </div>

            <div id="result-box">
                <div class="result-header">
                    <span class="section-title" style="margin:0;">Grounding Guard Response</span>
                    <span id="latency-val" class="latency-pill"></span>
                </div>
                <div id="answer-text" class="answer-content"></div>
                <div class="meta-badges">
                    <span id="sources-badge" class="source-pill"></span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">
                <span>Assessment Suite Benchmark</span>
                <button id="bench-btn" class="btn-secondary" onclick="runBenchmark()">▶ Run All 5 Questions</button>
            </div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                Executes the 5 assessment queries end-to-end to verify grounding, source attribution, and not-found handling.
            </p>
            <div id="bench-loading" style="display:none; color: var(--accent-primary); font-size: 0.85rem; padding: 0.5rem 0;">
                <span class="spinner" style="border-top-color: var(--accent-primary);"></span> Running evaluation across ChromaDB and RAG pipeline...
            </div>
            <div style="overflow-x: auto;">
                <table id="benchmark-table" style="display:none;">
                    <thead>
                        <tr>
                            <th style="width: 50px;">ID</th>
                            <th>Question</th>
                            <th>Generated Answer</th>
                            <th style="width: 90px;">Sources</th>
                            <th style="width: 70px;">Latency</th>
                        </tr>
                    </thead>
                    <tbody id="benchmark-tbody"></tbody>
                </table>
            </div>
        </div>

        <footer>
            AeroVantage X9 RAG Assistant &bull; ChromaDB + SentenceTransformers + BM25 &bull; Grounded Knowledge Base
        </footer>
    </div>

    <script>
        const PRESETS = {
            1: "What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?",
            2: "Can the AeroVantage X9 be used for underwater sonar mapping at a depth of 50 metres?",
            3: "At what minimum per-cell voltage, in volts, does the X9 trigger a forced emergency Return-to-Home (RTH) landing?",
            4: "What is the maximum sustained wind speed at which the X9 is certified to operate safely?",
            5: "Who is the current Chief Executive Officer of DJI Innovations?"
        };

        function setQuestion(num) {
            document.getElementById('question-input').value = PRESETS[num];
            submitQuery();
        }

        async function submitQuery() {
            const input = document.getElementById('question-input');
            const q = input.value.trim();
            if (!q) return;

            const btn = document.getElementById('ask-btn');
            const resultBox = document.getElementById('result-box');
            const answerText = document.getElementById('answer-text');
            const sourcesBadge = document.getElementById('sources-badge');
            const latencyVal = document.getElementById('latency-val');

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Querying...';
            resultBox.style.display = 'none';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q })
                });
                const data = await response.json();

                answerText.textContent = data.answer;
                latencyVal.textContent = `Latency: ${data.latency_sec}s`;

                if (data.answer === '[not found]') {
                    answerText.className = 'answer-content not-found';
                    sourcesBadge.className = 'source-pill empty';
                    sourcesBadge.textContent = 'Sources: None [not found]';
                } else {
                    answerText.className = 'answer-content';
                    sourcesBadge.className = 'source-pill';
                    const srcStr = data.sources && data.sources.length > 0 ? data.sources.map(p => `Page ${p}`).join(', ') : 'None';
                    sourcesBadge.textContent = `Verified Sources: ${srcStr}`;
                }

                resultBox.style.display = 'block';
            } catch (err) {
                answerText.className = 'answer-content not-found';
                answerText.textContent = 'Error connecting to RAG backend: ' + err.message;
                resultBox.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Ask Manual';
            }
        }

        async function runBenchmark() {
            const benchBtn = document.getElementById('bench-btn');
            const benchLoading = document.getElementById('bench-loading');
            const benchTable = document.getElementById('benchmark-table');
            const tbody = document.getElementById('benchmark-tbody');

            benchBtn.disabled = true;
            benchLoading.style.display = 'block';
            benchTable.style.display = 'none';
            tbody.innerHTML = '';

            try {
                const resp = await fetch('/api/benchmark');
                const list = await resp.json();

                list.forEach(item => {
                    const tr = document.createElement('tr');
                    const isNotFound = item.answer === '[not found]';
                    const srcBadge = isNotFound
                        ? '<span class="source-pill empty" style="font-size:0.7rem; padding:2px 6px;">[]</span>'
                        : item.sources.map(p => `<span class="source-pill" style="font-size:0.7rem; padding:2px 6px;">p.${p}</span>`).join(' ');

                    tr.innerHTML = `
                        <td><strong>${item.id}</strong></td>
                        <td style="color:#cbd5e1;">${item.question}</td>
                        <td style="${isNotFound ? 'color:#f87171; font-family:monospace; font-weight:bold;' : 'color:#f1f5f9;'}">${item.answer}</td>
                        <td>${srcBadge}</td>
                        <td style="color:#64748b; font-family:monospace; font-size:0.75rem;">${item.latency_sec}s</td>
                    `;
                    tbody.appendChild(tr);
                });

                benchTable.style.display = 'table';
            } catch (e) {
                alert('Benchmark failed: ' + e.message);
            } finally {
                benchLoading.style.display = 'none';
                benchBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting AeroVantage X9 Drone Assistant Web UI on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
