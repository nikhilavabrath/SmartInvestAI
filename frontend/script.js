const form = document.getElementById('ticker-form');
const tickerInput = document.getElementById('ticker-input');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingNote = document.getElementById('loading-note');
const errorNote = document.getElementById('error-note');

let forecastChart = null;

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const ticker = tickerInput.value.trim().toUpperCase();
    if (ticker) {
        localStorage.setItem('smartinvest_last_ticker', ticker);
        runAnalysis(ticker);
    }
});

async function runAnalysis(ticker) {
    setLoading(true);
    hideError();

    // Currency must resolve BEFORE anything that displays a price -- otherwise
    // there's a race where the price renders using the previous ticker's
    // currency symbol. Everything else can run in parallel after that.
    await fetchTickerName(ticker);

    fetchPredict(ticker);
    fetchPredictWeek(ticker);
    fetchExplain(ticker);
    fetchSentiment(ticker);
    fetchCandlestick(ticker);

    // Model-info depends on the other models having trained at least once,
    // so give predict/explain a moment's head start before checking it.
    setTimeout(() => fetchModelInfo(ticker), 1500);

    setLoading(false);
}

function setLoading(isLoading) {
    loadingNote.hidden = !isLoading;
    analyzeBtn.disabled = isLoading;
}

function showError(msg) {
    errorNote.textContent = msg;
    errorNote.hidden = false;
}
function hideError() {
    errorNote.hidden = true;
}

async function getJSON(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `Request failed: ${url}`);
    return data;
}

// --- /ticker-name ---
let currentCurrencySymbol = '$';

async function fetchTickerName(ticker) {
    const el = document.getElementById('hero-companyname');
    el.textContent = '';
    currentCurrencySymbol = '$'; // reset to a sane default until this resolves
    try {
        const data = await getJSON(`/ticker-name?ticker=${encodeURIComponent(ticker)}`);
        if (data.name) el.textContent = data.name;
        if (data.currency_symbol) currentCurrencySymbol = data.currency_symbol;
    } catch (err) {
        // Non-critical -- just leave the name slot empty rather than showing an error
    }
}

// --- /predict ---
async function fetchPredict(ticker) {
    try {
        const data = await getJSON(`/predict?ticker=${encodeURIComponent(ticker)}`);
        const delta = data.predicted_price - data.last_close;
        const pct = (delta / data.last_close) * 100;
        const dir = delta >= 0 ? 'up' : 'down';
        const arrow = delta >= 0 ? '▲' : '▼';

        document.getElementById('hero-price').textContent = `${currentCurrencySymbol}${data.predicted_price.toFixed(2)}`;
        const deltaEl = document.getElementById('hero-delta');
        deltaEl.textContent = `${arrow} ${Math.abs(pct).toFixed(2)}%`;
        deltaEl.className = `hero-delta ${dir}`;
        document.getElementById('hero-lastclose').textContent = `${currentCurrencySymbol}${data.last_close.toFixed(2)}`;
    } catch (err) {
        showError(`Prediction unavailable: ${err.message}`);
    }
}

// --- /predict-week ---
async function fetchPredictWeek(ticker) {
    try {
        const data = await getJSON(`/predict-week?ticker=${encodeURIComponent(ticker)}`);
        renderForecastChart(data);
    } catch (err) {
        showError(`5-day forecast unavailable: ${err.message}`);
    }
}

function renderForecastChart(data) {
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    const labels = ['Last close', ...data.labels];
    const values = [data.last_close, ...data.prices];

    if (forecastChart) forecastChart.destroy();
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: '#161D26',
                backgroundColor: 'rgba(22,29,38,0.05)',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '#161D26',
                tension: 0.15,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { ticks: { font: { family: 'IBM Plex Mono', size: 11 } }, grid: { color: '#D8DAD4' } },
                x: { ticks: { font: { family: 'IBM Plex Mono', size: 11 } }, grid: { display: false } },
            },
        },
    });
}

// --- /explain ---
async function fetchExplain(ticker) {
    try {
        const data = await getJSON(`/explain?ticker=${encodeURIComponent(ticker)}`);
        const pct = (data.probability_up * 100).toFixed(1);
        document.getElementById('explain-summary').innerHTML =
            `Model leans <strong class="${data.predicted_direction}">${data.predicted_direction}</strong> ` +
            `&mdash; <span class="mono">${pct}%</span> probability of a rise.`;

        const maxAbs = Math.max(...data.top_factors.map(f => Math.abs(f.shap_contribution)), 0.0001);
        const list = document.getElementById('factor-list');
        list.innerHTML = data.top_factors.map(f => {
            const widthPct = (Math.abs(f.shap_contribution) / maxAbs) * 100;
            return `
        <li class="factor-row">
          <span class="factor-name">${f.feature}</span>
          <span class="factor-bar-track">
            <span class="factor-bar-fill ${f.pushed_toward}" style="width:${widthPct}%"></span>
          </span>
          <span class="factor-value ${f.pushed_toward}">${f.shap_contribution > 0 ? '+' : ''}${f.shap_contribution.toFixed(3)}</span>
        </li>`;
        }).join('');
    } catch (err) {
        showError(`Explanation unavailable: ${err.message}`);
    }
}

// --- /sentiment ---
async function fetchSentiment(ticker) {
    try {
        const data = await getJSON(`/sentiment?ticker=${encodeURIComponent(ticker)}`);
        const el = document.getElementById('sentiment-readout');
        const score = data.latest_sentiment;
        const cls = score > 0.05 ? 'up' : score < -0.05 ? 'down' : 'neutral';
        el.textContent = score.toFixed(3);
        el.className = `sentiment-readout ${cls}`;
        if (data.note) el.title = data.note;
    } catch (err) {
        // Sentiment is non-critical -- fail quietly into a neutral display rather than an error banner
        const el = document.getElementById('sentiment-readout');
        el.textContent = 'n/a';
        el.className = 'sentiment-readout neutral';
    }
}

// --- /model-info ---
async function fetchModelInfo(ticker) {
    try {
        const data = await getJSON(`/model-info?ticker=${encodeURIComponent(ticker)}`);

        const dirEl = document.getElementById('metric-direction');
        if (data.direction_model) {
            const m = data.direction_model.model;
            const b = data.direction_model.naive_baseline;
            const beat = data.direction_model.beats_baseline_f1;
            dirEl.innerHTML =
                `F1 <strong>${m.F1.toFixed(3)}</strong> vs baseline ${b.F1.toFixed(3)} ` +
                `<span class="${beat ? 'beat' : 'miss'}">(${beat ? 'beats baseline' : 'at/below baseline'})</span>`;
        } else {
            dirEl.textContent = 'Not yet trained for this ticker — check back after /explain has run.';
        }

        const priceEl = document.getElementById('metric-price');
        if (data.price_model) {
            const m = data.price_model.model;
            const b = data.price_model.naive_baseline;
            const beat = data.price_model.beats_baseline_rmse;
            priceEl.innerHTML =
                `RMSE <strong>${m.RMSE.toFixed(2)}</strong> vs baseline ${b.RMSE.toFixed(2)} ` +
                `<span class="${beat ? 'beat' : 'miss'}">(${beat ? 'beats baseline' : 'at/below baseline'})</span>`;
        } else {
            priceEl.textContent = 'Not yet trained for this ticker — check back after /predict has run.';
        }

        const baselineText = document.getElementById('hero-baseline-text');
        if (data.price_model) {
            const m = data.price_model.model;
            const b = data.price_model.naive_baseline;
            baselineText.innerHTML =
                `This model's 1-day RMSE is <span class="mono">${m.RMSE.toFixed(2)}</span> vs a naive ` +
                `"no change" baseline of <span class="mono">${b.RMSE.toFixed(2)}</span> &mdash; ` +
                `${data.price_model.beats_baseline_rmse ? 'a real, if modest, edge.' : 'essentially a statistical tie, which is expected for short-horizon prices.'}`;
        }
    } catch (err) {
        // Non-critical panel -- leave placeholders in place
    }
}

// --- /candlestick-data ---
async function fetchCandlestick(ticker) {
    try {
        const data = await getJSON(`/candlestick-data?ticker=${encodeURIComponent(ticker)}&period=6mo`);
        const candles = data.candles;

        const trace1 = {
            x: candles.map(c => c.time),
            open: candles.map(c => c.open),
            high: candles.map(c => c.high),
            low: candles.map(c => c.low),
            close: candles.map(c => c.close),
            type: 'candlestick',
            name: ticker,
            increasing: { line: { color: '#2F6F5E' } },
            decreasing: { line: { color: '#9B3B2E' } },
            yaxis: 'y2',
        };
        const trace2 = {
            x: candles.map(c => c.time),
            y: candles.map(c => c.volume),
            type: 'bar',
            name: 'Volume',
            marker: { color: '#D8DAD4' },
            yaxis: 'y',
        };

        const layout = {
            dragmode: 'pan',
            margin: { t: 10, l: 50, r: 10, b: 30 },
            font: { family: 'Public Sans, sans-serif', size: 12, color: '#161D26' },
            paper_bgcolor: '#FFFFFF',
            plot_bgcolor: '#FFFFFF',
            xaxis: { rangeslider: { visible: false }, gridcolor: '#EFEFE9' },
            yaxis: { domain: [0, 0.18], gridcolor: '#EFEFE9' },
            yaxis2: { domain: [0.22, 1], gridcolor: '#EFEFE9' },
            showlegend: false,
        };

        Plotly.newPlot('candlestick-chart', [trace1, trace2], layout, { responsive: true, displayModeBar: false });
    } catch (err) {
        document.getElementById('candlestick-chart').textContent = `Price history unavailable: ${err.message}`;
    }
}

// --- Agent chat ---
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const chatWindow = document.getElementById('chat-window');
const chatEmpty = document.getElementById('chat-empty');

// One session id per browser tab load -- ties this conversation's
// multi-turn history together server-side (see /agent/chat).
const agentSessionId = crypto.randomUUID();

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    appendChatMessage('user', message);
    chatInput.value = '';
    chatInput.disabled = true;
    chatSendBtn.disabled = true;

    try {
        const res = await fetch('/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: agentSessionId, message }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Agent request failed');
        appendChatMessage('assistant', data.response, data.tool_calls);
    } catch (err) {
        appendChatMessage('assistant', `Sorry, something went wrong: ${err.message}`);
    } finally {
        chatInput.disabled = false;
        chatSendBtn.disabled = false;
        chatInput.focus();
    }
});

function appendChatMessage(role, text, toolCalls) {
    if (chatEmpty) chatEmpty.remove();

    const wrap = document.createElement('div');
    wrap.className = `chat-msg chat-msg-${role}`;

    const roleLabel = document.createElement('div');
    roleLabel.className = 'chat-msg-role';
    roleLabel.textContent = role === 'user' ? 'You' : 'Agent';
    wrap.appendChild(roleLabel);

    const body = document.createElement('div');
    body.className = 'chat-msg-body';
    if (role === 'assistant') {
        // Agent replies come back as Markdown (bold, tables) -- render it
        // properly instead of dumping raw "**text**" and "|" syntax on screen.
        // Sanitized with DOMPurify since this is LLM output being inserted as
        // HTML, not text we wrote ourselves.
        body.innerHTML = DOMPurify.sanitize(marked.parse(text));
    } else {
        body.textContent = text; // the user's own typed message -- never needs markdown rendering
    }
    wrap.appendChild(body);

    if (toolCalls && toolCalls.length > 0) {
        const details = document.createElement('details');
        details.className = 'chat-trace';
        const summary = document.createElement('summary');
        summary.textContent = `${toolCalls.length} tool call${toolCalls.length > 1 ? 's' : ''} made`;
        details.appendChild(summary);

        toolCalls.forEach(tc => {
            const item = document.createElement('div');
            item.className = 'chat-trace-item';
            item.textContent = `${tc.tool}(${JSON.stringify(tc.input)}) →\n${JSON.stringify(tc.result, null, 2)}`;
            details.appendChild(item);
        });
        wrap.appendChild(details);
    }

    chatWindow.appendChild(wrap);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Load the last-searched ticker (persisted across refreshes), falling back
// to whatever's in the input HTML (AAPL) if nothing's been searched yet.
const savedTicker = localStorage.getItem('smartinvest_last_ticker');
if (savedTicker) tickerInput.value = savedTicker;
runAnalysis(tickerInput.value.trim().toUpperCase());