// Modern Glassmorphic App Client for Training Data Factory

document.addEventListener('DOMContentLoaded', () => {
  // State
  let selectedFiles = [];
  let currentJobId = null;
  let pollInterval = null;
  let currentDatasetId = null;

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const fileList = document.getElementById('file-list');
  const btnProcess = document.getElementById('btn-process');

  const progressCard = document.getElementById('progress-card');
  const progressBar = document.getElementById('progress-bar');
  const progressTitle = document.getElementById('progress-title');
  const progressStatusBadge = document.getElementById('progress-status-badge');
  const progressDetail = document.getElementById('progress-detail');

  const resultsCard = document.getElementById('results-card');
  const statExamples = document.getElementById('stat-examples');
  const statQuality = document.getElementById('stat-quality');
  const statStatus = document.getElementById('stat-status');
  const previewTableBody = document.getElementById('preview-table-body');

  const btnDownloadJsonl = document.getElementById('btn-download-jsonl');
  const btnDownloadCsv = document.getElementById('btn-download-csv');
  const btnDownloadSplit = document.getElementById('btn-download-split');

  const datasetsGrid = document.getElementById('datasets-grid');
  const jobsList = document.getElementById('jobs-list');
  const btnRefreshDatasets = document.getElementById('btn-refresh-datasets');
  const btnRefreshJobs = document.getElementById('btn-refresh-jobs');

  // Status Badges
  const badgeStatus = document.getElementById('badge-status');
  const badgeProvider = document.getElementById('badge-provider');
  const badgeVersion = document.getElementById('badge-version');
  const datasetsCount = document.getElementById('datasets-count');
  const jobsCount = document.getElementById('jobs-count');

  // Initialize
  fetchStatus();
  fetchDatasets();
  fetchJobs();

  // --- Tabs Navigation ---
  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-tab');
      tabBtns.forEach((b) => b.classList.remove('active'));
      tabPanes.forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(target).classList.add('active');

      if (target === 'tab-datasets') fetchDatasets();
      if (target === 'tab-jobs') fetchJobs();
    });
  });

  // --- Pill Option Groups ---
  document.querySelectorAll('.pill-options').forEach((group) => {
    group.addEventListener('click', (e) => {
      const option = e.target.closest('.pill-option');
      if (!option) return;

      const input = option.querySelector('input');
      if (!input) return;

      if (input.type === 'radio') {
        group.querySelectorAll('.pill-option').forEach((opt) => opt.classList.remove('active'));
        option.classList.add('active');
        input.checked = true;
      } else if (input.type === 'checkbox') {
        input.checked = !input.checked;
        option.classList.toggle('active', input.checked);
      }
    });
  });

  // --- File Dropzone ---
  dropzone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    addFiles(Array.from(e.target.files));
    fileInput.value = '';
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  });

  function addFiles(newFiles) {
    newFiles.forEach((file) => {
      // Prevent duplicates by filename and size
      if (!selectedFiles.some((f) => f.name === file.name && f.size === file.size)) {
        selectedFiles.push(file);
      }
    });
    renderFileList();
  }

  function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
  }

  function renderFileList() {
    fileList.innerHTML = '';
    if (selectedFiles.length === 0) {
      return;
    }

    selectedFiles.forEach((file, index) => {
      const item = document.createElement('div');
      item.className = 'file-item';

      const sizeStr = (file.size / 1024).toFixed(1) + ' KB';
      item.innerHTML = `
        <div class="file-info">
          <span style="color: var(--text-3);">&#9679;</span>
          <strong>${escapeHtml(file.name)}</strong>
          <span style="color: var(--text-muted); font-size: 11px;">(${sizeStr})</span>
        </div>
        <button type="button" class="file-remove-btn" title="Remove">&times;</button>
      `;

      item.querySelector('.file-remove-btn').addEventListener('click', () => removeFile(index));
      fileList.appendChild(item);
    });
  }

  // --- Status & Statistics ---
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();
      badgeVersion.textContent = `v${data.version}`;
      badgeProvider.textContent = `Provider: ${data.default_provider.toUpperCase()}`;
      datasetsCount.textContent = data.persisted_datasets;
      jobsCount.textContent = data.persisted_jobs;
    } catch (err) {
      console.warn('Could not fetch status:', err);
    }
  }

  // --- Processing Pipeline Execution ---
  btnProcess.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
      alert('Please upload at least one document to process.');
      return;
    }

    // Selected Tasks
    const checkedTasks = Array.from(document.querySelectorAll('#tasks-group input:checked')).map(
      (i) => i.value
    );
    if (checkedTasks.length === 0) {
      alert('Please select at least one task to generate.');
      return;
    }

    // Selected Provider
    const provider = document.querySelector('#provider-group input:checked').value;
    const qualityFilter = document.getElementById('toggle-quality').checked;
    const split = document.getElementById('toggle-split').checked;
    const format = document.querySelector('#format-group input:checked').value;

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append('files', file));
    formData.append('tasks', checkedTasks.join(','));
    formData.append('provider', provider);
    formData.append('quality_filter', qualityFilter);
    formData.append('split', split);
    formData.append('format', format);

    // Prepare UI
    btnProcess.disabled = true;
    btnProcess.innerHTML = `<span class="spinner"></span> Processing...`;
    resultsCard.classList.remove('visible');
    progressCard.classList.add('visible');
    progressBar.style.width = '15%';
    progressStatusBadge.textContent = 'Queued';
    progressStatusBadge.className = 'badge';
    progressDetail.textContent = 'Submitting uploads to pipeline...';
    resetStagePills();

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Processing request failed');
      }

      const data = await res.json();
      currentJobId = data.job_id;
      startJobPolling(currentJobId);
    } catch (err) {
      alert(`Error starting pipeline: ${err.message}`);
      resetProcessButton();
      progressCard.classList.remove('visible');
    }
  });

  function startJobPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    updateStages('stage-upload');
    progressBar.style.width = '30%';

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        progressStatusBadge.textContent = job.status.toUpperCase();
        if (job.status === 'processing') {
          progressStatusBadge.className = 'badge';
          progressBar.style.width = '65%';
          updateStages('stage-ai');
          progressDetail.textContent = `Running task templates and generating examples...`;
        } else if (job.status === 'completed' || job.status === 'partial') {
          clearInterval(pollInterval);
          progressBar.style.width = '100%';
          completeAllStages();
          progressStatusBadge.className = `badge status-tag ${job.status === 'completed' ? 'status-completed' : 'status-partial'}`;
          progressDetail.textContent = `Processing completed. Loading generated dataset...`;

          setTimeout(() => {
            resetProcessButton();
            if (job.dataset_id) {
              loadDatasetResults(job.dataset_id, job.status);
            }
            fetchStatus();
            fetchDatasets();
            fetchJobs();
          }, 600);
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          progressBar.style.width = '100%';
          progressBar.style.backgroundColor = 'var(--accent-red)';
          progressStatusBadge.className = 'badge status-tag status-failed';
          progressDetail.textContent = `Failure: ${job.error_message || 'Pipeline execution failed'}`;
          resetProcessButton();
          fetchJobs();
        }
      } catch (err) {
        console.warn('Polling error:', err);
      }
    }, 800);
  }

  function resetProcessButton() {
    btnProcess.disabled = false;
    btnProcess.innerHTML = `<span>Run pipeline</span>`;
  }

  function resetStagePills() {
    document.querySelectorAll('.stage-pill').forEach((pill) => {
      pill.className = 'stage-pill';
    });
    document.getElementById('stage-upload').classList.add('active');
  }

  function updateStages(activeId) {
    const stageIds = ['stage-upload', 'stage-chunk', 'stage-ai', 'stage-eval', 'stage-export'];
    let passedActive = false;
    stageIds.forEach((id) => {
      const el = document.getElementById(id);
      if (id === activeId) {
        el.className = 'stage-pill active';
        passedActive = true;
      } else if (!passedActive) {
        el.className = 'stage-pill done';
      } else {
        el.className = 'stage-pill';
      }
    });
  }

  function completeAllStages() {
    document.querySelectorAll('.stage-pill').forEach((pill) => {
      pill.className = 'stage-pill done';
    });
  }

  // --- Load Dataset Results & Preview ---
  async function loadDatasetResults(datasetId, outcomeStatus) {
    currentDatasetId = datasetId;
    try {
      const res = await fetch(`/api/datasets/${datasetId}`);
      if (!res.ok) return;
      const ds = await res.json();

      statExamples.textContent = ds.example_count;
      statStatus.textContent = outcomeStatus === 'completed' ? 'Success' : 'Partial';
      statStatus.style.color = outcomeStatus === 'completed' ? 'var(--accent-green)' : 'var(--accent-amber)';

      // Preview Table
      previewTableBody.innerHTML = '';
      if (ds.examples && ds.examples.length > 0) {
        ds.examples.slice(0, 15).forEach((ex) => {
          const row = document.createElement('tr');
          const taskLabel = (ex.task_type || 'qa').toUpperCase();
          const qualityPassed = ex.quality_approved !== false;
          const score = ex.quality_scores?.overall != null
            ? (ex.quality_scores.overall * 100).toFixed(0) + '%'
            : 'Pass';

          row.innerHTML = `
            <td><span class="badge">${taskLabel}</span></td>
            <td><div class="snippet-cell">${escapeHtml(ex.input_text)}</div></td>
            <td><div class="snippet-cell">${escapeHtml(ex.output_text)}</div></td>
            <td>
              <span class="status-tag ${qualityPassed ? 'status-completed' : 'status-failed'}">
                ${score}
              </span>
            </td>
          `;
          previewTableBody.appendChild(row);
        });
      } else {
        previewTableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 24px;">No examples generated.</td></tr>`;
      }

      // Download Buttons
      btnDownloadJsonl.onclick = () => downloadDataset(datasetId, 'jsonl', false);
      btnDownloadCsv.onclick = () => downloadDataset(datasetId, 'csv', false);
      btnDownloadSplit.onclick = () => downloadDataset(datasetId, 'jsonl', true);

      resultsCard.classList.add('visible');
    } catch (err) {
      console.error('Failed to load dataset preview:', err);
    }
  }

  function downloadDataset(datasetId, format, split) {
    const url = `/api/datasets/${datasetId}/export?format=${format}&split=${split}`;
    window.location.href = url;
  }

  // --- Saved Datasets Tab ---
  async function fetchDatasets() {
    try {
      const res = await fetch('/api/datasets');
      if (!res.ok) return;
      const datasets = await res.json();
      datasetsCount.textContent = datasets.length;
      datasetsGrid.innerHTML = '';

      if (datasets.length === 0) {
        datasetsGrid.innerHTML = `
          <div class="glass-card" style="grid-column: 1/-1; text-align: center; padding: 40px;">
            <p style="color: var(--text-secondary); margin-bottom: 12px;">No persisted datasets found yet.</p>
            <p style="font-size: 13px; color: var(--text-muted);">Use the Process tab to ingest files and generate datasets.</p>
          </div>
        `;
        return;
      }

      datasets.forEach((ds) => {
        const card = document.createElement('div');
        card.className = 'glass-card dataset-card';
        const dateStr = new Date(ds.created_at).toLocaleString();

        card.innerHTML = `
          <div>
            <h3>${escapeHtml(ds.name)}</h3>
            <p>${escapeHtml(ds.description || 'Curated dataset')}</p>
            <div class="dataset-meta">
              <span><strong>${ds.example_count}</strong> examples</span>
              <span>${dateStr}</span>
            </div>
            <div style="font-family: monospace; font-size: 11px; color: var(--text-muted); margin-bottom: 14px;">
              ID: ${ds.id}
            </div>
          </div>
          <div class="actions-bar" style="margin-top: auto;">
            <button class="btn-secondary btn-ds-jsonl" title="Download JSONL">JSONL</button>
            <button class="btn-secondary btn-ds-csv" title="Download CSV">CSV</button>
            <button class="btn-secondary btn-ds-split" title="Download Train/Val/Test Split Zip">Split Zip</button>
          </div>
        `;

        card.querySelector('.btn-ds-jsonl').addEventListener('click', () => downloadDataset(ds.id, 'jsonl', false));
        card.querySelector('.btn-ds-csv').addEventListener('click', () => downloadDataset(ds.id, 'csv', false));
        card.querySelector('.btn-ds-split').addEventListener('click', () => downloadDataset(ds.id, 'jsonl', true));

        datasetsGrid.appendChild(card);
      });
    } catch (err) {
      console.warn('Failed to load datasets:', err);
    }
  }

  btnRefreshDatasets.addEventListener('click', fetchDatasets);

  // --- Job History Tab ---
  async function fetchJobs() {
    try {
      const res = await fetch('/api/jobs');
      if (!res.ok) return;
      const jobs = await res.json();
      jobsCount.textContent = jobs.length;
      jobsList.innerHTML = '';

      if (jobs.length === 0) {
        jobsList.innerHTML = `<p style="text-align: center; color: var(--text-muted); padding: 20px;">No jobs recorded yet.</p>`;
        return;
      }

      jobs.forEach((job) => {
        const row = document.createElement('div');
        row.className = 'job-row glass-card';
        const dateStr = new Date(job.created_at).toLocaleString();

        let statusClass = 'status-processing';
        if (job.status === 'completed') statusClass = 'status-completed';
        else if (job.status === 'partial') statusClass = 'status-partial';
        else if (job.status === 'failed') statusClass = 'status-failed';

        const errorHtml = job.error_message
          ? `<div style="width: 100%; color: var(--accent-red); font-size: 12px; margin-top: 6px;">Error: ${escapeHtml(job.error_message)}</div>`
          : '';

        row.innerHTML = `
          <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
              <strong>${escapeHtml(job.name)}</strong>
              <span class="status-tag ${statusClass}">${escapeHtml(job.status)}</span>
            </div>
            <div style="font-size: 12px; color: var(--text-muted);">
              ${dateStr} &bull; ID: <code>${job.id}</code>
            </div>
            ${errorHtml}
          </div>
          <div style="text-align: right; font-size: 12px; color: var(--text-secondary);">
            ${job.output_data?.examples_generated ? `<span><strong>${job.output_data.examples_generated}</strong> examples generated</span>` : ''}
          </div>
        `;
        jobsList.appendChild(row);
      });
    } catch (err) {
      console.warn('Failed to load jobs:', err);
    }
  }

  btnRefreshJobs.addEventListener('click', fetchJobs);

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
