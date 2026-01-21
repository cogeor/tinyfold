/**
 * Main application logic for TinyFold Visualizer
 */
class App {
    constructor() {
        this.viewer = new ProteinViewer('viewer');
        this.currentSample = null;
        this.currentPage = 1;
        this.totalSamples = 0;
        this.perPage = 50;

        // Detect mode from URL params or global config
        this.mode = this.detectMode();
        this.applyMode();

        if (this.mode === 'full') {
            this.setupEventListeners();
            this.loadSamples();
        } else {
            this.setupEmbedListeners();
            this.setupPostMessageAPI();
        }
    }

    /**
     * Detect operating mode: 'full' or 'embed'
     */
    detectMode() {
        // Check URL params first
        const params = new URLSearchParams(window.location.search);
        if (params.get('mode') === 'embed' || params.get('mode') === 'light') {
            return 'embed';
        }
        // Check global config (can be injected by server)
        if (window.TINYFOLD_MODE === 'embed' || window.TINYFOLD_MODE === 'light') {
            return 'embed';
        }
        return 'full';
    }

    /**
     * Apply mode-specific CSS classes
     */
    applyMode() {
        const container = document.getElementById('app-container');
        if (this.mode === 'embed') {
            container.classList.add('embed-mode');
            // Check for minimal mode (no controls at all)
            const params = new URLSearchParams(window.location.search);
            if (params.get('minimal') === 'true') {
                container.classList.add('minimal');
            }
        }
    }

    /**
     * Setup minimal event listeners for embed mode
     */
    setupEmbedListeners() {
        // Style buttons
        document.querySelectorAll('[data-style]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-style]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.viewer.setStyle(btn.dataset.style);
            });
        });

        // Color scheme
        document.getElementById('color-scheme').addEventListener('change', (e) => {
            this.viewer.setColorScheme(e.target.value);
        });

        // Visibility toggles
        document.getElementById('show-gt').addEventListener('change', (e) => {
            this.viewer.setGroundTruthVisible(e.target.checked);
        });

        document.getElementById('show-pred').addEventListener('change', (e) => {
            this.viewer.setPredictionVisible(e.target.checked);
        });
    }

    /**
     * Setup postMessage API for external communication
     */
    setupPostMessageAPI() {
        window.addEventListener('message', (event) => {
            const data = event.data;
            if (!data || !data.type) return;

            switch (data.type) {
                case 'load':
                    this.loadFromData(data);
                    break;
                case 'clear':
                    this.viewer.clear();
                    break;
                case 'setStyle':
                    this.viewer.setStyle(data.style);
                    break;
                case 'setColorScheme':
                    this.viewer.setColorScheme(data.scheme);
                    break;
                case 'resetView':
                    this.viewer.resetView();
                    break;
            }
        });

        // Also expose API directly on window for same-origin usage
        window.tinyfold = {
            load: (data) => this.loadFromData(data),
            clear: () => this.viewer.clear(),
            setStyle: (style) => this.viewer.setStyle(style),
            setColorScheme: (scheme) => this.viewer.setColorScheme(scheme),
            resetView: () => this.viewer.resetView(),
            viewer: this.viewer,
        };
    }

    /**
     * Load structures from coordinate data
     * @param {Object} data - { groundTruth: coords, prediction: coords, atomTypes: [...], sequence: [...] }
     * coords format: [[x,y,z], ...] - Nx3 array
     * atomTypes: ['N', 'CA', 'C', 'O', ...] - optional, defaults to backbone pattern
     * sequence: ['ALA', 'GLY', ...] - optional, for residue names
     */
    loadFromData(data) {
        if (data.groundTruth) {
            const pdb = this.coordsToPDB(data.groundTruth, data.atomTypes, data.sequence, 'A');
            this.viewer.loadGroundTruth(pdb);
            document.getElementById('show-gt').checked = true;
        }

        if (data.prediction) {
            const pdb = this.coordsToPDB(data.prediction, data.atomTypes, data.sequence, 'A');
            this.viewer.loadPrediction(pdb);
            document.getElementById('show-pred').disabled = false;
            document.getElementById('show-pred').checked = true;
        }
    }

    /**
     * Convert coordinate array to PDB string
     */
    coordsToPDB(coords, atomTypes, sequence, chainId = 'A') {
        const defaultAtoms = ['N', 'CA', 'C', 'O'];
        const atoms = atomTypes || defaultAtoms;
        const atomsPerResidue = atoms.length === coords.length ? 1 :
            (atoms.length <= 4 ? 4 : atoms.length);

        let pdb = '';
        let atomNum = 1;
        let resNum = 1;

        for (let i = 0; i < coords.length; i++) {
            const [x, y, z] = coords[i];
            const atomIdx = i % atomsPerResidue;
            const atomName = atoms[atomIdx % atoms.length];
            const resName = sequence ? sequence[Math.floor(i / atomsPerResidue)] : 'ALA';

            // PDB ATOM format
            const line = 'ATOM  ' +
                String(atomNum).padStart(5) + ' ' +
                atomName.padEnd(4) + ' ' +
                resName.padStart(3) + ' ' +
                chainId +
                String(resNum).padStart(4) + '    ' +
                x.toFixed(3).padStart(8) +
                y.toFixed(3).padStart(8) +
                z.toFixed(3).padStart(8) +
                '  1.00  0.00           ' +
                atomName[0] + '\n';

            pdb += line;
            atomNum++;

            // Increment residue number after each set of atoms
            if ((i + 1) % atomsPerResidue === 0) {
                resNum++;
            }
        }

        pdb += 'END\n';
        return pdb;
    }

    setupEventListeners() {
        // Search input with debounce
        const searchInput = document.getElementById('search');
        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.currentPage = 1;
                this.loadSamples();
            }, 300);
        });

        // Split filter
        document.getElementById('split-filter').addEventListener('change', () => {
            this.currentPage = 1;
            this.loadSamples();
        });

        // Style buttons
        document.querySelectorAll('[data-style]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-style]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.viewer.setStyle(btn.dataset.style);
            });
        });

        // Color scheme
        document.getElementById('color-scheme').addEventListener('change', (e) => {
            this.viewer.setColorScheme(e.target.value);
        });

        // Visibility toggles
        document.getElementById('show-gt').addEventListener('change', (e) => {
            this.viewer.setGroundTruthVisible(e.target.checked);
        });

        document.getElementById('show-pred').addEventListener('change', (e) => {
            this.viewer.setPredictionVisible(e.target.checked);
        });

        // Predict button
        document.getElementById('btn-predict').addEventListener('click', () => {
            this.runPrediction();
        });

        // Reset view button
        document.getElementById('btn-reset').addEventListener('click', () => {
            this.viewer.resetView();
        });

        // Pagination
        document.getElementById('prev-page').addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.loadSamples();
            }
        });

        document.getElementById('next-page').addEventListener('click', () => {
            const maxPage = Math.ceil(this.totalSamples / this.perPage);
            if (this.currentPage < maxPage) {
                this.currentPage++;
                this.loadSamples();
            }
        });
    }

    async loadSamples() {
        const search = document.getElementById('search').value;
        const split = document.getElementById('split-filter').value;

        try {
            const data = await API.getSamples({
                search: search || undefined,
                split: split,
                page: this.currentPage,
                perPage: this.perPage,
            });

            this.totalSamples = data.total;
            this.renderSampleList(data.samples);
            this.updatePagination();

        } catch (err) {
            console.error('Failed to load samples:', err);
            document.getElementById('sample-list').innerHTML =
                '<div class="sample-item" style="color: #e94560;">Error loading samples</div>';
        }
    }

    renderSampleList(samples) {
        const container = document.getElementById('sample-list');

        if (samples.length === 0) {
            container.innerHTML = '<div class="sample-item" style="color: #888;">No samples found</div>';
            document.getElementById('sample-count').textContent = '0 samples';
            return;
        }

        container.innerHTML = samples.map(s => `
            <div class="sample-item" data-id="${s.sample_id}" data-has-pred="${s.has_prediction}">
                <span class="sample-id">${s.sample_id}</span>
                <span class="sample-meta">
                    ${s.has_prediction ? '<span class="pred-indicator" title="Prediction cached">P</span>' : ''}
                    <span class="atoms">${s.n_atoms} atoms</span>
                    <span class="split ${s.split}">${s.split}</span>
                </span>
            </div>
        `).join('');

        document.getElementById('sample-count').textContent = `${this.totalSamples} samples`;

        // Add click handlers
        container.querySelectorAll('.sample-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectSample(item.dataset.id);
            });
        });

        // Highlight current selection if still in list
        if (this.currentSample) {
            const current = container.querySelector(`[data-id="${this.currentSample.sample_id}"]`);
            if (current) {
                current.classList.add('selected');
            }
        }
    }

    updatePagination() {
        const maxPage = Math.ceil(this.totalSamples / this.perPage) || 1;

        document.getElementById('prev-page').disabled = this.currentPage <= 1;
        document.getElementById('next-page').disabled = this.currentPage >= maxPage;
        document.getElementById('page-info').textContent = `Page ${this.currentPage} / ${maxPage}`;
    }

    async selectSample(sampleId) {
        // Update UI selection
        document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
        const selected = document.querySelector(`[data-id="${sampleId}"]`);
        if (selected) {
            selected.classList.add('selected');
        }

        // Show loading state
        document.getElementById('sample-title').textContent = 'Loading...';
        document.getElementById('sample-info').innerHTML = '';
        document.getElementById('pred-info').innerHTML = '';

        try {
            // Fetch sample
            const sample = await API.getSample(sampleId);
            this.currentSample = sample;

            // Update info panel
            document.getElementById('sample-title').textContent = sampleId;
            document.getElementById('sample-info').innerHTML = `
                <span><span class="label">Atoms:</span> ${sample.n_atoms}</span>
                <span><span class="label">Residues:</span> ${sample.n_residues}</span>
                <span><span class="label">Split:</span> ${sample.split}</span>
            `;

            // Clear previous prediction
            this.viewer.clearPrediction();
            document.getElementById('show-pred').checked = false;
            document.getElementById('show-pred').disabled = true;
            document.getElementById('pred-info').innerHTML = '';

            // Load structure
            this.viewer.loadGroundTruth(sample.pdb_string);

            // Enable predict button
            document.getElementById('btn-predict').disabled = false;

            // Update button text based on cache status
            const hasCachedPred = document.querySelector(`[data-id="${sampleId}"]`)?.dataset.hasPred === 'true';
            document.getElementById('btn-predict').textContent = hasCachedPred ? 'Load Prediction' : 'Run Prediction';

        } catch (err) {
            console.error('Failed to load sample:', err);
            document.getElementById('sample-title').textContent = 'Error';
            document.getElementById('sample-info').innerHTML = `<span style="color: #e94560;">${err.message}</span>`;
        }
    }

    async runPrediction() {
        if (!this.currentSample) {
            return;
        }

        const btn = document.getElementById('btn-predict');
        btn.disabled = true;
        btn.textContent = 'Running...';
        document.getElementById('pred-info').innerHTML = '<span class="time">Running inference...</span>';

        try {
            const result = await API.predict(this.currentSample.sample_id);

            // Load prediction
            this.viewer.loadPrediction(result.pdb_string);

            // Enable prediction checkbox
            document.getElementById('show-pred').disabled = false;
            document.getElementById('show-pred').checked = true;

            // Show results
            const rmsdClass = result.rmsd > 5 ? 'bad' : '';
            const cachedTag = result.cached ? '<span class="cached-tag">cached</span>' : '';
            document.getElementById('pred-info').innerHTML = `
                <span class="rmsd ${rmsdClass}">RMSD: ${result.rmsd.toFixed(2)} \u00C5</span>
                <span class="time">${result.inference_time_ms.toFixed(0)} ms</span>
                ${cachedTag}
            `;

        } catch (err) {
            console.error('Prediction failed:', err);
            document.getElementById('pred-info').innerHTML =
                `<span style="color: #e94560;">Prediction failed: ${err.message}</span>`;
        } finally {
            btn.disabled = false;
            // Restore appropriate button text
            const hasCachedPred = document.querySelector(`[data-id="${this.currentSample?.sample_id}"]`)?.dataset.hasPred === 'true';
            btn.textContent = hasCachedPred ? 'Load Prediction' : 'Run Prediction';
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
