/**
 * API client for TinyFold backend
 */
const API = {
    baseUrl: '/api',

    /**
     * Get list of samples with optional filtering
     */
    async getSamples(options = {}) {
        const params = new URLSearchParams();
        if (options.split && options.split !== 'all') {
            params.append('split', options.split);
        }
        if (options.search) {
            params.append('search', options.search);
        }
        if (options.page) {
            params.append('page', options.page);
        }
        if (options.perPage) {
            params.append('per_page', options.perPage);
        }

        const response = await fetch(`${this.baseUrl}/samples?${params}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch samples: ${response.statusText}`);
        }
        return response.json();
    },

    /**
     * Get sample details including PDB
     */
    async getSample(sampleId) {
        const response = await fetch(`${this.baseUrl}/sample/${encodeURIComponent(sampleId)}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch sample: ${response.statusText}`);
        }
        return response.json();
    },

    /**
     * Get raw PDB string for a sample
     */
    async getSamplePDB(sampleId) {
        const response = await fetch(`${this.baseUrl}/sample/${encodeURIComponent(sampleId)}/pdb`);
        if (!response.ok) {
            throw new Error(`Failed to fetch PDB: ${response.statusText}`);
        }
        return response.text();
    },

    /**
     * Run prediction on a sample
     */
    async predict(sampleId) {
        const response = await fetch(`${this.baseUrl}/predict/${encodeURIComponent(sampleId)}`, {
            method: 'POST',
        });
        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.statusText}`);
        }
        return response.json();
    },
};
