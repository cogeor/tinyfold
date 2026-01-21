/**
 * 3Dmol.js wrapper for protein visualization
 */
class ProteinViewer {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.viewer = $3Dmol.createViewer(this.element, {
            backgroundColor: 'white',
        });
        this.models = {
            groundTruth: null,
            prediction: null,
        };
        this.style = 'cartoon';
        this.colorScheme = 'chain';
        this.showGT = true;
        this.showPred = true;
    }

    /**
     * Load ground truth structure from PDB string
     */
    loadGroundTruth(pdbString) {
        if (this.models.groundTruth !== null) {
            this.viewer.removeModel(this.models.groundTruth);
        }
        this.models.groundTruth = this.viewer.addModel(pdbString, 'pdb');
        this.applyStyle('groundTruth');
        this.viewer.zoomTo();
        this.viewer.render();
    }

    /**
     * Load prediction structure from PDB string
     */
    loadPrediction(pdbString) {
        if (this.models.prediction !== null) {
            this.viewer.removeModel(this.models.prediction);
        }
        this.models.prediction = this.viewer.addModel(pdbString, 'pdb');
        this.applyStyle('prediction');
        this.viewer.render();
    }

    /**
     * Clear prediction model
     */
    clearPrediction() {
        if (this.models.prediction !== null) {
            this.viewer.removeModel(this.models.prediction);
            this.models.prediction = null;
            this.viewer.render();
        }
    }

    /**
     * Apply visual style to a model
     */
    applyStyle(modelKey) {
        const model = this.models[modelKey];
        if (!model) return;

        // Check visibility
        const isVisible = modelKey === 'groundTruth' ? this.showGT : this.showPred;
        if (!isVisible) {
            model.setStyle({}, {});
            return;
        }

        const styleSpec = this.getStyleSpec(modelKey);
        model.setStyle({}, styleSpec);
    }

    /**
     * Get style specification for a model
     */
    getStyleSpec(modelKey) {
        const colorSpec = this.getColorSpec(modelKey);

        switch (this.style) {
            case 'cartoon':
                return { cartoon: colorSpec };
            case 'stick':
                return { stick: { radius: 0.15, ...colorSpec } };
            case 'sphere':
                return { sphere: { radius: 0.4, ...colorSpec } };
            case 'line':
                return { line: colorSpec };
            default:
                return { cartoon: colorSpec };
        }
    }

    /**
     * Get color specification based on current scheme
     */
    getColorSpec(modelKey) {
        const isPred = modelKey === 'prediction';

        switch (this.colorScheme) {
            case 'chain':
                // Ground truth: blue (A) / green (B)
                // Prediction: red (A) / orange (B)
                if (isPred) {
                    return {
                        colorfunc: (atom) => atom.chain === 'A' ? '#e74c3c' : '#f39c12'
                    };
                } else {
                    return {
                        colorfunc: (atom) => atom.chain === 'A' ? '#3498db' : '#2ecc71'
                    };
                }
            case 'spectrum':
                return { color: 'spectrum' };
            case 'ss':
                return { color: 'ss' };
            default:
                return {};
        }
    }

    /**
     * Set rendering style for all models
     */
    setStyle(style) {
        this.style = style;
        this.applyAllStyles();
    }

    /**
     * Set color scheme
     */
    setColorScheme(scheme) {
        this.colorScheme = scheme;
        this.applyAllStyles();
    }

    /**
     * Apply styles to all models and render
     */
    applyAllStyles() {
        Object.keys(this.models).forEach(key => this.applyStyle(key));
        this.viewer.render();
    }

    /**
     * Toggle ground truth visibility
     */
    setGroundTruthVisible(visible) {
        this.showGT = visible;
        this.applyStyle('groundTruth');
        this.viewer.render();
    }

    /**
     * Toggle prediction visibility
     */
    setPredictionVisible(visible) {
        this.showPred = visible;
        this.applyStyle('prediction');
        this.viewer.render();
    }

    /**
     * Clear all models
     */
    clear() {
        this.viewer.removeAllModels();
        this.models = { groundTruth: null, prediction: null };
        this.viewer.render();
    }

    /**
     * Reset view to fit all
     */
    resetView() {
        this.viewer.zoomTo();
        this.viewer.render();
    }

    /**
     * Check if prediction is loaded
     */
    hasPrediction() {
        return this.models.prediction !== null;
    }
}
