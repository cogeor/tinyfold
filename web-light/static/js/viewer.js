class ShowcaseViewer {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.viewer = $3Dmol.createViewer(this.element, { backgroundColor: "white" });
        this.models = { gt: null, pred: null };
        this.style = "cartoon";
    }

    load(gtPdb, predPdb) {
        this.clear();
        this.models.gt = this.viewer.addModel(gtPdb, "pdb");
        this.models.pred = this.viewer.addModel(predPdb, "pdb");
        this.applyStyle();
        this.viewer.zoomTo();
        this.viewer.render();
    }

    applyStyle() {
        if (this.models.gt) {
            this.models.gt.setStyle({}, this._styleSpec(false));
        }
        if (this.models.pred) {
            this.models.pred.setStyle({}, this._styleSpec(true));
        }
        this.viewer.render();
    }

    setStyle(style) {
        this.style = style;
        this.applyStyle();
    }

    reset() {
        this.viewer.zoomTo();
        this.viewer.render();
    }

    clear() {
        this.viewer.removeAllModels();
        this.models = { gt: null, pred: null };
    }

    _styleSpec(isPrediction) {
        const colorfunc = (atom) => {
            if (isPrediction) {
                return atom.chain === "A" ? "#d94841" : "#f08c00";
            }
            return atom.chain === "A" ? "#1c7ed6" : "#2f9e44";
        };

        switch (this.style) {
            case "stick":
                return { stick: { radius: 0.15, colorfunc } };
            case "sphere":
                return { sphere: { radius: 0.45, colorfunc } };
            case "line":
                return { line: { colorfunc } };
            case "cartoon":
            default:
                return { cartoon: { colorfunc } };
        }
    }
}

