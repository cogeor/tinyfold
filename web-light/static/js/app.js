class ShowcaseApp {
    constructor() {
        this.viewer = new ShowcaseViewer("viewer");
        this.data = [];
        this.filtered = [];
        this.selectedId = null;

        this.setupEvents();
        this.loadData();
    }

    setupEvents() {
        document.getElementById("split-filter").addEventListener("change", () => {
            this.applyFilter();
            this.renderList();
        });

        document.querySelectorAll("[data-style]").forEach((btn) => {
            btn.addEventListener("click", () => {
                document.querySelectorAll("[data-style]").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                this.viewer.setStyle(btn.dataset.style);
            });
        });

        document.getElementById("btn-reset").addEventListener("click", () => this.viewer.reset());
    }

    async loadData() {
        const res = await fetch("/assets/showcase_samples.json");
        const payload = await res.json();
        this.data = payload.samples || [];

        document.getElementById("meta").textContent =
            `${this.data.length} samples loaded`;

        this.applyFilter();
        this.renderList();
        if (this.filtered.length > 0) {
            this.select(this.filtered[0].sample_id);
        }
    }

    applyFilter() {
        const split = document.getElementById("split-filter").value;
        this.filtered = split === "all" ? this.data : this.data.filter((s) => s.split === split);
    }

    renderList() {
        const list = document.getElementById("sample-list");
        list.innerHTML = this.filtered
            .map((s) => {
                const selected = s.sample_id === this.selectedId ? "selected" : "";
                return `
                    <div class="sample ${selected}" data-id="${s.sample_id}">
                        <div class="top">${s.sample_id}</div>
                        <div class="bottom">${s.split} | RMSD ${s.rmsd.toFixed(2)} A | ${s.n_residues} residues</div>
                    </div>
                `;
            })
            .join("");

        list.querySelectorAll(".sample").forEach((el) => {
            el.addEventListener("click", () => this.select(el.dataset.id));
        });
    }

    select(sampleId) {
        const sample = this.filtered.find((s) => s.sample_id === sampleId) || this.data.find((s) => s.sample_id === sampleId);
        if (!sample) return;

        this.selectedId = sample.sample_id;
        this.renderList();
        this.viewer.load(sample.ground_truth_pdb, sample.prediction_pdb);
        document.getElementById("info").textContent =
            `${sample.sample_id} | split=${sample.split} | RMSD=${sample.rmsd.toFixed(2)} A | inference=${sample.inference_time.toFixed(3)} s`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.showcaseApp = new ShowcaseApp();
});
