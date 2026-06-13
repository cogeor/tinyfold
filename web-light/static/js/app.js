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
        try {
            const res = await fetch("/assets/showcase_samples.json");
            if (!res.ok) throw new Error(`HTTP ${res.status} fetching showcase data`);
            const payload = await res.json();
            this.data = payload.samples || [];
            if (this.data.length === 0) throw new Error("showcase_samples.json has no samples");

            document.getElementById("meta").textContent =
                `${this.data.length} held-out test complexes${payload.model ? ` · ${payload.model}` : ""}`;

            this.applyFilter();
            this.renderList();
            this.select(this.filtered[0].sample_id);
        } catch (err) {
            document.getElementById("meta").textContent = "";
            document.getElementById("info").textContent =
                `Could not load showcase data: ${err.message}. ` +
                `Regenerate with scripts/web/build_showcase.py, then reload.`;
            console.error("loadData failed:", err);
        }
    }

    applyFilter() {
        this.filtered = this.data;
    }

    renderList() {
        const list = document.getElementById("sample-list");
        list.innerHTML = this.filtered
            .map((s) => {
                const selected = s.sample_id === this.selectedId ? "selected" : "";
                const dockq = s.dockq != null ? s.dockq.toFixed(3) : "—";
                const band = s.capri || "";
                return `
                    <div class="sample ${selected}" data-id="${s.sample_id}">
                        <div class="top">${s.sample_id} <span class="badge ${band}">${band}</span></div>
                        <div class="bottom">DockQ ${dockq} | C-RMSD ${s.c_rmsd != null ? s.c_rmsd.toFixed(1) : "—"} Å | ${s.n_residues} res</div>
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
        const dq = sample.dockq != null ? sample.dockq.toFixed(3) : "—";
        const cr = sample.c_rmsd != null ? `${sample.c_rmsd.toFixed(2)} Å` : "—";
        document.getElementById("info").innerHTML =
            `<b>${sample.sample_id}</b> · held-out test · ` +
            `<b>DockQ ${dq}</b> (${sample.capri || "—"}) · C-RMSD ${cr} · ` +
            `${sample.n_residues} residues · ${sample.inference_time.toFixed(3)}s/sample &nbsp; ` +
            `<span class="legend"><span class="sw gt"></span>ground truth ` +
            `<span class="sw pred"></span>prediction</span>`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.showcaseApp = new ShowcaseApp();
});
