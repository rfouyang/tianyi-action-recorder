class WorkspaceTabs {
  constructor(root = document) {
    this.tabs = [...root.querySelectorAll("[data-panel-target]")];
    this.panels = [...root.querySelectorAll("[data-panel]")];
  }

  start() {
    for (const tab of this.tabs) {
      tab.addEventListener("click", () => this.activate(tab.dataset.panelTarget));
    }
    const requestedPanel = window.location.hash.replace("#", "");
    if (this.panels.some((panel) => panel.dataset.panel === requestedPanel)) {
      this.activate(requestedPanel, false);
    }
  }

  activate(panelName, updateLocation = true) {
    for (const tab of this.tabs) {
      const active = tab.dataset.panelTarget === panelName;
      tab.classList.toggle("tab-active", active);
      tab.setAttribute("aria-selected", String(active));
    }
    for (const panel of this.panels) {
      panel.classList.toggle("hidden", panel.dataset.panel !== panelName);
    }
    if (updateLocation) {
      window.history.replaceState(null, "", `#${panelName}`);
    }
    document.dispatchEvent(
      new CustomEvent("tianyi:panel-activated", { detail: { panelName } }),
    );
  }
}

class ViserViewer {
  constructor(root = document) {
    this.root = root;
    this.frame = root.querySelector("#viser-frame");
    this.status = root.querySelector("#viser-status");
    this.statusLabel = root.querySelector("#viser-status-label");
    this.cameraButtons = [...root.querySelectorAll("[data-camera-view]")];
    this.comparisonLegend = root.querySelector("#retarget-comparison-legend");
    this.transitionTimer = null;
    this.cameraRequest = 0;
    this.modeRequest = 0;
    this.comparisonVisible = false;
    this.hasRetargetPreview = false;
    this.activePanel = "pose-recorder";
  }

  start() {
    if (!this.frame) {
      return;
    }
    const port = this.frame.dataset.viserPort;
    this.frame.src = `${window.location.protocol}//${window.location.hostname}:${port}/`;
    this.frame.addEventListener("load", () => this.setConnectionState("ready"));
    this.frame.addEventListener("error", () => this.setConnectionState("error"));
    for (const button of this.cameraButtons) {
      button.addEventListener("click", () => this.selectCamera(button));
    }
    document.addEventListener("tianyi:panel-activated", (event) => {
      this.selectWorkspace(event.detail.panelName);
    });
    document.addEventListener("tianyi:retarget-preview-updated", () => {
      this.hasRetargetPreview = true;
      this.selectWorkspace(this.activePanel);
    });
    document.addEventListener("tianyi:retarget-preview-invalidated", () => {
      this.hasRetargetPreview = false;
      this.selectWorkspace(this.activePanel);
    });
    const activePanel = this.root.querySelector("[data-panel]:not(.hidden)");
    this.selectWorkspace(activePanel?.dataset.panel || "pose-recorder");
  }

  async selectWorkspace(panelName) {
    this.activePanel = panelName;
    const viewerMode = panelName === "pose-retargeting" && this.hasRetargetPreview
      ? "retarget-comparison"
      : "tianyi";
    const request = ++this.modeRequest;
    try {
      const response = await fetch(`/ui/tianyi-3d/viewer/mode/${viewerMode}`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Viewer mode request failed with ${response.status}`);
      }
      const result = await response.json();
      if (request !== this.modeRequest) {
        return;
      }
      this.setComparisonState(result.comparison_visible);
    } catch (error) {
      console.error(error);
      if (request === this.modeRequest) {
        this.setConnectionState("error");
      }
    }
  }

  setComparisonState(visible) {
    this.comparisonVisible = visible;
    this.comparisonLegend.classList.toggle("hidden", !visible);
    this.comparisonLegend.classList.toggle("grid", visible);
    if (visible) {
      this.statusLabel.textContent = "Live G1 ↔ Tianyi comparison";
    } else if (this.status.classList.contains("status-success")) {
      this.statusLabel.textContent = "Live 3D scene";
    }
  }

  async selectCamera(button) {
    const cameraView = button.dataset.cameraView;
    const request = ++this.cameraRequest;
    window.clearTimeout(this.transitionTimer);
    this.setConnectionState("pending");
    try {
      const response = await fetch(`/ui/tianyi-3d/viewer/camera/${cameraView}`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Camera request failed with ${response.status}`);
      }
      const result = await response.json();
      if (request !== this.cameraRequest) {
        return;
      }
      if (result.connected_clients < 1) {
        this.setConnectionState("waiting");
        return;
      }
      for (const candidate of this.cameraButtons) {
        const active = candidate === button;
        candidate.classList.toggle("btn-active", active);
        candidate.classList.toggle("btn-ghost", !active);
        candidate.classList.toggle("bg-neutral-content", active);
        candidate.classList.toggle("text-neutral", active);
        candidate.classList.toggle("text-neutral-content/70", !active);
        candidate.setAttribute("aria-pressed", String(active));
      }
      this.transitionTimer = window.setTimeout(
        () => this.setConnectionState("ready"),
        Number(result.transition_seconds || 0) * 1000,
      );
    } catch (error) {
      console.error(error);
      this.setConnectionState("error");
    }
  }

  setConnectionState(state) {
    const states = {
      ready: [
        "status-success",
        this.comparisonVisible ? "Live G1 ↔ Tianyi comparison" : "Live 3D scene",
      ],
      pending: ["status-info", "Changing camera"],
      waiting: ["status-warning", "Waiting for viewer"],
      error: ["status-error", "3D scene unavailable"],
    };
    const [statusClass, label] = states[state];
    this.status.classList.remove(
      "status-success",
      "status-info",
      "status-warning",
      "status-error",
    );
    this.status.classList.add(statusClass);
    this.statusLabel.textContent = label;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new ViserViewer().start();
  new WorkspaceTabs().start();
  new window.PoseRecorderPanel().start();
  new window.PoseRetargetingPanel().start();
  new window.PoseComposerPanel().start();
  new window.ActionComposerPanel().start();
});
