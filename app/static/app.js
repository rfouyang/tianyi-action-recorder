const ui = {
  connectionBadge: document.querySelector("#connection-badge"),
  revisionBadge: document.querySelector("#revision-badge"),
  robotImage: document.querySelector("#robot-image"),
  renderLoading: document.querySelector("#render-loading"),
  renderMetric: document.querySelector("#render-metric"),
  cameraDescription: document.querySelector("#camera-description"),
  jointControls: document.querySelector("#joint-controls"),
  resetButton: document.querySelector("#reset-button"),
  lockedDescription: document.querySelector("#locked-description"),
  statusMessage: document.querySelector("#status-message"),
  statusText: document.querySelector("#status-text"),
  modelId: document.querySelector("#model-id"),
  meshSummary: document.querySelector("#mesh-summary"),
};

const cameraLabels = {
  front: "Front",
  front_left: "Front-left",
  left: "Left",
  right: "Right",
  front_right: "Front-right",
  back: "Back",
};

let modelContract = null;
let simulationState = null;
let selectedGroup = "head";
let selectedCamera = "front";
let renderRequest = 0;
let displayedImageUrl = null;
let updateTimer = null;
let updateInFlight = false;
let pendingJointUpdates = {};

function readableName(name) {
  return name
    .replace(/_joint$/, "")
    .replace(/_l$/, " left")
    .replace(/_r$/, " right")
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

function showStatus(message, kind = "info") {
  ui.statusMessage.className = `alert alert-${kind} alert-soft`;
  ui.statusText.textContent = message;
}

function hideStatus() {
  ui.statusMessage.className = "alert alert-soft hidden";
  ui.statusText.textContent = "";
}

function syncStateIndicators() {
  ui.revisionBadge.textContent = `Revision ${simulationState.revision}`;
  ui.connectionBadge.textContent = simulationState.locked ? "Simulation ready" : "Lock violation";
  ui.connectionBadge.className = simulationState.locked
    ? "badge badge-success badge-soft"
    : "badge badge-error badge-soft";
}

function renderJointControls() {
  ui.jointControls.replaceChildren();
  const jointNames = modelContract.joint_groups[selectedGroup];
  for (const jointName of jointNames) {
    const definition = modelContract.joints.find((joint) => joint.name === jointName);
    const value = simulationState.joint_positions[jointName];

    const container = document.createElement("div");
    container.className = "rounded-box border border-base-300 bg-base-200 p-3";

    const heading = document.createElement("div");
    heading.className = "mb-2 flex items-center justify-between gap-3";
    const label = document.createElement("label");
    label.className = "text-sm font-medium";
    label.htmlFor = `range-${jointName}`;
    label.textContent = readableName(jointName);
    const bounds = document.createElement("span");
    bounds.className = "text-xs text-base-content/50 tabular-nums";
    bounds.textContent = `${definition.lower_limit.toFixed(2)} to ${definition.upper_limit.toFixed(2)}`;
    heading.append(label, bounds);

    const controls = document.createElement("div");
    controls.className = "grid grid-cols-[minmax(0,1fr)_6.25rem] items-center gap-3";
    const range = document.createElement("input");
    range.id = `range-${jointName}`;
    range.className = "range range-sm";
    range.type = "range";
    range.min = definition.lower_limit;
    range.max = definition.upper_limit;
    range.step = "0.001";
    range.value = value;
    range.dataset.jointName = jointName;

    const number = document.createElement("input");
    number.className = "input input-sm joint-value w-full";
    number.type = "number";
    number.min = definition.lower_limit;
    number.max = definition.upper_limit;
    number.step = "0.001";
    number.value = value.toFixed(3);
    number.setAttribute("aria-label", `${readableName(jointName)} angle in radians`);

    range.addEventListener("input", () => {
      number.value = Number(range.value).toFixed(3);
      scheduleJointUpdate(jointName, Number(range.value));
    });
    number.addEventListener("change", () => {
      const nextValue = Number(number.value);
      if (!Number.isFinite(nextValue)) {
        number.value = Number(range.value).toFixed(3);
        return;
      }
      range.value = String(nextValue);
      number.value = Number(range.value).toFixed(3);
      scheduleJointUpdate(jointName, Number(range.value));
    });
    controls.append(range, number);
    container.append(heading, controls);
    ui.jointControls.append(container);
  }
}

function scheduleJointUpdate(jointName, value) {
  pendingJointUpdates[jointName] = value;
  window.clearTimeout(updateTimer);
  updateTimer = window.setTimeout(flushJointUpdates, 90);
}

async function flushJointUpdates() {
  if (updateInFlight || Object.keys(pendingJointUpdates).length === 0) {
    return;
  }
  updateInFlight = true;
  const updates = pendingJointUpdates;
  pendingJointUpdates = {};
  try {
    simulationState = await fetchJson("/api/v1/simulation/joints", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ joint_positions: updates }),
    });
    hideStatus();
    syncStateIndicators();
    await refreshRobotImage();
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    updateInFlight = false;
    if (Object.keys(pendingJointUpdates).length > 0) {
      window.clearTimeout(updateTimer);
      updateTimer = window.setTimeout(flushJointUpdates, 30);
    }
  }
}

async function refreshRobotImage() {
  const requestId = ++renderRequest;
  ui.renderLoading.classList.remove("hidden");
  try {
    const response = await fetch(
      `/api/v1/render/${selectedCamera}.png?width=720&height=720&revision=${simulationState.revision}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Robot render failed");
    }
    const blob = await response.blob();
    if (requestId !== renderRequest) {
      return;
    }
    const nextUrl = URL.createObjectURL(blob);
    ui.robotImage.src = nextUrl;
    ui.robotImage.alt = `${cameraLabels[selectedCamera]} view of the Tianyi robot in MuJoCo`;
    if (displayedImageUrl) {
      URL.revokeObjectURL(displayedImageUrl);
    }
    displayedImageUrl = nextUrl;
    const renderMs = response.headers.get("X-Render-Milliseconds");
    ui.renderMetric.textContent = renderMs ? `Rendered in ${renderMs} ms` : "Render complete";
  } finally {
    if (requestId === renderRequest) {
      ui.renderLoading.classList.add("hidden");
    }
  }
}

async function resetNeutral() {
  ui.resetButton.disabled = true;
  try {
    simulationState = await fetchJson("/api/v1/simulation/reset", { method: "POST" });
    pendingJointUpdates = {};
    hideStatus();
    syncStateIndicators();
    renderJointControls();
    await refreshRobotImage();
    showStatus("Neutral pose restored.", "success");
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    ui.resetButton.disabled = false;
  }
}

function bindInteractions() {
  for (const button of document.querySelectorAll(".camera-button")) {
    button.addEventListener("click", async () => {
      selectedCamera = button.dataset.camera;
      for (const candidate of document.querySelectorAll(".camera-button")) {
        candidate.classList.toggle("btn-active", candidate === button);
      }
      ui.cameraDescription.textContent = `${cameraLabels[selectedCamera]} camera, named from the robot frame.`;
      try {
        await refreshRobotImage();
        hideStatus();
      } catch (error) {
        showStatus(error.message, "error");
      }
    });
  }

  for (const tab of document.querySelectorAll("#joint-tabs .tab")) {
    tab.addEventListener("click", () => {
      selectedGroup = tab.dataset.group;
      for (const candidate of document.querySelectorAll("#joint-tabs .tab")) {
        const active = candidate === tab;
        candidate.classList.toggle("tab-active", active);
        candidate.setAttribute("aria-selected", String(active));
      }
      renderJointControls();
    });
  }
  ui.resetButton.addEventListener("click", resetNeutral);
}

async function boot() {
  bindInteractions();
  try {
    [modelContract, simulationState] = await Promise.all([
      fetchJson("/api/v1/model"),
      fetchJson("/api/v1/simulation"),
    ]);
    ui.modelId.textContent = modelContract.model_id;
    ui.meshSummary.textContent =
      `${modelContract.visualization.optimized_triangles.toLocaleString()} visual triangles ` +
      `(${modelContract.visualization.triangle_reduction_percent}% reduction)`;
    ui.lockedDescription.textContent = Object.entries(
      modelContract.locked_joint_positions,
    )
      .map(([name, value]) => {
        const motorId = modelContract.locked_joint_motor_ids[name];
        return `${readableName(name)} (ID ${motorId}): ${Number(value).toFixed(6)} rad`;
      })
      .join(", ");
    syncStateIndicators();
    renderJointControls();
    await refreshRobotImage();
  } catch (error) {
    ui.connectionBadge.textContent = "Unavailable";
    ui.connectionBadge.className = "badge badge-error badge-soft";
    showStatus(error.message, "error");
    ui.renderLoading.classList.add("hidden");
  }
}

boot();
