class PoseRecorderPanel {
  constructor(root = document) {
    this.panel = root.querySelector("[data-pose-recorder]");
    this.defaultPoseName = this.panel?.dataset.defaultPoseName || "default";
    this.groupSelect = root.querySelector("#pose-group");
    this.loadType = root.querySelector("#recorder-load-type");
    this.loadName = root.querySelector("#recorder-load-name");
    this.loadButton = root.querySelector("#recorder-load");
    this.loadLabel = root.querySelector("[data-recorder-load-label]");
    this.loadLoading = root.querySelector("[data-recorder-load-loading]");
    this.savedPoseNames = JSON.parse(this.loadName?.dataset.poseNames || "{}");
    this.rows = [...root.querySelectorAll("[data-joint-row]")];
    this.sliders = [...root.querySelectorAll("[data-joint-target]")];
    this.numberInputs = [...root.querySelectorAll("[data-joint-target-number]")];
    this.resetButtons = [...root.querySelectorAll("[data-joint-reset]")];
    this.actualValues = new Map(
      [...root.querySelectorAll("[data-joint-actual]")].map((element) => [
        element.dataset.jointActual,
        element,
      ]),
    );
    this.status = root.querySelector("#simulation-status");
    this.statusLabel = root.querySelector("#simulation-status-label");
    this.visibleCount = root.querySelector("#visible-joint-count");
    this.error = root.querySelector("#simulation-error");
    this.success = root.querySelector("#simulation-success");
    this.resetButton = root.querySelector("#reset-pose");
    this.saveButton = root.querySelector("#save-pose");
    this.saveLabel = root.querySelector("[data-save-label]");
    this.saveLoading = root.querySelector("[data-save-loading]");
    this.poseName = root.querySelector("#pose-name");
    this.poseNotes = root.querySelector("#pose-notes");
    this.poseOverwrite = root.querySelector("#pose-overwrite");
    this.poseMetadataFields = root.querySelector("#pose-metadata-fields");
    this.poseOverwriteField = root.querySelector("#pose-overwrite-field");
    this.calibrationNotice = root.querySelector("#calibration-notice");
    this.mirrorControls = root.querySelector("#arm-mirror-controls");
    this.mirrorButtons = [...root.querySelectorAll("[data-mirror-arm]")];
    this.defaultTargets = new Map(
      this.resetButtons.map((button) => [
        button.dataset.jointName,
        Number(button.dataset.resetValue),
      ]),
    );
    this.socket = null;
  }

  start() {
    if (!this.panel || !window.TianyiSimulationSocket) {
      return;
    }
    this.socket = new window.TianyiSimulationSocket({
      onState: (message) => this.applyState(message),
      onStatus: (state) => this.setConnectionState(state),
      onError: (detail) => this.showError(detail),
    });

    this.groupSelect.addEventListener("change", () => this.filterRows());
    this.loadType.addEventListener("change", () => this.refreshLoadNames());
    this.loadButton.addEventListener("click", () => this.loadPose());
    for (const slider of this.sliders) {
      slider.addEventListener("input", () => this.changeTarget(slider));
    }
    for (const input of this.numberInputs) {
      input.addEventListener("change", () => this.changeTarget(input));
    }
    for (const button of this.resetButtons) {
      button.addEventListener("click", () => this.resetJoint(button));
    }
    for (const button of this.mirrorButtons) {
      button.addEventListener("click", () => this.mirrorArm(button));
    }
    this.resetButton.addEventListener("click", () => this.resetTargets());
    this.saveButton.addEventListener("click", () => this.savePose());
    document.addEventListener("tianyi:panel-activated", (event) => {
      if (event.detail.panelName === "pose-recorder") {
        this.refreshSavedPoses();
      }
    });
    this.refreshLoadNames();
    this.filterRows();
    this.socket.start();
  }

  async refreshSavedPoses() {
    const selectedType = this.loadType.value;
    const selectedName = this.loadName.value;
    try {
      const response = await fetch("/ui/tianyi-3d/poses", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Could not refresh saved poses (${response.status}).`);
      }
      this.savedPoseNames = await response.json();
      this.loadType.value = selectedType;
      this.refreshLoadNames(selectedName);
    } catch (error) {
      this.showError(error.message);
    }
  }

  refreshLoadNames(preferredName = "") {
    const names = this.savedPoseNames[this.loadType.value] || [];
    this.loadName.replaceChildren();
    if (names.length === 0) {
      this.loadName.add(new Option("No saved poses", ""));
    }
    for (const name of names) {
      this.loadName.add(new Option(name, name));
    }
    if (names.includes(preferredName)) {
      this.loadName.value = preferredName;
    }
    this.loadButton.disabled = names.length === 0;
  }

  async loadPose() {
    if (!this.loadName.value) {
      this.showError("Select a saved pose to load.");
      return;
    }
    this.setLoading(true);
    this.hideFeedback();
    try {
      const response = await fetch("/ui/tianyi-3d/pose-recorder/load", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pose_type: this.loadType.value,
          name: this.loadName.value,
        }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Load failed with ${response.status}`);
      }
      for (const [jointName, value] of Object.entries(result.joint_positions)) {
        this.setTargetInputs(jointName, Number(value));
      }
      this.groupSelect.value = result.edit_pose_type;
      this.poseName.value = result.pose_name;
      this.poseNotes.value = result.notes;
      this.filterRows();
      const composedNote = result.pose_type === "composed"
        ? " It is now an editable base pose; saving creates a base file."
        : "";
      this.showSuccess(
        `Loaded ${this.poseTypeLabel(result.pose_type)} pose “${result.pose_name}” for editing.${composedNote}`,
      );
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.setLoading(false);
    }
  }

  filterRows() {
    const group = this.groupSelect.value;
    const calibration = group === "calibration";
    let visible = 0;
    for (const row of this.rows) {
      const show =
        (group === "base" && row.dataset.jointRecordable === "true") ||
        row.dataset.jointGroup === group;
      row.classList.toggle("hidden", !show);
      visible += Number(show);
    }
    this.visibleCount.textContent = `${visible} joints`;
    this.calibrationNotice.classList.toggle("hidden", !calibration);
    this.mirrorControls.classList.toggle("opacity-45", calibration);
    for (const button of this.mirrorButtons) {
      button.disabled = calibration;
      button.setAttribute("aria-disabled", String(calibration));
    }
    this.poseMetadataFields.classList.toggle("opacity-45", calibration);
    this.poseOverwriteField.classList.toggle("opacity-45", calibration);
    this.poseName.disabled = calibration;
    this.poseNotes.disabled = calibration;
    this.poseOverwrite.disabled = calibration;
    this.saveButton.disabled = calibration;
    this.saveButton.setAttribute("aria-disabled", String(calibration));
    this.saveLabel.textContent = calibration ? "Never recorded" : "Save pose";
    this.resetButton.textContent = calibration
      ? "Reset calibration"
      : `Reset to ${this.defaultPoseName}`;
    this.hideFeedback();
  }

  async mirrorArm(button) {
    const sourcePoseType = button.dataset.mirrorArm;
    for (const mirrorButton of this.mirrorButtons) {
      mirrorButton.disabled = true;
    }
    this.hideFeedback();
    try {
      const response = await fetch("/ui/tianyi-3d/pose-recorder/mirror-arm", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_pose_type: sourcePoseType,
        }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Mirror failed with ${response.status}`);
      }
      for (const [jointName, value] of Object.entries({
        ...result.source_joint_positions,
        ...result.joint_positions,
      })) {
        this.setTargetInputs(jointName, Number(value));
      }
      this.groupSelect.value = result.target_pose_type;
      this.filterRows();
      this.showSuccess(
        `Mirrored ${this.poseTypeLabel(result.source_pose_type)} to ${this.poseTypeLabel(result.target_pose_type)}. Save the destination arm when ready.`,
      );
    } catch (error) {
      this.showError(error.message);
    } finally {
      const calibration = this.groupSelect.value === "calibration";
      for (const mirrorButton of this.mirrorButtons) {
        mirrorButton.disabled = calibration;
      }
    }
  }

  changeTarget(input) {
    const jointName = input.dataset.jointName;
    const value = this.clampToInput(input, Number(input.value));
    if (!Number.isFinite(value)) {
      this.showError("Joint target must be a finite number.");
      return;
    }
    this.setTargetInputs(jointName, value);
    this.hideFeedback();
    this.sendTarget(input, jointName, value);
  }

  resetJoint(button) {
    const jointName = button.dataset.jointName;
    const value = Number(button.dataset.resetValue);
    this.setTargetInputs(jointName, value);
    this.hideFeedback();
    this.sendTarget(button, jointName, value);
  }

  resetTargets() {
    const group = this.groupSelect.value;
    for (const [jointName, value] of this.defaultTargets) {
      const row = this.rowForJoint(jointName);
      if (group !== "base" && row.dataset.jointGroup !== group) {
        continue;
      }
      this.setTargetInputs(jointName, value);
      const input = this.sliders.find(
        (candidate) => candidate.dataset.jointName === jointName,
      );
      this.sendTarget(input, jointName, value);
    }
    this.hideFeedback();
  }

  async savePose() {
    if (this.groupSelect.value === "calibration") {
      this.showError("Calibration joints are never recorded in pose files.");
      return;
    }
    const name = this.poseName.value.trim();
    if (!name) {
      this.showError("Enter a pose name before saving.");
      return;
    }

    const poseType = this.groupSelect.value;
    const jointPositions = {};
    for (const slider of this.sliders) {
      const row = slider.closest("[data-joint-row]");
      if (
        slider.dataset.jointRecordable === "true" &&
        (poseType === "base" || row.dataset.jointGroup === poseType)
      ) {
        jointPositions[slider.dataset.jointName] = Number(slider.value);
      }
    }

    this.setSaving(true);
    this.hideFeedback();
    try {
      const response = await fetch("/ui/tianyi-3d/poses", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pose_type: poseType,
          name,
          notes: this.poseNotes.value,
          joint_positions: jointPositions,
          overwrite: this.poseOverwrite.checked,
        }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `Save failed with ${response.status}`);
      }
      this.showSuccess(
        `Saved ${this.poseTypeLabel(result.pose_type)} pose “${result.pose_name}”.`,
      );
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.setSaving(false);
    }
  }

  applyState(message) {
    for (const [jointName, rawValue] of Object.entries(message.joint_positions)) {
      const actual = this.actualValues.get(jointName);
      if (actual) {
        const precision = actual.dataset.jointCalibration === "true" ? 9 : 3;
        actual.textContent = Number(rawValue).toFixed(precision);
      }
    }
  }

  setTargetInputs(jointName, value) {
    const slider = this.sliders.find(
      (candidate) => candidate.dataset.jointName === jointName,
    );
    const numberInput = this.numberInputs.find(
      (candidate) => candidate.dataset.jointName === jointName,
    );
    slider.value = String(value);
    const precision = slider.dataset.jointCalibration === "true" ? 12 : 3;
    numberInput.value = value.toFixed(precision);
  }

  sendTarget(input, jointName, value) {
    if (input.dataset.jointCalibration === "true") {
      this.socket.setCalibrationTarget(jointName, value);
    } else {
      this.socket.setJointTarget(jointName, value);
    }
  }

  rowForJoint(jointName) {
    return this.rows.find((row) =>
      row.querySelector(`[data-joint-name="${jointName}"]`),
    );
  }

  clampToInput(input, value) {
    if (!Number.isFinite(value)) {
      return value;
    }
    return Math.min(Number(input.max), Math.max(Number(input.min), value));
  }

  setConnectionState(state) {
    const states = {
      ready: ["status-success", "MuJoCo live"],
      connecting: ["status-warning", "Connecting"],
      waiting: ["status-warning", "Reconnecting"],
      error: ["status-error", "Connection error"],
    };
    const [statusClass, label] = states[state];
    this.status.classList.remove(
      "status-success",
      "status-warning",
      "status-error",
    );
    this.status.classList.add(statusClass);
    this.statusLabel.textContent = label;
  }

  showError(detail) {
    this.hideSuccess();
    this.error.textContent = detail;
    this.error.classList.remove("hidden");
  }

  showSuccess(detail) {
    this.hideError();
    this.success.textContent = detail;
    this.success.classList.remove("hidden");
  }

  hideError() {
    this.error.classList.add("hidden");
  }

  hideSuccess() {
    this.success.classList.add("hidden");
  }

  hideFeedback() {
    this.hideError();
    this.hideSuccess();
  }

  setSaving(saving) {
    const calibration = this.groupSelect.value === "calibration";
    this.saveButton.disabled = saving || calibration;
    this.saveButton.setAttribute("aria-busy", String(saving));
    this.saveLabel.textContent = calibration
      ? "Never recorded"
      : saving
        ? "Saving"
        : "Save pose";
    this.saveLoading.classList.toggle("hidden", !saving);
  }

  setLoading(loading) {
    this.loadButton.disabled = loading || !this.loadName.value;
    this.loadButton.setAttribute("aria-busy", String(loading));
    this.loadLabel.textContent = loading ? "Loading pose" : "Load for editing";
    this.loadLoading.classList.toggle("hidden", !loading);
  }

  poseTypeLabel(poseType) {
    const labels = {
      base: "base",
      left_arm: "robot-left arm",
      right_arm: "robot-right arm",
      composed: "composed",
    };
    return labels[poseType] || poseType;
  }
}

window.PoseRecorderPanel = PoseRecorderPanel;
