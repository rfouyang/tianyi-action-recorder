class PoseRetargetingPanel {
  constructor(root = document) {
    this.panel = root.querySelector("[data-pose-retargeting]");
    this.sourceFile = root.querySelector("#retarget-source-file");
    this.fileSummary = root.querySelector("#retarget-file-summary");
    this.name = root.querySelector("#retarget-name");
    this.outputType = root.querySelector("#retarget-output-type");
    this.notes = root.querySelector("#retarget-notes");
    this.overwrite = root.querySelector("#retarget-overwrite");
    this.previewButton = root.querySelector("#retarget-preview");
    this.previewLabel = root.querySelector("[data-retarget-preview-label]");
    this.previewLoading = root.querySelector("[data-retarget-preview-loading]");
    this.saveButton = root.querySelector("#retarget-save");
    this.solveBadge = root.querySelector("#retarget-solve-badge");
    this.report = root.querySelector("#retarget-report");
    this.contactReport = root.querySelector("#retarget-contact-report");
    this.status = root.querySelector("#retarget-status");
    this.success = root.querySelector("#retarget-success");
    this.error = root.querySelector("#retarget-error");
    this.sourceContent = "";
    this.sourceFileName = "";
    this.currentSourceName = "";
    this.hasPreview = false;
    this.suggestedName = "";
  }

  start() {
    if (!this.panel) {
      return;
    }
    this.sourceFile.addEventListener("change", () => this.selectSourceFile());
    this.outputType.addEventListener("change", () => this.selectOutputType());
    this.previewButton.addEventListener("click", () => this.preview());
    this.saveButton.addEventListener("click", () => this.save());
    this.selectOutputType();
  }

  async selectSourceFile() {
    const file = this.sourceFile.files[0];
    this.sourceContent = "";
    this.sourceFileName = "";
    this.currentSourceName = "";
    this.invalidatePreview();
    if (!file) {
      this.fileSummary.textContent = "No file selected. Supported: base, composed, robot-left arm, and robot-right arm.";
      return;
    }
    if (!file.name.toLowerCase().endsWith(".json")) {
      this.fileSummary.textContent = "Select a G1 pose file with a .json extension.";
      this.showError("The selected file must be a G1 pose JSON file.");
      return;
    }
    if (file.size > 256 * 1024) {
      this.fileSummary.textContent = "The selected file is larger than 256 KiB.";
      this.showError("G1 pose JSON files cannot exceed 256 KiB.");
      return;
    }
    try {
      const content = await file.text();
      const pose = JSON.parse(content);
      if (!pose || !["base", "composed", "left_arm", "right_arm"].includes(pose.pose_type)) {
        throw new Error("The G1 pose type must be base, composed, left_arm, or right_arm.");
      }
      this.sourceContent = content;
      this.sourceFileName = file.name;
      this.fileSummary.textContent = `${file.name} · ${pose.pose_type}/${pose.name || "unnamed"}`;
      this.previewButton.disabled = false;
      this.hideFeedback();
    } catch (error) {
      this.fileSummary.textContent = `${file.name} could not be read as a supported G1 pose.`;
      this.showError(error.message || "The selected file is not valid JSON.");
    }
  }

  invalidatePreview() {
    this.hasPreview = false;
    this.saveButton.disabled = true;
    this.previewButton.disabled = !this.sourceContent;
    this.solveBadge.textContent = "Not run";
    this.solveBadge.className = "badge badge-ghost badge-sm";
    document.dispatchEvent(new CustomEvent("tianyi:retarget-preview-invalidated"));
  }

  async preview() {
    if (!this.sourceContent) {
      this.showError("Choose a supported G1 pose JSON file first.");
      return;
    }
    this.setPreviewing(true);
    this.hideFeedback();
    try {
      const result = await this.send("/ui/tianyi-3d/pose-retargeting/preview", false);
      this.hasPreview = true;
      this.currentSourceName = result.source_name;
      this.saveButton.disabled = false;
      if (!this.name.value.trim()) {
        this.suggestedName = this.outputName(result.source_name);
        this.name.value = this.suggestedName;
      }
      this.renderReport(result);
      document.dispatchEvent(
        new CustomEvent("tianyi:retarget-preview-updated", {
          detail: { sourceName: result.source_name },
        }),
      );
      this.showStatus(
        `Comparing original G1 ${result.source_pose_type}/${result.source_name} with the retargeted Tianyi pose.`,
      );
    } catch (error) {
      this.invalidatePreview();
      this.showError(error.message);
    } finally {
      this.setPreviewing(false);
    }
  }

  async save() {
    if (!this.hasPreview) {
      this.showError("Retarget and inspect the pose before saving.");
      return;
    }
    if (!this.name.value.trim()) {
      this.showError("Enter a Tianyi arm pose name before saving.");
      return;
    }
    this.saveButton.disabled = true;
    this.hideFeedback();
    try {
      const result = await this.send("/ui/tianyi-3d/pose-retargeting/save", true);
      this.renderReport(result);
      this.showSuccess(
        `Saved ${this.poseTypeLabel(result.saved_pose_type)} pose “${result.saved_pose_name}”.`,
      );
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.saveButton.disabled = false;
    }
  }

  async send(url, save) {
    const response = await fetch(url, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({
        source_file_name: this.sourceFileName,
        source_pose_json: this.sourceContent,
        output_pose_type: save ? this.outputType.value : null,
        name: save ? this.name.value.trim() : "",
        notes: save ? this.notes.value : "",
        overwrite: save && this.overwrite.checked,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || `Retargeting failed (${response.status}).`);
    }
    return result;
  }

  renderReport(result) {
    this.report.replaceChildren();
    for (const arm of result.arm_reports) {
      const row = document.createElement("tr");
      const side = arm.side === "left" ? "Robot-left" : "Robot-right";
      const limit = arm.reached_joint_limits.length
        ? ` · limit: ${arm.reached_joint_limits.join(", ")}`
        : "";
      row.innerHTML = `
        <th>${side}</th>
        <td class="font-mono tabular-nums">${(arm.hand_position_error_m * 1000).toFixed(1)} mm</td>
        <td class="font-mono tabular-nums">${(arm.elbow_position_error_m * 1000).toFixed(1)} mm${limit}</td>
        <td class="font-mono tabular-nums">${(arm.upper_arm_direction_error_rad * 180 / Math.PI).toFixed(1)}°</td>
        <td class="font-mono tabular-nums">${(arm.forearm_direction_error_rad * 180 / Math.PI).toFixed(1)}°</td>
        <td class="font-mono tabular-nums">${(arm.hand_orientation_error_rad * 180 / Math.PI).toFixed(1)}°</td>
      `;
      this.report.append(row);
    }
    this.solveBadge.textContent = result.converged ? "Solved" : "Review";
    this.solveBadge.className = result.converged
      ? "badge badge-success badge-sm"
      : "badge badge-warning badge-sm";
    this.contactReport.textContent = `Contacts · ${result.contact_count}`;
  }

  setPreviewing(previewing) {
    this.previewButton.disabled = previewing || !this.sourceContent;
    this.previewButton.setAttribute("aria-busy", String(previewing));
    this.previewLabel.textContent = previewing ? "Solving both arms" : "Retarget and preview";
    this.previewLoading.classList.toggle("hidden", !previewing);
  }

  selectOutputType() {
    const nextSuggestion = this.outputName(this.currentSourceName || "");
    if (!this.name.value.trim() || this.name.value === this.suggestedName) {
      this.name.value = nextSuggestion;
    }
    this.suggestedName = nextSuggestion;
    this.saveButton.textContent = this.outputType.value === "left_arm"
      ? "Save robot-left arm pose"
      : "Save robot-right arm pose";
  }

  outputName(sourceName) {
    const suffix = this.outputType.value === "left_arm" ? "left" : "right";
    return sourceName ? `${sourceName}_${suffix}` : "";
  }

  poseTypeLabel(poseType) {
    return poseType === "left_arm" ? "robot-left arm" : "robot-right arm";
  }

  showStatus(message) {
    this.status.textContent = message;
    this.status.classList.remove("hidden");
    this.error.classList.add("hidden");
  }

  showSuccess(message) {
    this.success.textContent = message;
    this.success.classList.remove("hidden");
    this.error.classList.add("hidden");
  }

  showError(message) {
    this.error.textContent = message;
    this.error.classList.remove("hidden");
    this.success.classList.add("hidden");
  }

  hideFeedback() {
    this.success.classList.add("hidden");
    this.error.classList.add("hidden");
  }
}

window.PoseRetargetingPanel = PoseRetargetingPanel;
