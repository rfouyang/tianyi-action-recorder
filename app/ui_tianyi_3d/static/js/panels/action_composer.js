class ActionComposerPanel {
  constructor(root = document) {
    this.panel = root.querySelector("[data-action-panel]");
    this.saved = root.querySelector("#action-saved");
    this.loadButton = root.querySelector("#action-load");
    this.name = root.querySelector("#action-name");
    this.notes = root.querySelector("#action-notes");
    this.pose = root.querySelector("#action-pose");
    this.travel = root.querySelector("#action-travel");
    this.hold = root.querySelector("#action-hold");
    this.addButton = root.querySelector("#action-add-frame");
    this.clearButton = root.querySelector("#action-clear");
    this.sequence = root.querySelector("#action-sequence");
    this.endFrame = root.querySelector("[data-action-end]");
    this.endIndex = root.querySelector("[data-action-end-index]");
    this.returnDuration = root.querySelector("#action-return");
    this.frequency = root.querySelector("#action-frequency");
    this.overwrite = root.querySelector("#action-overwrite");
    this.saveButton = root.querySelector("#action-save");
    this.compileButton = root.querySelector("#action-compile");
    this.download = root.querySelector("#action-download");
    this.trajectory = root.querySelector("#action-trajectory");
    this.playButton = root.querySelector("#action-play");
    this.pauseButton = root.querySelector("#action-pause");
    this.stopButton = root.querySelector("#action-stop");
    this.loop = root.querySelector("#action-loop");
    this.playbackState = root.querySelector("#action-playback-state");
    this.progress = root.querySelector("#action-progress");
    this.time = root.querySelector("#action-time");
    this.status = root.querySelector("#action-status");
    this.success = root.querySelector("#action-success");
    this.error = root.querySelector("#action-error");
    this.frames = [];
    this.state = "idle";
    this.pollTimer = null;
  }

  start() {
    if (!this.panel) return;
    this.loadButton.addEventListener("click", () => this.load());
    this.addButton.addEventListener("click", () => this.addFrame());
    this.clearButton.addEventListener("click", () => this.clearFrames());
    this.saveButton.addEventListener("click", () => this.save());
    this.compileButton.addEventListener("click", () => this.compile());
    this.playButton.addEventListener("click", () => this.play());
    this.pauseButton.addEventListener("click", () => this.pauseOrResume());
    this.stopButton.addEventListener("click", () => this.playbackCommand("stop"));
    document.addEventListener("tianyi:panel-activated", (event) => {
      if (event.detail.panelName === "action-composer") this.refreshSources();
    });
    this.pollTimer = window.setInterval(() => this.pollPlayback(), 250);
  }

  async refreshSources() {
    const selected = {
      saved: this.saved.value,
      pose: this.pose.value,
      trajectory: this.trajectory.value,
    };
    try {
      const response = await fetch("/ui/tianyi-3d/action/sources", { headers: { Accept: "application/json" } });
      const result = await this.responseJson(response, "Could not refresh action sources");
      this.fill(this.saved, result.saved_action_names, selected.saved, "Select a definition");
      this.fillObjects(this.pose, result.action_pose_choices, selected.pose, "Select a base or composed pose");
      this.fill(this.trajectory, result.compiled_action_names, selected.trajectory, "Select a compiled NPZ");
    } catch (error) {
      this.showError(error.message);
    }
  }

  fill(select, values, selected, emptyLabel, prefix = "") {
    select.replaceChildren(new Option(emptyLabel, ""));
    for (const value of values) select.add(new Option(`${prefix}${value}`, value));
    if (values.includes(selected)) select.value = selected;
  }

  fillObjects(select, values, selected, emptyLabel) {
    select.replaceChildren(new Option(emptyLabel, ""));
    for (const value of values) select.add(new Option(value.label, value.value));
    if (values.some((value) => value.value === selected)) select.value = selected;
  }

  addFrame() {
    try {
      const reference = this.decodeReference(this.pose.value);
      this.frames.push({
        pose_type: reference.poseType,
        name: reference.name,
        duration_seconds: this.number(this.travel, "Travel time", false),
        hold_seconds: this.number(this.hold, "Hold time", true),
      });
      this.renderFrames();
      this.previewPose(reference.poseType, reference.name);
      this.showStatus(`Added ${reference.poseType}/${reference.name}. Timing remains editable in the timeline.`);
    } catch (error) {
      this.showError(error.message);
    }
  }

  clearFrames() {
    this.frames = [];
    this.renderFrames();
    this.showStatus("Cleared all intermediate keyframes.");
  }

  renderFrames() {
    for (const row of this.sequence.querySelectorAll("[data-action-frame]")) row.remove();
    this.frames.forEach((frame, index) => {
      const row = document.createElement("li");
      row.className = "list-row border border-base-300 bg-base-100 px-3 py-2";
      row.dataset.actionFrame = String(index);
      const badge = document.createElement("span");
      badge.className = frame.pose_type === "composed" ? "badge badge-secondary badge-sm" : "badge badge-outline badge-sm";
      badge.textContent = String(index + 2).padStart(2, "0");
      const detail = document.createElement("div");
      detail.className = "min-w-0";
      const poseButton = document.createElement("button");
      poseButton.className = "block max-w-full truncate text-left text-sm font-semibold hover:underline";
      poseButton.type = "button";
      poseButton.textContent = `${frame.pose_type}/${frame.name}`;
      poseButton.addEventListener("click", () => this.previewPose(frame.pose_type, frame.name));
      const timings = document.createElement("div");
      timings.className = "mt-1 flex gap-2";
      timings.append(
        this.timingInput(frame, "duration_seconds", "Travel", 0.01),
        this.timingInput(frame, "hold_seconds", "Hold", 0),
      );
      detail.append(poseButton, timings);
      const remove = document.createElement("button");
      remove.className = "btn btn-ghost btn-xs text-error";
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        this.frames.splice(index, 1);
        this.renderFrames();
      });
      row.append(badge, detail, remove);
      this.sequence.insertBefore(row, this.endFrame);
    });
    this.endIndex.textContent = String(this.frames.length + 2).padStart(2, "0");
    this.download.classList.add("hidden");
  }

  timingInput(frame, key, label, min) {
    const wrapper = document.createElement("label");
    wrapper.className = "flex items-center gap-1 font-mono text-[0.65rem] uppercase text-base-content/55";
    wrapper.append(`${label} `);
    const input = document.createElement("input");
    input.className = "input input-xs w-16";
    input.type = "number";
    input.min = String(min);
    input.step = "0.1";
    input.value = String(frame[key]);
    input.addEventListener("change", () => {
      try {
        frame[key] = this.number(input, label, min === 0);
        this.download.classList.add("hidden");
      } catch (error) {
        this.showError(error.message);
      }
    });
    wrapper.append(input, "s");
    return wrapper;
  }

  async previewPose(poseType, name) {
    try {
      const response = await fetch("/ui/tianyi-3d/action/preview-pose", {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ pose_type: poseType, name }),
      });
      await this.responseJson(response, "Could not preview pose");
    } catch (error) {
      this.showError(error.message);
    }
  }

  payload(compile) {
    if (!this.name.value.trim()) throw new Error("Enter an action name.");
    if (!this.frames.length) throw new Error("Add at least one intermediate keyframe.");
    const payload = {
      name: this.name.value.trim(),
      notes: this.notes.value,
      frames: this.frames,
      return_duration_seconds: this.number(this.returnDuration, "Return travel time", false),
      overwrite: this.overwrite.checked,
    };
    if (compile) payload.sample_frequency_hz = this.number(this.frequency, "Trajectory frequency", false);
    return payload;
  }

  async save() {
    await this.submit("save", this.saveButton, false);
  }

  async compile() {
    await this.submit("compile", this.compileButton, true);
  }

  async submit(endpoint, button, compile) {
    let payload;
    try { payload = this.payload(compile); } catch (error) { this.showError(error.message); return; }
    this.busy(button, true);
    this.hideFeedback();
    try {
      const response = await fetch(`/ui/tianyi-3d/action/${endpoint}`, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await this.responseJson(response, `Could not ${endpoint} action`);
      await this.refreshSources();
      if (compile) {
        this.trajectory.value = result.action_name;
        this.download.href = result.download_url;
        this.download.download = `${result.action_name}.npz`;
        this.download.classList.remove("hidden");
        this.showSuccess(`Compiled ${result.sample_count} direct joint samples at ${result.sample_frequency_hz} Hz · ${result.duration_seconds.toFixed(2)}s.`);
      } else {
        this.saved.value = result.action_name;
        this.showSuccess(`Saved “${result.action_name}” · ${result.keyframe_count} keyframes · ${result.total_duration_seconds.toFixed(2)}s.`);
      }
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.busy(button, false);
    }
  }

  async load() {
    if (!this.saved.value) { this.showError("Select a saved action first."); return; }
    this.busy(this.loadButton, true);
    try {
      const response = await fetch(`/ui/tianyi-3d/action/definitions/${encodeURIComponent(this.saved.value)}`);
      const result = await this.responseJson(response, "Could not load action");
      this.name.value = result.name;
      this.notes.value = result.notes;
      this.frames = result.frames;
      this.returnDuration.value = result.return_duration_seconds;
      this.renderFrames();
      this.showSuccess(`Loaded “${result.name}” · ${result.frames.length} intermediate keyframe(s).`);
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.busy(this.loadButton, false);
    }
  }

  async play() {
    if (!this.trajectory.value) { this.showError("Select or compile a trajectory first."); return; }
    try {
      const response = await fetch("/ui/tianyi-3d/action/playback/play", {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ action_name: this.trajectory.value, loop: this.loop.checked }),
      });
      this.updatePlayback(await this.responseJson(response, "Could not play preview"));
    } catch (error) { this.showError(error.message); }
  }

  async pauseOrResume() {
    await this.playbackCommand(this.state === "paused" ? "resume" : "pause");
  }

  async playbackCommand(command) {
    try {
      const response = await fetch(`/ui/tianyi-3d/action/playback/${command}`, { method: "POST", headers: { Accept: "application/json" } });
      this.updatePlayback(await this.responseJson(response, `Could not ${command} preview`));
    } catch (error) { this.showError(error.message); }
  }

  async pollPlayback() {
    if (document.hidden || !this.panel || this.panel.classList.contains("hidden")) return;
    try {
      const response = await fetch("/ui/tianyi-3d/action/playback", { headers: { Accept: "application/json" } });
      if (response.ok) this.updatePlayback(await response.json());
    } catch (_) { /* next poll retries */ }
  }

  updatePlayback(playback) {
    this.state = playback.state;
    const labels = { idle: "Idle", playing: "Playing", paused: "Paused", stopped: "Stopped", completed: "Complete", failed: "Failed" };
    this.playbackState.textContent = labels[playback.state] || playback.state;
    this.playbackState.className = `badge badge-sm ${playback.state === "playing" ? "badge-success" : playback.state === "failed" ? "badge-error" : "badge-ghost"}`;
    this.progress.value = Number(playback.progress || 0) * 100;
    this.time.textContent = `${Number(playback.elapsed_seconds).toFixed(2)} / ${Number(playback.duration_seconds).toFixed(2)} s`;
    this.pauseButton.disabled = !["playing", "paused"].includes(playback.state);
    this.pauseButton.textContent = playback.state === "paused" ? "Resume" : "Pause";
    this.stopButton.disabled = !["playing", "paused", "completed"].includes(playback.state);
  }

  decodeReference(value) {
    const separator = value.indexOf("/");
    if (separator < 1) throw new Error("Select a complete pose.");
    return { poseType: value.slice(0, separator), name: value.slice(separator + 1) };
  }

  number(input, label, allowZero) {
    const value = Number(input.value);
    if (!Number.isFinite(value) || (allowZero ? value < 0 : value <= 0)) throw new Error(`${label} must be ${allowZero ? "zero or greater" : "greater than zero"}.`);
    return value;
  }

  async responseJson(response, fallback) {
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || fallback);
    return result;
  }

  busy(button, enabled) {
    button.disabled = enabled;
    button.classList.toggle("loading", enabled);
  }

  hideFeedback() {
    this.success.classList.add("hidden");
    this.error.classList.add("hidden");
  }

  showStatus(message) {
    this.hideFeedback();
    this.status.textContent = message;
  }

  showSuccess(message) {
    this.hideFeedback();
    this.success.textContent = message;
    this.success.classList.remove("hidden");
  }

  showError(message) {
    this.hideFeedback();
    this.error.textContent = message;
    this.error.classList.remove("hidden");
  }
}

window.ActionComposerPanel = ActionComposerPanel;
