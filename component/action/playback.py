"""Interruptible MuJoCo-only playback for Action Composer previews."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

from component.action.models import ActionTrajectory
from component.simulation import SimulationService


class ActionPlaybackState(str, Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ActionPlaybackSnapshot:
    revision: int
    state: ActionPlaybackState
    action_name: str | None
    sample_index: int
    sample_count: int
    elapsed_seconds: float
    duration_seconds: float
    loop: bool
    error: str | None = None

    @property
    def progress(self) -> float:
        return 0.0 if self.duration_seconds <= 0.0 else self.elapsed_seconds / self.duration_seconds


class ActionPlaybackService:
    """Preview compiled 17-joint trajectories in the shared simulation."""

    def __init__(self, *, simulation: SimulationService) -> None:
        self.simulation = simulation
        self._condition = threading.Condition()
        self._command_lock = threading.Lock()
        self._generation = 0
        self._revision = 0
        self._state = ActionPlaybackState.IDLE
        self._trajectory: ActionTrajectory | None = None
        self._sample_index = 0
        self._loop = False
        self._error: str | None = None
        self._thread: threading.Thread | None = None

    def load(self, trajectory: ActionTrajectory) -> ActionPlaybackSnapshot:
        if trajectory.joint_names != self.simulation.schema.BASE_JOINT_NAMES:
            raise ValueError("Playback trajectory must contain exactly 17 authored joints")
        with self._command_lock:
            self._cancel_worker()
            with self._condition:
                self._trajectory = trajectory
                self._state = ActionPlaybackState.IDLE
                self._sample_index = 0
                self._loop = False
                self._error = None
                self._revision += 1
                return self._snapshot_unlocked()

    def play(self, *, loop: bool = False) -> ActionPlaybackSnapshot:
        with self._command_lock:
            self._cancel_worker()
            with self._condition:
                trajectory = self._trajectory
                if trajectory is None:
                    raise ValueError("Load a compiled trajectory before playback")
            self._apply(trajectory, 0)
            with self._condition:
                self._generation += 1
                generation = self._generation
                self._sample_index = 0
                self._loop = bool(loop)
                self._error = None
                self._state = ActionPlaybackState.PLAYING
                self._revision += 1
                thread = threading.Thread(
                    target=self._run,
                    args=(trajectory, generation),
                    name=f"tianyi-preview-{trajectory.action_name}",
                    daemon=True,
                )
                self._thread = thread
                snapshot = self._snapshot_unlocked()
            thread.start()
            return snapshot

    def pause(self) -> ActionPlaybackSnapshot:
        with self._condition:
            if self._state is not ActionPlaybackState.PLAYING:
                raise ValueError("Only playing preview can be paused")
            self._state = ActionPlaybackState.PAUSED
            self._revision += 1
            self._condition.notify_all()
            return self._snapshot_unlocked()

    def resume(self) -> ActionPlaybackSnapshot:
        with self._condition:
            if self._state is not ActionPlaybackState.PAUSED:
                raise ValueError("Only paused preview can be resumed")
            self._state = ActionPlaybackState.PLAYING
            self._revision += 1
            self._condition.notify_all()
            return self._snapshot_unlocked()

    def stop(self) -> ActionPlaybackSnapshot:
        with self._command_lock:
            self._cancel_worker()
            with self._condition:
                trajectory = self._trajectory
            if trajectory is not None:
                self._apply(trajectory, 0)
            with self._condition:
                self._sample_index = 0
                self._state = ActionPlaybackState.STOPPED
                self._error = None
                self._revision += 1
                return self._snapshot_unlocked()

    def snapshot(self) -> ActionPlaybackSnapshot:
        with self._condition:
            return self._snapshot_unlocked()

    def close(self) -> None:
        with self._command_lock:
            self._cancel_worker()
            with self._condition:
                self._trajectory = None
                self._state = ActionPlaybackState.IDLE
                self._revision += 1

    def _cancel_worker(self) -> None:
        with self._condition:
            self._generation += 1
            thread = self._thread
            self._thread = None
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self, trajectory: ActionTrajectory, generation: int) -> None:
        cycle_start = time.monotonic()
        index = 1
        try:
            while True:
                with self._condition:
                    if generation != self._generation:
                        return
                    if self._state is ActionPlaybackState.PAUSED:
                        paused_at = time.monotonic()
                        self._condition.wait_for(
                            lambda: generation != self._generation
                            or self._state is not ActionPlaybackState.PAUSED
                        )
                        if generation != self._generation:
                            return
                        cycle_start += time.monotonic() - paused_at
                    if self._state is not ActionPlaybackState.PLAYING:
                        continue
                target_time = cycle_start + float(trajectory.timestamps[index])
                wait = target_time - time.monotonic()
                if wait > 0.0:
                    with self._condition:
                        self._condition.wait(timeout=wait)
                    continue
                self._apply(trajectory, index)
                with self._condition:
                    if generation != self._generation:
                        return
                    self._sample_index = index
                    self._revision += 1
                    if index == trajectory.sample_count - 1:
                        if not self._loop:
                            self._state = ActionPlaybackState.COMPLETED
                            self._thread = None
                            return
                        self._apply(trajectory, 0)
                        self._sample_index = 0
                        self._revision += 1
                        cycle_start = time.monotonic()
                        index = 1
                        continue
                    index += 1
        except Exception as error:  # pragma: no cover - defensive worker boundary
            with self._condition:
                self._state = ActionPlaybackState.FAILED
                self._error = str(error)
                self._revision += 1
                self._thread = None

    def _apply(self, trajectory: ActionTrajectory, index: int) -> None:
        self.simulation.update_joint_positions(
            dict(zip(trajectory.joint_names, trajectory.joint_positions[index], strict=True))
        )

    def _snapshot_unlocked(self) -> ActionPlaybackSnapshot:
        trajectory = self._trajectory
        elapsed = 0.0 if trajectory is None else float(trajectory.timestamps[self._sample_index])
        return ActionPlaybackSnapshot(
            revision=self._revision,
            state=self._state,
            action_name=None if trajectory is None else trajectory.action_name,
            sample_index=self._sample_index,
            sample_count=0 if trajectory is None else trajectory.sample_count,
            elapsed_seconds=elapsed,
            duration_seconds=0.0 if trajectory is None else trajectory.duration_seconds,
            loop=self._loop,
            error=self._error,
        )
