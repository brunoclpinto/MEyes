//
//  Camera.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import CoreImage
import MWDATCore
import MWDATCamera
internal import UIKit

// MARK: - FrameGate

/// Thread-safe gate that allows only one frame through at a time.
/// While a frame is being processed, all subsequent frames are discarded.
private nonisolated final class FrameGate: @unchecked Sendable {
  private var processing = false
  private let lock = NSLock()

  /// Returns `true` if no frame is currently being processed,
  /// and atomically marks the gate as busy.
  func tryEnter() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard !processing else { return false }
    processing = true
    return true
  }

  func leave() {
    lock.lock()
    processing = false
    lock.unlock()
  }

  func reset() {
    lock.lock()
    processing = false
    lock.unlock()
  }
}

// MARK: - CameraMeta

public actor CameraMeta: Camera {
  public private(set) var state: CameraState = .disconnected(.notInit)
  public let name: String
  public let zoom: String = ""

  private let wearables: WearablesInterface
  private let deviceSelector: AutoDeviceSelector
  private var stateContinuations: [AsyncStream<CameraState>.Continuation] = []

  // SDK session objects
  private var deviceSession: DeviceSession?
  private var streamSession: StreamSession?

  // Listener tokens
  private var deviceStateToken: (any AnyListenerToken)?
  private var deviceErrorToken: (any AnyListenerToken)?
  private var streamStateToken: (any AnyListenerToken)?
  private var videoToken: (any AnyListenerToken)?
  private var streamErrorToken: (any AnyListenerToken)?

  // Monitoring
  private var deviceMonitorTask: Task<Void, Never>?
  private var nextFrameCallback: ((CIImage) -> Void)?

  /// Discards incoming frames while one is still being processed downstream.
  private let frameGate = FrameGate()

  init(wearables: WearablesInterface) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    self.name = String(localized: "Meta Glasses")
  }

  // MARK: - Connect

  public func connect(nextFrame: @escaping (CIImage) -> Void) async {
    switch state {
      case
          .connected,
          .connecting,
          .disconnecting,
          .started,
          .starting,
          .stopped,
          .stopping,
          .forceDisconnect:
        return
      case .disconnected:
        break
    }
    setState(.connecting)
    nextFrameCallback = nextFrame

    // 1 — Check / request camera permission.
    do {
      let status = try await wearables.checkPermissionStatus(.camera)
      if status != .granted {
        let granted = try await wearables.requestPermission(.camera)
        guard granted == .granted else {
          setState(.disconnected(.noPermissions))
          return
        }
      }
    } catch {
      setState(.disconnected(.noPermissions))
      return
    }

    // 2 — Wait for a device to become available via AutoDeviceSelector.
    print("[CameraMeta] waiting for active device…")
    let hasDevice = await waitForActiveDevice(timeout: .seconds(15))
    guard hasDevice else {
      print("[CameraMeta] no active device found within timeout")
      setState(.disconnected(.noDevice))
      return
    }
    print("[CameraMeta] active device available: \(deviceSelector.activeDevice as Any)")

    // 3 — Create and start the DeviceSession.
    guard let session = await createAndStartSession() else {
      setState(.disconnected(.noSession))
      return
    }

    // 4 — Add the stream capability (session is now .started).
    guard let stream = addStream(to: session) else {
      session.stop()
      deviceSession = nil
      setState(.disconnected(.noSession))
      return
    }

    // 5 — Set up runtime observers.
    setupDeviceSessionObservers(session)
    setupStreamObservers(stream, nextFrame: nextFrame)

    // 6 — Start device monitoring for disconnect/reconnect.
    startDeviceMonitoring()

    setState(.connected)
  }

  // MARK: - Disconnect

  public func disconnect() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected:
        return
      case .started:
        await stop()
      case
          .forceDisconnect,
          .stopped,
          .connected:
        break
    }
    setState(.disconnecting)
    await teardown()
    setState(.disconnected(nil))
  }

  // MARK: - Start

  public func start() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected(_),
          .forceDisconnect,
          .started:
        return
      case
          .connected,
          .stopped:
        break
    }
    setState(.starting)
    guard let streamSession else {
      setState(.disconnected(.noSession))
      return
    }
    await streamSession.start()
    setState(.started)
  }

  // MARK: - Stop

  public func stop() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected(_),
          .forceDisconnect,
          .stopped,
          .connected:
        return
      case .started:
        break
    }
    setState(.stopping)
    await streamSession?.stop()
    setState(.stopped)
  }

  // MARK: - Session Setup

  /// Creates a DeviceSession using AutoDeviceSelector, starts it,
  /// and waits for it to reach `.started` before returning.
  private func createAndStartSession() async -> DeviceSession? {
    let session: DeviceSession
    do {
      session = try wearables.createSession(deviceSelector: deviceSelector)
    } catch {
      print("[CameraMeta] createSession failed: \(error)")
      return nil
    }
    self.deviceSession = session

    // Get stateStream BEFORE calling start() to avoid missing the .started emission.
    let stateStream = session.stateStream()
    do {
      try session.start()
      print("[CameraMeta] DeviceSession.start() called, state: \(session.state)")
    } catch {
      print("[CameraMeta] DeviceSession.start() failed: \(error)")
      deviceSession = nil
      return nil
    }

    // Wait for .started (or bail on .stopped).
    for await sdkState in stateStream {
      print("[CameraMeta] DeviceSession state: \(sdkState)")
      if sdkState == .started { return session }
      if sdkState == .stopped {
        print("[CameraMeta] DeviceSession went to .stopped before .started")
        deviceSession = nil
        return nil
      }
    }

    print("[CameraMeta] DeviceSession stateStream ended unexpectedly")
    deviceSession = nil
    return nil
  }

  /// Adds a StreamSession capability to an already-started DeviceSession.
  private func addStream(to session: DeviceSession) -> StreamSession? {
    let config = StreamSessionConfig(
      videoCodec: .raw,
      resolution: .high,
      frameRate: 30
    )
    let stream: StreamSession?
    do {
      stream = try session.addStream(config: config)
    } catch {
      print("[CameraMeta] addStream failed: \(error)")
      return nil
    }
    guard let stream else {
      print("[CameraMeta] addStream returned nil (session state: \(session.state))")
      return nil
    }
    self.streamSession = stream
    print("[CameraMeta] StreamSession created, state: \(stream.state)")
    return stream
  }

  // MARK: - Observers

  private func setupDeviceSessionObservers(_ session: DeviceSession) {
    deviceStateToken = session.statePublisher.listen { [weak self] sdkState in
      Task { [weak self] in
        await self?.handleDeviceSessionState(sdkState)
      }
    }
    deviceErrorToken = session.errorPublisher.listen { [weak self] sdkError in
      Task { [weak self] in
        print("[CameraMeta] DeviceSession error: \(sdkError)")
        await self?.forceDisconnect()
      }
    }
  }

  private func setupStreamObservers(_ stream: StreamSession, nextFrame: @escaping (CIImage) -> Void) {
    streamStateToken = stream.statePublisher.listen { [weak self] sdkState in
      Task { [weak self] in
        await self?.handleStreamSessionState(sdkState)
      }
    }
    let gate = self.frameGate
    videoToken = stream.videoFramePublisher.listen { videoFrame in
      guard gate.tryEnter() else { return }
      guard
        let image = videoFrame.makeUIImage(),
        let ciImage = CIImage(image: image)
      else {
        gate.leave()
        return
      }
      nextFrame(ciImage)
      gate.leave()
    }
    streamErrorToken = stream.errorPublisher.listen { [weak self] sdkError in
      Task { [weak self] in
        print("[CameraMeta] StreamSession error: \(sdkError)")
        await self?.forceDisconnect()
      }
    }
  }

  // MARK: - SDK State Handling

  private func handleDeviceSessionState(_ sdkState: DeviceSessionState) async {
    switch sdkState {
      case .started:
        break
      case .paused:
        if state == .started {
          setState(.stopped)
        }
      case .stopped:
        // DeviceSession.stopped is terminal — must tear down.
        await forceDisconnect()
      case .idle, .starting, .stopping:
        break
    }
  }

  private func handleStreamSessionState(_ sdkState: StreamSessionState) async {
    switch sdkState {
      case .streaming:
        if state == .starting { return }
        setState(.started)
      case .stopped, .paused:
        if state == .started || state == .starting {
          setState(.stopped)
        }
      case .waitingForDevice, .starting, .stopping:
        break
    }
  }

  // MARK: - Device Monitoring

  /// Waits for AutoDeviceSelector to report an active device.
  private nonisolated func waitForActiveDevice(timeout: Duration) async -> Bool {
    await withTaskGroup(of: Bool.self) { group in
      group.addTask { [deviceSelector] in
        for await device in deviceSelector.activeDeviceStream() {
          if device != nil { return true }
        }
        return false
      }
      group.addTask {
        try? await Task.sleep(for: timeout)
        return false
      }
      let result = await group.next() ?? false
      group.cancelAll()
      return result
    }
  }

  /// Monitors device availability. If the device is lost, tears down the session.
  private func startDeviceMonitoring() {
    deviceMonitorTask?.cancel()
    deviceMonitorTask = Task { [weak self, deviceSelector] in
      // Skip the first emission (current state) — we only care about changes.
      var isFirst = true
      for await device in deviceSelector.activeDeviceStream() {
        guard let self else { return }
        if isFirst { isFirst = false; continue }
        if device == nil {
          print("[CameraMeta] device lost, forcing disconnect")
          await self.forceDisconnect()
          return
        }
      }
    }
  }

  // MARK: - Teardown

  private func forceDisconnect() async {
    state = .forceDisconnect
    await disconnect()
  }

  private func teardown() async {
    deviceMonitorTask?.cancel()
    deviceMonitorTask = nil
    nextFrameCallback = nil
    await cancelAllTokens()
    await streamSession?.stop()
    streamSession = nil
    deviceSession?.stop()
    deviceSession = nil
    frameGate.reset()
  }

  private func cancelAllTokens() async {
    await deviceStateToken?.cancel()
    await deviceErrorToken?.cancel()
    await streamStateToken?.cancel()
    await videoToken?.cancel()
    await streamErrorToken?.cancel()
    deviceStateToken = nil
    deviceErrorToken = nil
    streamStateToken = nil
    videoToken = nil
    streamErrorToken = nil
  }
}

// MARK: - Publisher

public extension CameraMeta {
  func stateUpdates() -> AsyncStream<CameraState> {
    AsyncStream { cont in
      stateContinuations.append(cont)
      cont.yield(state)
    }
  }

  private func setState(_ newState: CameraState) {
    state = newState
    stateContinuations.forEach { $0.yield(newState) }
  }
}
