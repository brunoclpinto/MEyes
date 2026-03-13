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

// MARK: - MetaStreamBridge

/// Owns all @MainActor-isolated Meta SDK objects.
/// CameraMeta delegates SDK calls here to satisfy the SDK's main-actor isolation.
@MainActor
private final class MetaStreamBridge: @unchecked Sendable {
  private var session: StreamSession?
  private var stateToken: AnyListenerToken?
  private var videoToken: AnyListenerToken?
  private var errorToken: AnyListenerToken?

  /// Timestamp of the last frame forwarded to the consumer.
  /// Used to enforce the configured frame rate on the app side because
  /// the SDK does not always honour StreamSessionConfig.frameRate.
  private var lastFrameTime: CFAbsoluteTime = 0

  nonisolated init() {}

  func setup(
    wearables: WearablesInterface,
    onState: @escaping (StreamSessionState) -> Void,
    onFrame: @escaping (CIImage) -> Void,
    onError: @escaping () -> Void
  ) {
    let selector = AutoDeviceSelector(wearables: wearables)
    let frameRate: UInt = 30
    let config = StreamSessionConfig(videoCodec: .raw, resolution: .high, frameRate: frameRate)
    let session = StreamSession(streamSessionConfig: config, deviceSelector: selector)

    let minInterval: CFAbsoluteTime = 1.0 / CFAbsoluteTime(frameRate)

    stateToken = session.statePublisher.listen { sdkState in
      onState(sdkState)
    }
    videoToken = session.videoFramePublisher.listen { videoFrame in
      Task { @MainActor in
        let now = CFAbsoluteTimeGetCurrent()
        guard now - self.lastFrameTime >= minInterval else { return }
        self.lastFrameTime = now
        guard
          let image = videoFrame.makeUIImage(),
          let ciImage = CIImage(image: image)
        else {
          return
        }
        onFrame(ciImage)
      }
    }
    errorToken = session.errorPublisher.listen { _ in
      onError()
    }
    self.session = session
  }

  func cancelListeners() {
    stateToken = nil
    videoToken = nil
    errorToken = nil
  }

  var isReady: Bool { return session != nil }
  func start() async { await session?.start() }
  func stop() async { await session?.stop() }

  func teardown() async {
    cancelListeners()
    await session?.stop()
    session = nil
    lastFrameTime = 0
  }
}

// MARK: - CameraMeta

public actor CameraMeta: Camera {
  public private(set) var state: CameraState = .disconnected(.notInit)
  public let name: String
  public let zoom: String = ""

  private let wearables: WearablesInterface
  private let bridge = MetaStreamBridge()
  private var stateContinuations: [AsyncStream<CameraState>.Continuation] = []

  init(deviceId: DeviceIdentifier, wearables: WearablesInterface) {
    self.wearables = wearables
    self.name = wearables
      .deviceForIdentifier(deviceId)?
      .nameOrId() ?? String(localized: "Unnamed Meta Wearable")
  }

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

    await bridge.setup(
      wearables: wearables,
      onState: { [weak self] sdkState in
        Task { [weak self] in
          await self?.handleSDKState(sdkState)
        }
      },
      onFrame: nextFrame,
      onError: { [weak self] in
        Task { [weak self] in
          await self?.forceDisconnect()
        }
      }
    )
    setState(.connected)
  }

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
    await bridge.teardown()
    setState(.disconnected(nil))
  }

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
    guard await bridge.isReady else {
      setState(.disconnected(.noSession))
      return
    }
    await bridge.start()
    setState(.started)
  }

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
    await bridge.stop()
    setState(.stopped)
  }

  private func handleSDKState(_ sdkState: StreamSessionState) async {
    switch sdkState {
      case .streaming:
        switch state {
          case .starting:
            return
          default:
            break
        }
        setState(.started)
      case .stopped, .paused:
        setState(.stopped)
      case .waitingForDevice, .starting, .stopping:
        break
      @unknown default:
        break
    }
  }

  private func forceDisconnect() async {
    state = .forceDisconnect
    await disconnect()
  }
}

/// Publisher
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
