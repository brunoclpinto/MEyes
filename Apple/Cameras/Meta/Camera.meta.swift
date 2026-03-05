//
//  Camera.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import UIKit
import MWDATCore
import MWDATCamera

// MARK: - MetaStreamBridge

/// Owns all @MainActor-isolated Meta SDK objects.
/// CameraMeta delegates SDK calls here to satisfy the SDK's main-actor isolation.
@MainActor
private final class MetaStreamBridge: @unchecked Sendable {
  private var session: StreamSession?
  private var stateToken: AnyListenerToken?
  private var videoToken: AnyListenerToken?
  private var errorToken: AnyListenerToken?

  nonisolated init() {}

  func setup(
    wearables: WearablesInterface,
    onState: @escaping (StreamSessionState) -> Void,
    onFrame: @escaping (CGImage?) -> Void,
    onError: @escaping () -> Void
  ) {
    let selector = AutoDeviceSelector(wearables: wearables)
    let config = StreamSessionConfig(videoCodec: .raw, resolution: .high, frameRate: 30)
    let session = StreamSession(streamSessionConfig: config, deviceSelector: selector)

    stateToken = session.statePublisher.listen { sdkState in
      onState(sdkState)
    }
    videoToken = session.videoFramePublisher.listen { videoFrame in
      Task { @MainActor in
        onFrame(videoFrame.makeUIImage()?.toCGImage())
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

  var isReady: Bool { session != nil }
  func start() async { await session?.start() }
  func stop() async { await session?.stop() }

  func teardown() async {
    cancelListeners()
    await session?.stop()
    session = nil
  }
}

// MARK: - CameraMeta

public actor CameraMeta: Camera {
  public private(set) var state: CameraState = .disconnected(.notInit)
  public let name: String
  public let zoom: String = "1x"

  private let wearables: WearablesInterface
  private let bridge = MetaStreamBridge()
  private var stateContinuations: [AsyncStream<CameraState>.Continuation] = []

  init(deviceId: DeviceIdentifier, wearables: WearablesInterface) {
    self.wearables = wearables
    self.name = wearables.deviceForIdentifier(deviceId)?.nameOrId() ?? "Meta Glasses"
  }

  public func connect(nextFrame: @escaping (CGImage?) -> Void) async {
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
