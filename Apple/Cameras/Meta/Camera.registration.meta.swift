//
//  Camera.registration.meta.swift
//  HackEye
//
//  Created by Bruno Pinto on 05/03/2026.
//

import CoreImage
import MWDATCore

/// A placeholder camera that drives the Meta Wearables registration flow.
/// Appears in the camera list when the Meta AI app is installed but the user
/// has not yet registered this app with the SDK.
///
/// - `connect()` triggers `startRegistration()` and waits for completion.
/// - Once registered, the state moves to `.connected` so the UI layer
///   can reload the camera list and discover real Meta cameras.
public actor CameraMetaRegistration: Camera {
  public private(set) var state: CameraState = .disconnected(nil)
  public let name: String = "Register Meta Wearable"
  public let zoom: String = ""

  private let wearables: WearablesInterface
  private var stateContinuations: [AsyncStream<CameraState>.Continuation] = []

  init(wearables: WearablesInterface) {
    self.wearables = wearables
  }

  // MARK: - Camera protocol

  public func connect(nextFrame: @escaping (CIImage) -> Void) async {
    setState(.connected)
  }

  public func disconnect() async {
    setState(.disconnected(nil))
  }

  public func start() async {
    guard case .connected = state else { return }
    setState(.starting)

    // Start the OAuth registration flow (opens the Meta AI companion app).
    // Must dispatch to MainActor — the SDK opens a URL via UIApplication.
    do {
      try await Self.startRegistrationOnMain(wearables: wearables)
    } catch {
      print("[CameraMetaRegistration] startRegistration failed: \(String(describing: error))")
      if let regError = error as? RegistrationError {
        print("[CameraMetaRegistration] RegistrationError description: \(regError.description)")
      }
      setState(.disconnected(.noPermissions))
      return
    }

    // Wait for the registration callback to complete (up to 2 minutes).
    let registered = await waitForRegistration(timeout: .seconds(120))
    if registered {
      setState(.started)
    } else {
      setState(.disconnected(.noPermissions))
    }
  }

  public func stop() async {
    // No-op.
  }

  public func stateUpdates() -> AsyncStream<CameraState> {
    AsyncStream { cont in
      stateContinuations.append(cont)
      cont.yield(state)
    }
  }

  // MARK: - Private

  private func setState(_ newState: CameraState) {
    state = newState
    stateContinuations.forEach { $0.yield(newState) }
  }

  @MainActor
  private static func startRegistrationOnMain(wearables: WearablesInterface) async throws {
    try await wearables.startRegistration()
  }

  private func waitForRegistration(timeout: Duration) async -> Bool {
    await withTaskGroup(of: Bool.self) { group in
      group.addTask { [wearables] in
        for await registrationState in wearables.registrationStateStream() {
          if registrationState == .registered { return true }
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
}
