//
//  Camera.view.model.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI
internal import Combine
import AVFoundation

/// Describes what the camera view should display in its action area.
enum CameraAction {
  /// An interactive button the user can tap.
  case button(icon: String, label: String, hint: String)
  /// A non-interactive status message (with a spinner).
  case status(message: String)
}

@MainActor
class CameraViewModel: ObservableObject {
  @Published var camera: CameraSnapshot
  @Published var state: CameraState = .disconnected(.notInit)
  @Published var action: CameraAction = .status(message: "Initializing")
  
  private var stateTask: Task<Void, Never>?
  
  init(camera: CameraSnapshot) {
    self.camera = camera
  }
  
  func performAction() async {
    guard let device = camera.device else {
      switch state {
        case .disconnected(_):
          break
        default:
          state = .disconnected(.notInit)
      }
      return
    }

    if camera.isRegistration {
      // Registration camera: connect() drives the OAuth flow.
      if case .disconnected = state {
        await device.connect { _ in }
      }
      return
    }

    switch state {
      case .connected, .stopped:
        await device.start()
      case .started:
        await device.stop()
      default:
        break
    }
  }
  
  func startObservingState() async {
    guard let camera = self.camera.device else { return }
    stateTask?.cancel()
    stateTask = Task { [weak self] in
      guard let self else { return }
      for await state in await camera.stateUpdates() {
        self.state = state
        self.action = self.actionForState(state)
      }
    }
  }

  func stopObservingState() {
    stateTask?.cancel()
    stateTask = nil
  }

  // MARK: - Private

  private func actionForState(_ state: CameraState) -> CameraAction {
    if camera.isRegistration {
      return actionForRegistrationState(state)
    }
    return actionForCameraState(state)
  }

  private func actionForCameraState(_ state: CameraState) -> CameraAction {
    switch state {
      case .connected, .stopped:
        return .button(
          icon: "play.fill",
          label: String(localized: "Start"),
          hint: String(localized: "Start using camera")
        )
      case .started:
        return .button(
          icon: "stop.fill",
          label: String(localized: "Stop"),
          hint: String(localized: "Stop using camera")
        )
      case .connecting:
        return .status(message: String(localized: "Connecting to camera"))
      case .starting:
        return .status(message: String(localized: "Starting camera feed"))
      case .stopping:
        return .status(message: String(localized: "Stopping camera feed"))
      case .disconnecting:
        return .status(message: String(localized: "Disconnecting from camera"))
      case .disconnected(let error):
        if let error {
          return .status(message: error.rawValue)
        }
        return .status(message: String(localized: "Disconnected"))
      case .forceDisconnect:
        return .status(message: String(localized: "Connection lost"))
    }
  }

  private func actionForRegistrationState(_ state: CameraState) -> CameraAction {
    switch state {
      case .disconnected:
        return .button(
          icon: "link.badge.plus",
          label: String(localized: "Register"),
          hint: String(localized: "Opens Meta AI to register this app with your glasses")
        )
      case .connecting:
        return .status(
          message: String(localized: "Waiting for registration in Meta AI. Approve the request and return to this app.")
        )
      case .connected:
        return .status(
          message: String(localized: "Registration complete. Discovering cameras.")
        )
      default:
        return .status(message: state.stringValue)
    }
  }
}
