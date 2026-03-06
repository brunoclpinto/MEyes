//
//  Camera.view.model.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI
internal import Combine
import AVFoundation
import CoreGraphics

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
  private let tracker: BusApproachTracker?
  private let speaker = Speaker()
  
  init(camera: CameraSnapshot) {
    self.camera = camera

    let stage1 = try? YOLOModel(.bundle(name: "yolo26sINT8512x896"))
    let stage2 = try? YOLOModel(.bundle(name: "busInfoYolo26sINT8512x896"))
    self.tracker = BusApproachTracker(stage1Model: stage1, stage2Model: stage2)
  }
  
  func processFrame(_ frame: CGImage) async {
    guard let tracker else { return }
    do {
      let results = try await tracker.processFrame(frame)
      for bus in results {
        let number = bus.ocrText.leadingNaturalNumber()
        guard !number.isEmpty else { continue }
        speaker.speak(number)
      }
    } catch {
      print("[CameraViewModel] processFrame error: \(error)")
    }
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
      case .connected:
        return .button(
          icon: "link.badge.plus",
          label: String(localized: "Register"),
          hint: String(localized: "Opens Meta AI to register this app with your glasses")
        )
      case .connecting, .starting:
        return .status(
          message: String(localized: "Waiting for registration in Meta AI. Approve the request and return to this app.")
        )
      case .started:
        return .status(
          message: String(localized: "Registration complete. Discovering cameras.")
        )
      default:
        return .status(message: state.stringValue)
    }
  }
}
